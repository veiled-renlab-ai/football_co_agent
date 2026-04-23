"""PlayerAgent — the per-player container for the multi-agent runtime.

ONE PlayerAgent per controlled footballer on the pitch. Bundles everything
that's "this specific player" into a single, technically-isolated object:

  - persona              (immutable identity)
  - LLMPlayer            (per-player brain + 3-tier memory)
  - EgocentricFilter     (per-player perception state)
  - current MotorController + last_skill_name/status (per-player motor state)
  - obs_queue / skill_queue (per-player worker thread channels, single-slot)
  - dedicated worker thread (only calls LLMPlayer.choose_skill, nothing else)

Strict isolation contract (per user requirement: "memory etc. must not cross"):
  - Each PlayerAgent's brain / perception / motor state is touched ONLY by
    the main thread (filter, motor step, install_skill) AND its own worker
    thread (which exclusively calls LLMPlayer.choose_skill — no env access,
    no other agent state, no shared mutables).
  - No state crosses between PlayerAgent instances. Memory lists, perception
    facing direction, tracked entities — all per-instance.
  - Acceptable shared resources: LLMClient (HTTP, OpenAI SDK is thread-safe);
    gfootball env (single-threaded, accessed only by main thread); future
    TeamMessageBus (Phase 5c, partitioned by team).
  - Anything else shared = bug.

Threading model (per user requirement: "each agent its own thread"):
  - One dedicated `threading.Thread` per PlayerAgent. NOT a ThreadPoolExecutor.
  - This is N OS threads for N agents (5v5 = 10, 11v11 = 22). Memory and
    GIL impact verified acceptable in design docs.
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Optional

from .agent import LLMPlayer
from .llm_client import LLMClient
from .motor import MotorController, SkillStatus, make_controller
from .perception import EgocentricFilter, Observation, Role, Team
from .prompts import PlayerPersona
from .skills import ScanBehind, Skill, Track

logger = logging.getLogger(__name__)

TeamSide = str  # "left" | "right"


def _replace_in_queue(q: Queue, item) -> None:
    """Single-slot queue with replace-stale semantics — newest item wins,
    older item is silently dropped. Used so a busy LLM doesn't pile up
    stale observations / skills.
    """
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass


class PlayerAgent:
    """One LLM-driven player on the pitch. Self-contained, isolated.

    Constructed by MultiAgentRunner; lives for one episode; cleaned up on stop().

    Public API (called from MultiAgentRunner main thread):
      - perceive(raw_obs, tick) -> Observation     # build egocentric view
      - install_skill(skill)                       # swap in new motor controller
      - step_motor(raw_obs) -> (action_id, status) # advance current controller one tick
      - push_observation(obs)                      # send fresh obs to worker thread
      - try_pop_skill() -> Optional[(skill, dt, tick)]  # drain decided skill
      - start() / stop()                           # worker thread lifecycle
    """

    def __init__(
        self,
        *,
        slot: int,
        player_id: int,
        team_side: TeamSide,
        role: Role,
        persona: PlayerPersona,
        llm_client: LLMClient,
    ) -> None:
        # Identity (immutable for the agent's lifetime)
        self.slot = slot                # gfootball action-list index (0..N-1)
        self.player_id = player_id      # gfootball team-array index (== slot in multi-agent mode)
        self.team_side = team_side      # "left" | "right"
        self.role = role
        self.persona = persona

        # Brain — per-player, owns its own 3-tier memory.
        # LLMClient is shared across all agents (thread-safe HTTP client).
        self.llm_player = LLMPlayer(
            player_id=player_id,
            role=role,
            llm_client=llm_client,
            persona=persona,
        )

        # Perception — per-player, owns _last_facing_deg / _tracked_entity_ids.
        # team_side ("left"/"right") -> team label ("team_a"/"team_b") for filter.
        team_label: Team = "team_a" if team_side == "left" else "team_b"
        self.perception = EgocentricFilter(
            player_id=player_id,
            team=team_label,
            role=role,
        )

        # Motor state — per-player, set by install_skill().
        self.current_controller: Optional[MotorController] = None
        self.last_skill_name: Optional[str] = None
        self.last_skill_status: Optional[SkillStatus] = None

        # Worker thread channels (per-agent, isolated).
        self.obs_queue: "Queue[Observation]" = Queue(maxsize=1)
        self.skill_queue: "Queue[tuple[Skill, float, int]]" = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Main-thread API: called by MultiAgentRunner each env tick
    # ------------------------------------------------------------------

    def perceive(self, raw_obs: dict, tick: int) -> Observation:
        """Build this agent's egocentric Observation from the raw god-view.

        Filter runs on MAIN THREAD because EgocentricFilter._last_facing_deg
        is a per-instance mutation and we want single-threaded access to it.
        Filter is cheap (~µs) so cost is negligible.
        """
        obs = self.perception.filter(raw_obs, tick)
        obs.last_skill = self.last_skill_name
        obs.last_skill_status = self.last_skill_status
        return obs

    def install_skill(self, skill: Skill) -> None:
        """Replace current motor controller with one for the given skill.

        Special-cases routed to perception layer (no body motion):
          - Track:      perception.track_entity(eid) — force-include in FOV
          - ScanBehind: perception.scan_behind() — bypass FOV cone for next obs
        These were originally motor actions (Track moved a phantom direction;
        ScanBehind pressed LEFT for one tick to physically turn). Both caused
        visible jerks in the render. Now they're pure perception flips: the
        body keeps doing whatever sticky state it had, only the brain's view
        of the world expands for that one decision tick.
        """
        if isinstance(skill, Track):
            try:
                self.perception.track_entity(skill.entity_id)
            except Exception as e:
                logger.warning(
                    "agent[pid=%d] track_entity(%s) failed: %s",
                    self.player_id, skill.entity_id, e,
                )
        elif isinstance(skill, ScanBehind):
            try:
                self.perception.scan_behind()
            except Exception as e:
                logger.warning(
                    "agent[pid=%d] scan_behind() failed: %s",
                    self.player_id, e,
                )
        self.current_controller = make_controller(
            skill, team_side=self.team_side, player_id=self.player_id,
        )
        self.last_skill_name = type(skill).__name__
        self.last_skill_status = "in_progress"

    def step_motor(self, raw_obs: dict) -> tuple[int, SkillStatus]:
        """Advance current motor controller by one env tick. Main thread.

        Returns (action_id, status). If no controller is installed yet
        (haven't received first LLM decision and no fallback armed), returns
        (IDLE, in_progress) — runner is responsible for arming a fallback.
        """
        if self.current_controller is None:
            return 0, "in_progress"  # gfootball IDLE
        action, status = self.current_controller.step(raw_obs)
        self.last_skill_status = status
        return action, status

    def push_observation(self, obs: Observation) -> None:
        """Send a fresh observation to this agent's worker thread.
        Replace-stale: if worker hasn't consumed the previous obs, it's
        silently overwritten (newest obs wins).
        """
        _replace_in_queue(self.obs_queue, obs)

    def try_pop_skill(self) -> Optional[tuple[Skill, float, int]]:
        """Non-blocking drain of decided skill from worker.
        Returns (skill, llm_dt_seconds, obs_tick_when_decided) or None.
        """
        try:
            return self.skill_queue.get_nowait()
        except Empty:
            return None

    # ------------------------------------------------------------------
    # Worker thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the dedicated worker thread for this agent's LLM calls."""
        if self._thread is not None:
            raise RuntimeError(
                f"PlayerAgent[pid={self.player_id}] already started"
            )
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"agent-pid{self.player_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Signal worker to stop and wait briefly for it to exit.
        Idempotent; safe to call before start() (no-op).
        """
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        """Worker thread body.

        ONLY this thread touches self.llm_player. Main thread reads
        self.skill_queue (queue.Queue is thread-safe). No other state
        crosses the thread boundary.

        Polls obs_queue with 0.2s timeout so stop_event is checked at least
        every 200ms — clean shutdown without busy-waiting.
        """
        while not self.stop_event.is_set():
            try:
                obs = self.obs_queue.get(timeout=0.2)
            except Empty:
                continue
            t0 = time.monotonic()
            try:
                skill = self.llm_player.choose_skill(obs)
            except Exception as e:
                logger.warning(
                    "agent[pid=%d] LLM call raised: %s — agent will keep "
                    "running its current motor controller (fallback re-arms "
                    "automatically when current skill completes)",
                    self.player_id, e,
                )
                continue
            llm_dt = time.monotonic() - t0
            _replace_in_queue(self.skill_queue, (skill, llm_dt, obs.tick))
