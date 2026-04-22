"""AsyncRunner — decouples LLM decision-making from env ticking.

Standard game-AI pattern: the render/physics loop runs continuously on the
main thread; the LLM "brain" runs in a background thread and atomically swaps
in new skills as it produces them. The 3D window stays smooth at gfootball's
native tick rate; LLM latency only affects how stale the agent's tactical
intent is, not the visual fluidity.

Thread model:
  - **Main thread**: env.reset(), env.step_action(), env.observe(), agent skill
    dispatch via MotorController. ALL gfootball calls happen here (the C++
    engine is not safe to call from multiple threads).
  - **LLM worker thread**: pulls latest Observation from a queue, calls
    LLMPlayer.choose_skill, pushes the resulting Skill back through another
    queue. Pure Python + HTTP — no gfootball calls.

Communication:
  - obs_queue (main → worker, maxsize=1):  freshest Observation snapshot.
    If worker hasn't consumed yet when main has a new one, replace it.
  - skill_queue (worker → main, maxsize=1):  newest decided Skill. Same
    replace-if-stale semantics.

This is how OpenAI Realtime, Inworld AI, Convai NPC, etc. all handle async LLM
in real-time game loops.
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Callable, Optional

from .agent import LLMPlayer
from .env import FootballEnvAdapter
from .motor import MotorController, make_controller
from .perception import Observation
from .skills import DribbleToward, HoldPosition, MoveTo, Skill

# A fallback policy is a pure-Python rule that picks a sensible default
# Skill from the current Observation. Used between LLM decisions so the
# player never just stands there waiting.
FallbackPolicy = Callable[[Observation], Skill]


def body_rest_state_fallback(obs: Observation) -> Skill:
    """The 'motor cortex autopilot' — what the body does while the brain
    (LLM) thinks. Slow, conservative, conserves stamina, keeps options open.
    Maps the LLM's ~1-2s thinking gap to the rest-state behavior real players
    exhibit between deliberate tactical moments.

    Rule (≤2 branches by design — anything more is a state machine):
      - Have ball → slow dribble toward opponent goal (jog, not sprint)
      - No ball  → jog toward the ball (or toward midfield if ball not visible)

    The LLM provides BURSTS of explicit intent (sprint, shoot, press, pass,
    specific runs) on top of this default rhythm.
    """
    if obs.self_state.has_ball:
        return DribbleToward(target_x=0.95, target_y=0.0, urgency="jog")
    ball = obs.ball()
    if ball is not None:
        return MoveTo(
            target_x=float(ball.position.x),
            target_y=float(ball.position.y),
            urgency="jog",
        )
    # Ball not visible — jog forward toward midfield to stay involved
    return MoveTo(target_x=0.5, target_y=0.0, urgency="jog")

logger = logging.getLogger(__name__)


class AsyncRunner:
    """Run env continuously while LLM decides in background.

    Usage:
        runner = AsyncRunner(env, agent, on_decision=lambda log: print(log))
        result = runner.run()
    """

    def __init__(
        self,
        env: FootballEnvAdapter,
        agent: LLMPlayer,
        *,
        fallback_policy: Optional[FallbackPolicy] = None,
        obs_refresh_every_ticks: int = 4,
        max_decisions: int = 60,
        max_wall_seconds: float = 120.0,
        on_decision: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.obs_refresh_every_ticks = obs_refresh_every_ticks
        self.max_decisions = max_decisions
        self.max_wall_seconds = max_wall_seconds
        self._on_decision_cb = on_decision
        self._fallback_policy = fallback_policy  # None = use HoldPosition

        self._obs_queue: "Queue[Observation]" = Queue(maxsize=1)
        self._skill_queue: "Queue[tuple[Skill, float, int]]" = Queue(maxsize=1)
        self._stop_flag = threading.Event()

        # Main-thread state
        self._current_controller: Optional[MotorController] = None
        self._decisions_completed: int = 0
        self._decision_log: list[dict] = []

    # ------------------------------------------------------------------
    # Worker thread (LLM)
    # ------------------------------------------------------------------

    def _llm_worker(self) -> None:
        while not self._stop_flag.is_set():
            try:
                obs = self._obs_queue.get(timeout=0.2)
            except Empty:
                continue

            t0 = time.monotonic()
            try:
                skill = self.agent.choose_skill(obs)
            except Exception as e:
                logger.warning("LLM call raised: %s", e)
                continue
            llm_dt = time.monotonic() - t0

            # Replace any stale skill the main thread hasn't consumed
            self._replace_in_queue(self._skill_queue, (skill, llm_dt, obs.tick))

    @staticmethod
    def _replace_in_queue(q: Queue, item) -> None:
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _make_fallback_controller(self) -> MotorController:
        """Build the rest-state controller for when no LLM skill is in flight."""
        policy = self._fallback_policy or body_rest_state_fallback
        try:
            fb_obs = self.env.observe()
            fb_skill = policy(fb_obs)
        except Exception as e:
            logger.warning("fallback policy raised: %s; using HoldPosition", e)
            fb_skill = HoldPosition()
        return make_controller(fb_skill, team_side=self.env.team_side)

    def run(self) -> dict:
        # Default skill so env can tick before the first LLM decision lands.
        # Use the fallback policy if provided — keeps player active from tick 0.
        self._current_controller = self._make_fallback_controller()
        self.env.set_last_skill(
            type(self._current_controller.skill).__name__, "in_progress"
        )

        # Seed the worker with the very first observation
        self._replace_in_queue(self._obs_queue, self.env.observe())

        worker = threading.Thread(target=self._llm_worker, daemon=True)
        worker.start()

        wall_start = time.monotonic()
        last_obs_push_tick = 0

        try:
            while True:
                # 1. Consume any newly-decided skill
                try:
                    skill, llm_dt, obs_tick = self._skill_queue.get_nowait()
                    self._swap_in_skill(skill, llm_dt, obs_tick)
                except Empty:
                    pass

                # 2. Termination
                if self.env.done:
                    break
                if self._decisions_completed >= self.max_decisions:
                    break
                if time.monotonic() - wall_start >= self.max_wall_seconds:
                    break

                # 3. One env tick using current controller
                action, status = self._current_controller.step(self.env.raw_obs)
                self.env.step_action(action)
                if status != "in_progress":
                    self.env.set_last_skill(
                        type(self._current_controller.skill).__name__, status
                    )
                    # Skill done — fall back to scripted policy (or HoldPosition)
                    self._current_controller = self._make_fallback_controller()

                # 4. Periodically push fresh obs to the worker
                if self.env.tick - last_obs_push_tick >= self.obs_refresh_every_ticks:
                    self._replace_in_queue(self._obs_queue, self.env.observe())
                    last_obs_push_tick = self.env.tick
        finally:
            self._stop_flag.set()
            worker.join(timeout=2.0)

        return {
            "wall_seconds": time.monotonic() - wall_start,
            "env_ticks": self.env.tick,
            "decisions": self._decisions_completed,
            "cumulative_reward": self.env.cumulative_reward,
            "log": self._decision_log,
        }

    def _swap_in_skill(self, skill: Skill, llm_dt: float, obs_tick: int) -> None:
        self._current_controller = make_controller(
            skill, team_side=self.env.team_side
        )
        self.env.set_last_skill(type(skill).__name__, "in_progress")
        self._decisions_completed += 1

        log = {
            "decision": self._decisions_completed,
            "env_tick": self.env.tick,
            "obs_tick": obs_tick,
            "lag_ticks": self.env.tick - obs_tick,
            "llm_seconds": llm_dt,
            "skill": skill,
        }
        self._decision_log.append(log)
        if self._on_decision_cb is not None:
            try:
                self._on_decision_cb(log)
            except Exception as e:
                logger.warning("on_decision callback raised: %s", e)
