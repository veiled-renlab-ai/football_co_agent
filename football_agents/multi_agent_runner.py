"""MultiAgentRunner — coordinates N PlayerAgents inside one gfootball env.

Replaces AsyncRunner for multi-agent mode (N >= 1; works at N=1 for
regression testing against the old single-agent path).

Threading model (per user requirement):
  - Main thread: env tick loop. Owns all gfootball / raw_obs access.
    Per tick:
      1. drain each agent's skill_queue (non-blocking)
      2. for each agent: step its motor controller -> (action_id, status);
         if status != "in_progress", re-arm with body_rest_state_fallback
      3. assemble action list (length = env.n_controlled_left, IDLE for
         slots without an agent assigned), call env.step_actions(actions)
      4. periodically push fresh per-agent egocentric Observation to each
         agent's obs_queue (filter runs on main thread for thread-safety)
  - Worker threads: N dedicated threads, one per PlayerAgent. Each ONLY
    calls LLMPlayer.choose_skill (HTTP). No env / other-agent access.

Strict per-agent isolation (per user requirement: "memory must not cross"):
  - Each PlayerAgent owns its brain / perception / motor state / queues.
  - Shared resources: LLMClient (thread-safe HTTP), gfootball env (main-
    thread only), runner-level decision log (main-thread mutation only).

Backward compat:
  - At N=1 (single PlayerAgent at slot=primary_player_slot of an env with
    n_controlled_left=2), behavior should match the legacy AsyncRunner.
    Used as the regression target for v0.5a-encapsulated.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .env import FootballEnvAdapter
from .perception import Observation
from .player_agent import PlayerAgent
from .skills import DribbleToward, HoldPosition, MoveTo, Skill

logger = logging.getLogger(__name__)

FallbackPolicy = Callable[[Observation], Skill]


def body_rest_state_fallback(obs: Observation) -> Skill:
    """The motor cortex autopilot — what the body does between LLM decisions.

    Identical to the v0.walk-fallback policy from async_runner.py. Lifted
    here so MultiAgentRunner doesn't depend on AsyncRunner's module.

    All branches use `walk` urgency (~17% jog speed via direction throttling
    in motor.py) so the body drifts deliberately during the 2-4s LLM gap.

      • has_ball   → walk-dribble toward midfield on current y-lane
      • sees ball  → walk 40% of the way toward the ball
      • no ball    → HoldPosition: stop and look around
    """
    if obs.self_state.has_ball:
        self_y = float(obs.self_state.position.y)
        return DribbleToward(target_x=0.6, target_y=self_y, urgency="walk")
    ball = obs.ball()
    if ball is not None:
        self_pos = obs.self_state.position
        target_x = float(self_pos.x) + 0.4 * (float(ball.position.x) - float(self_pos.x))
        target_y = float(self_pos.y) + 0.4 * (float(ball.position.y) - float(self_pos.y))
        return MoveTo(target_x=target_x, target_y=target_y, urgency="walk")
    return HoldPosition()


class MultiAgentRunner:
    """Coordinator for N PlayerAgents inside one gfootball env.

    Construction:
        env = FootballEnvAdapter(scenario=..., n_controlled_left=N, ...)
        agents = [PlayerAgent(slot=i, player_id=i, ...) for i in slots]
        runner = MultiAgentRunner(env, agents)
        runner.run()
    """

    # gfootball atomic action id for IDLE (slots without an agent assigned).
    _IDLE_ACTION = 0

    def __init__(
        self,
        env: FootballEnvAdapter,
        agents: list[PlayerAgent],
        *,
        fallback_policy: Optional[FallbackPolicy] = None,
        obs_refresh_every_ticks: int = 4,
        max_decisions_total: int = 200,
        max_wall_seconds: float = 300.0,
        target_wall_fps: float = 50.0,
        on_decision: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Game-time pace is controlled by gfootball's `physics_steps_per_frame`
        config (set in FootballEnvAdapter). gfootball has PHYSICS_STEPS_PER_SECOND=100
        internal physics ticks per game second; physics_steps_per_frame=N
        means each env.step advances the game by N/100 seconds.

        For the visible clock (HUD in render) to match wall time:
          wall_fps * physics_steps_per_frame * 0.01 sec = 1.0
          → wall_fps * pps = 100
          e.g. pps=2 + wall_fps=50 → exactly 1.0x real-time.

        `target_wall_fps` caps the main env-step loop to this wall-clock rate
        via a small sleep after env.step_actions. Default 50 Hz pairs with
        env's default physics_steps_per_frame=2 for exact real-time pacing.
        Render is still at this rate (one render per env.step), but 50 fps
        is smooth to the eye (vs the previously-tried 10 fps which felt choppy).

        Set to 0 to disable the cap (env runs as fast as it can; with the
        default pps=2 that's ~1.16x real-time on most machines).
        """
        # --- validate agent slots (silent corruption guard) ---
        slots = [a.slot for a in agents]
        if len(set(slots)) != len(slots):
            raise ValueError(f"duplicate agent slots: {slots}")
        n_total = env.n_controlled_total
        if any(s < 0 or s >= n_total for s in slots):
            raise ValueError(
                f"agent slot out of range [0, {n_total}) "
                f"(L={env.n_controlled_left}, R={env.n_controlled_right}): {slots}"
            )
        # Sort agents by slot so action list assembly is deterministic.
        agents = sorted(agents, key=lambda a: a.slot)

        self.env = env
        self.agents = agents
        self.n_slots = n_total
        self._fallback_policy = fallback_policy or body_rest_state_fallback
        self.obs_refresh_every_ticks = obs_refresh_every_ticks
        self.max_decisions_total = max_decisions_total
        self.max_wall_seconds = max_wall_seconds
        self._target_tick_dt = (
            1.0 / target_wall_fps if target_wall_fps and target_wall_fps > 0 else 0.0
        )
        self._on_decision_cb = on_decision

        self._decisions_completed: int = 0
        self._decision_log: list[dict] = []
        self._stop_flag = threading.Event()

    # ---- per-agent helpers -------------------------------------------

    def _arm_fallback_for(self, agent: PlayerAgent) -> None:
        """Install body_rest_state_fallback for one agent. Main thread."""
        try:
            fb_obs = agent.perceive(self.env.raw_obs_for_slot(agent.slot), self.env.tick)
            fb_skill = self._fallback_policy(fb_obs)
        except Exception as e:
            logger.warning(
                "agent[pid=%d] fallback raised: %s; using HoldPosition",
                agent.player_id, e,
            )
            fb_skill = HoldPosition()
        # Pass tick + raw_obs so Call (if ever in fallback) can post to bus.
        # Fallback policy doesn't currently produce Call, but pass anyway for
        # consistency — and so future fallback variants don't silently break.
        agent.install_skill(
            fb_skill, tick=self.env.tick, raw_obs=self.env.raw_obs_for_slot(agent.slot),
        )

    # ---- main loop ---------------------------------------------------

    def run(self) -> dict:
        """Run the simulation until env.done / max_decisions / max_wall_seconds."""

        # 1. Initial fallback controllers so every agent has SOMETHING running
        #    from tick 0 (else step_motor returns IDLE which is fine but
        #    matches v0.walk-fallback's behavior to arm fallback first).
        for a in self.agents:
            self._arm_fallback_for(a)

        # 2. Seed each worker with its first observation, then start threads.
        #    Startup jitter (50ms × i) spreads the initial LLM burst so 22
        #    agents don't all hit the API in the same instant on tick 0.
        for i, a in enumerate(self.agents):
            obs = a.perceive(self.env.raw_obs_for_slot(a.slot), self.env.tick)
            a.push_observation(obs)
            a.start()
            if i < len(self.agents) - 1:
                time.sleep(0.05)

        wall_start = time.monotonic()
        last_tick_wall = time.monotonic()
        # Initial last-push-tick; vary by agent so refresh cadence stays staggered
        # even after the startup jitter.
        # KEYED BY SLOT (not player_id) — player_id is per-team-relative
        # (left team has pids 0-4, right team also has pids 0-4), so keying
        # by player_id would let blue and red collide on the same key. The
        # team that iterates first wins every push and the other team's obs
        # queue never refreshes → only first decision ever fires.
        # slot is globally unique (0..n_total-1).
        last_obs_push_tick: dict[int, int] = {
            a.slot: -((i * self.obs_refresh_every_ticks) // max(1, len(self.agents)))
            for i, a in enumerate(self.agents)
        }

        try:
            while True:
                # ---- 1. Drain decided skills from each agent's worker ----
                for a in self.agents:
                    item = a.try_pop_skill()
                    if item is None:
                        continue
                    skill, llm_dt, obs_tick = item
                    # Pass tick + raw_obs so Call skills can post to TeamMessageBus
                    # (PlayerAgent.install_skill needs them to build a Message;
                    # without them, Call silently no-ops with a logged warning).
                    a.install_skill(
                        skill, tick=self.env.tick, raw_obs=self.env.raw_obs_for_slot(a.slot),
                    )
                    self._decisions_completed += 1
                    log_entry = {
                        "decision": self._decisions_completed,
                        "player_id": a.player_id,
                        "slot": a.slot,
                        "env_tick": self.env.tick,
                        "obs_tick": obs_tick,
                        "lag_ticks": self.env.tick - obs_tick,
                        "llm_seconds": llm_dt,
                        "skill": skill,
                    }
                    self._decision_log.append(log_entry)
                    if self._on_decision_cb is not None:
                        try:
                            self._on_decision_cb(log_entry)
                        except Exception as e:
                            logger.warning("on_decision callback raised: %s", e)

                # ---- 2. Termination ----
                if self.env.done:
                    break
                if self._decisions_completed >= self.max_decisions_total:
                    break
                if time.monotonic() - wall_start >= self.max_wall_seconds:
                    break

                # ---- 3. Step every agent's motor controller; assemble actions ----
                actions: list[int] = [self._IDLE_ACTION] * self.n_slots
                for a in self.agents:
                    action, status = a.step_motor(self.env.raw_obs_for_slot(a.slot))
                    actions[a.slot] = action
                    if status != "in_progress":
                        # Skill finished/failed — re-arm with fallback so the
                        # body keeps doing something until next LLM intent.
                        self._arm_fallback_for(a)

                # ---- 4. One env tick with full action list ----
                self.env.step_actions(actions)

                # ---- 5. Periodically push fresh obs to each agent's worker ----
                for a in self.agents:
                    if self.env.tick - last_obs_push_tick[a.slot] >= self.obs_refresh_every_ticks:
                        obs = a.perceive(self.env.raw_obs_for_slot(a.slot), self.env.tick)
                        a.push_observation(obs)
                        last_obs_push_tick[a.slot] = self.env.tick

                # ---- 6. Cap wall-clock tick rate for game=wall alignment ----
                # With default pps=2 + 50 fps wall cap: 50*2*0.01 = 1.0 game
                # sec / wall sec exactly. 50 fps is smooth enough to not feel
                # choppy. Disable by constructing with target_wall_fps=0.
                if self._target_tick_dt > 0:
                    elapsed = time.monotonic() - last_tick_wall
                    sleep_for = self._target_tick_dt - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    last_tick_wall = time.monotonic()
        finally:
            self._stop_flag.set()
            for a in self.agents:
                a.stop(timeout=0.5)

        return {
            "wall_seconds": time.monotonic() - wall_start,
            "env_ticks": self.env.tick,
            "decisions_total": self._decisions_completed,
            "n_agents": len(self.agents),
            "cumulative_reward": self.env.cumulative_reward,
            "log": self._decision_log,
        }
