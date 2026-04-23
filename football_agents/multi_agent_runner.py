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
        target_tick_hz: float = 10.0,
        on_decision: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        target_tick_hz: cap on how many env ticks per WALL second the main
            loop will produce. gfootball runs as fast as we feed it, so
            without a cap we run at ~58 ticks/s wall (way faster than real
            time). Real football pace is roughly 10 game ticks per game
            second, so target_tick_hz=10 makes 1 wall second ≈ 1 game
            second — the LLM's 2.5s thinking time then only advances the
            world by ~25 ticks (a real player's reaction window), not 145
            ticks (a full attacking move).
            Set to 0 or negative to disable the cap (legacy behavior).
            Tune higher if you want smoother render at the cost of faster
            game pace; lower for slower game (more LLM-relevant decisions).
        """
        # --- validate agent slots (silent corruption guard) ---
        slots = [a.slot for a in agents]
        if len(set(slots)) != len(slots):
            raise ValueError(f"duplicate agent slots: {slots}")
        if any(s < 0 or s >= env.n_controlled_left for s in slots):
            raise ValueError(
                f"agent slot out of range [0, {env.n_controlled_left}): {slots}"
            )
        # Sort agents by slot so action list assembly is deterministic.
        agents = sorted(agents, key=lambda a: a.slot)

        self.env = env
        self.agents = agents
        self.n_slots = env.n_controlled_left
        self._fallback_policy = fallback_policy or body_rest_state_fallback
        self.obs_refresh_every_ticks = obs_refresh_every_ticks
        self.max_decisions_total = max_decisions_total
        self.max_wall_seconds = max_wall_seconds
        self._target_tick_dt = (
            1.0 / target_tick_hz if target_tick_hz and target_tick_hz > 0 else 0.0
        )
        self._on_decision_cb = on_decision

        self._decisions_completed: int = 0
        self._decision_log: list[dict] = []
        self._stop_flag = threading.Event()

    # ---- per-agent helpers -------------------------------------------

    def _arm_fallback_for(self, agent: PlayerAgent) -> None:
        """Install body_rest_state_fallback for one agent. Main thread."""
        try:
            fb_obs = agent.perceive(self.env.raw_obs, self.env.tick)
            fb_skill = self._fallback_policy(fb_obs)
        except Exception as e:
            logger.warning(
                "agent[pid=%d] fallback raised: %s; using HoldPosition",
                agent.player_id, e,
            )
            fb_skill = HoldPosition()
        agent.install_skill(fb_skill)

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
            obs = a.perceive(self.env.raw_obs, self.env.tick)
            a.push_observation(obs)
            a.start()
            if i < len(self.agents) - 1:
                time.sleep(0.05)

        wall_start = time.monotonic()
        last_tick_wall = time.monotonic()
        # Initial last-push-tick; vary by agent so refresh cadence stays staggered
        # even after the startup jitter.
        last_obs_push_tick: dict[int, int] = {
            a.player_id: -((i * self.obs_refresh_every_ticks) // max(1, len(self.agents)))
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
                    a.install_skill(skill)
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
                    action, status = a.step_motor(self.env.raw_obs)
                    actions[a.slot] = action
                    if status != "in_progress":
                        # Skill finished/failed — re-arm with fallback so the
                        # body keeps doing something until next LLM intent.
                        self._arm_fallback_for(a)

                # ---- 4. One env tick with full action list ----
                self.env.step_actions(actions)

                # ---- 5. Periodically push fresh obs to each agent's worker ----
                for a in self.agents:
                    if self.env.tick - last_obs_push_tick[a.player_id] >= self.obs_refresh_every_ticks:
                        obs = a.perceive(self.env.raw_obs, self.env.tick)
                        a.push_observation(obs)
                        last_obs_push_tick[a.player_id] = self.env.tick

                # ---- 6. Cap env tick rate so wall time ≈ game time ----
                # gfootball runs as fast as we feed it; without this cap the
                # simulation flies past at ~58 ticks/s wall, meaning 2.5s of
                # LLM thinking = 145 ticks of game advance. With cap at 10
                # ticks/s wall, 2.5s LLM = 25 ticks (a realistic reaction
                # window). The simulation feels like real-time football.
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
