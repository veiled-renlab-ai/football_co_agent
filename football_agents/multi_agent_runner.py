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
from .fallbacks import FallbackContext, get_fallback
from .perception import Observation
from .player_agent import PlayerAgent
from .skills import DribbleToward, HoldPosition, MoveTo, Skill

logger = logging.getLogger(__name__)

FallbackPolicy = Callable[[Observation], Skill]


def body_rest_state_fallback(obs: Observation) -> Skill:
    """The motor cortex autopilot — what the body does between LLM decisions.

    This fallback carries basic game intelligence: when the LLM hasn't
    decided yet, the body still responds to the most urgent situations.
    Unlike a real footballer who always has intent, this covers the 2-4s
    gaps while the LLM is thinking.

    Priority order:
      1. OPPONENT has ball + is moving toward/away from our goal → SPRINT
         toward the carrier to apply pressure (chase reflex).
      2. BALL is loose (no carrier) → SPRINT toward the ball to contest it.
      3. TEAMMATE has ball + is moving toward opponent goal → maintain
         formation (don't cluster); slow walk toward appropriate formation
         position.
      4. DEFAULT → HoldPosition (stand and wait for next LLM decision).

    All SPRINT urgency is used intentionally: fallback only fires when the
    LLM is silent, so urgency is the body's best guess at engagement need.
    """
    import math

    ball = obs.ball()
    self_pos = obs.self_state.position

    opponent_with_ball: tuple[EntityView, float] | None = None
    for e in obs.opponents():
        if e.has_ball:
            opponent_with_ball = (e, e.distance)
            break

    if opponent_with_ball is not None:
        opp, dist = opponent_with_ball
        if opp.velocity is not None:
            speed = math.hypot(opp.velocity.x, opp.velocity.y)
            if speed > 0.001:
                return MoveTo(
                    target_x=float(opp.position.x),
                    target_y=float(opp.position.y),
                    urgency="sprint",
                )
        if dist > 0.05:
            return MoveTo(
                target_x=float(opp.position.x),
                target_y=float(opp.position.y),
                urgency="sprint",
            )

    if ball is not None and not any(e.has_ball for e in obs.perceived_entities):
        ball_speed = 0.0
        if ball.velocity is not None:
            ball_speed = math.hypot(ball.velocity.x, ball.velocity.y)
        if ball_speed > 0.0005 or ball.distance > 0.08:
            return MoveTo(
                target_x=float(ball.position.x),
                target_y=float(ball.position.y),
                urgency="sprint",
            )

    teammate_with_ball: EntityView | None = None
    for e in obs.teammates():
        if e.has_ball:
            teammate_with_ball = e
            break

    if teammate_with_ball is not None:
        if teammate_with_ball.velocity is not None:
            speed = math.hypot(
                teammate_with_ball.velocity.x, teammate_with_ball.velocity.y
            )
            if speed > 0.001:
                return HoldPosition()

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
        # Legacy override: pre-per-persona demos passed a global fallback.
        # When set, it short-circuits the per-persona registry in _arm_fallback_for.
        # New demos should leave this None so each agent gets its own fallback.
        self._legacy_fallback_policy = fallback_policy
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
        """Install the per-persona fallback for one agent. Main thread.

        Per-player fallbacks live in football_agents/fallbacks/ and are
        resolved via get_fallback(persona). The shared-guard pipeline
        (game_mode / ball-invisible / recent-LLM-intent / stamina) is
        already wrapped around each registered function.

        Legacy override: if the runner was constructed with an explicit
        fallback_policy (old Callable[[Observation], Skill] signature),
        that function is applied to ALL agents instead — backward compat
        for demos that pre-date the per-persona registry.
        """
        try:
            raw = self.env.raw_obs_for_slot(agent.slot)
            fb_obs = agent.perceive(raw, self.env.tick)
            if self._legacy_fallback_policy is not None:
                fb_skill = self._legacy_fallback_policy(fb_obs)
            else:
                ctx = FallbackContext(
                    persona=agent.persona,
                    obs=fb_obs,
                    recent_llm_intent=agent.get_recent_llm_intent(
                        current_tick=self.env.tick, window_ticks=100,
                    ),
                )
                fallback_fn = get_fallback(agent.persona)
                fb_skill = fallback_fn(ctx)
        except Exception as e:
            logger.warning(
                "agent[pid=%d] fallback raised: %s; using HoldPosition",
                agent.player_id, e,
            )
            fb_skill = HoldPosition()
        # Pass from_llm=False so PlayerAgent does NOT update last_llm_intent
        # (the fallback's own choice shouldn't masquerade as an LLM decision
        # and then veto later fallbacks via the shared guard).
        agent.install_skill(
            fb_skill,
            tick=self.env.tick,
            raw_obs=self.env.raw_obs_for_slot(agent.slot),
            from_llm=False,
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
                    # from_llm=True updates the recent-LLM-intent so the next
                    # fallback invocation defers to this decision for ~2s.
                    a.install_skill(
                        skill,
                        tick=self.env.tick,
                        raw_obs=self.env.raw_obs_for_slot(a.slot),
                        from_llm=True,
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
