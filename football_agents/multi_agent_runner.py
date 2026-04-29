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
        obs_refresh_every_ticks: int = 25,
        max_decisions_total: int = 200,
        max_wall_seconds: float = 300.0,
        target_wall_fps: float = 50.0,
        on_decision: Optional[Callable[[dict], None]] = None,
        on_tick: Optional[Callable[[dict], None]] = None,
        tick_stream_every_ticks: int = 5,
        on_frame: Optional[Callable[[object], None]] = None,
        frame_stream_every_ticks: int = 3,
        on_status: Optional[Callable[[str, dict], None]] = None,
        kickoff_wait_seconds: float = 8.0,
        stop_world_mode: bool = False,
        stop_world_timeout: float = 8.0,
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
        self._on_tick_cb = on_tick
        self._tick_stream_every = max(1, int(tick_stream_every_ticks))
        self._last_tick_stream: int = -10**9
        # Frame stream (for embedding gfootball's 3D render in the UI).
        # Only meaningful when env was constructed with render=True.
        self._on_frame_cb = on_frame
        self._frame_stream_every = max(1, int(frame_stream_every_ticks))
        self._last_frame_stream: int = -10**9
        # Status stream (phase events: "waiting_for_kickoff", "kickoff", ...).
        # The harness already wires "starting" / "running" / "finished"; the
        # runner now adds finer-grained phase events between starting and
        # running.
        self._on_status_cb = on_status
        # Hard cap on the pre-kickoff wait. env.tick is NOT advanced during
        # the wait, so this only delays wall-clock — never the game.
        self._kickoff_wait_seconds = max(0.0, float(kickoff_wait_seconds))

        self._decisions_completed: int = 0
        self._decision_log: list[dict] = []
        self._stop_flag = threading.Event()
        # Stop-world mode: pause env.step while all agents decide, then execute K ticks.
        self.stop_world_mode = stop_world_mode
        self._stop_world_timeout = max(1.0, float(stop_world_timeout))

    def request_stop(self) -> None:
        """External signal to break out of the run loop on the next tick.

        Thread-safe — `_stop_flag` is a `threading.Event`. The runner checks
        it once per env tick, so worst-case latency is one frame (~20ms at
        50 fps).
        """
        self._stop_flag.set()

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
                    motor_status=agent.last_skill_status,
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

    def _arm_fallback_for_stop_world(self, agent: PlayerAgent) -> None:
        """Arm fallback in stop-world mode with recent_llm_intent=None (blocker #1 fix).

        During freeze, env.tick does NOT advance, so the 100-tick guard on
        recent_llm_intent never expires → fallback would keep re-installing the
        last LLM skill forever. Passing None forces the position-based fallback
        to run instead.

        motor_status is passed so the guard can detect if the current skill is
        still running (in_progress) and avoid interrupting it. In Phase 3, this
        method is only called when step_motor() returned status != "in_progress",
        so motor_status will be "completed" or "failed" — the guard will allow
        the fallback to proceed.
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
                    recent_llm_intent=None,  # bypass 100-tick intent lock in stop-world
                    motor_status=agent.last_skill_status,
                )
                fallback_fn = get_fallback(agent.persona)
                fb_skill = fallback_fn(ctx)
        except Exception as e:
            logger.warning(
                "agent[pid=%d] stop-world fallback raised: %s; using HoldPosition",
                agent.player_id, e,
            )
            fb_skill = HoldPosition()
        agent.install_skill(
            fb_skill,
            tick=self.env.tick,
            raw_obs=self.env.raw_obs_for_slot(agent.slot),
            from_llm=False,
        )

    # ---- pre-kickoff wait --------------------------------------------

    def _emit_status(self, phase: str, payload: dict) -> None:
        """Best-effort status emit. Same signature as harness on_status."""
        if self._on_status_cb is None:
            return
        try:
            self._on_status_cb(phase, payload)
        except Exception as e:
            logger.warning("on_status callback raised: %s", e)

    def _wait_for_kickoff(self) -> None:
        """Block until every agent has produced its first LLM skill, or
        until self._kickoff_wait_seconds elapses — whichever comes first.

        IMPORTANT INVARIANTS:
          - env.step_actions is NEVER called here. env.tick stays at 0.
            The wait only consumes wall-clock, not game-clock.
          - Skills that arrive during the wait ARE installed via
            agent.install_skill(...) (from_llm=True), so the body is
            already armed with each LLM's first intent when env.step
            begins firing in the main loop.
          - For agents whose worker hasn't returned a skill by the
            timeout, we leave the per-persona fallback (already armed
            in run() step 1) in place. A single hung LLM does NOT
            block the rest of the match.
        """
        if self._kickoff_wait_seconds <= 0.0 or not self.agents:
            self._emit_status("kickoff", {
                "ready": len(self.agents),
                "total": len(self.agents),
                "wait_seconds": 0.0,
                "timed_out": False,
            })
            return

        n_total = len(self.agents)
        ready: set[int] = set()  # agent slots that have produced a skill
        wait_start = time.monotonic()
        last_progress_emit = -1

        # Initial status so the UI can show the banner immediately.
        self._emit_status("waiting_for_kickoff", {
            "ready": 0,
            "total": n_total,
            "timeout_seconds": self._kickoff_wait_seconds,
        })

        # Poll every 50ms — fast enough that ready-count progress feels
        # live, slow enough that we don't burn a CPU.
        poll_dt = 0.05
        while True:
            for a in self.agents:
                if a.slot in ready:
                    continue
                item = a.try_pop_skill()
                if item is None:
                    continue
                skill, llm_dt, obs_tick = item
                # Install the LLM's first decision so the body uses it
                # from env.tick=0 onward (replacing the fallback that
                # was armed in run() step 1).
                a.install_skill(
                    skill,
                    tick=self.env.tick,  # always 0 here
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
                ready.add(a.slot)

            # Emit progress when ready count changes.
            if len(ready) != last_progress_emit:
                self._emit_status("waiting_for_kickoff", {
                    "ready": len(ready),
                    "total": n_total,
                    "timeout_seconds": self._kickoff_wait_seconds,
                })
                last_progress_emit = len(ready)

            if len(ready) >= n_total:
                break
            if self._stop_flag.is_set():
                break
            if time.monotonic() - wait_start >= self._kickoff_wait_seconds:
                break
            time.sleep(poll_dt)

        elapsed = time.monotonic() - wait_start
        timed_out = len(ready) < n_total
        if timed_out:
            missing = [a.slot for a in self.agents if a.slot not in ready]
            logger.warning(
                "Kickoff after timeout, %d/%d agents ready (slots without "
                "first LLM decision, falling back: %s); waited %.2fs",
                len(ready), n_total, missing, elapsed,
            )
        else:
            logger.info(
                "All %d agents ready, kicking off. (waited %.2fs)",
                n_total, elapsed,
            )
        self._emit_status("kickoff", {
            "ready": len(ready),
            "total": n_total,
            "wait_seconds": elapsed,
            "timed_out": timed_out,
        })

    # ---- main loop ---------------------------------------------------

    def run(self) -> dict:
        """Run the simulation until env.done / max_decisions / max_wall_seconds."""
        if self.stop_world_mode:
            return self._run_stop_world()

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

        # 2b. WAIT FOR KICKOFF — block until every agent has produced its
        #     first LLM-decided skill, OR the wall-clock cap fires. While
        #     we wait, env.step is NEVER called → env.tick stays at 0 and
        #     the simulator does not advance. The fallback that was armed
        #     in step 1 is what would run if we didn't wait; the whole
        #     point of this phase is to install LLM intents BEFORE the
        #     first env.step so kickoff doesn't look identical every match.
        self._wait_for_kickoff()

        wall_start = time.monotonic()
        last_tick_wall = time.monotonic()
        # Initial last-push-tick; vary by agent so refresh cadence stays staggered
        # even after the startup jitter.
        # KEYED BY SLOT (not player_id) — slot is globally unique (0..n_total-1).
        # SYNCHRONIZED PUSH: all agents pushed on the same tick, so every LLM sees
        # a consistent global snapshot at the same moment. (Old code staggered the
        # init by `i * refresh / N` to spread API requests in time, but staggered
        # snapshots break team coordination — agents reasoning off slightly different
        # world states. Startup jitter in PlayerAgent.start() (50ms × i) still
        # spreads the initial API burst.)
        last_obs_push_tick: dict[int, int] = {
            a.slot: -self.obs_refresh_every_ticks for a in self.agents
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
                if self._stop_flag.is_set():
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

                # ---- 5c. Stream rendered 3D frame to UI (when render=True) ----
                if (self._on_frame_cb is not None
                        and self.env.tick - self._last_frame_stream >= self._frame_stream_every):
                    frame = self.env.latest_frame
                    if frame is not None:
                        self._last_frame_stream = self.env.tick
                        try:
                            self._on_frame_cb(frame)
                        except Exception as e:
                            logger.warning("on_frame callback raised: %s", e)

                # ---- 5b. Stream raw position snapshot to UI (live pitch view) ----
                if (self._on_tick_cb is not None
                        and self.env.tick - self._last_tick_stream >= self._tick_stream_every):
                    self._last_tick_stream = self.env.tick
                    try:
                        raw = self.env.raw_obs
                        snap = {
                            "env_tick": self.env.tick,
                            "left_team": [[float(p[0]), float(p[1])] for p in raw["left_team"]],
                            "right_team": [[float(p[0]), float(p[1])] for p in raw["right_team"]],
                            "ball": [float(raw["ball"][0]), float(raw["ball"][1])],
                            "ball_owned_team": int(raw.get("ball_owned_team", -1)),
                            "ball_owned_player": int(raw.get("ball_owned_player", -1)),
                            "score_left": int(raw.get("score", [0, 0])[0]),
                            "score_right": int(raw.get("score", [0, 0])[1]),
                            "game_mode": int(raw.get("game_mode", 0)),
                        }
                        self._on_tick_cb(snap)
                    except Exception as e:
                        logger.warning("on_tick callback raised: %s", e)

                # ---- 6. Cap wall-clock tick rate for game=wall alignment ----
                # With default pps=2 + 50 fps wall cap: 50*2*0.01 = 1.0 game
                # sec / wall sec exactly. 50 fps is smooth enough to not feel
                # choppy. Disable by constructing with target_wall_fps=0.
                #
                # DEADLINE-CARRY PACING (fixed 2026-04): the previous code
                # measured `elapsed = now - last_tick_wall` and reset
                # `last_tick_wall = now` AFTER sleep. On Linux time.sleep()
                # routinely overshoots by 0.5–2ms; resetting the baseline
                # after sleep BAKES that overshoot into every subsequent
                # tick — at 50 fps × 1ms overshoot = 50ms drift per second
                # (~5% slow). Verified on WSL: ratio measured 0.91 with
                # the old code instead of 1.00.
                #
                # Fix: track the next ABSOLUTE deadline. Each tick's target
                # is `last_tick_wall + target_tick_dt` regardless of how
                # long the previous tick / sleep took. Overshoot in tick N
                # is compensated by less sleep in tick N+1, so over time
                # the average pace stays exactly at target_wall_fps.
                #
                # If we're chronically slipping (work + sleep > tick_dt),
                # we still don't sleep, but we resync the deadline so we
                # don't accumulate unbounded debt that would later cause
                # a long no-sleep burst when load drops. Threshold: 3
                # tick periods of slip → declare hopeless and resync.
                if self._target_tick_dt > 0:
                    next_deadline = last_tick_wall + self._target_tick_dt
                    now = time.monotonic()
                    sleep_for = next_deadline - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                        last_tick_wall = next_deadline
                    elif sleep_for > -3.0 * self._target_tick_dt:
                        # Mild slip — let the deadline carry forward so
                        # the next-faster tick can catch up.
                        last_tick_wall = next_deadline
                    else:
                        # Hopeless slip (loop is structurally too slow for
                        # the configured fps); resync to now so we don't
                        # build unbounded debt.
                        last_tick_wall = now
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

    # ---- stop-world mode ------------------------------------------------

    def _run_stop_world(self) -> dict:
        """Stop-the-world mode: discrete decide → execute K-tick cycles.

        Each cycle:
          1. Push obs to ALL agents simultaneously (30ms stagger per agent
             to avoid rate-limit burst — blocker #2 fix).
          2. Wait up to stop_world_timeout seconds for ALL to return decisions.
             Timeout agents get a fallback (with recent_llm_intent=None —
             blocker #1 fix) rather than blocking the whole match.
          3. Execute obs_refresh_every_ticks env ticks with those decisions.

        The env.tick counter does NOT advance during phase 1-2, so agents
        reason on a stable world snapshot — no more "I decided to pass but
        the ball moved 10m while I was thinking".
        """
        # ---- Initial setup (mirrors the async run() preamble) ----
        for a in self.agents:
            self._arm_fallback_for(a)

        for i, a in enumerate(self.agents):
            obs = a.perceive(self.env.raw_obs_for_slot(a.slot), self.env.tick)
            a.push_observation(obs)
            a.start()
            if i < len(self.agents) - 1:
                time.sleep(0.05)

        self._wait_for_kickoff()

        wall_start = time.monotonic()
        last_tick_wall = time.monotonic()

        try:
            while True:
                # ---- Termination ----
                if self.env.done:
                    break
                if self._stop_flag.is_set():
                    break
                if self._decisions_completed >= self.max_decisions_total:
                    break
                if time.monotonic() - wall_start >= self.max_wall_seconds:
                    break

                # ---- Phase 1: Push obs to ALL agents (staggered for rate limit) ----
                for i, a in enumerate(self.agents):
                    raw = self.env.raw_obs_for_slot(a.slot)
                    obs = a.perceive(raw, self.env.tick)
                    a.push_observation(obs)
                    if i < len(self.agents) - 1:
                        time.sleep(0.03)  # 30ms → ~3 req/s burst not 10 simultaneous

                # ---- Phase 2: Wait for ALL decisions (or per-agent timeout) ----
                decided: set[int] = set()
                deadline = time.monotonic() + self._stop_world_timeout

                while len(decided) < len(self.agents):
                    if self._stop_flag.is_set() or time.monotonic() > deadline:
                        break
                    for a in self.agents:
                        if a.slot in decided:
                            continue
                        item = a.try_pop_skill()
                        if item is None:
                            continue
                        skill, llm_dt, obs_tick = item
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
                            "lag_ticks": 0,  # env didn't advance during freeze
                            "llm_seconds": llm_dt,
                            "skill": skill,
                        }
                        self._decision_log.append(log_entry)
                        if self._on_decision_cb is not None:
                            try:
                                self._on_decision_cb(log_entry)
                            except Exception as e:
                                logger.warning("on_decision callback raised: %s", e)
                        decided.add(a.slot)
                    if len(decided) < len(self.agents):
                        time.sleep(0.02)

                # Fallback for timed-out agents (blocker #1 fix: recent_llm_intent=None)
                for a in self.agents:
                    if a.slot not in decided:
                        logger.warning(
                            "stop-world: agent[pid=%d] timed out (%.1fs), arming fallback",
                            a.player_id, self._stop_world_timeout,
                        )
                        self._arm_fallback_for_stop_world(a)

                # ---- Phase 3: Execute K env ticks with current decisions ----
                last_tick_wall = time.monotonic()
                for _ in range(self.obs_refresh_every_ticks):
                    if self.env.done or self._stop_flag.is_set():
                        break
                    if (self._decisions_completed >= self.max_decisions_total
                            or time.monotonic() - wall_start >= self.max_wall_seconds):
                        break

                    actions: list[int] = [self._IDLE_ACTION] * self.n_slots
                    for a in self.agents:
                        raw = self.env.raw_obs_for_slot(a.slot)
                        action, status = a.step_motor(raw)
                        actions[a.slot] = action
                        if status != "in_progress":
                            self._arm_fallback_for_stop_world(a)

                    self.env.step_actions(actions)

                    # 3D frame stream
                    if (self._on_frame_cb is not None
                            and self.env.tick - self._last_frame_stream >= self._frame_stream_every):
                        frame = self.env.latest_frame
                        if frame is not None:
                            self._last_frame_stream = self.env.tick
                            try:
                                self._on_frame_cb(frame)
                            except Exception as e:
                                logger.warning("on_frame callback raised: %s", e)

                    # Position snapshot for live pitch view
                    if (self._on_tick_cb is not None
                            and self.env.tick - self._last_tick_stream >= self._tick_stream_every):
                        self._last_tick_stream = self.env.tick
                        try:
                            raw = self.env.raw_obs
                            snap = {
                                "env_tick": self.env.tick,
                                "left_team": [[float(p[0]), float(p[1])] for p in raw["left_team"]],
                                "right_team": [[float(p[0]), float(p[1])] for p in raw["right_team"]],
                                "ball": [float(raw["ball"][0]), float(raw["ball"][1])],
                                "ball_owned_team": int(raw.get("ball_owned_team", -1)),
                                "ball_owned_player": int(raw.get("ball_owned_player", -1)),
                                "score_left": int(raw.get("score", [0, 0])[0]),
                                "score_right": int(raw.get("score", [0, 0])[1]),
                                "game_mode": int(raw.get("game_mode", 0)),
                            }
                            self._on_tick_cb(snap)
                        except Exception as e:
                            logger.warning("on_tick callback raised: %s", e)

                    # Rate cap (deadline-carry pacing)
                    if self._target_tick_dt > 0:
                        next_deadline = last_tick_wall + self._target_tick_dt
                        now = time.monotonic()
                        sleep_for = next_deadline - now
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                            last_tick_wall = next_deadline
                        elif sleep_for > -3.0 * self._target_tick_dt:
                            last_tick_wall = next_deadline
                        else:
                            last_tick_wall = now

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
