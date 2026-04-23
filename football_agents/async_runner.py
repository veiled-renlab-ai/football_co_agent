"""AsyncRunner — env ticks at native 30fps on main thread, LLM decides
in a background thread. The 'body rest-state' fallback fills the LLM
thinking gap with the most basic real-player default (slow dribble or jog).

Maps to real cognition:
  - Motor cortex (per env tick): runs autopilot routines (jog, slow dribble)
  - Prefrontal cortex (LLM, ~1-2s): fires occasional tactical bursts
    (sprint, shoot, press, pass, specific runs)

The fallback is NOT a state machine substitute for LLM decisions — it is
the body's default rest behavior while the brain is between deliberate
thoughts. Approved as the only allowed fallback (see feedback memory).
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
from .skills import DribbleToward, HoldPosition, MoveTo, Skill, Track

logger = logging.getLogger(__name__)

FallbackPolicy = Callable[[Observation], Skill]


def body_rest_state_fallback(obs: Observation) -> Skill:
    """The motor cortex autopilot — what the body does between LLM decisions.

    Designed to feel like a real footballer THINKING-while-moving during
    the 2-4s LLM gap, not sprinting on autopilot. All branches use 'walk'
    urgency (~25% jog speed via direction throttling in motor.py) so the
    body drifts slowly enough that the brain's deliberate decisions still
    look like the dominant force.

    Three branches (still simple, still 3 cases — no rules expansion):
      • has_ball   → walk-dribble toward midfield on current y-lane
                     (don't auto-commit to a goal-bound sprint)
      • sees ball  → walk 40% of the way toward the ball
                     (drift toward, don't desperate-chase)
      • no ball    → HoldPosition: stop and look around
                     (the most natural "I'm thinking" footballer behavior)
    """
    if obs.self_state.has_ball:
        # Carry ball slowly toward midfield, staying on current y-lane.
        # Halfway-line target (0.6) keeps options open vs. committing to goal.
        self_y = float(obs.self_state.position.y)
        return DribbleToward(target_x=0.6, target_y=self_y, urgency="walk")

    ball = obs.ball()
    if ball is not None:
        # Drift 40% of the vector toward the ball — deliberate, not desperate.
        self_pos = obs.self_state.position
        target_x = float(self_pos.x) + 0.4 * (float(ball.position.x) - float(self_pos.x))
        target_y = float(self_pos.y) + 0.4 * (float(ball.position.y) - float(self_pos.y))
        return MoveTo(target_x=target_x, target_y=target_y, urgency="walk")

    # No ball visible — the most thinking-looking thing a real player does
    # is stop and scan the field. HoldPosition releases sticky direction;
    # momentum decays and the body settles.
    return HoldPosition()


class AsyncRunner:
    """Run env continuously while LLM decides in background.

    Thread model:
      - Main thread: env.reset(), env.step_action(), env.observe(), motor.step
        (all gfootball calls — single-threaded contract)
      - Worker thread: only LLM HTTP calls (no gfootball)

    Communication via two single-slot queues (replace-stale semantics).
    """

    def __init__(
        self,
        env: FootballEnvAdapter,
        agent: LLMPlayer,
        *,
        fallback_policy: Optional[FallbackPolicy] = None,
        obs_refresh_every_ticks: int = 4,
        max_decisions: int = 60,
        max_wall_seconds: float = 180.0,
        on_decision: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self._fallback_policy = fallback_policy or body_rest_state_fallback
        self.obs_refresh_every_ticks = obs_refresh_every_ticks
        self.max_decisions = max_decisions
        self.max_wall_seconds = max_wall_seconds
        self._on_decision_cb = on_decision

        self._obs_queue: "Queue[Observation]" = Queue(maxsize=1)
        self._skill_queue: "Queue[tuple[Skill, float, int]]" = Queue(maxsize=1)
        self._stop_flag = threading.Event()

        self._current_controller: Optional[MotorController] = None
        self._decisions_completed: int = 0
        self._decision_log: list[dict] = []

    # ---- worker thread ------------------------------------------------

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

    # ---- main loop ----------------------------------------------------

    def _make_fallback_controller(self) -> MotorController:
        try:
            fb_obs = self.env.observe()
            fb_skill = self._fallback_policy(fb_obs)
        except Exception as e:
            logger.warning("fallback policy raised: %s; using HoldPosition", e)
            fb_skill = HoldPosition()
        return make_controller(fb_skill, team_side=self.env.team_side)

    def run(self) -> dict:
        # Start with the fallback so the body is doing something from tick 0
        self._current_controller = self._make_fallback_controller()
        self.env.set_last_skill(
            type(self._current_controller.skill).__name__, "in_progress"
        )

        # Seed worker with first observation
        self._replace_in_queue(self._obs_queue, self.env.observe())

        worker = threading.Thread(target=self._llm_worker, daemon=True)
        worker.start()

        wall_start = time.monotonic()
        last_obs_push_tick = 0

        try:
            while True:
                # 1. Consume new LLM-decided skill if any
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
                    # Skill finished or failed — re-arm with fallback (body
                    # keeps doing its rest-state until LLM provides next intent)
                    self._current_controller = self._make_fallback_controller()

                # 4. Push fresh obs to worker periodically
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
        # Track is a perception-layer side-effect, not a motor action.
        # Apply it directly to the EgocentricFilter so subsequent
        # observations include the tracked entity.
        if isinstance(skill, Track):
            try:
                self.env.track_entity(skill.entity_id)
            except Exception as e:
                logger.warning("track_entity failed: %s", e)
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
