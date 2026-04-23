"""Env wrapper — gfootball under our clean observe() / act() API.

Hides gfootball's gym semantics, dict observation format, and the auto-switch
of the `active` controlled player. Exposes a single-player API where the
agent just sees Observations and dispatches Skills.

Phase 3 scope: single-player control. Phase 5 will add multi-player control
(passing N actions per env.step) for full team simulation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from gfootball.env import create_environment

from .motor import MotorController, SkillStatus, make_controller
from .perception import EgocentricFilter, Observation
from .skills import Skill

TeamSide = str  # "left" | "right"


class FootballEnvAdapter:
    """Single-player gfootball wrapper exposing our Skill / Observation API.

    Typical agent loop:

        env = FootballEnvAdapter(scenario="academy_empty_goal_close")
        obs = env.reset()
        while not env.done:
            skill  = agent.choose_skill(obs)
            status = env.dispatch_skill(skill, max_env_ticks=8)
            obs    = env.observe()
        print("score:", env.cumulative_reward)
    """

    def __init__(
        self,
        scenario: str = "academy_empty_goal_close",
        *,
        team_side: TeamSide = "left",
        render: bool = False,
        write_video: bool = False,
        logdir: Optional[str] = None,
        n_controlled_left: int = 1,
        primary_player_slot: int = 0,
        controlled_player_id: Optional[int] = None,  # legacy, only affects filter
        physics_steps_per_frame: int = 2,
        real_time: bool = True,
    ) -> None:
        """
        Args:
          n_controlled_left: gfootball multi-agent control. With value N, the
              env expects a list of N actions per step (slot i → player #i).
              See gfootball/doc/multi_agent.md.
          primary_player_slot: which of the N slots receives the agent's
              chosen action; the other slots get IDLE. For academy attacking
              scenarios, set n_controlled_left=2, primary_player_slot=1
              (player #0 is the GK, #1 is the attacker we want the LLM to drive).
          physics_steps_per_frame: how many internal physics ticks gfootball
              runs per env.step (and per rendered frame). gfootball's internal
              constant PHYSICS_STEPS_PER_SECOND=100, so each env.step advances
              the GAME TIME by `physics_steps_per_frame / 100` seconds.
              Default 10 (gfootball's default) means each env.step = 100ms
              game time, which combined with our ~58 step/s wall rate gives
              ~5.8x real-time game pace (the world flies past while LLM thinks).
              We default to 2 here (each step = 20ms game time) so 58 step/s
              wall ≈ 1.16x real-time — close to natural pace, no choppy render.
              Render frame rate stays at native (one frame per env.step).
              Set to 1 for slower-than-real-time; 10 for legacy fast-mode.
          real_time: pass through to gfootball's internal real_time flag
              (used by the C++ engine for vsync-style render throttling
              when render=True). True ≈ engine tries to pace itself.
        """
        self.scenario = scenario
        self.team_side = team_side
        self.n_controlled_left = n_controlled_left
        self.primary_player_slot = primary_player_slot
        # Per gfootball/doc/saving_replays.md: write_video=True works WITHOUT
        # render=True; gfootball auto-generates a simple 2D animation. Real-time
        # 3D window only needs render=True (and a display, e.g. WSLg).
        env_kwargs: dict = dict(
            env_name=scenario,
            representation="raw",
            render=render,
            number_of_left_players_agent_controls=n_controlled_left,
            other_config_options={
                "physics_steps_per_frame": int(physics_steps_per_frame),
                "real_time": bool(real_time),
            },
        )
        if write_video:
            env_kwargs["write_video"] = True
            env_kwargs["write_full_episode_dumps"] = True  # AVI + pickled trace
            if logdir:
                import os
                os.makedirs(logdir, exist_ok=True)
                env_kwargs["logdir"] = logdir
        self._env = create_environment(**env_kwargs)
        self._raw_obs: Optional[dict] = None
        self._tick: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._active_controller: Optional[MotorController] = None
        self._last_skill_name: Optional[str] = None
        self._last_skill_status: Optional[SkillStatus] = None
        self._filter: Optional[EgocentricFilter] = None
        self._fixed_player_id = controlled_player_id  # None = follow gfootball's `active`

    # ---- properties ----------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def controlled_player_id(self) -> int:
        if self._raw_obs is None:
            raise RuntimeError("Call reset() first.")
        return int(self._raw_obs["active"])

    # ---- lifecycle -----------------------------------------------------

    def reset(self) -> Observation:
        raw = self._env.reset()
        # In multi-agent mode raw is a list of N dicts (one per slot);
        # in single-agent it's a single dict (or list of 1).
        if isinstance(raw, list):
            self._raw_obs = raw[self.primary_player_slot]
        else:
            self._raw_obs = raw
        self._tick = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._active_controller = None
        self._last_skill_name = None
        self._last_skill_status = None
        self._build_filter()
        return self.observe()

    def close(self) -> None:
        self._env.close()

    # ---- observation ---------------------------------------------------

    def observe(self) -> Observation:
        if self._raw_obs is None or self._filter is None:
            raise RuntimeError("Call reset() first.")
        # Auto-switch: if gfootball changed which player we control, rebuild filter
        active = int(self._raw_obs["active"])
        if active != self._filter.player_id:
            self._build_filter()
        obs = self._filter.filter(self._raw_obs, self._tick)
        obs.last_skill = self._last_skill_name
        obs.last_skill_status = self._last_skill_status
        return obs

    # ---- action --------------------------------------------------------

    def dispatch_skill(self, skill: Skill, max_env_ticks: int = 10) -> SkillStatus:
        """Execute one Skill for up to `max_env_ticks` env ticks.

        Stops early if the skill completes/fails or the episode ends.
        Returns the final status after the run.
        """
        self._active_controller = make_controller(skill, team_side=self.team_side)  # type: ignore[arg-type]
        self._last_skill_name = type(skill).__name__
        self._last_skill_status = "in_progress"

        for _ in range(max_env_ticks):
            if self._done:
                break
            action, status = self._active_controller.step(self._raw_obs)
            self._step_env(action)
            self._last_skill_status = status
            if status != "in_progress":
                self._active_controller = None
                break
            if self._done:
                break

        return self._last_skill_status  # type: ignore[return-value]

    # ---- internals -----------------------------------------------------

    # ---- public hooks for AsyncRunner --------------------------------

    @property
    def raw_obs(self) -> dict:
        """Latest gfootball raw observation dict (for controllers / async runner)."""
        if self._raw_obs is None:
            raise RuntimeError("Call reset() first.")
        return self._raw_obs

    def step_action(self, action: int) -> None:
        """Public: step the env once with the given atomic gfootball action id.

        Used by AsyncRunner where the env-tick loop is decoupled from skill
        dispatch (which is what dispatch_skill() bundles synchronously).

        SINGLE-AGENT path: when n_controlled_left > 1, fills non-primary
        slots with IDLE so the primary slot gets `action`.
        """
        self._step_env(action)

    def step_actions(self, actions: list[int]) -> None:
        """MULTI-AGENT: step the env with one action per controlled slot.

        For MultiAgentRunner where each of N players is independently
        controlled by its own PlayerAgent. The list length MUST equal
        n_controlled_left, ordered by slot index.

        Differences from step_action(int):
          - Takes ALL N actions explicitly (no IDLE-filling).
          - Stores raw_obs as raw[0] — the canonical god-view. All N raw
            views share identical world state; they differ only in the
            `active` field, which is meaningless when controlling all N
            players. Per-player perception filters in PlayerAgent use
            their explicit player_id, not raw_obs["active"].
        """
        if len(actions) != self.n_controlled_left:
            raise ValueError(
                f"step_actions: expected {self.n_controlled_left} actions, "
                f"got {len(actions)}"
            )
        step_arg = [int(a) for a in actions]
        result = self._env.step(step_arg)
        if len(result) == 5:
            raw, reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            raw, reward, done, _info = result
            done = bool(done)
        # Multi-agent canonical god-view: take raw[0] (or raw if not a list).
        # All slot views are identical except for `active`, which we don't use.
        self._raw_obs = raw[0] if isinstance(raw, list) else raw
        self._cumulative_reward += float(np.asarray(reward).sum())
        self._done = done
        self._tick += 1

    def set_last_skill(self, name: str, status: SkillStatus) -> None:
        """Mark the most-recently-active skill so the next observe() reflects it."""
        self._last_skill_name = name
        self._last_skill_status = status

    def track_entity(self, entity_id: int) -> None:
        """Forward Track skill's perception side-effect to the EgocentricFilter."""
        if self._filter is None:
            raise RuntimeError("Call reset() first.")
        self._filter.track_entity(entity_id)

    def untrack_entity(self, entity_id: int) -> None:
        if self._filter is None:
            raise RuntimeError("Call reset() first.")
        self._filter.untrack_entity(entity_id)

    # ---- internals ---------------------------------------------------

    def _step_env(self, action: int) -> None:
        # In multi-agent mode env.step expects a list of N actions
        # (slot i → player #i). The primary slot gets the agent's action,
        # all other slots get IDLE so they hold position.
        if self.n_controlled_left == 1:
            step_arg: object = int(action)
        else:
            actions = [0] * self.n_controlled_left  # 0 = IDLE
            actions[self.primary_player_slot] = int(action)
            step_arg = actions
        result = self._env.step(step_arg)
        if len(result) == 5:
            raw, reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            raw, reward, done, _info = result
            done = bool(done)
        if isinstance(raw, list):
            self._raw_obs = raw[self.primary_player_slot]
        else:
            self._raw_obs = raw
        self._cumulative_reward += float(np.asarray(reward).sum())
        self._done = done
        self._tick += 1

    def _build_filter(self) -> None:
        assert self._raw_obs is not None
        team_key = f"{self.team_side}_team"
        # In multi-agent mode the primary slot's `active` IS the player index
        # of that slot. Honour an explicit override if set.
        pid = self._fixed_player_id if self._fixed_player_id is not None \
            else int(self._raw_obs["active"])
        roles = self._raw_obs[f"{team_key}_roles"]
        role_id = int(roles[pid])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        team_label = "team_a" if self.team_side == "left" else "team_b"
        self._filter = EgocentricFilter(
            player_id=pid,
            team=team_label,
            role=role_name,
        )
