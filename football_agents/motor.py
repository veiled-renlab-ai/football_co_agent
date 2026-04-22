"""Motor layer — translates intent-level Skills into gfootball atomic actions.

Each Skill class has a corresponding MotorController state machine. The
controller is constructed once per skill invocation, then `.step(obs)` is
called every env tick, returning an (action_id, status) tuple.

gfootball Discrete(19) action ids and their meanings are listed in `class A`.
Movement is "sticky": pressing LEFT keeps you moving left until another
direction or RELEASE_DIRECTION fires. Same for SPRINT and DRIBBLE.

Phase 2 implements 5 controllers: MoveTo, DribbleToward, PassTo, Shoot,
HoldPosition. Skills without dedicated controllers fall back to IdleController
so the system stays runnable end-to-end. The remaining skills get real
controllers in Phase 3+.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .skills import (
    Call,
    DribbleToward,
    HoldPosition,
    Mark,
    MoveTo,
    PassTo,
    Press,
    ReceiveBall,
    ScanBehind,
    Shoot,
    Skill,
    Tackle,
    Track,
)

SkillStatus = Literal["in_progress", "completed", "failed"]
TeamSide = Literal["left", "right"]


# ---------------------------------------------------------------------------
# gfootball Discrete(19) action ids (from gfootball.env.football_action_set)
# ---------------------------------------------------------------------------

class A:
    IDLE = 0
    LEFT = 1
    TOP_LEFT = 2
    TOP = 3
    TOP_RIGHT = 4
    RIGHT = 5
    BOTTOM_RIGHT = 6
    BOTTOM = 7
    BOTTOM_LEFT = 8
    LONG_PASS = 9
    HIGH_PASS = 10
    SHORT_PASS = 11
    SHOT = 12
    SPRINT = 13
    RELEASE_DIRECTION = 14
    RELEASE_SPRINT = 15
    SLIDING = 16
    DRIBBLE = 17
    RELEASE_DRIBBLE = 18


# Human-readable names for logging
ACTION_NAMES: dict[int, str] = {
    0: "IDLE", 1: "LEFT", 2: "TOP_LEFT", 3: "TOP", 4: "TOP_RIGHT",
    5: "RIGHT", 6: "BOTTOM_RIGHT", 7: "BOTTOM", 8: "BOTTOM_LEFT",
    9: "LONG_PASS", 10: "HIGH_PASS", 11: "SHORT_PASS", 12: "SHOT",
    13: "SPRINT", 14: "RELEASE_DIR", 15: "RELEASE_SPRINT",
    16: "SLIDING", 17: "DRIBBLE", 18: "RELEASE_DRIBBLE",
}


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

# 8 sectors of 45° each, ordered counterclockwise starting at +x (RIGHT).
# atan2 returns angles in [-pi, pi] where +x = 0, +y = pi/2.
# In gfootball field coords, +y is "down" on screen (BOTTOM direction),
# -y is "up" (TOP), +x is RIGHT, -x is LEFT.
_SECTOR_TO_ACTION = (
    A.RIGHT,         # sector 0:  angle ~  0   (+x, 0y)
    A.BOTTOM_RIGHT,  # sector 1:  angle ~  +π/4
    A.BOTTOM,        # sector 2:  angle ~  +π/2 (0x, +y)
    A.BOTTOM_LEFT,   # sector 3:  angle ~  +3π/4
    A.LEFT,          # sector 4:  angle ~  ±π   (-x, 0y)
    A.TOP_LEFT,      # sector 5:  angle ~  -3π/4
    A.TOP,           # sector 6:  angle ~  -π/2 (0x, -y)
    A.TOP_RIGHT,     # sector 7:  angle ~  -π/4
)


def vector_to_action(dx: float, dy: float, idle_threshold: float = 1e-4) -> int:
    """Pick the closest of 8 directional gfootball actions for a (dx, dy) heading."""
    if abs(dx) < idle_threshold and abs(dy) < idle_threshold:
        return A.IDLE
    angle = math.atan2(dy, dx)  # [-pi, pi]
    sector = round(angle / (math.pi / 4)) % 8
    return _SECTOR_TO_ACTION[sector]


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------

class MotorController:
    """Base for Skill state machines.

    Subclasses implement `.step(obs)` returning (action_id, status). The
    controller is owned by the env wrapper, which calls `.step()` every env
    tick until status != 'in_progress'.
    """
    # Default tolerance for "arrived at point" checks (in gfootball units).
    EPSILON: float = 0.03

    def __init__(self, skill: Skill, team_side: TeamSide = "left") -> None:
        self.skill = skill
        self.team_side = team_side
        self.tick_count: int = 0

    # --- helpers ----------------------------------------------------------

    def _self_pos_vel(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        team_key = f"{self.team_side}_team"
        active_idx = obs["active"]
        return obs[team_key][active_idx], obs[f"{team_key}_direction"][active_idx]

    def _has_ball(self, obs: dict) -> bool:
        team_idx = 0 if self.team_side == "left" else 1
        return (
            obs["ball_owned_team"] == team_idx
            and obs["ball_owned_player"] == obs["active"]
        )

    def _opponent_goal_x(self) -> float:
        return 1.0 if self.team_side == "left" else -1.0

    # --- override me ------------------------------------------------------

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Movement controllers
# ---------------------------------------------------------------------------

class MoveToController(MotorController):
    """Walk/sprint toward (target_x, target_y). Done within EPSILON of target."""

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        self.tick_count += 1
        skill: MoveTo = self.skill  # type: ignore[assignment]
        pos, _ = self._self_pos_vel(obs)
        dx = skill.target_x - float(pos[0])
        dy = skill.target_y - float(pos[1])
        if math.hypot(dx, dy) < self.EPSILON:
            return A.RELEASE_DIRECTION, "completed"
        if skill.urgency == "sprint" and self.tick_count == 1:
            return A.SPRINT, "in_progress"
        return vector_to_action(dx, dy), "in_progress"


class DribbleTowardController(MotorController):
    """Like MoveTo but with DRIBBLE sticky on; fails if we lose possession."""

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        self.tick_count += 1
        skill: DribbleToward = self.skill  # type: ignore[assignment]
        if not self._has_ball(obs):
            return A.RELEASE_DRIBBLE, "failed"
        pos, _ = self._self_pos_vel(obs)
        dx = skill.target_x - pos[0]
        dy = skill.target_y - pos[1]
        if math.hypot(dx, dy) < self.EPSILON:
            return A.RELEASE_DRIBBLE, "completed"
        # First tick: enable dribble sticky
        if self.tick_count == 1:
            return A.DRIBBLE, "in_progress"
        return vector_to_action(dx, dy), "in_progress"


class HoldPositionController(MotorController):
    """One-shot release of any sticky direction, then idle."""

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        self.tick_count += 1
        if self.tick_count == 1:
            return A.RELEASE_DIRECTION, "in_progress"
        return A.IDLE, "completed"


# ---------------------------------------------------------------------------
# Ball controllers
# ---------------------------------------------------------------------------

class PassToController(MotorController):
    """Two-tick: face teammate -> press appropriate pass action."""

    PASS_ACTION = {
        "short": A.SHORT_PASS,
        "long": A.LONG_PASS,
        "through": A.HIGH_PASS,  # closest gfootball equivalent for v0
    }

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        self.tick_count += 1
        skill: PassTo = self.skill  # type: ignore[assignment]
        if not self._has_ball(obs):
            return A.IDLE, "failed"

        team_key = f"{self.team_side}_team"

        # Tick 1: orient toward the target teammate
        if self.tick_count == 1:
            self_pos, _ = self._self_pos_vel(obs)
            try:
                target_pos = obs[team_key][skill.target_player_id]
            except IndexError:
                return A.IDLE, "failed"
            dx = float(target_pos[0]) - float(self_pos[0])
            dy = float(target_pos[1]) - float(self_pos[1])
            return vector_to_action(dx, dy), "in_progress"

        # Tick 2: trigger the pass
        return self.PASS_ACTION[skill.pass_type], "completed"


class ShootController(MotorController):
    """Two-tick: face goal (with target_zone bias) -> press SHOT."""

    # Map target_zone to a y-offset relative to goal center.
    # gfootball field y range is [-0.42, +0.42] where y < 0 is "top" of screen
    # and y > 0 is "bottom". Goal posts are roughly at y = ±0.04.
    # "top" zones aim for negative y, "bottom" zones aim for positive y;
    # left/right within a zone is too fine-grained for gfootball's shot mechanic
    # (shots resolve via player facing + power), so we only encode top/bottom.
    ZONE_Y_BIAS = {
        "top_left": -0.04, "top_center": -0.04, "top_right": -0.04,
        "bottom_left": +0.04, "bottom_center": +0.04, "bottom_right": +0.04,
    }

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        self.tick_count += 1
        skill: Shoot = self.skill  # type: ignore[assignment]
        if not self._has_ball(obs):
            return A.IDLE, "failed"

        # Tick 1: orient toward goal
        if self.tick_count == 1:
            self_pos, _ = self._self_pos_vel(obs)
            goal_x = self._opponent_goal_x()
            goal_y = self.ZONE_Y_BIAS.get(skill.target_zone, 0.0)
            dx = goal_x - float(self_pos[0])
            dy = goal_y - float(self_pos[1])
            return vector_to_action(dx, dy), "in_progress"

        # Tick 2: shoot
        return A.SHOT, "completed"


# ---------------------------------------------------------------------------
# Placeholder controllers (TODO Phase 3+)
# ---------------------------------------------------------------------------

class IdleController(MotorController):
    """Fallback for skills not yet implemented; emits IDLE for one tick."""

    def step(self, obs: dict) -> tuple[int, SkillStatus]:
        return A.IDLE, "completed"


# ---------------------------------------------------------------------------
# Skill -> Controller dispatch
# ---------------------------------------------------------------------------

SKILL_TO_CONTROLLER: dict[type[Skill], type[MotorController]] = {
    MoveTo: MoveToController,
    DribbleToward: DribbleTowardController,
    HoldPosition: HoldPositionController,
    PassTo: PassToController,
    Shoot: ShootController,
    # TODO Phase 3+: real implementations of these
    ReceiveBall: IdleController,
    Mark: IdleController,
    Press: IdleController,
    Tackle: IdleController,
    ScanBehind: IdleController,
    Track: IdleController,
    Call: IdleController,
}


def make_controller(skill: Skill, team_side: TeamSide = "left") -> MotorController:
    """Factory — given a Skill instance, build the right MotorController."""
    cls = SKILL_TO_CONTROLLER.get(type(skill), IdleController)
    return cls(skill=skill, team_side=team_side)
