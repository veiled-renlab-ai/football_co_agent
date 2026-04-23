"""Perception layer — converts gfootball god-view state into egocentric,
FOV-limited observations that simulate what a real player can see.

The Observation produced here is what the LLM agent reads each decision tick
to choose a Skill.

Phase 1 (this file): data schemas only. The actual EgocentricFilter
implementation lands in Phase 3 — see DEV_PLAN.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Field constants (gfootball coordinate system)
#   x in [-1.0, 1.0]   length of the field; -1 = own goal line, +1 = opponent
#   y in [-0.42, 0.42] width;                -0.42 = top side,  +0.42 = bottom
# ---------------------------------------------------------------------------

FIELD_X_MIN, FIELD_X_MAX = -1.0, 1.0
FIELD_Y_MIN, FIELD_Y_MAX = -0.42, 0.42

# Perception tuning knobs (DEV_PLAN.md §6 lists these as open questions).
FOV_HALF_ANGLE_DEG = 105.0   # human peripheral vision: ~105° each side of facing
SIGHT_DISTANCE_FULL = 0.30   # within this radius, full info on entity
SIGHT_DISTANCE_MAX = 0.60    # beyond this, entity invisible unless tracked
ATTENTION_CAP = 7            # max entities a player can simultaneously track
SHORT_TERM_MEMORY_TICKS = 30 # ~3 seconds at 10 Hz env tick rate

Team = Literal["team_a", "team_b"]
Role = Literal[
    "GK",   # goalkeeper
    "CB", "LB", "RB", "LWB", "RWB",   # defenders
    "CDM", "CM", "CAM", "LM", "RM",   # midfielders
    "LW", "RW", "CF", "ST",           # forwards
]


@dataclass(frozen=True)
class Vec2:
    """2D point on the gfootball field (x along length, y along width)."""
    x: float
    y: float

    def distance_to(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_to_deg(self, other: "Vec2") -> float:
        """Bearing from self to other in degrees (0° = +x, 90° = +y)."""
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class EntityView:
    """What this player perceives about another entity (player or ball)."""
    entity_id: int                          # player number, or 99 for ball
    role: Literal["teammate", "opponent", "ball"]
    position: Vec2
    velocity: Optional[Vec2] = None         # None if too far / out of FOV
    distance: float = 0.0                   # distance from observer
    in_current_fov: bool = True             # vs. recalled from memory
    age_ticks: int = 0                      # 0 = seen this tick, >0 = stale
    has_ball: bool = False                  # for player entities


@dataclass(frozen=True)
class SelfState:
    """The player's own state — always fully known (proprioception)."""
    player_id: int
    team: Team
    role: Role
    position: Vec2
    velocity: Vec2
    facing_deg: float                       # 0° = +x, 90° = +y
    stamina: float                          # 0.0 - 1.0
    has_ball: bool


@dataclass(frozen=True)
class RecentEvent:
    """Something that happened in the last few seconds, surfaced to the brain."""
    tick: int
    description: str                        # "teammate #10 received ball from #5"


@dataclass
class Observation:
    """The egocentric per-tick observation the LLM brain consumes.

    OUTPUT of the Perception module: god-view dict from gfootball
    filtered through FOV / distance / occlusion + merged with this
    player's short-term memory.
    """
    tick: int
    match_clock: str                        # "23:15"
    score: tuple[int, int]                  # (team_a, team_b)
    self_state: SelfState
    perceived_entities: list[EntityView] = field(default_factory=list)
    recent_events: list[RecentEvent] = field(default_factory=list)
    last_skill: Optional[str] = None
    last_skill_status: Optional[Literal["in_progress", "completed", "failed"]] = None

    # ------------------------------------------------------------------
    # Convenience accessors — used by the LLM client when formatting
    # the observation into a prompt.
    # ------------------------------------------------------------------

    def teammates(self) -> list[EntityView]:
        return [e for e in self.perceived_entities if e.role == "teammate"]

    def opponents(self) -> list[EntityView]:
        return [e for e in self.perceived_entities if e.role == "opponent"]

    def ball(self) -> Optional[EntityView]:
        for e in self.perceived_entities:
            if e.role == "ball":
                return e
        return None


# ---------------------------------------------------------------------------
# Perception filter
# ---------------------------------------------------------------------------

class EgocentricFilter:
    """Filters gfootball's god-view dict into what a specific player can see.

    Construct one per player; call `.filter(god_view, tick)` each env tick.

    v0 (this implementation):
      - FOV cone test (entity must be within ±FOV_HALF_ANGLE_DEG of facing)
      - distance attenuation (full info < FULL, position-only < MAX, invisible beyond)
      - attention cap (closest ATTENTION_CAP entities prioritized)
      - facing direction inferred from velocity (defaults to +x = opponent goal)
      - ball is special: always perceived if within MAX (a moving ball is loud)

    v1 (Phase 5+ TODO):
      - occlusion (entity behind another entity)
      - short-term memory of recently-seen entities, with positional drift
      - explicit `track(entity_id)` skill to bypass FOV for tracked entities
      - `scan_behind` skill to flip FOV cone for one tick
    """

    # gfootball uses integer role ids in *_team_roles arrays.
    # See gfootball.env.football_action_set / game_constants.
    GFOOTBALL_ROLE_TO_NAME: dict[int, Role] = {
        0: "GK", 1: "CB", 2: "LB", 3: "RB", 4: "CDM",
        5: "CM", 6: "LM", 7: "RM", 8: "CAM", 9: "CF",
    }

    BALL_ENTITY_ID = 99

    def __init__(self, player_id: int, team: Team, role: Role) -> None:
        self.player_id = player_id
        self.team = team
        self.role = role
        # Default facing: toward the opponent goal (+x for team_a / left side, -x for team_b)
        self._last_facing_deg = 0.0 if team == "team_a" else 180.0
        # Tracked entity ids — always force-included in perceived_entities
        # regardless of FOV / distance / attention cap. Set via track_entity().
        # The id is interpreted in BOTH teams (LLM doesn't distinguish on Track skill);
        # whichever exists gets added.
        self._tracked_entity_ids: set[int] = set()

    # ----- public side-effect API used when LLM picks Track skill -----

    def track_entity(self, entity_id: int) -> None:
        """Add an entity to the tracked set. From now on, this entity is
        always included in perceived_entities regardless of FOV, distance,
        or attention cap.
        """
        self._tracked_entity_ids.add(int(entity_id))

    def untrack_entity(self, entity_id: int) -> None:
        """Remove an entity from the tracked set."""
        self._tracked_entity_ids.discard(int(entity_id))

    def filter(self, god_view: dict, tick: int) -> Observation:
        team_key = "left_team" if self.team == "team_a" else "right_team"
        opp_key = "right_team" if self.team == "team_a" else "left_team"
        team_idx = 0 if self.team == "team_a" else 1
        opp_team_idx = 1 - team_idx

        # ---- self ---------------------------------------------------------
        self_pos_arr = god_view[team_key][self.player_id]
        self_vel_arr = god_view[f"{team_key}_direction"][self.player_id]
        self_pos = Vec2(float(self_pos_arr[0]), float(self_pos_arr[1]))
        self_vel = Vec2(float(self_vel_arr[0]), float(self_vel_arr[1]))

        # Update facing if we're moving fast enough to reorient
        speed = math.hypot(self_vel.x, self_vel.y)
        if speed > 1e-3:
            self._last_facing_deg = math.degrees(math.atan2(self_vel.y, self_vel.x))
        facing_deg = self._last_facing_deg

        has_ball = (
            int(god_view["ball_owned_team"]) == team_idx
            and int(god_view["ball_owned_player"]) == self.player_id
        )
        tired = float(god_view[f"{team_key}_tired_factor"][self.player_id])

        self_state = SelfState(
            player_id=self.player_id,
            team=self.team,
            role=self.role,
            position=self_pos,
            velocity=self_vel,
            facing_deg=facing_deg,
            stamina=max(0.0, min(1.0, 1.0 - tired)),
            has_ball=has_ball,
        )

        # ---- candidate entities ------------------------------------------
        candidates: list[tuple[float, EntityView]] = []  # (distance, view)

        # Teammates (skip self)
        for i, pos in enumerate(god_view[team_key]):
            if i == self.player_id:
                continue
            ev = self._make_entity_view(
                entity_id=i, role="teammate",
                pos=pos, vel=god_view[f"{team_key}_direction"][i],
                self_pos=self_pos, facing_deg=facing_deg,
                has_ball=(
                    int(god_view["ball_owned_team"]) == team_idx
                    and int(god_view["ball_owned_player"]) == i
                ),
            )
            if ev is not None:
                candidates.append((ev.distance, ev))

        # Opponents
        for i, pos in enumerate(god_view[opp_key]):
            ev = self._make_entity_view(
                entity_id=i, role="opponent",
                pos=pos, vel=god_view[f"{opp_key}_direction"][i],
                self_pos=self_pos, facing_deg=facing_deg,
                has_ball=(
                    int(god_view["ball_owned_team"]) == opp_team_idx
                    and int(god_view["ball_owned_player"]) == i
                ),
            )
            if ev is not None:
                candidates.append((ev.distance, ev))

        # Ball — special: not subject to FOV, always perceived if within MAX
        ball_xyz = god_view["ball"]
        ball_pos = Vec2(float(ball_xyz[0]), float(ball_xyz[1]))
        ball_dist = self_pos.distance_to(ball_pos)
        if ball_dist < SIGHT_DISTANCE_MAX:
            ball_dir = god_view["ball_direction"]
            candidates.append((
                ball_dist,
                EntityView(
                    entity_id=self.BALL_ENTITY_ID,
                    role="ball",
                    position=ball_pos,
                    velocity=Vec2(float(ball_dir[0]), float(ball_dir[1])),
                    distance=ball_dist,
                    in_current_fov=True,
                    has_ball=False,
                ),
            ))

        # ---- attention cap ----------------------------------------------
        candidates.sort(key=lambda x: x[0])  # nearest first
        perceived = [ev for _, ev in candidates[:ATTENTION_CAP]]

        # ---- forced inclusion of tracked entities (perception layer's
        # implementation of the Track skill — bypasses FOV/distance/cap) ----
        if self._tracked_entity_ids:
            already_keys = {(e.role, e.entity_id) for e in perceived}
            for tid in self._tracked_entity_ids:
                if tid == self.BALL_ENTITY_ID:
                    continue  # ball special-cased above
                # Look up tid in BOTH teams (LLM didn't specify team)
                for tk, role_label, owner_team_idx in (
                    (team_key, "teammate", team_idx),
                    (opp_key, "opponent", opp_team_idx),
                ):
                    arr = god_view.get(tk)
                    if arr is None or tid >= len(arr):
                        continue
                    if (role_label, tid) in already_keys:
                        continue
                    pos = arr[tid]
                    vel = god_view[f"{tk}_direction"][tid]
                    forced = EntityView(
                        entity_id=tid,
                        role=role_label,
                        position=Vec2(float(pos[0]), float(pos[1])),
                        velocity=Vec2(float(vel[0]), float(vel[1])),
                        distance=self_pos.distance_to(Vec2(float(pos[0]), float(pos[1]))),
                        in_current_fov=False,  # tracked, not actually seen this tick
                        age_ticks=0,
                        has_ball=(
                            int(god_view["ball_owned_team"]) == owner_team_idx
                            and int(god_view["ball_owned_player"]) == tid
                        ),
                    )
                    perceived.append(forced)
                    already_keys.add((role_label, tid))

        # ---- assemble observation ---------------------------------------
        score = god_view.get("score", [0, 0])
        return Observation(
            tick=tick,
            match_clock=self._format_clock(tick),
            score=(int(score[0]), int(score[1])),
            self_state=self_state,
            perceived_entities=perceived,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _make_entity_view(
        self,
        *,
        entity_id: int,
        role: Literal["teammate", "opponent"],
        pos,                         # ndarray-like (2,)
        vel,                         # ndarray-like (2,)
        self_pos: Vec2,
        facing_deg: float,
        has_ball: bool,
    ) -> Optional[EntityView]:
        dx = float(pos[0]) - self_pos.x
        dy = float(pos[1]) - self_pos.y
        distance = math.hypot(dx, dy)

        if distance > SIGHT_DISTANCE_MAX:
            return None

        # Bearing relative to facing — normalize to [-180, 180]
        bearing_deg = math.degrees(math.atan2(dy, dx))
        relative = (bearing_deg - facing_deg + 540.0) % 360.0 - 180.0
        if abs(relative) > FOV_HALF_ANGLE_DEG:
            return None  # not in FOV (no v0 memory)

        # Within FULL: include velocity. Beyond FULL but within MAX: position only.
        velocity = None
        if distance <= SIGHT_DISTANCE_FULL:
            velocity = Vec2(float(vel[0]), float(vel[1]))

        return EntityView(
            entity_id=entity_id,
            role=role,
            position=Vec2(float(pos[0]), float(pos[1])),
            velocity=velocity,
            distance=distance,
            in_current_fov=True,
            age_ticks=0,
            has_ball=has_ball,
        )

    @staticmethod
    def _format_clock(tick: int) -> str:
        # gfootball runs ~10 env ticks per simulated second by default.
        sec = tick // 10
        return f"{sec // 60:02d}:{sec % 60:02d}"
