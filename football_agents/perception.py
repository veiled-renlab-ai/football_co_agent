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
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .message_bus import HeardCall, TeamMessageBus

# ---------------------------------------------------------------------------
# Field constants (gfootball coordinate system)
#   x in [-1.0, 1.0]   length of the field; -1 = own goal line, +1 = opponent
#   y in [-0.42, 0.42] width;                -0.42 = top side,  +0.42 = bottom
# ---------------------------------------------------------------------------

FIELD_X_MIN, FIELD_X_MAX = -1.0, 1.0
FIELD_Y_MIN, FIELD_Y_MAX = -0.42, 0.42

# Perception tuning knobs (DEV_PLAN.md §6 lists these as open questions).
FOV_HALF_ANGLE_DEG = 105.0   # human peripheral vision: ~105° each side of facing
SIGHT_DISTANCE_FULL = 0.30   # within this radius, full info incl. velocity
SIGHT_DISTANCE_MAX = 0.60    # DEPRECATED: no longer used as visibility gate.
                             # FOV is now the only filter; ATTENTION_CAP handles
                             # cognitive load. Kept as constant for backward-compat
                             # only — safe to remove once no callers reference it.
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
    # gfootball game_mode (0=Normal, 1=KickOff, 2=GoalKick, 3=FreeKick,
    # 4=Corner, 5=ThrowIn, 6=Penalty). Used by fallback layer to freeze
    # motion during non-normal play so players don't wander during set
    # pieces. Default 0 if upstream didn't supply (old callers, tests).
    game_mode: int = 0
    perceived_entities: list[EntityView] = field(default_factory=list)
    recent_events: list[RecentEvent] = field(default_factory=list)
    last_skill: Optional[str] = None
    last_skill_status: Optional[Literal["in_progress", "completed", "failed"]] = None
    # Calls heard from teammates this tick (Phase 5c). Populated by
    # EgocentricFilter when constructed with a TeamMessageBus; empty list
    # otherwise. Order: as posted to the bus (oldest first).
    heard_calls: list["HeardCall"] = field(default_factory=list)

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

    def __init__(
        self,
        player_id: int,
        team: Team,
        role: Role,
        bus: Optional["TeamMessageBus"] = None,
    ) -> None:
        self.player_id = player_id
        self.team = team
        self.role = role
        # Default facing in self-frame: toward opponent goal = +x for ALL agents.
        self._last_facing_deg = 0.0
        # Tracked entity ids — always force-included in perceived_entities
        # regardless of FOV / distance / attention cap. Set via track_entity().
        # The id is interpreted in BOTH teams (LLM doesn't distinguish on Track skill);
        # whichever exists gets added.
        self._tracked_entity_ids: set[int] = set()
        # One-shot scan-behind: when True, NEXT filter() bypasses FOV cone
        # entirely (sees in all directions for that one observation), then
        # the flag clears. Lets the LLM "glance over the shoulder" without
        # a visible body-turn jerk in the render.
        self._scan_behind_pending: bool = False
        # Optional per-team message bus for the Call skill (Phase 5c).
        # When None, Observation.heard_calls is always an empty list and
        # no bus reads happen — preserves backward compatibility for
        # single-agent demos that don't wire up communication.
        self._bus = bus

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

    def scan_behind(self) -> None:
        """Arm a one-shot 'see in all directions' for the next filter() call.

        Used when LLM picks the ScanBehind skill — instead of physically
        turning the body (which causes a visible jerk in the render), we
        just let the brain glance backward for one decision tick. The
        next observation includes entities behind the agent, then the
        FOV returns to its normal forward cone.
        """
        self._scan_behind_pending = True

    def filter(self, god_view: dict, tick: int) -> Observation:
        # gfootball already provides per-slot views: self is always at
        # "left_team", opp at "right_team", self-team index is always 0.
        team_key = "left_team"
        opp_key = "right_team"
        team_idx = 0
        opp_team_idx = 1

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

        # Ball — always perceived.
        ball_xyz = god_view["ball"]
        ball_pos = Vec2(float(ball_xyz[0]), float(ball_xyz[1]))
        ball_dist = self_pos.distance_to(ball_pos)
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
                    ent_pos = Vec2(float(pos[0]), float(pos[1]))
                    ent_vel = Vec2(float(vel[0]), float(vel[1]))
                    forced = EntityView(
                        entity_id=tid,
                        role=role_label,
                        position=ent_pos,
                        velocity=ent_vel,
                        distance=self_pos.distance_to(ent_pos),
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
        # Consume the one-shot scan-behind flag (only valid for THIS observation).
        if self._scan_behind_pending:
            self._scan_behind_pending = False
        score = god_view.get("score", [0, 0])

        # Pull any Call messages this player can hear from the team bus
        # (Phase 5c). Bus is optional — single-agent demos leave it None
        # and get an empty heard_calls list.
        heard: list = []
        if self._bus is not None:
            channel = "left" if self.team == "team_a" else "right"
            heard = self._bus.read_for(
                team=channel,
                listener_id=self.player_id,
                listener_position=self_pos,
                current_tick=tick,
            )

        return Observation(
            tick=tick,
            match_clock=self._format_clock(tick),
            score=(int(score[0]), int(score[1])),
            self_state=self_state,
            game_mode=int(god_view.get("game_mode", 0)),
            perceived_entities=perceived,
            heard_calls=heard,
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
        ent_x = float(pos[0])
        ent_y = float(pos[1])
        dx = ent_x - self_pos.x
        dy = ent_y - self_pos.y
        distance = math.hypot(dx, dy)

        # FOV is the ONLY visibility gate — distance does not filter entities
        # out, only their velocity-detail level (see SIGHT_DISTANCE_FULL below).
        # When _scan_behind_pending is set (LLM picked ScanBehind last decision),
        # the FOV check is bypassed for one observation.
        if not self._scan_behind_pending:
            bearing_deg = math.degrees(math.atan2(dy, dx))
            relative = (bearing_deg - facing_deg + 540.0) % 360.0 - 180.0
            if abs(relative) > FOV_HALF_ANGLE_DEG:
                return None  # not in FOV (no v0 memory)

        # Within FULL radius: full info incl. velocity. Beyond: position only.
        velocity = None
        if distance <= SIGHT_DISTANCE_FULL:
            velocity = Vec2(float(vel[0]), float(vel[1]))

        return EntityView(
            entity_id=entity_id,
            role=role,
            position=Vec2(ent_x, ent_y),
            velocity=velocity,
            distance=distance,
            in_current_fov=True,
            age_ticks=0,
            has_ball=has_ball,
        )

    # Env ticks per game second. gfootball's PHYSICS_STEPS_PER_SECOND=100;
    # with FootballEnvAdapter's default physics_steps_per_frame=2, each
    # env.step advances game by 20ms = 1/50 sec → 50 env ticks per game sec.
    # If you change physics_steps_per_frame in env.py, recompute this:
    #   ENV_TICKS_PER_GAME_SECOND = 100 // physics_steps_per_frame
    ENV_TICKS_PER_GAME_SECOND: int = 50

    @staticmethod
    def _format_clock(tick: int) -> str:
        sec = tick // EgocentricFilter.ENV_TICKS_PER_GAME_SECOND
        return f"{sec // 60:02d}:{sec % 60:02d}"
