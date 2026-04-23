"""Skills — the player's intent-level action API.

A Skill is what the LLM brain CHOOSES each decision tick. It expresses an
INTENT ("pass to player 7") not a low-level action ("press button A").
The Motor layer (motor.py) translates each Skill into a sequence of
gfootball Discrete(19) atomic actions over the next several ticks.

Phase 1 (this file): Skill protocol + 12 v0 skills as immutable dataclasses,
plus tool-schema export for LLM function calling.
The actual MotorController state machines land in Phase 2 — see DEV_PLAN.md.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, ClassVar, Literal, TYPE_CHECKING, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from .perception import Observation, Vec2

# ---------------------------------------------------------------------------
# Skill argument literal types
# ---------------------------------------------------------------------------

PassType = Literal["short", "long", "through"]
Urgency = Literal["jog", "sprint"]
ShootZone = Literal[
    "top_left", "top_center", "top_right",
    "bottom_left", "bottom_center", "bottom_right",
]
CallAudience = Literal["team", "nearby"]


# ---------------------------------------------------------------------------
# Skill base class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    """Base for all skills. Subclasses are frozen dataclasses carrying intent.

    Each subclass must set:
      - `tool_name` (ClassVar[str]): snake_case name exposed to the LLM
      - `description` (ClassVar[str]): one-sentence purpose for the LLM tool def

    Subclasses may override `is_valid(obs)` to declare prerequisites
    (e.g., shoot() requires possession).
    """
    tool_name: ClassVar[str] = "skill"
    description: ClassVar[str] = ""

    def is_valid(self, obs: "Observation") -> bool:  # noqa: ARG002
        """Whether this skill is applicable given the current observation.

        Default: always valid. Subclasses override (e.g., Shoot requires has_ball).
        """
        return True


# ---------------------------------------------------------------------------
# Movement skills
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveTo(Skill):
    """Move (without ball) to a target field point."""
    tool_name: ClassVar[str] = "move_to"
    description: ClassVar[str] = (
        "Run to a specific field coordinate. Use to take up position, "
        "make a run into space, or get back on defense."
    )
    target_x: float                   # gfootball coords, [-1, 1]
    target_y: float                   # gfootball coords, [-0.42, 0.42]
    urgency: Urgency = "jog"


@dataclass(frozen=True)
class HoldPosition(Skill):
    """Stay where you are; useful for defensive shape or waiting for a pass."""
    tool_name: ClassVar[str] = "hold_position"
    description: ClassVar[str] = "Stand still and maintain current position."


@dataclass(frozen=True)
class DribbleToward(Skill):
    """Carry the ball toward a field point. Requires possession."""
    tool_name: ClassVar[str] = "dribble_toward"
    description: ClassVar[str] = (
        "Dribble the ball toward a target coordinate. Requires possession."
    )
    target_x: float
    target_y: float
    urgency: Urgency = "jog"

    def is_valid(self, obs: "Observation") -> bool:
        return obs.self_state.has_ball


# ---------------------------------------------------------------------------
# Ball skills
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PassTo(Skill):
    """Pass the ball to a specific teammate. Requires possession."""
    tool_name: ClassVar[str] = "pass_to"
    description: ClassVar[str] = (
        "Pass the ball to teammate by player id. "
        "type='short' (ground), 'long' (lofted), 'through' (in behind defenders)."
    )
    target_player_id: int
    pass_type: PassType = "short"

    def is_valid(self, obs: "Observation") -> bool:
        if not obs.self_state.has_ball:
            return False
        return any(t.entity_id == self.target_player_id for t in obs.teammates())


@dataclass(frozen=True)
class Shoot(Skill):
    """Shoot at the goal. Requires possession."""
    tool_name: ClassVar[str] = "shoot"
    description: ClassVar[str] = (
        "Shoot at the opponent's goal, aiming for a target zone."
    )
    target_zone: ShootZone = "top_center"

    def is_valid(self, obs: "Observation") -> bool:
        return obs.self_state.has_ball


@dataclass(frozen=True)
class ReceiveBall(Skill):
    """Position to receive an incoming pass (open up body, control first touch)."""
    tool_name: ClassVar[str] = "receive_ball"
    description: ClassVar[str] = (
        "Prepare to receive a pass — orient body, ready first touch."
    )


# ---------------------------------------------------------------------------
# Defense skills
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mark(Skill):
    """Goalside-mark a specific opponent (stay between them and own goal)."""
    tool_name: ClassVar[str] = "mark"
    description: ClassVar[str] = (
        "Mark a specific opponent — stay goal-side of them and shadow movement."
    )
    opponent_id: int


@dataclass(frozen=True)
class Press(Skill):
    """Aggressively close down a specific opponent (close space, force error)."""
    tool_name: ClassVar[str] = "press"
    description: ClassVar[str] = (
        "Aggressively press an opponent — close space fast to force a mistake."
    )
    opponent_id: int


@dataclass(frozen=True)
class Tackle(Skill):
    """Attempt a tackle on the opponent in possession (sliding or standing)."""
    tool_name: ClassVar[str] = "tackle"
    description: ClassVar[str] = (
        "Attempt a tackle on the ball carrier in front of you."
    )


# ---------------------------------------------------------------------------
# Active perception skills — modeling shoulder checks / scanning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanBehind(Skill):
    """Quick over-the-shoulder check. Costs the current decision but gives
    you the next observation including entities BEHIND your facing direction."""
    tool_name: ClassVar[str] = "scan_behind"
    description: ClassVar[str] = (
        "Glance over your shoulder. Cannot do anything else this tick, but "
        "next observation will include what was behind you."
    )


@dataclass(frozen=True)
class Track(Skill):
    """Lock attention on a specific entity — stays in your perception cap
    even if other entities are physically closer."""
    tool_name: ClassVar[str] = "track"
    description: ClassVar[str] = (
        "Lock visual attention on a specific player or the ball; they stay "
        "in your perceived state even when not nearest."
    )
    entity_id: int


# ---------------------------------------------------------------------------
# Communication skills
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Call(Skill):
    """Vocalize to teammates (or just nearby players). Heard by teammates
    in range; opponents within ~10m may overhear (audience='nearby')."""
    tool_name: ClassVar[str] = "call"
    description: ClassVar[str] = (
        "Shout to teammates: instructions, warnings, or requests "
        "('over here', 'man on', 'switch'). "
        "audience='team' (broadcast) or 'nearby' (only ~10m radius, opponents may overhear)."
    )
    message: str
    audience: CallAudience = "team"


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------

ALL_SKILLS: tuple[type[Skill], ...] = (
    MoveTo, HoldPosition, DribbleToward,
    PassTo, Shoot, ReceiveBall,
    Mark, Press, Tackle,
    ScanBehind, Track,
    Call,
)

SKILLS_BY_NAME: dict[str, type[Skill]] = {s.tool_name: s for s in ALL_SKILLS}


# ---------------------------------------------------------------------------
# OpenAI / 火山 / MiniMax tool-schema export
# ---------------------------------------------------------------------------

# Mapping from Python primitive types to JSON-Schema fragments.
_PRIMITIVE_TYPE_MAP: dict[type, dict[str, Any]] = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
}


def _resolved_type_to_schema(t: Any) -> dict[str, Any]:
    """Translate a *resolved* (not-stringified) type annotation into JSON Schema."""
    origin = get_origin(t)
    args = get_args(t)

    # Literal["a", "b", ...] -> string enum
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Primitives
    for py_type, schema in _PRIMITIVE_TYPE_MAP.items():
        if t is py_type:
            return dict(schema)

    return {"type": "string"}  # safe fallback


def skill_to_tool_schema(skill_cls: type[Skill]) -> dict[str, Any]:
    """Convert a Skill class into an OpenAI-style tool definition.

    Uses `strict: True` + all-required to engage constrained decoding on
    providers that support it (OpenAI, GLM-4.7 on 火山方舟 Coding Plan).
    With strict mode the model is physically constrained to emit args
    matching the schema exactly — no more truncated `{` returns.

    Trade-off: with strict, ALL parameters must be required (OpenAI rule).
    Skills with optional fields like `urgency` now force the LLM to specify
    them on every call. Slightly more verbose, but eliminates malformed args.
    """
    type_hints = get_type_hints(skill_cls)

    properties: dict[str, Any] = {}
    for f in fields(skill_cls):
        resolved = type_hints.get(f.name, str)
        properties[f.name] = _resolved_type_to_schema(resolved)

    return {
        "type": "function",
        "function": {
            "name": skill_cls.tool_name,
            "description": skill_cls.description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),  # strict requires all
                "additionalProperties": False,
            },
        },
    }


def all_tool_schemas() -> list[dict[str, Any]]:
    """All v0 skills as OpenAI tool definitions, ready for `tools=` arg."""
    return [skill_to_tool_schema(s) for s in ALL_SKILLS]
