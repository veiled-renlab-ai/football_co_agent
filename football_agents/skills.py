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
# Urgency tiers (slow → fast):
#   walk   — throttled drift; used by fallback during LLM thinking gaps,
#            also LLM-selectable when conserving stamina or holding shape
#   jog    — gfootball base movement (no SPRINT pressed); steady running
#   sprint — gfootball SPRINT sticky on; top speed, drains stamina
Urgency = Literal["walk", "jog", "sprint"]
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
        "跑到场上指定坐标。无球时调整位置、跑空当、或回防时用。"
    )
    target_x: float                   # gfootball coords, [-1, 1]
    target_y: float                   # gfootball coords, [-0.42, 0.42]
    urgency: Urgency = "jog"


@dataclass(frozen=True)
class HoldPosition(Skill):
    """Stay where you are; useful for defensive shape or waiting for a pass."""
    tool_name: ClassVar[str] = "hold_position"
    description: ClassVar[str] = "站住不动，保持当前位置。等队友传球或维持阵型时用。"


@dataclass(frozen=True)
class DribbleToward(Skill):
    """Carry the ball toward a field point. Requires possession."""
    tool_name: ClassVar[str] = "dribble_toward"
    description: ClassVar[str] = (
        "带球朝指定坐标突破推进。需要脚下有球。"
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
        "把球传给指定球衣号码的队友。"
        "pass_type='short' 地面短传 / 'long' 高吊长传 / 'through' 直塞防线身后。"
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
        "朝对方球门射门，可瞄准球门的某个区域（上左/上中/上右/下左/下中/下右）。需要脚下有球。"
    )
    target_zone: ShootZone = "top_center"

    def is_valid(self, obs: "Observation") -> bool:
        return obs.self_state.has_ball


@dataclass(frozen=True)
class ReceiveBall(Skill):
    """Position to receive an incoming pass (open up body, control first touch)."""
    tool_name: ClassVar[str] = "receive_ball"
    description: ClassVar[str] = (
        "准备接球 —— 调整身体方向，准备第一脚处理。队友要传球给你时用。"
    )


# ---------------------------------------------------------------------------
# Defense skills
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mark(Skill):
    """Goalside-mark a specific opponent (stay between them and own goal)."""
    tool_name: ClassVar[str] = "mark"
    description: ClassVar[str] = (
        "盯防指定对手 —— 站位在他和己方球门之间，跟随其移动，断他的接球路线。"
    )
    opponent_id: int


@dataclass(frozen=True)
class Press(Skill):
    """Aggressively close down a specific opponent (close space, force error)."""
    tool_name: ClassVar[str] = "press"
    description: ClassVar[str] = (
        "上抢指定对手 —— 全速冲过去逼抢，压缩他处理球的空间，迫使失误。"
    )
    opponent_id: int


@dataclass(frozen=True)
class Tackle(Skill):
    """Attempt a tackle on the opponent in possession (sliding or standing)."""
    tool_name: ClassVar[str] = "tackle"
    description: ClassVar[str] = (
        "对身前的持球对手发起铲球。距离近且对方持球时用，有失败被过的风险。"
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
        "回头扫一眼背后。这一帧不能做别的，但下一次观察里能看到原本在你身后的情况。"
    )


@dataclass(frozen=True)
class Track(Skill):
    """Lock attention on a specific entity — stays in your perception cap
    even if other entities are physically closer."""
    tool_name: ClassVar[str] = "track"
    description: ClassVar[str] = (
        "锁定视觉注意力到某个特定球员或球；即使他不是最近的实体，也始终保留在你的视野里。"
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
        "对队友喊话：指令、提醒、呼应（'传给我' / '身后有人' / '换边' 之类）。"
        "audience='team' 全场广播 / 'nearby' 只有 ~10m 内的人能听见（对手有概率偷听）。"
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
# Progressive disclosure — Skill categories (Claude-Skills style)
# ---------------------------------------------------------------------------
# Layer 1 sends only 5 category meta-tools (no args). Layer 2 (after LLM
# picks a category) sends only that category's actual skill schemas.
# Smaller per-call payload → faster decisions, less GLM-4.7 truncation.

SKILL_CATEGORY: dict[type[Skill], str] = {
    # MOVE: positional / off-ball running
    MoveTo: "MOVE",
    HoldPosition: "MOVE",
    ReceiveBall: "MOVE",
    # ATTACK: on-ball offense
    DribbleToward: "ATTACK",
    PassTo: "ATTACK",
    Shoot: "ATTACK",
    # DEFEND: disrupt opposing possession
    Mark: "DEFEND",
    Press: "DEFEND",
    Tackle: "DEFEND",
    # PERCEIVE: gather info (head check / focus)
    ScanBehind: "PERCEIVE",
    Track: "PERCEIVE",
    # COMMUNICATE: shout to teammates
    Call: "COMMUNICATE",
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "ATTACK": "进攻类动作（射门 / 带球突破 / 传球）。脚下有球或即将拿到球，要往对方半场施压时选。",
    "DEFEND": "防守类动作（盯人 / 上抢 / 铲球）。对方持球或我方刚失球，要干扰对手时选。",
    "MOVE": "跑位类动作（跑到某点 / 站位不动 / 准备接球）。无球时调整位置或保持阵型时选。",
    "PERCEIVE": "感知类动作（回头观察 / 锁定关注某球员）。需要先看清场上情况再决定时选。",
    "COMMUNICATE": "沟通类动作（喊话呼应队友）。需要给队友发指令或信号时选（多 agent 时才有效）。",
}

# Reverse mapping: tool_name → category
CATEGORY_TOOL_NAMES: dict[str, str] = {
    f"choose_{cat.lower()}": cat for cat in CATEGORY_DESCRIPTIONS
}


def make_invoke_skill_tool(valid_skill_names: list[str]) -> dict[str, Any]:
    """The single meta-tool exposed to the LLM (Anthropic Skills style).

    Skill metadata (name + description + params) lives in the system prompt
    (always-loaded Level 1). The LLM picks one by name via this single tool;
    we validate args at our side. Per-turn `valid_skill_names` lets us
    enforce mechanical possibility (no shoot without ball) at the schema layer.
    """
    return {
        "type": "function",
        "function": {
            "name": "invoke_skill",
            "description": (
                "执行一个动作。skill_name 必须是 system prompt 里 'Available Skills' "
                "列表中某一个 name；args 是该 skill 的参数字典（键名和类型见 metadata 描述）。"
                "如果 skill 不接受参数，args 传 {}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": valid_skill_names,
                        "description": "要执行的动作的 name (snake_case)。",
                    },
                    "args": {
                        "type": "object",
                        "description": "该动作的参数字典；键名按 metadata 描述。无参数时传 {}。",
                    },
                },
                "required": ["skill_name", "args"],
            },
        },
    }


def skill_metadata_block(skills: list[type[Skill]] | None = None) -> str:
    """Generate the 'Available Skills' section for the system prompt.

    Mirrors Anthropic Skills Level 1 (always loaded): name + description +
    one-line param hint per skill. ~80-150 tokens per skill, total ~1.5k
    for all 12 skills. Sits in the system prompt so it's loaded once per
    LLM call (no per-tool schema overhead).

    Output is plain Chinese markdown.
    """
    if skills is None:
        skills = list(ALL_SKILLS)

    lines: list[str] = ["# 你能用的动作 (Available Skills)\n"]
    for s in skills:
        cat = SKILL_CATEGORY.get(s, "?")
        lines.append(f"## `{s.tool_name}`  ({cat})")
        lines.append(f"{s.description}")
        # Param hints
        try:
            type_hints = get_type_hints(s)
        except Exception:
            type_hints = {}
        param_lines: list[str] = []
        for f in fields(s):
            resolved = type_hints.get(f.name, str)
            origin = get_origin(resolved)
            type_args = get_args(resolved)
            if origin is Literal:
                kind = " | ".join(repr(a) for a in type_args)
            elif resolved is float:
                kind = "float"
            elif resolved is int:
                kind = "int"
            elif resolved is str:
                kind = "str"
            elif resolved is bool:
                kind = "bool"
            else:
                name_attr = getattr(resolved, "__name__", None)
                kind = name_attr if name_attr else str(resolved)
            default_part = ""
            if f.default is not MISSING:
                default_part = f"（默认 {f.default!r}）"
            param_lines.append(f"  - `{f.name}`: {kind}{default_part}")
        if param_lines:
            lines.append("参数:")
            lines.extend(param_lines)
        else:
            lines.append("（无参数）")
        lines.append("")
    return "\n".join(lines)


def layer_1_category_tools() -> list[dict[str, Any]]:
    """Return the 5 category meta-tools — no params, one-line descriptions.

    LLM picks one of these in stage-1 to declare its intent category. Then
    we send only that category's actual skill schemas in stage-2.
    """
    out: list[dict[str, Any]] = []
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        out.append({
            "type": "function",
            "function": {
                "name": f"choose_{cat.lower()}",
                "description": desc,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        })
    return out


def skills_in_category(category: str) -> list[type[Skill]]:
    """All Skill classes belonging to the given category."""
    return [s for s, c in SKILL_CATEGORY.items() if c == category]


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
