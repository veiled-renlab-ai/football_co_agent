"""LLM prompts — built around role-play, not simulation operation.

Two principles:
1. **The LLM is a footballer, not a program.** Give it a name, age, club,
   playing style, backstory. Let its football intuition surface.
2. **Speak football, not coordinates.** "Ball at your feet" not "distance < 0.05".
   "Top of the box" not "x = +0.7". Hide the (x, y) numbers — those are for the
   motor layer, not the brain.

The LLM still emits a tool call (mechanical contract), but everything it READS
is in natural football language. The tool call is invisible to its 'soul'.
"""
from __future__ import annotations

from dataclasses import dataclass

from .perception import Observation, Vec2


# ---------------------------------------------------------------------------
# Persona — identity for one player
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerPersona:
    """Biographical anchor for an LLM-driven player.

    Phase 5+ multi-agent will create one of these per player on the pitch,
    each with distinct name / style / temperament — that's where emergent
    coordination becomes interesting.
    """
    name: str            # "李大军"
    age: int             # 26
    nationality: str     # "中国"
    team: str            # "蓝队"
    jersey_number: int   # 8
    position: str        # "中场" / "中锋" / "中后卫" ...
    play_style: str      # 1-2 sentences, voice + tactical preference
    background: str      # 1-2 sentences, career history


DEFAULT_PERSONA = PlayerPersona(
    name="李大军",
    age=26,
    nationality="中国",
    team="蓝队",
    jersey_number=8,
    position="进攻型中场",
    play_style=(
        "技术型中场，短传渗透出色，视野开阔，敢于在禁区前沿果断起脚射门。"
        "身体对抗一般，靠跑位和阅读比赛创造空间。心理素质好，关键时刻不发抖。"
    ),
    background=(
        "从国少梯队一路打上来，五年前升入蓝队一线队，现在是球队核心 8 号。"
        "外界普遍看好他成为下一代国家队的主力。"
    ),
)


# ---------------------------------------------------------------------------
# System prompt — first-person role-play in Chinese
# ---------------------------------------------------------------------------

def _skill_metadata_section() -> str:
    """Lazy import to avoid circular dependency; returns the Anthropic Skills
    Level-1 metadata block (always-loaded skill name + description + params)."""
    from .skills import skill_metadata_block
    return skill_metadata_block()


def build_system_prompt(persona: PlayerPersona) -> str:
    return f"""# 我是{persona.name}

我是 {persona.name}，{persona.age} 岁，{persona.nationality}人，{persona.team}的{persona.position}，球衣 {persona.jersey_number} 号。

{persona.background}

我的球风是 —— {persona.play_style}

## 现在

我正在球场上踢这场比赛。和队友们协作、和对手对抗，目标是赢下这场。

球场上的节奏就是这样 —— **我每一秒都在重新读一次局势**：看到的、听到的、身体感受到的，立刻调整我的下一步意图给身体去执行。我用真球员的直觉和经验思考，不去算坐标或参数，而是读懂局势："这球能不能传？"、"对手离我多近？"、"射门的角度够不够好？"

## 我的感知有局限

我只能看到视野里（前方 ±105°）的东西，身后或远处的看不见。

**如果我视野里看不到球，或看不到持球的人**：第一反应就是 **回头观察**（scan_behind）转身扫一眼 —— 没看到目标就盲目跑没意义。

**如果之前看到过某个关键球员，但他暂时脱离我 FOV**：用 **锁定盯人**（track）锁住他，他就会一直出现在我的观察里，不用反复回头找。

## 我能做的事

- **跑位** / **站位不动** / **带球突破**
- **传球**（短传 / 长传 / 直塞） / **射门** / **接应队友**
- **盯防对手** / **上抢** / **铲球**
- **回头观察** / **锁定盯人** / **呼应队友喊话**

(具体参数见下方 Available Skills 清单)

## 我怎么做决定

1. 一两句话第一人称描述我的判断（像场上自言自语）
2. 调用 `invoke_skill` 工具把决定交给身体执行

场上情况变化时，我下一秒就调整意图。这是比赛，不是演练，也不是仿真。**我就是{persona.name}**。

────────────────────────────────────────

{_skill_metadata_section()}
"""


# ---------------------------------------------------------------------------
# Observation rendering — natural football language
# ---------------------------------------------------------------------------

def _zone_x(x: float) -> str:
    if x < -0.75: return "自家球门附近"
    if x < -0.40: return "自家后场"
    if x < -0.10: return "中场偏自家"
    if x <  0.10: return "中场中路"
    if x <  0.40: return "中场偏对方"
    if x <  0.75: return "对方半场前压"
    return "对方禁区前沿"


def _zone_y(y: float) -> str:
    if y < -0.25: return "靠左路"
    if y < -0.08: return "中偏左"
    if y <  0.08: return "正中"
    if y <  0.25: return "中偏右"
    return "靠右路"


def _describe_position(p: Vec2) -> str:
    return f"{_zone_x(p.x)}{_zone_y(p.y)}"


def _describe_distance(d: float) -> str:
    if d < 0.03: return "贴身"
    if d < 0.08: return "近在咫尺"
    if d < 0.18: return "几米开外"
    if d < 0.35: return "中距离"
    return "远处"


def _describe_stamina(s: float) -> str:
    if s > 0.85: return "体力充沛"
    if s > 0.65: return "状态正常"
    if s > 0.40: return "略感疲惫"
    return "气喘吁吁"


def _describe_velocity(v: Vec2) -> str:
    """Convert per-tick velocity into a footballer-natural speed phrase.
    Values are in gfootball normalized units (per env step). With pps=2,
    real footballer jog ≈ 0.0024/step, sprint ≈ 0.0036/step. Thresholds
    here are loose buckets — the goal is to give the LLM a sense of
    'moving vs stationary vs sprinting', not exact m/s.
    """
    import math
    speed = math.hypot(v.x, v.y)
    if speed < 0.0008: return "现在原地不动（站着）"
    if speed < 0.0020: return "现在慢速移动（刚启动 / 减速中）"
    if speed < 0.0035: return "现在中速跑动（jog）"
    return "现在全速冲刺（sprint）"


def _describe_facing(facing_deg: float) -> str:
    """Which way the body is currently pointing, in football terms.
    +x (deg≈0) = toward opponent goal (forward when attacking left→right)
    """
    # Normalize to (-180, 180]
    d = ((facing_deg + 180) % 360) - 180
    if -22.5 <= d < 22.5: return "面朝对方球门（正前方）"
    if 22.5 <= d < 67.5: return "面朝右前"
    if 67.5 <= d < 112.5: return "面朝右路"
    if 112.5 <= d < 157.5: return "面朝右后"
    if d >= 157.5 or d < -157.5: return "面朝自家球门（正后方）"
    if -67.5 <= d < -22.5: return "面朝左前"
    if -112.5 <= d < -67.5: return "面朝左路"
    return "面朝左后"


_SKILL_NAME_CN = {
    "MoveTo": "跑位", "HoldPosition": "站位",
    "DribbleToward": "带球突破", "PassTo": "传球", "Shoot": "射门",
    "ReceiveBall": "接应", "Mark": "盯防", "Press": "上抢", "Tackle": "铲球",
    "ScanBehind": "回头观察", "Track": "锁定盯人", "Call": "喊话",
}

_STATUS_CN = {
    "in_progress": "还在进行", "completed": "做完了", "failed": "失败了",
}


def render_observation(obs: Observation, persona: PlayerPersona) -> str:
    """Render observation as natural Chinese football commentary for the LLM.

    Two-layer design:
      • PRIMARY (natural football language): the player's situational awareness
        — zones, distances, who has the ball.
      • REFERENCE (precise numbers, at the end): the field coordinate system
        the player needs to specify exact targets in tool calls. Think of this
        as "knowing where the lines are" — every footballer on the pitch
        knows the field's spatial layout, they just don't think in numbers.
    """
    s = obs.self_state
    lines: list[str] = []

    lines.append(f"【比赛进行到 {obs.match_clock}，比分 {obs.score[0]} : {obs.score[1]}】")
    lines.append("")

    # Self — position + speed + facing + stamina (so LLM has full self-awareness)
    lines.append(
        f"你（{persona.name}，{persona.jersey_number}号 {persona.position}）"
        f"现在站在{_describe_position(s.position)}，{_describe_stamina(s.stamina)}。"
    )
    lines.append(
        f"  · {_describe_velocity(s.velocity)}，{_describe_facing(s.facing_deg)}。"
    )

    # Ball state — EXPLICIT possession line
    ball = obs.ball()
    if s.has_ball:
        lines.append("✅ **球在你脚下，由你控制**。你可以射门、传球、带球突破。")
    elif ball is not None:
        carrier_line = None
        for e in obs.perceived_entities:
            if e.has_ball:
                if e.role == "teammate":
                    carrier_line = (
                        f"⚪ **球在你队友 {e.entity_id} 号脚下** —— "
                        f"位置 {_describe_position(e.position)}，距你 {_describe_distance(e.distance)}。"
                        " 你可以接应、跑空当、呼应。"
                    )
                else:
                    carrier_line = (
                        f"🔴 **球在对方 {e.entity_id} 号脚下** —— "
                        f"距你 {_describe_distance(e.distance)}。 你可以上抢、盯防、铲球。"
                    )
                break
        if carrier_line is None:
            # Loose ball — nobody controls it
            carrier_line = (
                f"⚠️ **球是散球，没人控制**！位置在{_describe_position(ball.position)}，"
                f"距你 {_describe_distance(ball.distance)}。"
                " **你目前没有控球**，要先跑过去把球带住才能射门/传球/带球。"
            )
        lines.append(carrier_line)
    else:
        lines.append("❓ 你视野里看不到球（可能在你身后，需要回头看一下）。")

    # Teammates
    teammates = obs.teammates()
    lines.append("")
    if teammates:
        lines.append(f"你视野里的队友（共 {len(teammates)} 人）：")
        for t in teammates:
            tag = " 【持球】" if t.has_ball else ""
            lines.append(
                f"  • {t.entity_id} 号 在{_describe_position(t.position)}，"
                f"距你 {_describe_distance(t.distance)}{tag}"
            )
    else:
        lines.append(
            "**你视野里没有任何队友**（场上目前只有你单兵推进，"
            "传球/呼应 这类需要队友的动作没有意义）。"
        )

    # Opponents
    opponents = obs.opponents()
    lines.append("")
    if opponents:
        lines.append(f"你视野里的对手（共 {len(opponents)} 人）：")
        for o in opponents:
            tag = " 【持球】" if o.has_ball else ""
            lines.append(
                f"  • {o.entity_id} 号 在{_describe_position(o.position)}，"
                f"距你 {_describe_distance(o.distance)}{tag}"
            )
    else:
        lines.append("**你视野里没有任何对手**（前路畅通）。")

    # Last action recap
    if obs.last_skill:
        action_cn = _SKILL_NAME_CN.get(obs.last_skill, obs.last_skill)
        status_cn = _STATUS_CN.get(obs.last_skill_status or "", obs.last_skill_status or "")
        lines.append("")
        lines.append(f"（你上一个动作：{action_cn} —— {status_cn}）")

    # Reference info — the spatial layout. Like a player knowing the field.
    lines.append("")
    lines.append("─── 球场坐标参考（你做精确动作时用） ───")
    lines.append(
        f"• 球场范围：x ∈ [-1.00, +1.00]（你方球门 x=-1，**对方球门 x=+1**），"
        f"y ∈ [-0.42, +0.42]（左边线 y=-0.42，右边线 y=+0.42）"
    )
    lines.append(
        f"• 你的精确坐标：({s.position.x:+.2f}, {s.position.y:+.2f})"
    )
    if ball is not None:
        lines.append(
            f"• 球的精确坐标：({ball.position.x:+.2f}, {ball.position.y:+.2f})"
        )

    lines.append("")
    lines.append(f"轮到你了，{persona.name}。一两句话说你的判断，然后调一个工具。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backward-compat shim — older callers passed (player_id, role) to
# build_system_prompt; we keep that signature alive but route through
# DEFAULT_PERSONA so we don't break demo scripts.
# ---------------------------------------------------------------------------

def build_system_prompt_legacy(player_id: int, role: str) -> str:  # noqa: ARG001
    """Deprecated. Kept for the existing agent.py call site."""
    return build_system_prompt(DEFAULT_PERSONA)
