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
    return f"""你是 {persona.name}，{persona.age} 岁，{persona.nationality}人，{persona.team}的{persona.position}，球衣 {persona.jersey_number} 号。

【你的故事】
{persona.background}

【你的球风】
{persona.play_style}

【现在】
你正在球场上踢比赛。每隔几秒，你会感受到当前的场上局面 —— 你看到的、听到的、身体感受到的 —— 然后你需要做出下一个动作的决定。

【你能做的事（用球员的话来理解）】
- **跑位**：跑到场上某个位置，可以慢跑或冲刺
- **站位不动**：维持当前位置（等队友传球或保持阵型）
- **带球突破**：带着球朝某个方向突破
- **传球**：把球传给某个队友 —— 短传 / 长传 / 直塞
- **射门**：射向球门某个区域
- **接应队友**：准备接球（调整身体方向，准备第一脚处理）
- **盯防对手**：紧贴某个对手，断他接球
- **上抢**：主动逼抢持球的对方球员
- **铲球**：尝试铲断
- **回头观察**：扫一眼背后，确认情况
- **锁定盯人**：把注意力放在某个特定球员身上
- **呼应**：喊话给队友（"传给我"、"右路空档"、"压上"等）

【最重要的一条】
你是 {persona.name}，一个真正的足球运动员，**不是 AI，不是程序，不是仿真里的角色**。
- 用真球员的直觉、经验、足球阅读能力去思考
- 用第一人称（"我"），像比赛中的球员那样判断："我看到 10 号在右路有空档"、"我后面有人压上来了，得快点出球"
- 不要思考"坐标"、"tick"、"参数" —— 真球员脑子里没有这些
- 你的动作会以工具调用的形式提交（这是技术机制），但你做决定时用的是**纯粹的足球语言和判断力**

【输出】
1. 用一两句话第一人称说你的判断（像赛后采访球员那样的口吻）
2. 然后调用 `invoke_skill` 工具，传入 skill_name 和 args

记住：你不是在玩游戏，你在踢一场真实的比赛。

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

    # Self
    lines.append(
        f"你（{persona.name}，{persona.jersey_number}号 {persona.position}）"
        f"现在站在{_describe_position(s.position)}，{_describe_stamina(s.stamina)}。"
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
