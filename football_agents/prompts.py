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
from typing import Optional

from .perception import Observation, Vec2


# ---------------------------------------------------------------------------
# Persona — identity for one player
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamProfile:
    """Optional team-style identity that can be attached to a PlayerPersona.

    Carries the team's TACTICAL TENDENCY (~1-2 sentences) — gets injected
    into the system prompt as a 'we play this style' character trait, so the
    LLM thinks of it as part of who it is, not as an external rule.

    Future: replaced or augmented when real player profiles are injected.
    """
    name: str           # "蓝队"
    character: str      # 1-2 sentence Chinese description, e.g.:
                        # "传控渗透为主，阵地战擅长。中场拿球后求精确出球，不求快。"


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
    team_profile: Optional[TeamProfile] = None  # optional team-style trait
                                                # (Phase 5c). None = no team
                                                # section in system prompt.
    position_discipline: Optional[str] = None  # per-position behavioral
                                               # guide (Phase 5e). None =
                                               # no position section in prompt.


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
    universal_discipline_section = (
        "\n## 我对足球的理解\n"
        "\n"
        "足球是 **5 个位置共同覆盖球场** 的游戏，不是 5 人都追球的运动。\n"
        "\n"
        "每个位置在球场上有责任范围。如果所有人都冲球：\n"
        "- 我们的位置**塌缩到一个点**\n"
        "- 球场其他区域**完全空白**\n"
        "- 对手反击 = 单刀\n"
        "\n"
        "每次球转移，我考虑的不是\"我要不要去抢球\"，而是：\n"
        "- **我现在的位置是不是我该在的地方？**\n"
        "- 离球近的队友会去处理；**我做我位置该做的事**：拉空当 / 防对手 / 等接应。\n"
        "\n"
        "球队像一张网，每个人是一个节点。一个节点跑了，网就破了。\n"
        "\n"
        "**关于追球的铁律**：\n"
        "- 除非球就在我身边（距离 < 0.08），**不要主动去追球**\n"
        "- 对手持球跑？**我附近有队友在追，我就不追**——我回我的位置守好\n"
        "- 只有两种情况我要冲刺抢球：\n"
        "  1. 球是散球（没人控制）且**我离球最近**\n"
        "  2. 对手持球**直奔我而来**，且我身后没有队友帮我\n"
    )

    if persona.team_profile is not None:
        team_section = (
            f"\n## 我们球队（{persona.team_profile.name}）的风格\n"
            f"\n{persona.team_profile.character}\n"
            f"\n我会按这个团队风格去思考和决策 —— 但具体怎么执行还是看场上情况和我的判断。\n"
        )
    else:
        team_section = ""

    if persona.position_discipline is not None:
        position_discipline_section = (
            f"\n## 我的位置职责（{persona.position}）\n"
            f"\n{persona.position_discipline}\n"
        )
    else:
        position_discipline_section = ""

    return f"""# 我是{persona.name}

我是 {persona.name}，{persona.age} 岁，{persona.nationality}人，{persona.team}的{persona.position}，球衣 {persona.jersey_number} 号。

{persona.background}

我的球风是 —— {persona.play_style}
{universal_discipline_section}
{team_section}
{position_discipline_section}
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


def _describe_entity_motion(
    entity_pos: Vec2, entity_vel: Vec2, self_pos: Vec2
) -> str:
    """Brief Chinese motion phrase for a perceived entity, relative to listener.

    Compares entity velocity vector against (self_pos - entity_pos) direction:
      - dot > +0.7  → "正向你跑来" (toward listener)
      - dot < -0.7  → "正在远离" (away from listener)
      - else        → "横向移动" (perpendicular)

    If the entity is essentially stationary, returns "原地".
    Speed bucket prefix is omitted to keep it ≤ ~6 Chinese chars.
    Returns empty string if the input is degenerate (no useful info).
    """
    import math
    speed = math.hypot(entity_vel.x, entity_vel.y)
    if speed < 0.0008:
        return "原地"
    # Direction from entity toward self
    dx = self_pos.x - entity_pos.x
    dy = self_pos.y - entity_pos.y
    rel_dist = math.hypot(dx, dy)
    if rel_dist < 1e-6:
        # Right on top of listener — call it "贴近你"
        return "贴近你"
    # Unit vectors then dot product (cosine of angle between vel and toward-self)
    ux, uy = dx / rel_dist, dy / rel_dist
    vx, vy = entity_vel.x / speed, entity_vel.y / speed
    dot = ux * vx + uy * vy
    if dot > 0.7:
        verdict = "向你跑来"
    elif dot < -0.7:
        verdict = "正在远离"
    else:
        verdict = "横向移动"
    # Optional speed prefix — keep terse
    if speed >= 0.0035:
        prefix = "全速"
    elif speed >= 0.0020:
        prefix = "中速"
    else:
        prefix = ""
    return f"{prefix}{verdict}"


def _describe_ball_motion(ball_pos: Vec2, ball_vel: Vec2, self_pos: Vec2) -> str:
    """Brief Chinese motion phrase for the ball, relative to listener."""
    import math
    speed = math.hypot(ball_vel.x, ball_vel.y)
    if speed < 0.0008:
        return "球静止"
    dx = self_pos.x - ball_pos.x
    dy = self_pos.y - ball_pos.y
    rel_dist = math.hypot(dx, dy)
    if rel_dist < 1e-6:
        return "球贴近你"
    ux, uy = dx / rel_dist, dy / rel_dist
    vx, vy = ball_vel.x / speed, ball_vel.y / speed
    dot = ux * vx + uy * vy
    if dot > 0.7:
        return "球正滚向你"
    elif dot < -0.7:
        return "球正滚开"
    else:
        return "球横向滚动"


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
        import math as _math
        # 对方球门:x=+1.00, y∈[-0.044, +0.044]。距离 = 球员到球门中心的欧氏距离。
        # 因为 gfootball 已经做了左右队镜像,所有球员的 +x 都指向"对方球门"。
        dx_to_goal = 1.0 - float(s.position.x)
        dist_to_goal = _math.hypot(dx_to_goal, float(s.position.y))
        if dist_to_goal < 0.25:
            zone_hint = "近距离射门区"
        elif dist_to_goal < 0.35:
            zone_hint = "禁区内"
        elif dist_to_goal < 0.55:
            zone_hint = "禁区外、射门远"
        else:
            zone_hint = "中场以远，离球门很远"
        lines.append("✅ **球在你脚下，由你控制**。你可以射门、传球、带球突破。")
        lines.append(
            f"  · **对方球门**:x=+1.00, y∈[-0.044, +0.044]"
            f"(球门中心 (1.00, 0.00),向 +x 还有 {dx_to_goal:+.2f},向 +y 还有 {-float(s.position.y):+.2f})。"
        )
        lines.append(
            f"  · **你距对方球门** {dist_to_goal:.2f} 单位 —— {zone_hint}。"
        )
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
                        f"位置 {_describe_position(e.position)}，距你 {_describe_distance(e.distance)}。"
                        " 你可以上抢、盯防、铲球。"
                    )
                break
        if carrier_line is None:
            # Loose ball — nobody controls it
            ball_motion = ""
            if ball.velocity is not None:
                phrase = _describe_ball_motion(ball.position, ball.velocity, s.position)
                if phrase:
                    ball_motion = f"，{phrase}"
            carrier_line = (
                f"⚠️ **球是散球，没人控制**！位置在{_describe_position(ball.position)}，"
                f"距你 {_describe_distance(ball.distance)}{ball_motion}。"
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
            motion = ""
            if t.velocity is not None:
                phrase = _describe_entity_motion(t.position, t.velocity, s.position)
                if phrase:
                    motion = f"，{phrase}"
            lines.append(
                f"  • {t.entity_id} 号 在{_describe_position(t.position)}，"
                f"距你 {_describe_distance(t.distance)}{motion}{tag}"
            )
    else:
        lines.append(
            "**你视野里没有任何队友**（场上目前只有你单兵推进，"
            "传球/呼应 这类需要队友的动作没有意义）。"
        )

    # Heard calls — incoming Call messages from teammates (Phase 5c).
    # Absence of section IS the signal (no "no calls" line needed).
    if obs.heard_calls:
        lines.append("")
        lines.append(f"你听到队友的喊话（{len(obs.heard_calls)} 条）:")
        for call in obs.heard_calls:
            # 50 ticks per game second (matches ENV_TICKS_PER_GAME_SECOND in perception.py)
            age_sec = call.age_ticks / 50.0
            if age_sec < 0.3:
                age_str = "刚刚"
            else:
                age_str = f"{age_sec:.1f} 秒前"
            sender_zone = _describe_position(call.sender_position)
            lines.append(
                f"  • {call.sender_jersey} 号（在{sender_zone}，{age_str}）: \"{call.message}\""
            )

    # Opponents
    opponents = obs.opponents()
    lines.append("")
    if opponents:
        lines.append(f"你视野里的对手（共 {len(opponents)} 人）：")
        for o in opponents:
            tag = " 【持球】" if o.has_ball else ""
            motion = ""
            if o.velocity is not None:
                phrase = _describe_entity_motion(o.position, o.velocity, s.position)
                if phrase:
                    motion = f"，{phrase}"
            lines.append(
                f"  • {o.entity_id} 号 在{_describe_position(o.position)}，"
                f"距你 {_describe_distance(o.distance)}{motion}{tag}"
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
        f"• 球门尺寸：两边球门都是 y∈[-0.044, +0.044]（即球门宽度约 0.088 单位）。"
        f"对方球门线在 x=+1.00,自家球门线在 x=-1.00。"
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
