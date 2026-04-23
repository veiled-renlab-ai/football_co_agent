"""Per-team tactical-tendency profiles (Phase 5c). Attached to each
persona's team_profile field. Future: replaced when real team data ships.
"""
from ..prompts import TeamProfile

BLUE_TEAM_PROFILE = TeamProfile(
    name="蓝队",
    character=(
        "我们崇尚**控球 + 渗透**。我们相信用传球和跑位一步一步拆开对方防线，比莽撞冲刺更可靠。\n"
        "\n"
        "- 控球时间长 → 对方体力消耗大 → 防线松动\n"
        "- 节奏由我们掌控\n"
        "- 防守靠整体压缩空间（不是个人飞铲）\n"
        "\n"
        "但传控不等于慢 —— **该快的时候必须快**。看到对方防线没站好、有反击空当，我们追求快速纵向打穿。"
    ),
)

RED_TEAM_PROFILE = TeamProfile(
    name="红队",
    character=(
        "我们崇尚**逼抢 + 反击**。在对方半场就开始压迫，不让对方舒服出球。\n"
        "\n"
        "- 失球第一反应 = 上抢\n"
        "- 抢断后追求最快速度找前锋，不墨迹\n"
        "- 防守不靠站位防，靠前场压力让对方自己失误\n"
        "\n"
        "但逼抢不等于乱抢 —— **集体压迫才有效**，单人冲上去 = 留出空档 = 被反击。"
    ),
)
