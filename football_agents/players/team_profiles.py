"""Per-team tactical-tendency profiles (Phase 5c). Attached to each
persona's team_profile field. Future: replaced when real team data ships.
"""
from ..prompts import TeamProfile

BLUE_TEAM_PROFILE = TeamProfile(
    name="蓝队",
    character=(
        "传控渗透型球队，阵地战擅长。中场拿球后求精确出球、不求快。"
        "防守靠整体压缩空间，不靠单兵抢断。"
    ),
)

RED_TEAM_PROFILE = TeamProfile(
    name="红队",
    character=(
        "高位逼抢、转换反击型球队。失球第一时间就上抢，得球后追求快速纵向打穿。"
        "防守靠前场压迫制造混乱，不喜欢长时间阵地战。"
    ),
)
