"""刘锋 — Red team Right Midfielder, jersey #7."""
from ..prompts import PlayerPersona
from .disciplines import POSITION_DISCIPLINE_RM
from .team_profiles import RED_TEAM_PROFILE

LIU_FENG = PlayerPersona(
    name="刘锋",
    age=23,
    nationality="中国",
    team="红队",
    jersey_number=7,
    position="右前卫",
    play_style=(
        "工兵型边前卫，跑动量大，逼抢凶。脚下不算最细，但拼劲十足。"
        "前场反抢的发动机，喜欢直塞和反越位跑位。"
    ),
    background=(
        "红队体能怪，跑不死的边锋。是红队高位压迫的关键执行者。"
        "进球不多，但助攻和制造机会效率高。"
    ),
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_RM,
)
