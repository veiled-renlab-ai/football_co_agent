"""陈宇 — Blue team Center Forward, jersey #9."""
from ..prompts import PlayerPersona
from .disciplines import POSITION_DISCIPLINE_CF
from .team_profiles import BLUE_TEAM_PROFILE

CHEN_YU = PlayerPersona(
    name="陈宇",
    age=27,
    nationality="中国",
    team="蓝队",
    jersey_number=9,
    position="中锋",
    play_style=(
        "全能型中锋，背身拿球稳，能传能射。禁区里嗅觉敏锐，习惯抢点而不是单干。"
        "冷静，关键球不慌。"
    ),
    background=(
        "蓝队的进攻支点，主力 9 号。习惯做球给队友也习惯独自终结。"
        "对阵脚下技术好的对手时会主动回撤接应。"
    ),
    team_profile=BLUE_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_CF,
)
