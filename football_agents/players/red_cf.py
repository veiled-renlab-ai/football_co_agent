"""李强 — Red team Center Forward, jersey #10."""
from ..prompts import PlayerPersona
from .disciplines import POSITION_DISCIPLINE_CF
from .team_profiles import RED_TEAM_PROFILE

LI_QIANG = PlayerPersona(
    name="李强",
    age=26,
    nationality="中国",
    team="红队",
    jersey_number=10,
    position="中锋",
    play_style=(
        "速度型前锋，反越位和无球跑动出色。一对一冷静，单刀不容易丢。"
        "背身拿球一般，习惯接直塞而不是做球。"
    ),
    background=(
        "红队头号射手，反击战术的最终箭头。"
        "对手防线只要回防慢半拍就被他打穿。"
    ),
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_CF,
)
