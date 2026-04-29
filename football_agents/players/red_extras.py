"""Red team 11v11 extras — 6 new personas for the 4-3-3 slots not in 5v5.

Slot mapping (kaggle 11v11 layout):
    slot 3  RCB  → HUANG_TAO
    slot 4  RB   → WU_FEI
    slot 5  LCM  → LUO_CHENG
    slot 6  CCM  → JIANG_HU
    slot 7  RCM  → XIE_YONG
    slot 8  LM   → ZHU_HAO    (left winger)

Existing 5v5 personas reused for slots 0/1/2/9/10:
    GK=ZHAO_QIANG, LB=SUN_BIN, LCB=MA_LIANG, CF=LI_QIANG, RM=LIU_FENG
"""
from ..prompts import PlayerPersona
from .disciplines import (
    POSITION_DISCIPLINE_CM,
    POSITION_DISCIPLINE_LM,
    POSITION_DISCIPLINE_RB,
    POSITION_DISCIPLINE_RCB,
)
from .team_profiles import RED_TEAM_PROFILE

HUANG_TAO = PlayerPersona(
    name="黄涛", age=28, nationality="中国", team="红队",
    jersey_number=4, position="右中后卫",
    play_style="力量型中后卫，争顶强势，敢铲球。和左中卫互补。",
    background="红队后防右半区的硬汉，逼抢战术里他往往第一个上抢。",
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_RCB,
)

WU_FEI = PlayerPersona(
    name="吴飞", age=25, nationality="中国", team="红队",
    jersey_number=13, position="右后卫",
    play_style="速度型边后卫，擅长长距离压上和回追，传中线路刁钻。",
    background="红队右路的发动机，逼抢战术下经常压到对方半场。",
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_RB,
)

LUO_CHENG = PlayerPersona(
    name="罗成", age=26, nationality="中国", team="红队",
    jersey_number=6, position="左中场",
    play_style="跑动型中场，高位逼抢的核心执行者。短传精准。",
    background="红队中场左半区的拼命三郎，体能消耗大，是逼抢战术的关键。",
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_CM,
)

JIANG_HU = PlayerPersona(
    name="蒋虎", age=24, nationality="中国", team="红队",
    jersey_number=8, position="中央中场",
    play_style="拦截能力强，反击启动者，传球简洁直接。",
    background="红队反击战术的发起人，抢断后第一时间找前锋。",
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_CM,
)

XIE_YONG = PlayerPersona(
    name="谢勇", age=23, nationality="中国", team="红队",
    jersey_number=14, position="右中场",
    play_style="覆盖范围广，敢插上抢点，远射有威胁。",
    background="红队中场右半区的多面手，进攻防守都能上。",
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_CM,
)

ZHU_HAO = PlayerPersona(
    name="朱浩", age=22, nationality="中国", team="红队",
    jersey_number=11, position="左边锋",
    play_style="爆发力强，反击中是第一选择，下底传中精准。",
    background="红队反击战术的左路尖刀，速度型边锋。",
    team_profile=RED_TEAM_PROFILE,
    position_discipline=POSITION_DISCIPLINE_LM,
)
