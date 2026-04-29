"""Player personas for both 5v5 and 11v11 LLM football matches.

Existing 5v5 personas live in their own files (blue_gk.py / blue_cb.py / etc.).
The 6 extra personas per team needed for 11v11 are bundled into blue_extras.py
and red_extras.py to avoid file sprawl — they're mostly trivial PlayerPersona
constructors over shared disciplines (CM/RB/RCB/LM).
"""
from .blue_gk import LIN_TAO
from .blue_rm import WANG_HAO
from .blue_cf import CHEN_YU
from .blue_lb import ZHOU_JUN
from .blue_cb import GAO_LEI
from .blue_extras import ZHANG_WEI, LI_MING, WANG_GANG, SUN_JIAN, HAN_LEI, ZHOU_KAI
from .red_gk import ZHAO_QIANG
from .red_rm import LIU_FENG
from .red_cf import LI_QIANG
from .red_lb import SUN_BIN
from .red_cb import MA_LIANG
from .red_extras import HUANG_TAO, WU_FEI, LUO_CHENG, JIANG_HU, XIE_YONG, ZHU_HAO
from .team_profiles import BLUE_TEAM_PROFILE, RED_TEAM_PROFILE

# 5v5 (legacy demos)
TEAM_BLUE_5V5 = (LIN_TAO, WANG_HAO, CHEN_YU, ZHOU_JUN, GAO_LEI)
TEAM_RED_5V5  = (ZHAO_QIANG, LIU_FENG, LI_QIANG, SUN_BIN, MA_LIANG)

# 11v11 — slot order matches the llm_11v11_full scenario AddPlayer order:
#   0=GK, 1=LB, 2=LCB, 3=RCB, 4=RB, 5=LCM, 6=CCM, 7=RCM, 8=LM, 9=CF, 10=RM
TEAM_BLUE_11V11 = (
    LIN_TAO, ZHOU_JUN, GAO_LEI, ZHANG_WEI, LI_MING,
    WANG_GANG, SUN_JIAN, HAN_LEI,
    ZHOU_KAI, CHEN_YU, WANG_HAO,
)
TEAM_RED_11V11 = (
    ZHAO_QIANG, SUN_BIN, MA_LIANG, HUANG_TAO, WU_FEI,
    LUO_CHENG, JIANG_HU, XIE_YONG,
    ZHU_HAO, LI_QIANG, LIU_FENG,
)
