"""Player personas for the 5v5 LLM football match.

Each player has its OWN file in this folder so changing one player doesn't
risk touching others. Add new players by adding a new file + appending to
TEAM_BLUE_5V5 / TEAM_RED_5V5 below.
"""
from .blue_gk import LIN_TAO
from .blue_rm import WANG_HAO
from .blue_cf import CHEN_YU
from .blue_lb import ZHOU_JUN
from .blue_cb import GAO_LEI
from .red_gk import ZHAO_QIANG
from .red_rm import LIU_FENG
from .red_cf import LI_QIANG
from .red_lb import SUN_BIN
from .red_cb import MA_LIANG
from .team_profiles import BLUE_TEAM_PROFILE, RED_TEAM_PROFILE

# Slot order matches multi_agent_runner expectations:
# slot 0/5 = GK, 1/6 = RM, 2/7 = CF, 3/8 = LB, 4/9 = CB
TEAM_BLUE_5V5 = (LIN_TAO, WANG_HAO, CHEN_YU, ZHOU_JUN, GAO_LEI)
TEAM_RED_5V5  = (ZHAO_QIANG, LIU_FENG, LI_QIANG, SUN_BIN, MA_LIANG)
