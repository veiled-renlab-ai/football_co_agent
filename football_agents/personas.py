"""Backward-compat shim. Real personas now live in football_agents/players/.

Existing imports `from football_agents.personas import TEAM_BLUE_5V5,
TEAM_RED_5V5` continue to work unchanged.
"""
from .players import (
    TEAM_BLUE_5V5,
    TEAM_RED_5V5,
    BLUE_TEAM_PROFILE,
    RED_TEAM_PROFILE,
    LIN_TAO, WANG_HAO, CHEN_YU, ZHOU_JUN, GAO_LEI,
    ZHAO_QIANG, LIU_FENG, LI_QIANG, SUN_BIN, MA_LIANG,
)

__all__ = [
    "TEAM_BLUE_5V5", "TEAM_RED_5V5",
    "BLUE_TEAM_PROFILE", "RED_TEAM_PROFILE",
    "LIN_TAO", "WANG_HAO", "CHEN_YU", "ZHOU_JUN", "GAO_LEI",
    "ZHAO_QIANG", "LIU_FENG", "LI_QIANG", "SUN_BIN", "MA_LIANG",
]
