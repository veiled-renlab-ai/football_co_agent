"""Custom 11v11 scenario — both teams LLM-controlled, no AI assistance.

Mirrors gfootball's 11_vs_11_kaggle 4-3-3 starting layout but disables
both team_difficulty (so LLM agents fully control all 22 players) and
disables early-end conditions so the match runs the full game_duration.

Slot order (gfootball auto-assigns indices 0..10 in AddPlayer order):
    0: GK
    1: LB
    2: LCB
    3: RCB
    4: RB
    5: LCM
    6: CCM
    7: RCM
    8: LM    (left winger in 4-3-3)
    9: CF    (centre forward)
   10: RM    (right winger in 4-3-3)

Install path (run by deploy script):
    /home/<user>/.local/lib/python3.10/site-packages/gfootball/scenarios/llm_11v11_full.py
    /home/<user>/football-env/lib/python3.10/site-packages/gfootball/scenarios/llm_11v11_full.py
"""
from . import *


def build_scenario(builder):
    builder.config().game_duration = 3000
    builder.config().second_half = 1500
    builder.config().right_team_difficulty = 0.05
    builder.config().left_team_difficulty = 0.05
    builder.config().deterministic = False
    builder.config().end_episode_on_score = False
    builder.config().end_episode_on_out_of_play = False
    builder.config().end_episode_on_possession_change = False
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left
    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000,  0.000000, e_PlayerRole_GK, controllable=True)
    builder.AddPlayer(-0.422000, -0.195760, e_PlayerRole_LB)
    builder.AddPlayer(-0.500000, -0.063560, e_PlayerRole_CB)
    builder.AddPlayer(-0.500000,  0.063559, e_PlayerRole_CB)
    builder.AddPlayer(-0.422000,  0.195760, e_PlayerRole_RB)
    builder.AddPlayer(-0.184212, -0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.267574,  0.000000, e_PlayerRole_CM)
    builder.AddPlayer(-0.184212,  0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.010000, -0.216100, e_PlayerRole_LM)
    builder.AddPlayer( 0.000000,  0.000000, e_PlayerRole_CF)
    builder.AddPlayer(-0.010000,  0.216100, e_PlayerRole_RM)
    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000,  0.000000, e_PlayerRole_GK, controllable=True)
    builder.AddPlayer(-0.422000, -0.195760, e_PlayerRole_LB)
    builder.AddPlayer(-0.500000, -0.063560, e_PlayerRole_CB)
    builder.AddPlayer(-0.500000,  0.063559, e_PlayerRole_CB)
    builder.AddPlayer(-0.422000,  0.195760, e_PlayerRole_RB)
    builder.AddPlayer(-0.184212, -0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.267574,  0.000000, e_PlayerRole_CM)
    builder.AddPlayer(-0.184212,  0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.010000, -0.216100, e_PlayerRole_LM)
    builder.AddPlayer(-0.050000,  0.000000, e_PlayerRole_CF)
    builder.AddPlayer(-0.010000,  0.216100, e_PlayerRole_RM)
