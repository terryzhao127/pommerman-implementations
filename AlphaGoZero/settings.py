"""Some settings related to game environment"""
from pommerman import constants

# Environment-related variables
game_config_id = 'OneVsOne-v0'
board_size = constants.BOARD_SIZE_ONE_VS_ONE
win_reward = 1
tie_reward = 0
lose_reward = -1
RewardType = float
num_agents = 2
