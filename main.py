# maybe train embeddings later

# potential inputs
# 1) arr with snake body position
# 2) food pos
# 3) snake length
# 4) danger flags

import gymnasium as gym
import numpy as np

# 14 * 14 for game and * 4 one-hot encoding
obs_shape = 14 * 14 * 4

observation_space = gym.spaces.Box(low=0.0, high=01.0, shape=obs_shape, dtype=np.float32)

action_space = gym.spaces.Discrete(4) # 0=up, 1=right, 2=down, 3=left