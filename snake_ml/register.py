from gymnasium.envs.registration import register

#one_hot encoding
register(
    # rewards death food and win
    id="Snake-one-hot-v0",
    entry_point="snake_ml.envs.one_hot.snake_env_0:SnakeEnv",
)
register(
    # truncate after board_size * 1.2 steps
    # reward distance reduction by 0.05 otherwise -0.002
    id="Snake-one-hot-v1",
    entry_point="snake_ml.envs.one_hot.snake_env_1:SnakeEnv",
)
register(
    # only punish steps after steps_since_food > len(snake)*1.2
    # TODO consider passing steps as parameter
    id="Snake-one-hot-v2",
    entry_point="snake_ml.envs.one_hot.snake_env_2:SnakeEnv",
)
register(
    # reward scaled by length. Staying alive longer better
    # scale step panalty
    # TODO consider passing steps as parameter
    id="Snake-one-hot-v3",
    entry_point="snake_ml.envs.one_hot.snake_env_3:SnakeEnv",
)

# int encoding
register(
    # based on Snake-one-hot-v0
    id="Snake-int-v0",
    entry_point="snake_ml.envs.int_encoding.snake_env_0:SnakeEnv",
)