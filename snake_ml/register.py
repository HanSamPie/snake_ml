from gymnasium.envs.registration import register

#one_hot encoding
register(
    # rewards death food and win
    id="snake_one-hot",
    entry_point="snake_ml.envs.one_hot.snake_int:SnakeEnv",
)

# int encoding
register(
    # based on Snake-one-hot-v0
    id="snake_int",
    entry_point="snake_ml.envs.int_encoding.snake_int:SnakeEnv",
)
