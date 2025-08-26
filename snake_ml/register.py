from gymnasium.envs.registration import register

register(
    id="Snake-one-hot-v0",
    entry_point="snake_ml.envs.one_hot.snake_env_0:SnakeEnv",
)
register(
    id="Snake-one-hot-v1",
    entry_point="snake_ml.envs.one_hot.snake_env_1:SnakeEnv",
)
register(
    id="Snake-one-hot-v2",
    entry_point="snake_ml.envs.one_hot.snake_env_2:SnakeEnv",
)
register(
    id="Snake-one-hot-v3",
    entry_point="snake_ml.envs.one_hot.snake_env_3:SnakeEnv",
)