from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="snake_ml.envs.snake_env_0:SnakeEnv",
)