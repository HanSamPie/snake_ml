import gymnasium as gym
import snake_ml  # this runs register.py automatically
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

if __name__ == "__main__":
    env = make_vec_env("Snake-one-hot-v2", n_envs=n_envs)

    model = PPO.load("ppo_snake.zip")

    model.learn(total_timesteps=20_000_000)
    model.save("ppo_snake")
