import cProfile
import pstats
import gymnasium as gym
import snake_ml  # this runs register.py automatically
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

if __name__ == "__main__":
    n_envs = 32
    n_steps = 512          # rollout per env
    total_rollout = n_envs * n_steps  # = 16,384
    batch_size = 1024      # divides 16,384 evenly
    n_epochs = 10          # you can try 5–8 if speed is critical
    env = make_vec_env("Snake-one-hot-v0", n_envs=n_envs)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 256, 128],
            vf=[512, 256, 128]
        ),
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",       # your GPU will be used
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        #learning_rate=3e-4,
        #gamma=0.99,
        #gae_lambda=0.95,
        #clip_range=0.2,
        #ent_coef=0.01,
        #vf_coef=0.5,
        #max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # profiler = cProfile.Profile()
    # profiler.enable()

    model.learn(total_timesteps=10_000)
    model.save("ppo_snake")

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumtime").print_stats(50)  