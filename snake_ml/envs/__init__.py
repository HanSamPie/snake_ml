# from gymnasium.envs.registration import register


# register(
#     id="snake/Snake-v0",
#     entry_point="snake.snake_env_0:SnakeEnv",
# )


# from snake.snake_env_0 import SnakeEnv


# import gymnasium as gym
# from stable_baselines3 import PPO

# import torch
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
# from torch import nn


# def train():
#     register(
#         id="snake/Snake-v0",
#         entry_point="snake.snake_env_0:SnakeEnv",
#     )
#     n_envs = 8
#     env = make_vec_env("Snake-v0", n_envs=n_envs, vec_env_cls=SubprocVecEnv)
#     env = VecMonitor(env)  # episode stats across vector envs
#     env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

#     # --- policy ---
#     policy_kwargs = dict(
#         activation_fn=nn.ReLU,
#         net_arch=[dict(pi=[256, 256], vf=[256, 256])]
#     )

#     # --- PPO with vectorized envs ---
#     # Note: total rollout per update = n_envs * n_steps = 8 * 256 = 2048
#     model = PPO(
#         "MlpPolicy",
#         env,
#         device="cuda",               # keep if your PyTorch build supports your GPU; otherwise "cpu"
#         n_steps=256,
#         batch_size=256,
#         n_epochs=10,
#         learning_rate=3e-4,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         policy_kwargs=policy_kwargs,
#         verbose=1,
#     )

#     # (optional, avoids CPU oversubscription with many env processes)
#     torch.set_num_threads(1)  # or set OMP_NUM_THREADS=1 in your shell

#     model.learn(total_timesteps=2_000_000)

#     # Save model and VecNormalize statistics for proper evaluation later
#     model.save("ppo_snake")
#     env.save("vecnorm_snake.pkl")

# if __name__ == "__main__":
#     train()

# # env = gym.make("snake/Snake-v0", board_size=14)

# # policy_kwargs = dict(net_arch=[512, 512, 256, 128])
# # model = PPO("MlpPolicy", env, device="cuda", policy_kwargs=policy_kwargs, verbose=1)
# # model.learn(total_timesteps=500_000)

# # model.save("ppo_snake")



# # model = PPO(
# #     policy="MlpPolicy",   # MlpPolicy is fine; extractor provides features
# #     env=env,
# #     policy_kwargs=policy_kwargs,
# #     n_steps=1024,
# #     batch_size=256,
# #     gamma=0.99,
# #     gae_lambda=0.95,
# #     ent_coef=0.01,
# #     vf_coef=0.5,
# #     learning_rate=3e-4,
# #     clip_range=0.2,
# #     verbose=1,
# # )


# # from stable_baselines3 import PPO
# # import gymnasium as gym

# # model = PPO.load("ppo_snake")
# # env = gym.make("snake/Snake-v0", board_size=14, render_mode="human")
# # obs, info = env.reset()
# # done = False
# # while not done:
# #     action, _ = model.predict(obs)
# #     obs, reward, terminated, truncated, info = env.step(action)
# #     done = terminated or truncated
# #     env.render()