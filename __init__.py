from gymnasium.envs.registration import register


register(
    id="snake/Snake-v0",
    entry_point="snake.snake_env_0:SnakeEnv",
)


from snake.snake_env_0 import SnakeEnv


# import gymnasium as gym
# from stable_baselines3 import PPO

# env = gym.make("snake/Snake-v0", board_size=14)

# model = PPO(
#     policy="MlpPolicy",  # use "CnnPolicy" if keeping 3D obs
#     env=env,
#     verbose=1,
# )

# model.learn(total_timesteps=500_000)
# model.save("ppo_snake")



# model = PPO(
#     policy="MlpPolicy",   # MlpPolicy is fine; extractor provides features
#     env=env,
#     policy_kwargs=policy_kwargs,
#     n_steps=1024,
#     batch_size=256,
#     gamma=0.99,
#     gae_lambda=0.95,
#     ent_coef=0.01,
#     vf_coef=0.5,
#     learning_rate=3e-4,
#     clip_range=0.2,
#     verbose=1,
# )


from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("ppo_snake")
env = gym.make("snake/Snake-v0", board_size=14, render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()