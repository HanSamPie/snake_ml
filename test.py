import os
from pathlib import Path

from stable_baselines3 import PPO
import gymnasium as gym

from functools import partial
from multiprocessing import Pool


def results_exist(versions):
    """
    returns list of lists, where one list contians all the models of one version
    """
    models = []
    data = []
    for version in versions:
        files = os.listdir(version)
        if 'results.json' in files:
            data.append(os.path.join(version, 'results.json'))
        else:
            names = [os.path.join(version, f) for f in os.listdir(version)]
            models.append(names)
    return models, data

def test_model(model_path, max_steps, num_episodes):    
    path = Path(model_path)

    env_str = path.parts[-3]

    model = PPO.load(model_path)
    env = gym.make(env_str, render_mode="human")
    
    obs, info = env.reset()
    for step in range(max_steps):
        # Model predicts an action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated or step == max_steps - 1:
            print(f"Episode ended after {step+1} steps, reward={reward}, info={info}")
            break

    env.close()

if __name__ == '__main__':
    max_steps = 14*14*1.5
    num_episodes = 100

    model_dir = './models'
    models_types = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    
    versions = [
        os.path.join(model_type, f)
        for model_type in models_types
        for f in os.listdir(model_type)
    ]
    
    version_collection, result_files = results_exist(versions)
    results = []
    for version_list in version_collection:
        partial_test_model = partial(test_model, 
                                        max_steps=max_steps,
                                        num_episodes=num_episodes
                                    )
    
        with Pool() as pool:
            results.append(list(pool.imap_unordered(partial_test_model, version_list)))
    
    print(results)


# import snake_ml  # this runs register.py automatically
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# import torch

# def load_train():
#     env = make_vec_env(env_str, n_envs=n_envs)
#     model = PPO.load(
#         path=load_path,
#         env=env,
#         device=device,
#     )

#     model.learn(total_timesteps=timesteps)
#     model.save(save_path)



# def test(max_steps=200, render=True):
#     model = PPO.load(save_path)

#     env = gym.make(env_str, render_mode="human")
    
#     obs, info = env.reset()
#     for step in range(max_steps):
#         # Model predicts an action
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
        
#         if render:
#             env.render()
        
#         if terminated or truncated or step == max_steps - 1:
#             print(f"Episode ended after {step+1} steps, reward={reward}, info={info}")
#             break

#     env.close()


# if __name__ == "__main__":
#     env_str = "Snake-one-hot-v1"
#     load_version = 1.0
#     load_path = f"hot_large_small-reward_{load_version}.zip"
#     save_version = 1.1
#     save_path = f"hot_large_small-reward_{save_version}.zip"
#     device = "cuda"
    
#     timesteps = 10_000_000
#     ns_env = 48

#     print("=== Starting Snake PPO Script ===")
#     load_train()
#     test(max_steps=10000, render=True)
#     print("=== Script finished ===")


# if __name__ == "__main__":
#     import torch
#     from multiprocessing import Process

#     version = 1.0
#     save_path1 = f"models/snake_one-hot/v{version}"
#     save_path2 = f"models/snake_int/v{version}"
#     device = "cuda"
#     n_envs = 48

#     onehot_policy = dict(
#         net_arch=dict(
#             pi=[1024, 512, 512, 256, 64],
#             vf=[1024, 512, 512, 256, 64],
#         ),
#         activation_fn=torch.nn.ReLU
#     )

#     int_policy = dict(
#         net_arch=dict(
#             pi=[256, 256, 256, 128, 64],
#             vf=[256, 256, 256, 128, 64],
#         ),
#         activation_fn=torch.nn.ReLU
#     )

#     base_kwargs = dict(
#         n_steps=1024,
#         batch_size=2048,
#         n_epochs=10,
#         learning_rate=0.0003,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         vf_coef=0.5,
#         device=device,
#     )

#     timesteps = 20_000_000

#     # different kwargs for each policy
#     onehot_kwargs = dict(base_kwargs, policy_kwargs=onehot_policy)
#     int_kwargs = dict(base_kwargs, policy_kwargs=int_policy)

#     jobs = []
#     jobs.append(Process(target=train_model, args=("snake_one-hot", save_path1, timesteps), kwargs={'ppo_kwargs': onehot_kwargs}))
#     jobs.append(Process(target=train_model, args=("snake_int", save_path2, timesteps), kwargs={'ppo_kwargs': int_kwargs}))

#     for j in jobs: j.start()
#     for j in jobs: j.join()