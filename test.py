from functools import partial
import json
import os
from pathlib import Path
from stable_baselines3 import PPO
import gymnasium as gym
import torch
from multiprocessing import Pool
import snake_ml

# Avoid oversubscription
torch.set_num_threads(1)

def data_exist(versions):
    """
    returns list of lists, where one list contains all the models of one version
    """
    models = []
    data = []
    for version in versions:
        files = os.listdir(version)
        if 'data.json' in files:
            data.append(os.path.join(version, 'data.json'))
        else:
            names = [os.path.join(version, f) for f in os.listdir(version)]
            models.append(names)
    return models, data


def env_from_path(model_path: Path):
    """
    Decide which env to create based on the checkpoint path.
    """
    parts = [p.lower() for p in model_path.parts]
    if "snake_one-hot" in parts:
        return "snake_one-hot"
    elif "snake_int" in parts:
        return "snake_int"
    else:
        raise ValueError(f"Unknown env type for {model_path}")


def test_model(model_path, max_steps, num_episodes):
    """
    Evaluate one model for num_episodes and return episode rewards.
    """
    env_id = env_from_path(Path(model_path))

    # load on CPU to avoid many CUDA contexts
    model = PPO.load(model_path, device="cpu")
    env = gym.make(env_id, render_mode=None)

    episodes = {"path": model_path, "results": []}

    for i in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        food_info, death_info = [], {}
        while True:
            # Model predicts an action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if "path" in info:
                food_info.append(info)
                
            death_info = info

            if terminated or truncated or info['steps_since_food'] >= max_steps:
                episodes['results'].append({
                    "episode_id": i,
                    "total_reward": total_reward,
                    "food_info": food_info,
                    "death_info": death_info,
                })
                break

    env.close()
    return episodes


def test_all(max_steps, num_episodes, model_dir):
    models_types = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]

    versions = [
        os.path.join(model_type, f)
        for model_type in models_types
        for f in os.listdir(model_type)
    ]

    version_collection, result_files = data_exist(versions)
    results = []
    for version_list in version_collection:
        print("Working on version List:")
        print(version_list)
        partial_test_model = partial(test_model, 
                                        max_steps=max_steps,
                                        num_episodes=num_episodes
                                    )
    
        with Pool(processes=int(os.cpu_count())) as pool:
            results.append(list(pool.imap_unordered(partial_test_model, version_list)))
    
    with open('results.json', 'w') as file:
        json.dump(results, file)

if __name__ == "__main__":

    max_steps = 14*14*2
    num_episodes = 100

    model_dir = './models'

    test_all(max_steps, num_episodes, model_dir)