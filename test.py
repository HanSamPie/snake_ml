import cProfile
import json
import os
from pathlib import Path

from stable_baselines3 import PPO
import gymnasium as gym

from functools import partial
from multiprocessing import Pool
import snake_ml

def results_exist(versions):
    """
    returns list of lists, where one list contains all the models of one version
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
    env = gym.make(env_str, render_mode=None)
    
    episodes = {}
    episodes['env'] = env_str
    episodes['path'] = model_path
    episodes['results'] = []

    # for i in range(num_episodes):
    #     obs, info = env.reset()

    #     total_reward = 0.0
    #     food_info = []
    #     death_info = {}
    #     #TODO add data tracking for first state
    #     while(True):
    #         # Model predicts an action
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env.step(action)
            
    #         total_reward += reward
    #         if "path" in info:
    #             food_info.append(info)
    #         elif "death" in info:
    #             death_info = info

    #         if terminated or truncated or info['steps_since_food'] >= max_steps:
    #             episode_data = {
    #                 "episode_id": i,
    #                 "food_info": food_info, 
    #                 "death_info": death_info, 
    #                 "total_reward": total_reward 
    #             }
    #             # Append the dictionary to the results list
    #             episodes['results'].append(episode_data)
    #             break

    env.close()
    print(f"Finished run for: {model_path}")
    return episodes

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    
    max_steps = 14*14*2
    num_episodes = 1

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
        print("Working on version List:")
        print(version_list)
        partial_test_model = partial(test_model, 
                                        max_steps=max_steps,
                                        num_episodes=num_episodes
                                    )
    
        with Pool(processes=int(os.cpu_count()/2)) as pool:
            results.append(list(pool.imap_unordered(partial_test_model, version_list)))
    
    with open('data.json', 'w') as file:
        json.dump(results, file)

    profiler.disable()
    profiler.dump_stats("profiler_results.prof")
    profiler.print_stats(sort="time")


# score distribution -> track score at death
#   average score

# death cause -> return string value
# death location (heatmap) -> return cords

# path fitness -> measure of how close the path was to the theoretical shortest path

# Time to score n -> steps since last fruit
# shows the increase in steps needed to reach fruit n compared to n+1

# inputs to score n -> track inputs since last apple
# can provide insights into the number of evasive maneuvers taken 
# (though a stair movement is just as effective in reducing distance as a simple 90° and are therefore not as 
# conclusive to performance and depend more on the reward function and training method)
