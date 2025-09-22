from collections import defaultdict
import json
from pathlib import Path
import seaborn

def get_steps(data):
    fname = Path(data["path"]).name
    if fname == "model.zip":
        return float("inf")  # put final model last
    if "steps" in fname:
        return int(fname.split("_")[1])  # extract step number
    print('fuck: get_steps')
    return -1  # fallback if format unexpected


def sort_by_steps(data):
    versions = []
    for version in data:
        sorted_version = sorted(version, key=get_steps)
        versions.append(sorted_version)
    return versions


def data_per_episode(episode):
    # path
    # death cause
        # score
    # deprecated
    food_info = episode['food_info']
    paths = [info['path'] for info in food_info]
    
    death_info_raw = episode.get('death_info', {})
    deprecated = not bool(death_info_raw)

    death_info = {}
    if not deprecated:
        death_info = {
            'score': death_info_raw.get('score'),
            'death': death_info_raw.get('death')
        }

    return {
        "paths": paths,
        'deprecated': deprecated,
        'death_info': death_info,
        'total_reward': episode['total_reward']
    }


def data_per_checkpoint(checkpoints):
    for checkpoint in checkpoints:
        # 1. Process all episodes in the current checkpoint
        all_episode_data = [data_per_episode(episode) for episode in checkpoint.get('results', [])]

        # 2. Separate valid runs from deprecated ones
        valid_episodes = [data for data in all_episode_data if not data['deprecated']]
        num_deprecated = len(all_episode_data) - len(valid_episodes)
        
        # 3. Aggregate basic data (handle division by zero)
        scores = [data['death_info']['score'] for data in valid_episodes if data.get('death_info', {}).get('score') is not None]
        mean_score = sum(scores) / len(scores) if scores else 0
        rewards = [data['total_reward'] for data in valid_episodes if data.get('total_reward') is not None]


        # 4. Aggregate death cause data
        death_causes = defaultdict(int) # Automatically handles new keys
        for data in valid_episodes:
            cause = data.get('death_info', {}).get('death', {}).get('cause', 'unknown')
            death_causes[cause] += 1

        # aggregate data for deaths        
        death_positions = [data['death_info']['death']['position']
                          for data in all_episode_data if data['death_info'] != {} ]


        # 5. Aggregate path data
        # Use defaultdict to fix the KeyError
        path_ratios_by_fruit = defaultdict(list)
        for episode in valid_episodes:
            for path in episode['paths']:
                # Safely calculate ratio, avoiding division by zero
                optimal = path.get('optimal_path', 0)
                actual = path.get('actual_path', 0)
                if optimal > 0:
                    path_ratio = actual / optimal
                    fruit_num = path.get('fruit_num')
                    if fruit_num is not None:
                        path_ratios_by_fruit[fruit_num].append(path_ratio)

        # Calculate the average ratio for each fruit number
        avg_path_ratios = {
            num: sum(ratios) / len(ratios)
            for num, ratios in path_ratios_by_fruit.items()
        }


def learning_graphs(data):
    results = []
    for version in data:
        results = data_per_checkpoint(version)


def result_graphs(data):
    pass


if __name__ == '__main__':
    with open('results.json', 'r') as file:
        data = json.load(file)

    #sorted_data = sort_by_steps(data)
    learning_graphs(data)
    result_graphs(data)
