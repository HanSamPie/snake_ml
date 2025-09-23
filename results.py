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
    """
    Extracts and structures relevant data from a single episode dictionary.
    """
    food_info = episode['food_info']
    paths = [info['path'] for info in food_info]

    death_info_raw = episode['death_info']
    deprecated = not bool(death_info_raw)

    death_info = {}
    if not deprecated:
        death_info = {
            'score': death_info_raw['score'],
            'death': death_info_raw['death']
        }

    return {
        "paths": paths,
        'deprecated': deprecated,
        'death_info': death_info,
        'total_reward': episode['total_reward']
    }


def aggregate_checkpoints_data(checkpoints):
    """
    Processes a list of checkpoints and yields aggregated metrics for each one.
    """
    for checkpoint in checkpoints:
        # Process each episode within the current checkpoint
        all_episode_data = [data_per_episode(episode) for episode in checkpoint['results']]

        # Separate episodes into valid and deprecated runs
        valid_episodes = [data for data in all_episode_data if not data['deprecated']]
        num_deprecated = len(all_episode_data) - len(valid_episodes)
        
        # Aggregate scores and rewards from valid episodes
        scores = [data['death_info']['score'] for data in valid_episodes]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        
        rewards = [data['total_reward'] for data in valid_episodes]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Tally death causes and collect death positions
        death_causes = defaultdict(int)
        death_positions = []
        for data in valid_episodes:
            cause = data['death_info']['death']['cause']
            death_causes[cause] += 1

            position = data['death_info']['death']['position']
            if position is not None:
                death_positions.append(position)

        # Calculate path efficiency ratios for each fruit
        path_ratios_by_fruit = defaultdict(list)
        for episode in valid_episodes:
            for path in episode['paths']:
                optimal = path['optimal_path']
                actual = path['actual_path']
                if optimal > 0:
                    path_ratio = actual / optimal
                    fruit_num = path['fruit_num']
                    path_ratios_by_fruit[fruit_num].append(path_ratio)

        avg_path_ratios = {
            num: sum(ratios) / len(ratios)
            for num, ratios in path_ratios_by_fruit.items()
        }

        # Yield the aggregated data for the current checkpoint
        yield {
            "checkpoint_path": checkpoint['path'],
            "total_episodes": len(all_episode_data),
            "num_deprecated": num_deprecated,
            "mean_score": mean_score,
            "mean_reward": mean_reward,
            "death_causes": dict(death_causes),
            "death_positions": death_positions,
            "avg_path_ratios": avg_path_ratios
        }




def learning_graphs(data):
    results = []
    for version in data:
        results = aggregate_checkpoints_data(version)


def result_graphs(data):
    pass


if __name__ == '__main__':
    with open('results.json', 'r') as file:
        data = json.load(file)

    #sorted_data = sort_by_steps(data)
    learning_graphs(data)
    result_graphs(data)
