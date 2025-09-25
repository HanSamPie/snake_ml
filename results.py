from collections import defaultdict
import json
from pathlib import Path
import pandas as pd
import seaborn as sns


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
        death_causes['deprecated'] = num_deprecated

        NUM_EPISODES = 100
        if not sum(death_causes.values()) == NUM_EPISODES:
            raise RuntimeError(f"More than or less then {NUM_EPISODES} death causes: {sum(death_causes.values())}")

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
            "score_distribution": scores,
            "mean_score": mean_score,
            "mean_reward": mean_reward,
            "death_causes": dict(death_causes),
            "death_positions": death_positions,
            "avg_path_ratios": avg_path_ratios
        }


def get_steps(data):
    fname = Path(data["checkpoint_path"]).name
    if fname == "model.zip":
        return float("inf")  # put final model last
    if "steps" in fname:
        return int(fname.split("_")[1])  # extract step number
    print('fuck: get_steps')
    return -1  # fallback if format unexpected


def data_to_frame(version: dict) -> pd.DataFrame:
    data = sorted(version, key=get_steps)
    
    # 1. Load the initial data
    df = pd.DataFrame(data)

    # 2. Extract features from the checkpoint path using regex
    # This creates new columns for model_name, version, and steps
    path_details = df['checkpoint_path'].str.extract(
        r'models/(?P<model_name>.*?)/(?P<version>.*?)/model_(?P<steps>\d+)_steps.zip'
    )

    # Convert steps to a numeric type for plotting and sorting
    path_details['steps'] = pd.to_numeric(path_details['steps'])
    # 3. Flatten the nested dictionary columns
    # pd.json_normalize is perfect for turning a column of dicts into new columns
    death_causes_df = pd.json_normalize(df['death_causes']).add_prefix('death_cause_')


    # 4. Create the main DataFrame for agent-level analysis
    # We combine the extracted and flattened data, and drop the original complex columns
    df_checkpoints = pd.concat([
        df.drop(['checkpoint_path', 'death_causes', 'avg_path_ratios', 'death_positions'], axis=1),
        path_details,
        death_causes_df
    ], axis=1)

    print("--- Main Checkpoint DataFrame ---")
    print(df_checkpoints)
    
    # 5. Create a separate, exploded DataFrame for death positions
    # This is for analyzing data with a different granularity (one row per death event)
    df_positions = df[['checkpoint_path', 'death_positions']].copy()
    df_positions['steps'] = path_details['steps'] # Add steps for linking data

    # .explode() creates a new row for each item in the list
    df_positions_exploded = df_positions.explode('death_positions')

    # Split the [x, y] coordinates into separate columns
    df_positions_exploded[['death_x', 'death_y']] = pd.DataFrame(
        df_positions_exploded['death_positions'].tolist(),
        index=df_positions_exploded.index
    )

    # Clean up the final DataFrame
    df_positions_exploded = df_positions_exploded.drop(['checkpoint_path', 'death_positions'], axis=1)


    print("\n--- Exploded Death Positions DataFrame ---")
    print(df_positions_exploded.head()) # Print first 5 rows

    # 6. Create a separate, tidy DataFrame for average path ratios
    # This is better than the wide format as it handles variable dict lengths
    df_ratios = df[['checkpoint_path', 'avg_path_ratios']].copy()
    df_ratios['steps'] = path_details['steps']

    # Convert the dictionary into a list of (key, value) tuples to prepare for exploding
    df_ratios['path_ratio_items'] = df_ratios['avg_path_ratios'].apply(lambda d: list(d.items()))

    # Explode the list, so each (key, value) tuple gets its own row
    df_ratios_exploded = df_ratios.explode('path_ratio_items')

    # Split the tuple into separate 'score' and 'avg_path_length' columns
    df_ratios_exploded[['score', 'avg_path_length']] = pd.DataFrame(
        df_ratios_exploded['path_ratio_items'].tolist(),
        index=df_ratios_exploded.index
    )

    # Convert to numeric types for plotting
    df_ratios_exploded['score'] = pd.to_numeric(df_ratios_exploded['score'])
    df_ratios_exploded['avg_path_length'] = pd.to_numeric(df_ratios_exploded['avg_path_length'])

    # Clean up the final DataFrame
    df_path_ratios_tidy = df_ratios_exploded.drop(
        ['checkpoint_path', 'avg_path_ratios', 'path_ratio_items'], axis=1
    )

    print("\n--- Tidy Path Ratios DataFrame ---")
    print(df_path_ratios_tidy.head())
    print("\nThis 'long' format DataFrame is ideal for plotting average path length vs. score.")


def learning_graphs(data):
    aggregated_data = []
    for version in data:
        aggregated_data.append(list(aggregate_checkpoints_data(version)))

    data_frames = []
    for version in aggregated_data:
        data_to_frame(version)

    with open('aggregated-data.json', 'w') as file:
        json.dump(aggregated_data, file, indent=2)


def result_graphs(data):
    pass


if __name__ == '__main__':
    with open('results.json', 'r') as file:
        data = json.load(file)

    #sorted_data = sort_by_steps(data)
    learning_graphs(data)
    result_graphs(data)
