from collections import defaultdict
import json
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors


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


def data_to_frames(version: dict) -> pd.DataFrame:
    data = sorted(version, key=get_steps)
    
    df = pd.DataFrame(data)

    # Extract features from path
    path_details = df['checkpoint_path'].str.extract(
        r'models/(?P<model_name>.*?)/(?P<version>.*?)/model_(?P<steps>\d+)_steps.zip'
    )
    path_details['steps'] = pd.to_numeric(path_details['steps'])

    # --- Create Checkpoints DataFrame ---
    death_causes_df = pd.json_normalize(df['death_causes']).add_prefix('death_cause_')
    df_checkpoints = pd.concat([
        df.drop(['checkpoint_path', 'death_causes', 'avg_path_ratios', 'death_positions'], axis=1),
        path_details,
        death_causes_df
    ], axis=1)

    # --- Create Death Positions DataFrame ---
    df_positions = df[['checkpoint_path', 'death_positions']].copy()
    df_positions = pd.concat([df_positions, path_details], axis=1)
    df_positions_exploded = df_positions.explode('death_positions')
    df_positions_exploded[['death_x', 'death_y']] = pd.DataFrame(
        df_positions_exploded['death_positions'].tolist(), index=df_positions_exploded.index
    )
    df_positions_tidy = df_positions_exploded.drop(['checkpoint_path', 'death_positions'], axis=1)

    # --- Create Path Ratios DataFrame ---
    df_ratios = df[['checkpoint_path', 'avg_path_ratios']].copy()
    df_ratios = pd.concat([df_ratios, path_details], axis=1)
    df_ratios['path_ratio_items'] = df_ratios['avg_path_ratios'].apply(lambda d: list(d.items()))
    df_ratios_exploded = df_ratios.explode('path_ratio_items')
    df_ratios_exploded[['score', 'avg_path_length']] = pd.DataFrame(
        df_ratios_exploded['path_ratio_items'].tolist(), index=df_ratios_exploded.index
    )
    df_ratios_exploded['score'] = pd.to_numeric(df_ratios_exploded['score'])
    df_ratios_exploded['avg_path_length'] = pd.to_numeric(df_ratios_exploded['avg_path_length'])
    df_path_ratios_tidy = df_ratios_exploded.drop(['checkpoint_path', 'avg_path_ratios', 'path_ratio_items'], axis=1)

    return df_checkpoints, df_positions_tidy, df_path_ratios_tidy


def learning_graphs(data):
    aggregated_data = []
    for model in data:
        aggregated_data.append(list(aggregate_checkpoints_data(model)))

    checkpoints_list, positions_list, ratios_list = [], [], []

    # Process each checkpoint dictionary
    for checkpoint_data in aggregated_data:
        df_c, df_p, df_r = data_to_frames(checkpoint_data)
        checkpoints_list.append(df_c)
        positions_list.append(df_p)
        ratios_list.append(df_r)

    # Concatenate all dataframes into final, aggregated ones
    all_checkpoints = pd.concat(checkpoints_list, ignore_index=True)
    all_positions = pd.concat(positions_list, ignore_index=True)
    all_path_ratios = pd.concat(ratios_list, ignore_index=True)

    print("--- Aggregated Checkpoint DataFrame ---")
    print(all_checkpoints)

    print("\n--- Aggregated Path Ratios DataFrame ---")
    print(all_path_ratios.head())

    # --- Generate and Save Heatmaps ---
    # Group by model and version to create a plot for each unique model
    for (model_name, version), group_df in all_path_ratios.groupby(['model_name', 'version']):
        print(f"\nGenerating heatmap for model: {model_name} (v{version})...")

        # Create a copy to safely add a new column
        group_df_copy = group_df.copy()

        # Map the unique sorted 'steps' to a simple 1-based index
        unique_steps = sorted(group_df_copy['steps'].unique())
        step_to_index_map = {step: i + 1 for i, step in enumerate(unique_steps)}
        group_df_copy['checkpoint_index'] = group_df_copy['steps'].map(step_to_index_map)
        
        # Pivot using the new 'checkpoint_index' for the x-axis
        pivot_df = group_df_copy.pivot(index='score', columns='checkpoint_index', values='avg_path_length')

        plt.figure(figsize=(12, 8))
        # Use PowerNorm to emphasize differences in lower values without unreadable log labels
        sns.heatmap(
            pivot_df,
            cmap='viridis',
            annot=False,
            fmt=".2f",
            norm=mcolors.PowerNorm(gamma=0.5) # Apply power-law normalization
        )
        
        # Invert the Y-axis to have higher scores at the top
        plt.gca().invert_yaxis()

        plt.title(f'Avg Path Length | Model: {model_name} {version}', fontsize=16)
        plt.xlabel('Checkpoint Number') # Use the new, cleaner axis label
        plt.ylabel('Score')

        output_filename = f'heatmap_{model_name}_{version}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to avoid displaying it in a loop

        print(f"Graph saved to {output_filename}")

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
