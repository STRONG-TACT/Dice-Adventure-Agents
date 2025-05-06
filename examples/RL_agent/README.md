# Dice Adventure RL Agent Metrics Visualization

This directory contains scripts for training and analyzing the performance of the Dice Adventure RL agent.

## Metrics Visualization

The `plot_level_metrics.py` script visualizes the level completion metrics collected during training. It generates various plots to help you understand how the agent's performance improves over time.

### Features

- **Level Completion Trends**: Shows how the number of steps to complete each level changes over episodes
- **Summary Plot**: Displays all levels on a single graph for easy comparison
- **Distribution Plot**: Shows the distribution of steps for each level using box plots
- **Heatmap**: Visualizes average steps by level and episode range
- **Summary Statistics**: Generates statistical summaries of level completion performance

### Requirements

The script requires the following Python packages:
- pandas
- matplotlib
- seaborn

You can install them using pip:
```
pip install pandas matplotlib seaborn
```

### Usage

Run the script from the command line:

```
python plot_level_metrics.py
```

By default, the script will:
1. Look for metrics files in the `metrics` directory
2. Generate plots in the `plots` directory
3. Print summary statistics to the console

#### Command-line Options

- `--file PATH`: Specify a particular metrics file to analyze
- `--output DIR`: Specify the output directory for plots (default: `plots`)

Example:
```
python plot_level_metrics.py --file metrics/level_completion_metrics_Dwarf_0.csv --output my_plots
```

### Output

The script generates the following files:

1. `level_completion_trends_TIMESTAMP.png`: Individual trend plots for each level
2. `level_completion_summary_TIMESTAMP.png`: Combined plot showing all levels
3. `level_completion_distribution_TIMESTAMP.png`: Box plots showing the distribution of steps
4. `level_completion_heatmap_TIMESTAMP.png`: Heatmap of average steps by level and episode range
5. `level_completion_summary_TIMESTAMP.csv`: CSV file with summary statistics

## Training the Agent

To train the agent in a vectorized environment, use the `launch_games.sh` script to launch the games in parallel first, and then run the `train_agent_vec.py` script to train the agent.

```
python train_agent_vec.py --port_start 6060
```

## Training Environment
The training environment is defined in `dice_adventure_python_env_new.py`, any changes to the observation space, action space, or reward function should be made in this file. The `observation_config_new.json` file contains the configuration for the observation space, which is used to generate the observation for the agent.

## Feature Extraction
The `cnn_extractor.py` file contains the code for the CNN feature extractor. It extracts features from the observation space and passes them to the agent.
