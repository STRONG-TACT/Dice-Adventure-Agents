# Dice Adventure RL Agent 
This directory contains scripts for training the RL agent in a vectorized environment.

## Training the Agent

To train the agent in a vectorized environment, use the `launch_games.sh` script to launch the games in parallel first, and then run the `train_agent_vec.py` script to train the agent.

```
./launch_games.sh 12 6060
python train_agent_vec.py --port_start 6060
```
The above command will launch 12 instances of the game on port 6060 and then train the agent in a vectorized environment.

## Training Environment
The training environment is defined in `dice_adventure_python_env_new.py`, any changes to the observation space, action space, or reward function should be made in this file. The `observation_config_new.json` file contains the configuration for the observation space, which is used to generate the observation for the agent.

## Feature Extraction
The `cnn_extractor.py` file contains the code for the CNN feature extractor. It extracts features from the observation space and passes them to the agent.
