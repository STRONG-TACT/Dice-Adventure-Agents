import gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from cnn_extractor import CustomCNN
# Import the registration (this needs to happen before creating the env)
import gym
# from dice_adventure_python_env_simp import DiceAdventurePythonEnvRL
from dice_adventure_python_env_new import DiceAdventurePythonEnvRL
import json
import logging
import sys
import socket
from datetime import datetime
import argparse
from typing import List
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker



def _make_envs(num_envs: int, players: list, env_args: dict):
    # def make_env():
    #     env = DiceAdventurePythonEnvRL(**env_args)
    #     # env = Monitor(env)  # Add Monitor wrapper

    #     env = ActionMasker(env, action_mask_fn=env.action_masks)  # Wrap environment with ActionMasker
    #     return env

    envs = [
        _get_env(env_id=int(i * num_envs + j),
                player=p,
                env_args=env_args)
        for i, p in enumerate(players)
        for j in range(num_envs)
    ]
    return DummyVecEnv(envs)


def _get_env(env_id, player, env_args):
    def env_fxn():
        env =  DiceAdventurePythonEnvRL(
            env_id=env_id,
            game_executable_filepath=env_args["game_executable_filepath"],
            port=env_args["port"][env_id],
            player=player,
            train_mode=env_args["train_mode"]
        )
       
        return env
    return env_fxn



def train_vec_env(port_list: List[str], num_envs: int, players: List[str]):
    assert len(port_list) / num_envs == len(players), f"Number of environments must be a multiple of number of players, got {len(port_list)} and {num_envs}"

    vec_env = _make_envs(num_envs=num_envs, players=players, env_args={"game_executable_filepath": "game/V1.0.5/DiceAdventure.x86_64", "port": port_list, "train_mode": True})
    logging.info(f"Created {num_envs} environments for {len(players)} players, total {len(port_list)} environments")

     # Define policy kwargs with the custom CNN
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": 256
        }
    }
    logging.info("Policy kwargs defined")

    # Create the model with single environment
    model = MaskablePPO(
        "CnnPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,  
        tensorboard_log="./logs/ppo_vec_env_0426/",
        learning_rate=3e-4,
        n_steps=2048,  
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    logging.info("Model created")

    # Add checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, 
        save_path="./model_checkpoints_0426/",
        name_prefix="dice_adventure_model_0426",
        verbose=1
    )
    # RewardMonitorCallback()
    
    logging.info("Checkpoint callback created")
    logging.info("Starting training...")
    try:
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=1
        )
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        # logging.error(f"Final environment state: {env_state}")




    
    


    



def train_single_env(port: str):
    logging.info("Creating environment...")
    env = DiceAdventurePythonEnvRL(
        env_id=0,
        game_executable_filepath="game/V1.0.5/DiceAdventure.x86_64",
        port=port,
        player="dwarf",
        train_mode=True
    )
    logging.info("Environment created")
    env = Monitor(env, "./logs/monitor/")
    logging.info("Environment monitored")
    
    # Define policy kwargs with the custom CNN
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": 256
        }
    }
    logging.info("Policy kwargs defined")

    # Create the model with single environment
    model = MaskablePPO(
        "CnnPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=1,  
        tensorboard_log="./logs/ppo_single_env_0426/",
        learning_rate=3e-4,
        n_steps=2048,  
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    logging.info("Model created")

    # Add checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, 
        save_path="./model_checkpoints_0426/",
        name_prefix="dice_adventure_model_0426",
        verbose=1
    )
    # RewardMonitorCallback()
    
    logging.info("Checkpoint callback created")
    logging.info("Starting training...")
    try:
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=1
        )
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        # Print environment state
        env_state = env.get_state("dwarf")
        logging.error(f"Final environment state: {env_state}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a single Dice Adventure agent')
    parser.add_argument('--port_start', type=str, default='8070', help='Port to connect to the game instance')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of environments to train on')
    parser.add_argument('--players', type=str, default='dwarf,giant,human', help='Players to train on')


    args = parser.parse_args()
        # parse players as a list
    args.players = args.players.split(',')

    # create a list of ports of length 3 * num_envs starting from args.port_start and incrementing by 1
    port_list = [str(int(args.port_start) + i) for i in range(3 * args.num_envs)]

    try:
        logging.info("Starting training script...")
        train_vec_env(port_list, args.num_envs, args.players)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        logging.info("Script ended")