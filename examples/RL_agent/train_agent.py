from abc import ABC
from examples.RL_agent.dice_adventure_python_env import DiceAdventurePythonEnvRL
from os import listdir
from os import makedirs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
from json import loads
import tensorflow as tf
import os
import numpy as np

# Create log directory if it doesn't exist
os.makedirs('training_logs', exist_ok=True)

############
# TRAINING #
############

def train_ppo(config):
    # Force CPU usage
    config["TRAINING_SETTINGS"]["GLOBAL"]["device"] = "cpu"
    
    save_callback = SaveCallback(model_id=config["TRAINING_SETTINGS"]["GLOBAL"]["model_number"],
                                 total_time_steps=config["TRAINING_SETTINGS"]["GLOBAL"]["num_time_steps"],
                                 save_threshold=config["TRAINING_SETTINGS"]["GLOBAL"]["save_threshold"])
    print('save_callback')

    kwargs = {**config["ENV_SETTINGS"], **config["GAME_SETTINGS"]}
    # Create list of vectorized environments for agent
    vec_env = _make_envs(num_envs=config["TRAINING_SETTINGS"]["GLOBAL"]["num_envs"],
                         players=config["TRAINING_SETTINGS"]["GLOBAL"]["players"],
                         env_args=kwargs)
    # print('vec_env')

    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                device=config["TRAINING_SETTINGS"]["GLOBAL"]["device"],
                tensorboard_log='tensorboard_logs_small_with_reward/')

    print('model')

    model.learn(total_timesteps=config["TRAINING_SETTINGS"]["GLOBAL"]["num_time_steps"],
                callback=save_callback, log_interval=10)
    print('learn')
    # model.save(MODEL_DIR.format(save_callback.model_number) + "dice_adventure_ppo_model_final")
    print("DONE TRAINING!")


################
# ENVIRONMENTS #
################

def _make_envs(num_envs: int, players: list, env_args: dict):
    envs = [
        _get_env(env_id=str(i * num_envs + j),
                 player=p,
                 env_args=env_args) #,
                 # model_number=save_callback.model_number)
        for i, p in enumerate(players)
        for j in range(num_envs)
    ]
    return SubprocVecEnv(envs)


def _get_env(env_id, player, env_args):
    # Needs to be function so that it is callable
    def env_fxn():
        return DiceAdventurePythonEnvRL(id_=env_id, player=player, **env_args)

    return env_fxn

################
# MODEL SAVING #
################


class SaveCallback(BaseCallback, ABC):
    def __init__(self, model_id, total_time_steps, save_threshold):
        super().__init__()
        self.time_steps = 0
        self.save_threshold = save_threshold
        self.model_id = model_id
        self.save_dir = f"dice_adventure_ppo_models/{self.model_id}/"
        self.checkpoint = 1
        self.pbar = tqdm(total=total_time_steps)

        # Track time between rewards for each env
        self.num_envs = None  # Will be set in _on_training_start
        self.last_reward_steps = {}  # {env_idx: last_step}
        self.last_shrine_steps = {}  # {env_idx: last_step}
        self.last_tower_steps = {}   # {env_idx: last_step}
        
        self._setup_directories()

    def _setup_directories(self):
        makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self):
        # Get number of environments from vectorized env
        self.num_envs = self.training_env.num_envs
        # Initialize tracking for each env
        for i in range(self.num_envs):
            self.last_reward_steps[i] = 0
            self.last_shrine_steps[i] = 0
            self.last_tower_steps[i] = 0

    def _on_step(self):
        self.time_steps += 1
        self.pbar.update(1)

        # Process rewards from all envs
        rewards = self.locals["rewards"]  # Array of immediate rewards at this step

        # Track steps between rewards for each env
        for env_idx, reward in enumerate(rewards):
            if reward > 0:  # Only track when we actually get a reward
                steps_since_last = self.time_steps - self.last_reward_steps[env_idx]
                self.last_reward_steps[env_idx] = self.time_steps
                
                # Log to a text file for each environment
                with open(f"training_logs/env_{env_idx}_rewards.txt", "a") as log_file:
                    log_file.write(f"{self.time_steps},{steps_since_last}\n")
                
                if reward == 1:  # Shrine reward
                    shrine_steps = self.time_steps - self.last_shrine_steps[env_idx]
                    self.last_shrine_steps[env_idx] = self.time_steps
                    with open(f"training_logs/env_{env_idx}_shrines.txt", "a") as log_file:
                        log_file.write(f"{self.time_steps},{shrine_steps}\n")
                    
                elif reward == 10:  # Tower reward
                    tower_steps = self.time_steps - self.last_tower_steps[env_idx]
                    self.last_tower_steps[env_idx] = self.time_steps
                    with open(f"training_logs/env_{env_idx}_towers.txt", "a") as log_file:
                        log_file.write(f"{self.time_steps},{tower_steps}\n")

        if self.time_steps % self.save_threshold == 0:
            self._save_model()
        
        return True

        
        # Get current level for each env
        # env_states = self.training_env.get_attr("game")  # Get game attribute from all envs
        # current_levels = [
        #     env.get_state()["content"]["gameData"]["currLevel"] 
        #     if env is not None else None 
        #     for env in env_states
        # ]

        # Track steps between rewards for each env
        # for env_idx, (reward, curr_level) in enumerate(zip(rewards, current_levels)):
        #     if reward > 0 and curr_level is not None:  # Only track when we actually get a reward
        #         steps_since_last = self.time_steps - self.last_reward_steps.get(env_idx, 0)
        #         self.last_reward_steps[env_idx] = self.time_steps
                
        #         # Log to a text file for each environment
        #         with open(f"training_logs/env_{env_idx}_rewards.txt", "a") as log_file:
        #             log_file.write(f"{self.time_steps},{steps_since_last},{curr_level},{reward}\n")
                
        #         if reward == 1:  # Shrine reward
        #             shrine_steps = self.time_steps - self.last_shrine_steps.get(env_idx, 0)
        #             self.last_shrine_steps[env_idx] = self.time_steps
        #             with open(f"training_logs/env_{env_idx}_shrines.txt", "a") as log_file:
        #                 log_file.write(f"{self.time_steps},{shrine_steps},{curr_level}\n")
                    
        #         elif reward == 10:  # Tower/Level completion reward
        #             tower_steps = self.time_steps - self.last_tower_steps.get(env_idx, 0)
        #             self.last_tower_steps[env_idx] = self.time_steps
        #             with open(f"training_logs/env_{env_idx}_towers.txt", "a") as log_file:
        #                 log_file.write(f"{self.time_steps},{tower_steps},{curr_level}\n")

        # if self.time_steps % self.save_threshold == 0:
        #     self._save_model()
        
        # return True

    def _save_model(self):
        self.model.save(self.save_dir + "dice_adventure_ppo_modelchkpt-{}".format(self.checkpoint))
        self.checkpoint += 1


if __name__ == "__main__":
    train_ppo(loads(open("examples/RL_agent/train_config.json").read()))
