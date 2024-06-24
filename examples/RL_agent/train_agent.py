from abc import ABC
from examples.RL_agent.dice_adventure_python_env import DiceAdventurePythonEnvRL
from os import listdir
from os import makedirs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
from json import loads


############
# TRAINING #
############

def train_ppo(config):
    save_callback = SaveCallback(model_id=config["TRAINING_SETTINGS"]["GLOBAL"]["model_number"],
                                 total_time_steps=config["TRAINING_SETTINGS"]["GLOBAL"]["num_time_steps"],
                                 save_threshold=config["TRAINING_SETTINGS"]["GLOBAL"]["save_threshold"])

    kwargs = {**config["ENV_SETTINGS"], **config["GAME_SETTINGS"]}
    # Create list of vectorized environments for agent
    vec_env = _make_envs(num_envs=config["TRAINING_SETTINGS"]["GLOBAL"]["num_envs"],
                         players=config["TRAINING_SETTINGS"]["GLOBAL"]["players"],
                         env_args=kwargs)

    makedirs("examples/RL_agent/tensorboard_logs/", exist_ok=True)
    model = PPO("MlpPolicy",
                vec_env,
                verbose=0,
                tensorboard_log="tensorboard_logs/",
                device=config["TRAINING_SETTINGS"]["GLOBAL"]["device"],
                # Kwargs
                **config["TRAINING_SETTINGS"]["PPO"])

    model.learn(total_timesteps=config["TRAINING_SETTINGS"]["GLOBAL"]["num_time_steps"],
                callback=save_callback)

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

        self._setup_directories()

    def _setup_directories(self):
        makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        self.time_steps += 1
        self.pbar.update(1)

        if self.time_steps % self.save_threshold == 0:
            self._save_model()

    def _save_model(self):
        self.model.save(self.save_dir + "dice_adventure_ppo_modelchkpt-{}".format(self.checkpoint))
        self.checkpoint += 1


if __name__ == "__main__":
    train_ppo(loads(open("examples/RL_agent/train_config.json").read()))
