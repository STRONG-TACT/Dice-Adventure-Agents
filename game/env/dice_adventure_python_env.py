from game.dice_adventure import DiceAdventure
import game.env.rewards as rewards
import game.env.unity_socket as unity_socket

from copy import deepcopy
from datetime import datetime
from gymnasium import Env
from gymnasium import spaces
from json import dumps
from json import loads
import numpy as np
from os import listdir
from os import makedirs
from os import path
from random import choice
from random import seed
from stable_baselines3 import PPO
import re
import pprint
pp = pprint.PrettyPrinter(indent=2)


class DiceAdventurePythonEnv(Env):
    def __init__(self, id_,
                 player,
                 model_number,
                 env_metrics=False,
                 train_mode=False,
                 server="local",
                 observation_type="vector",
                 automate_players=True,
                 random_players=False,
                 set_random_seed=False,
                 **kwargs):
        self.id = id_
        print(f"INITIALIZING ENV {self.id}...")
        if set_random_seed:
            seed(self.id)

        self.game = None
        self.config = self.config = loads(open("game/config/main_config.json", "r").read())
        self.reward_codes = self.config["GYM_ENVIRONMENT"]["REWARD"]["CODES"]
        self.observation_object_positions = self.config["GYM_ENVIRONMENT"]["OBSERVATION"]["OBJECT_POSITIONS"]
        self.object_size_mappings = self.config["OBJECT_INFO"]["ENEMIES"]["ENEMY_SIZE_MAPPING"]
        self.kwargs = kwargs
        self.player_num = 0
        self.players = ["Dwarf", "Giant", "Human"]
        # self.players = {"1S": "Dwarf", "2S": "Giant", "3S": "Human", "Dwarf": "1S", "Giant": "2S", "Human": "3S"}
        self.player_ids = ["1S", "2S", "3S"]
        # self.player = self.players[player]
        self.player = player
        self.automate_players = automate_players
        self.random_players = random_players

        # self.masks = {"1S": 1, "2S": 3, "3S": 2}
        self.masks = {"Dwarf": 1, "Giant": 3, "Human": 2}
        self.max_mask_radius = max(self.masks.values())
        self.local_mask_radius = self.masks[self.player]
        self.action_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'wait',
                           5: 'submit', 6: 'pinga', 7: 'pingb', 8: 'pingc', 9: 'pingd', 10: 'undo'}
        # pingA, pingB, etc.

        self.goals = {"1S": False, "2S": False, "3S": False}
        self.pin_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.time_steps = 0
        self.model_number = model_number
        self.model_dir = "train/{}/model/".format(self.model_number)
        self.model_file = None
        self.model = None

        ##################
        # TRAIN SETTINGS #
        ##################

        self.train_mode = train_mode

        ################
        # ENV SETTINGS #
        ################

        num_actions = len(self.action_map)
        self.action_space = spaces.Discrete(num_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.mask_size = self.max_mask_radius * 2 + 1
        vector_len = (self.mask_size * self.mask_size * len(set(self.observation_object_positions.values())) * 4) + 6
        self.observation_space = spaces.Box(low=-5, high=100,
                                            shape=(vector_len,), dtype=np.float32)
        ###################
        # METRIC TRACKING #
        ###################
        self.metrics_dir = self.config["GYM_ENVIRONMENT"]["METRICS"]["DIRECTORY"].format(model_number)
        makedirs(self.metrics_dir, exist_ok=True)
        self.track_metrics = env_metrics
        self.metrics_save_threshold = 10000
        # Reward tracking
        self.rewards_tracker = []
        self.num_games = 0

        # Server type
        self.server = server
        self.unity_socket_url = self.config["GYM_ENVIRONMENT"]["UNITY"]["URL"]

        #if self.server == "local":
        #    self.create_game()
        self.prev_observed_state = None

    def step(self, action):
        next_state = self.execute_action(self.player, action)

        reward = self._get_reward()

        # new_obs, reward, terminated, truncated, info
        terminated = next_state["status"] == "Done"
        if terminated:
            new_obs, info = self.reset()
        else:
            new_obs = self.get_observation(next_state)
            info = {}
        truncated = False
        # Track metrics
        self.save_metrics()

        return new_obs, reward, terminated, truncated, info

    def close(self):
        pass

    def render(self, mode='console'):
        if self.server == "local":
            self.game.render()

    def reset(self, **kwargs):
        if self.server == "local":
            self.create_game()
        state = self.get_state()
        obs = self.get_observation(state)
        return obs, {}

    def execute_action(self, player, game_action):
        if self.server == "local":
            self.game.execute_action(player, game_action)
        else:
            url = self.unity_socket_url.format(player.lower())
            unity_socket.execute_action(url, game_action)
        return self.get_state()

    def get_state(self):
        if self.server == "local":
            state = self.game.get_state()
        else:
            url = self.unity_socket_url.format("Dwarf")
            state = unity_socket.get_state(url)
        return state

    @staticmethod
    def _get_reward():
        return 0

