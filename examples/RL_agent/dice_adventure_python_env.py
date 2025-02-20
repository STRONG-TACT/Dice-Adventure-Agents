from game.dice_adventure import DiceAdventure
import game.unity_socket as unity_socket
from gymnasium import Env
from gymnasium import spaces
from json import loads
import numpy as np
from random import choice
from game.unity_socket import DiceAdventureWebSocket
import json


class DiceAdventurePythonEnvRL(Env):
    """
    Implements a custom gyn environment for the Dice Adventure game.
    """
    def __init__(self,
                 player="Dwarf",
                 id_=0,
                 train_mode=False,
                 server="local",
                 state_version="player",
                 **kwargs):
        """
        Init function for Dice Adventure gym environment.
        :param player:      (string) The player that will be used to play the game.
        :param id_:         (int) An optional ID parameter to distinguish this environment from others.
        :param train_mode:  (bool) A helper parameter to switch between training mode and play mode. When we test agents,
                                   we will use a "play" mode, where the step function simply takes an action and returns
                                   the next state.
        :param server:      (string) Determines which game version to use. Can be one of {local, unity}.
        :param kwargs:      (dict) Additional keyword arguments to pass into Dice Adventure game. Only applies when
                                   'server' is 'local'.
        """
        self.config = loads(open("game/config/main_config.json", "r").read())
        self.player = player
        self.id = id_
        self.kwargs = kwargs
        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]
        self.player_names = ["Dwarf", "Giant", "Human"]

        #################
        # GAME SETTINGS #
        #################
        self.state_version = state_version if state_version in ["full", "player", "fow"] else "player"
        self.object_positions = self.config["GYM_ENVIRONMENT"]["OBSERVATION"]["OBJECT_POSITIONS"]
        self.num_game_objects = len(self.object_positions)

        ##################
        # TRAIN SETTINGS #
        ##################
        self.train_mode = train_mode
        self.masks = {"Dwarf": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C1"]["SIGHT_RANGE"],
                      "Giant": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C2"]["SIGHT_RANGE"],
                      "Human": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C3"]["SIGHT_RANGE"]}
        self.max_mask_radius = max(self.masks.values())
        self.local_mask_radius = self.masks[self.player]
        self.action_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'wait',
                           5: 'submit', 6: 'pinga', 7: 'pingb', 8: 'pingc', 9: 'pingd', 10: 'undo'}

        num_actions = len(self.action_map)
        self.action_space = spaces.Discrete(num_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.mask_size = self.max_mask_radius * 2 + 1
        vector_len = self.mask_size * self.mask_size * self.num_game_objects
        self.observation_space = spaces.Box(low=-5, high=100,
                                            shape=(vector_len,), dtype=np.float32)
        
        self.num_actions = 0

        ###################
        # SERVER SETTINGS #
        ###################
        self.server = server
        self.unity_socket_url = self.config["GYM_ENVIRONMENT"]["UNITY"]["URL"]
        # self.websocket = DiceAdventureWebSocket(url=self.unity_socket_url.format(self.player.lower()))
        self.game = None
        if self.server == "local":
            try:
                self.game = DiceAdventure(**self.kwargs)
            except Exception as e:
                print(f"Error initializing DiceAdventure: {e}")
                raise

        # Load observation config
        with open("examples/RL_agent/observation_config.json", "r") as f:
            self.obs_config = json.load(f)
        
        # Initialize observation space
        shape = self.obs_config["TENSOR_SHAPE"] + [self.obs_config["NUM_CHANNELS"]]
        self.observation_space = spaces.Box(low=-5, high=100,
                                         shape=shape, dtype=np.float32)

    def step(self, action) -> tuple:
        """
        :return: (observation, reward, terminated, truncated, info)
        """
        if self.train_mode:
            return self._step_train(action)
        # Wrap _step_play return in proper tuple format
        state = self._step_play(action)
        obs = self.get_observation(state)
        return obs, 0.0, False, False, {}

    def _step_train(self, action):
        try:
            action = int(action)
            self.num_actions += 1

            state = self.get_state()
            game_action = self.action_map[action]
            next_state = self.execute_action(self.player, game_action)

            reward = self._get_reward(state, next_state)

            # Simulate other players
            other_players = [p for p in self.player_names if p != self.player]
            for op in other_players:
                next_state = self.execute_action(player=op, game_action=choice(self.actions))

            terminated = next_state.get("status") == "GAME_OVER"
            truncated = False
            
            if terminated:
                new_obs, info = self.reset()
            else:
                new_obs = self.get_observation(next_state)
                info = {}

            return new_obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"Error in _step_train: {e}")
            # Return safe default values
            zero_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return zero_obs, 0, True, False, {}

    def get_observation(self, state):
        """
        Constructs an array observation for agent based on state. Dimensions:
        1. self.mask x self.mask (1-2)
        2. len(self.observation_type_positions) (3)
        3. 4 (4) - max number of object types is 4 [i.e., M4]
        4. six additional state variables
        Total Est.: 7x7x10x4+6= 1006
        :param state:
        :return:
        """
        try:
            # Initialize observation shape at class level
            if not hasattr(self, '_observation_shape'):
                self._observation_shape = self.mask_size * self.mask_size * self.num_game_objects
            
            if state is None:
                return np.zeros(self._observation_shape, dtype=np.float32)
            
            # Find player object safely
            player_obj = None
            for ele in state["content"]["scene"]:
                if ele.get("objKey") == self.get_player_code(self.player):
                    player_obj = ele
                    break
                
            if player_obj is None:
                raise ValueError("Player object not found in scene")

            x, y = player_obj["x"], player_obj["y"]

            x_bound_upper = x + self.local_mask_radius
            x_bound_lower = x - self.local_mask_radius
            y_bound_upper = y + self.local_mask_radius
            y_bound_lower = y - self.local_mask_radius

            # Create observation tensor with error checking
            tensor = np.zeros((self.mask_size, self.mask_size, self.num_game_objects), dtype=np.float32)
            
            for obj in state["content"]["scene"]:
                if obj.get("objKey") in self.object_positions and \
                   obj.get("x") is not None and obj.get("y") is not None:
                    
                    if x_bound_lower <= obj["x"] <= x_bound_upper and \
                       y_bound_lower <= obj["y"] <= y_bound_upper:
                        other_x = self.local_mask_radius - (x - obj["x"])
                        other_y = self.local_mask_radius - (y - obj["y"])
                        
                        # Add bounds checking
                        if 0 <= other_x < self.mask_size and 0 <= other_y < self.mask_size:
                            # TODO: check if this is correct - if we have multiple monsters in sight range,
                            # wouldn't this overwrite the first one?
                            pos = self.object_positions[obj["objKey"]]["POSITION"]
                            if pos < self.num_game_objects:
                                tensor[other_x][other_y][pos] = \
                                    self.object_positions[obj["objKey"]]["VALUE"]

            return np.ndarray.flatten(tensor)
        except Exception as e:
            print(f"Error creating observation: {e}")
            return np.zeros(self._observation_shape, dtype=np.float32)

    def _step_play(self, action) -> dict:
        """Returns state dictionary"""
        state = self.execute_action(self.player, action)
        if state is None:
            return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}
        return state

    def close(self):
        """
        close() function from standard gym environment. Not implemented.
        :return: N/A
        """
        pass

    def render(self, mode='console'):
        if self.server == "local" and self.game is not None:
            self.game.render()

    def reset(self, **kwargs):
        if self.server == "local":
            try:
                self.game = DiceAdventure(**self.kwargs)
            except Exception as e:
                print(f"Error resetting game: {e}")
                raise
        obs = self.get_state()
        return self.get_observation(obs), {}

    def execute_action(self, player, game_action):
        if self.server == "local":
            if self.game is None:
                raise RuntimeError("Game not initialized")
            self.game.execute_action(player, game_action)
            return self.get_state()
        else:
            url = self.unity_socket_url.format(player.lower())
            try:
                # Type checking for unity socket methods
                if not hasattr(unity_socket, 'execute_action') or not hasattr(unity_socket, 'get_state'):
                    raise AttributeError("Unity socket missing required methods")
                unity_socket.execute_action(url, game_action)  # type: ignore
                return unity_socket.get_state(url, self.state_version)  # type: ignore
            except Exception as e:
                print(f"Unity socket error: {e}")
                return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}

    def get_state(self, player=None, version=None, server=None):
        version = version if version is not None else self.state_version
        player = player if player is not None else self.player
        server = server if server else self.server

        try:
            if server == "local":
                if self.game is None:
                    raise RuntimeError("Game not initialized")
                return self.game.get_state(player, version)
            else:
                url = self.unity_socket_url.format(player.lower())
                return unity_socket.get_state(url, version)
        except Exception as e:
            print(f"Error getting state: {e}")
            # Return a minimal valid state
            return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}

    def get_actions(self):
        return self.actions

    def get_player_names(self):
        return self.player_names

    @staticmethod
    def get_player_code(player):
        codes = {"Dwarf": "C1", "Giant": "C2", "Human": "C3"}
        return codes[player]

    def _get_reward(self, state, next_state):
        """
        Calculates the reward the agent should receive based on action taken.
        :return: (int) The reward
        """
        reward = 0
        if state["content"]["gameData"]["currLevel"] != next_state["content"]["gameData"]["currLevel"]:
            reward = 100 * (10/self.num_actions)
        return reward

    def _get_characters_from_state(self, state):
        chars = []
        for player in self.player_names:
            player_code = self.get_player_code(player)
            for ele in state["content"]["scene"]:
                if ele.get("objKey") == player_code:
                    chars.append(ele)
        return chars


