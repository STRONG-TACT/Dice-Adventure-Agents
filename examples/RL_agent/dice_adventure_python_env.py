from game.dice_adventure import DiceAdventure
import game.unity_socket as unity_socket
from gymnasium import Env
from gymnasium import spaces
from json import loads
import numpy as np
from random import choice
from game.unity_socket import DiceAdventureWebSocket


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
        self.game = DiceAdventure(**self.kwargs) if self.server == "local" else None

    def step(self, action):
        """
        Applies the given action to the game.
        If self.train_mode == True, passes action to _step_train() and returns new information for RL model.
        If self.train_mode == False, simply returns state obtained after taking action.
        :param action:  (string) The action produced by the agent
        :return:        (dict, float, bool, bool, dict) or (dict)
        """
        if self.train_mode:
            return self._step_train(action)
        else:
            return self._step_play(action)

    def _step_train(self, action):
        """
        Applies the given action to the game. Determines the next observation and reward,
        whether the training should terminate, whether training should be truncated, and
        additional info.
        :param action:  (string) The action produced by the agent
        :return:        (dict, float, bool, bool, dict) See below

        new_obs (dict) - The resulting game state after applying 'action' to the game
        reward (float) - The reward obtained from applying 'action' to the game. This must be defined by the user. A
                         helper function _get_reward() has been provided for convenience.
        terminated (bool) - Whether the game has terminated after applying 'action' to the game
        truncated (bool) - (See https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
        info (dict) - Additional information that should be passed back to model

        Note: Although this framework is usually used for RL models, users can develop any kind of model with this code.
        """
        action = int(action)
        self.num_actions += 1

        state = self.get_state()
        # Execute action and get next state
        game_action = self.action_map[action]
        next_state = self.execute_action(self.player, game_action)

        reward = self._get_reward(state, next_state)

        # Simulate other players
        other_players = [p for p in self.player_names if p != self.player]
        for op in other_players:
            next_state = self.execute_action(player=op, game_action=choice(self.actions))

        terminated = next_state["status"] == "GAME_OVER"
        if terminated:
            new_obs, info = self.reset()
        else:
            new_obs = self.get_observation(next_state)
            info = {}
        truncated = False
        # Track metrics
        # self.save_metrics()
        # self.render()
        return new_obs, reward, terminated, truncated, info

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
        player_obj = None
        for ele in state["content"]["scene"]:
            if ele.get("objKey") == self.get_player_code(self.player):
                player_obj = ele
                break

        x, y = player_obj["x"], player_obj["y"]

        x_bound_upper = x + self.local_mask_radius
        x_bound_lower = x - self.local_mask_radius
        y_bound_upper = y + self.local_mask_radius
        y_bound_lower = y - self.local_mask_radius

        tensor = np.zeros((self.mask_size, self.mask_size, self.num_game_objects))
        for obj in state["content"]["scene"]:
            if obj["objKey"] in self.object_positions and obj["x"] and obj["y"]:
                if x_bound_lower <= obj["x"] <= x_bound_upper and \
                        y_bound_lower <= obj["y"] <= y_bound_upper:
                    other_x = self.local_mask_radius - (x - obj["x"])
                    other_y = self.local_mask_radius - (y - obj["y"])
                    tensor[other_x][other_y][self.object_positions[obj["objKey"]]["POSITION"]] = \
                        self.object_positions[obj["objKey"]]["VALUE"]

        return np.ndarray.flatten(tensor)

    def _step_play(self, action):
        """
        Applies the given action to the game and returns the resulting state
        :param action:  (string) The action produced by the agent
        :return:        (dict) The resulting state
        """
        return self.execute_action(self.player, action)

    def close(self):
        """
        close() function from standard gym environment. Not implemented.
        :return: N/A
        """
        pass

    def render(self, mode='console'):
        """
        Prints the current board state of the game. Only applies when `self.server` is 'local'.
        :param mode: (string) Determines the mode to use (not used)
        :return: N/A
        """
        if self.server == "local":
            self.game.render()

    def reset(self, **kwargs):
        """
        Resets the game. Only applies when `self.server` is 'local'.
        :param kwargs:  (dict) Additional arguments to pass into local game server
        :return:        (dict, dict) The initial state when the game is reset, An empty 'info' dict
        """
        if self.server == "local":
            self.game = DiceAdventure(**self.kwargs)
        obs = self.get_state()
        return self.get_observation(obs), {}

    def execute_action(self, player, game_action):
        """
        Executes the given action for the given player.
        :param player:      (string) The player that should take the action
        :param game_action: (string) The action to take
        :return:            (dict) The resulting state after taking the given action
        """
        if self.server == "local":
            self.game.execute_action(player, game_action)
            next_state = self.get_state()
        else:
            url = self.unity_socket_url.format(player.lower())
            # TODO CAPTURE RESPONSE AND RETURN TO USER
            unity_socket.execute_action(url, game_action)
            next_state = self.get_state()
        return next_state

    def get_state(self, player=None, version=None, server=None):
        """
        Gets the current state of the game.
        :param player: (string) The player whose perspective will be used to collect the state. Can be one of
                                {Dwarf, Giant, Human}.
        :param version: (string) The level of visibility. Can be one of {full, player, fow}
        :param server: (string) Determines whether to get state from Python version or Unity version of game. Can be
                                one of {local, unity}.
        :return: (dict) The state of the game

        The state is always given from the perspective of a player and defines how much of the level the
        player can currently "see". The following state version options define how much information this function
        returns.
        - [full]:   Returns all objects and player stats for current level. This ignores the 'player' parameter.

        - [player]: Returns all objects in the current sight range of the player. Limited information is provided about
                    other players present in the state.

        - [fow]:    Stands for Fog of War. In the Unity version of the game, you can see a visibility mask for each
                    character. Black positions have not been observed. Gray positions have been observed but are not
                    currently in the player's view. This option returns all objects in the current sight range (view) of
                    the player plus objects in positions that the player has seen before. Note that any object that can
                    move (such as monsters and other players) are only returned when they are in the player's current
                    view, but static objects such as walls, stones, and traps are returned if they've been previously
                    observed.
        """
        version = version if version is not None else self.state_version
        player = player if player is not None else self.player
        server = server if server else self.server

        if server == "local":
            state = self.game.get_state(player, version)
        else:
            url = self.unity_socket_url.format(player.lower())
            state = unity_socket.get_state(url, version)

        return state

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


