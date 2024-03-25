from game.dice_adventure import DiceAdventure
import game.env.unity_socket as unity_socket
from gymnasium import Env
from json import loads


class DiceAdventurePythonEnv(Env):
    """
    Implements a custom gyn environment for the Dice Adventure game.
    """
    def __init__(self,
                 player="Dwarf",
                 id_=0,
                 train_mode=False,
                 server="local",
                 state_version="character",
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

        ##################
        # STATE SETTINGS #
        ##################
        self.state_version = state_version
        self.mask_radii = {"Dwarf": self.config["GAMEPLAY"]["OBJECT_INFO"]["OBJECT_CODES"]["1S"]["SIGHT_RANGE"],
                           "Giant": self.config["GAMEPLAY"]["OBJECT_INFO"]["OBJECT_CODES"]["2S"]["SIGHT_RANGE"],
                           "Human": self.config["GAMEPLAY"]["OBJECT_INFO"]["OBJECT_CODES"]["3S"]["SIGHT_RANGE"]}

        ##################
        # TRAIN SETTINGS #
        ##################
        self.train_mode = train_mode

        ###################
        # SERVER SETTINGS #
        ###################
        self.server = server
        self.unity_socket_url = self.config["GYM_ENVIRONMENT"]["UNITY"]["URL"]
        self.game = None

        if self.server == "local":
            self.game = DiceAdventure(**self.kwargs)

    def step(self, action):
        """
        Applies the given action to the game. Determines the next observation and reward,
        whether the training should terminate, whether training should be truncated, and
        additional info.
        :param action:  (string) The action produced by the agent
        :return:        (dict, float, bool, bool, dict) See description
        """
        next_state = self.execute_action(self.player, action)

        reward = self._get_reward()

        # new_obs, reward, terminated, truncated, info
        terminated = next_state["status"] == "Done"
        if terminated:
            new_obs, info = self.reset()
        else:
            new_obs = self.get_state()
            info = {}
        truncated = False

        return new_obs, reward, terminated, truncated, info

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
        return obs, {}

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
            next_state = unity_socket.execute_action(url, game_action)
        return next_state

    def get_state(self, player=None, version=None):
        version = version if version else self.state_version
        player = player if player else self.player

        if self.server == "local":
            state = self.game.get_state(player, version)
        else:
            url = self.unity_socket_url.format(player)
            state = unity_socket.get_state(url, version)

        return state

    def _get_state_character(self, state, player):
        obj = self._locate_object_by_name(state["content"]["scene"], player)
        sight_range = self.mask_radii[player]
        px, py = obj["x"], obj["y"]
        # Get upper and lower bounds for x,y coordinates
        x_bound_upper = px + sight_range
        x_bound_lower = px - sight_range
        y_bound_upper = py + sight_range
        y_bound_lower = py - sight_range

        filtered = []
        for obj in state["content"]["scene"]:
            if x_bound_lower <= obj["x"] <= x_bound_upper and \
                    y_bound_lower <= obj["y"] <= y_bound_upper:
                if obj["type"] == "character":
                    pass

    @staticmethod
    def _get_reward():
        """
        Calculates the reward the agent should receive based on action taken.
        :return: (int) The reward
        """
        return 0

    ###########
    # HELPERS #
    ###########
    def _check_bounds(self, px, py, x, y):
        pass

    @staticmethod
    def _locate_object_by_name(scene, obj_name):
        obj = None
        for i in scene:
            if i["name"] == obj_name:
                obj = i
                break
        return obj


