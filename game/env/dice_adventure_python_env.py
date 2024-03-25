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
        self.mask_radii = {"Dwarf": self.config["OBJECT_INFO"]["OBJECT_CODES"]["1S"]["SIGHT_RANGE"],
                           "Giant": self.config["OBJECT_INFO"]["OBJECT_CODES"]["2S"]["SIGHT_RANGE"],
                           "Human": self.config["OBJECT_INFO"]["OBJECT_CODES"]["3S"]["SIGHT_RANGE"]}

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

    def get_state(self, player=None, version=None, server=None):
        """
        Gets the current state of the game.
        :param player: (string) The player whose perspective will be used to collect the state. Can be one of
                                {Dwarf, Giant, Human}.
        :param version: (string) The level of visibility. Can be one of {full, player, fow}
        :param server: (string) Determines whether to get state from Python version or unity version of game. Can be
                                one of {local, unity}.
        :return: (dict) The state of the game

        The state is always given from the perspective of a player and defines how much of the level the
        player can currently "see". The following state version options define how much information this function
        returns.
        - [full]:   Returns all objects and player stats. This ignores the 'player' parameter.

        - [player]: Returns all objects in the current sight range of the player. Limited information is provided about
                    other players present in the state.

        - [fow]:    Stands for Fog of War. In the Unity version of the game, you can see a visibility mask for each
                    character. Black positions have not been observed. Gray positions have been observed but are not
                    currently in the player's view. This option returns all objects in the current sight range (view) of
                    the player plus objects in positions that the player has seen before. Note that any object that can
                    move (such as monsters and other players) are only returned when they are in the player's current
                    view.
        """
        version = version if version else self.state_version
        player = player if player else self.player
        server = server if server else self.server

        if server == "local":
            state = self.game.get_state(player, version)
        else:
            url = self.unity_socket_url.format(player)
            state = unity_socket.get_state(url, version)

        return state

    @staticmethod
    def _get_reward():
        """
        Calculates the reward the agent should receive based on action taken.
        :return: (int) The reward
        """
        return 0
