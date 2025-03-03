from json import loads
from typing import Any, Tuple, Union, Dict
from gymnasium import Env
import game.unity_socket as unity_socket


class DiceAdventurePythonEnv(Env):
    """
    Implements a custom gym environment for the Dice Adventure Unity game.
    """
    def __init__(self, player: str, port: str, train_mode: True):
        """
        Init function for Dice Adventure gym environment.
        :param player:      The player that will be used to play the game.
        :param port:        The custom port that will be used to connect the agent to the game instance.
                            You should choose a port that is currently not in use by your system.
        :param train_mode:  A helper parameter to switch between training mode and play mode. When you test agents,
                            you can set train_mode = False so that the step function simply takes an action and returns
                            the next state.
        """
        self.player = player
        self.port = port
        self.train_mode = train_mode
        self.socket_url = "ws://localhost:{}/hmt/{}".format(self.port, self.player)
        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]
        self.player_names = ["dwarf", "giant", "human"]

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

    def _step_train(self, action: str) -> tuple[Any,
                                                float,
                                                Union[bool, Any],
                                                bool,
                                                Union[dict[str, Any]]]:
        """
        Applies the given action to the game. Determines the next observation and reward, whether the training should
        terminate, whether training should be truncated, and additional info.
        :param action:  The action produced by the agent
        :return:        See below

        new_obs (dict) - The resulting game state after applying 'action' to the game
        reward (float) - The reward obtained from applying 'action' to the game. This must be defined by the user. A
                         helper function _get_reward() has been provided for convenience.
        terminated (bool) - Whether the game has terminated after applying 'action' to the game
        truncated (bool) - (See https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
        info (dict) - Additional information that should be passed back to model

        Note: Although this framework is usually used for RL models, users can develop any kind of model with this code.
        """
        next_state = self.execute_action(self.player, action)

        reward = self._get_reward()

        # new_obs, reward, terminated, truncated, info
        terminated = next_state["status"] == "Done"
        if terminated:
            new_obs, info = self.reset()
        else:
            new_obs, info = self.get_state(), {}
        truncated = False

        return new_obs, reward, terminated, truncated, info

    def _step_play(self, action: str) -> list[dict]:
        """
        Applies the given action to the game and returns the resulting state
        :param action:  The action produced by the agent
        :return:        The resulting state
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
        Not implemented.
        """
        pass

    def reset(self, **kwargs):
        """
        Not implemented.
        """
        pass

    def execute_action(self, player: str, game_action: str):
        """
        Executes the given action for the given player.
        :param player:      The player being controlled
        :param game_action: The action to take
        :return:            The resulting state after taking the given action
        """
        # TODO CAPTURE RESPONSE AND RETURN TO USER
        unity_socket.execute_action(self.socket_url, game_action)
        return self.get_state()

    def get_state(self):
        """
        Gets the current state of the game.

        The state is always given from the perspective of self.player, which defines how much of the environment (level)
        the agent can currently "see". The following describes how the Fog-Of-War mechanic limits the environment
        view for agents.

            In the Unity game, you can see a visibility mask for each character.
            Black positions have not been observed. Gray positions have been observed but are not
            currently in the player's view. This function returns all objects in the current sight range (view) of
            the player plus objects in positions that the player has seen before. Note that any object that can
            move (such as monsters and other players) are only returned when they are in the player's current
            view (i.e. not obscured by black or gray squares), but static objects such as walls, stones, and traps,
            and shrines are returned if they've been previously observed.
        """
        return unity_socket.get_state(self.socket_url)

    def get_actions(self):
        return self.actions

    def get_player_names(self):
        return self.player_names

    @staticmethod
    def get_player_code(player):
        codes = {"Dwarf": "C11", "Giant": "C21", "Human": "C31"}
        return codes[player]

    @staticmethod
    def _get_reward():
        """
        Calculates the reward the agent should receive based on action taken.
        :return: (int) The reward
        """
        return 0
