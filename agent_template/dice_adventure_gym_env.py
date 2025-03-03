from json import loads
from typing import Any, Tuple, Union, Dict
from gymnasium import Env
from game.unity_socket import UnityWebSocket


class DiceAdventureGymEnv(Env):
    """
    Implements a custom gym environment for the Dice Adventure Unity game.
    """
    def __init__(self, port: str, train_mode: True, game_executable_filepath: str):
        """
        Init function for Dice Adventure gym environment.
        :param player:      The player that will be used to play the game.
        :param port:        The custom port that will be used to connect the agent to the game instance.
                            You should choose a port that is currently not in use by your system.
        :param train_mode:  A helper parameter to switch between training mode and play mode. When you test agents,
                            you can set train_mode = False so that the step function simply takes an action and returns
                            the next state.
        :param game_executable_filepath:  The location of the game executable
        """
        self.player = player
        self.port = port
        self.train_mode = train_mode
        self.socket_url = "ws://localhost:{}/hmt/{}"
        self.sockets = {'dwarf': None, 'giant': None, 'human': None}

        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]
        self.player_names = ["dwarf", "giant", "human"]

        self._launch_game(game_executable_filepath)

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
        next_state = self.execute_action(action)

        reward = self._get_reward()

        # new_obs, reward, terminated, truncated, info
        terminated = next_state[0]["status"] == "Done"
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
        return self.execute_action(action)

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
        :param game_action: The action to take
        :return:            The resulting state after taking the given action
        """
        # TODO CAPTURE RESPONSE AND RETURN TO USER
        self._get_socket(player).execute_action(game_action)
        return self.get_state(player)

    def get_state(self, player: str):
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
        return _simplify_state(self._get_socket(player).get_state(), self.player)
    
    def _get_socket(self, player):
        if self.sockets[player] is None:
            self.sockets[player] = UnityWebSocket(self.socket_url.format(self.port, player))
        return self.sockets[player]

    def get_actions(self):
        return self.actions

    def get_player_names(self):
        return self.player_names

    @staticmethod
    def _get_reward():
        """
        Calculates the reward the agent should receive based on action taken.
        :return: (int) The reward
        """
        return 0

    def _launch_game(self, game_executable_filepath):
        command = [game_executable_filepath,
                   "-localMode",
                   "-hmtsocketurl", "ws://localhost",
                   "-hmtsocketport", "{}".format(self.port)]
        # subprocess.run(command)
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#############
# UTILITIES #
#############
def _simplify_state(state: dict, player: str) -> list[dict]:
    """
    Simplifies the game state into a list of dictionary objects, each representing an object in the game.
    :param state: The game state
    :param player: The player the state is related to
    :return: The modified state as a list of dictionaries
    """
    state['content']['gameData']['id'] = 'gameData'
    game_data_obj = state['content']['gameData']
    scene = [game_data_obj] + state['content']['scene']

    return _add_sight_status(scene, player, game_data_obj)


def _add_sight_status(scene: list[dict], player: str, game_data_obj: dict) -> list[dict]:
    """
    Modifies the state by adding whether objects are visible or hidden. It also adds 'Cell' objects with an 'unexplored'
    status to represent grid squares that have not yet been observed.
    :param scene: The list of objects in the state
    :param player: The name of the player
    :param game_data_obj: The state object containing high level information about the current state of the game
    :return: The modified scene list
    """
    player_obj = _find_player_obj(scene, player)
    x_lower = player_obj['x'] - player_obj['sightRange']
    x_upper = player_obj['x'] + player_obj['sightRange']
    y_lower = player_obj['y'] - player_obj['sightRange']
    y_upper = player_obj['y'] + player_obj['sightRange']

    all_cells = {(i, j) for i in range(game_data_obj['boardWidth']) for j in range(game_data_obj['boardHeight'])}
    scene_cells = {(obj.get('x'), obj.get('y')) for obj in scene}
    unexplored = all_cells - scene_cells

    for obj in scene:
        x, y = obj.get('x'), obj.get('y')
        if x is None or y is None:
            continue
        if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
            obj['sight_status'] = 'visible'
        else:
            obj['sight_status'] = 'hidden'

    for i, pos in enumerate(unexplored):
        scene.append({'id': f'UE{i}', 'entityType': 'Cell',
                      'objKey': 'UE', 'sight_status': 'unexplored',
                      'x': pos[0], 'y': pos[1]})
    return scene


def _find_player_obj(scene: list[dict], player: str) -> [dict, None]:
    """
    Locates the player's object dictionary in the scene list.
    :param scene: The list of objects in the state
    :param player: The name of the player to be returned
    :return: The player dictionary object
    """
    pid = get_player_id(player)
    for obj in scene:
        if obj.get('id') == pid:
            return copy(obj)


def get_player_id(player: str) -> str:
    """
    Gets the player ID.
    :param player: The player whose ID will be returned
    :return: The player's ID
    """
    ids = {"dwarf": "C11", "giant": "C21", "human": "C31"}
    return ids[player]




