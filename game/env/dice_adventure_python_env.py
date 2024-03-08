from game.dice_adventure import DiceAdventure
import game.env.unity_socket as unity_socket
from gymnasium import Env
from json import loads


class DiceAdventurePythonEnv(Env):
    def __init__(self,
                 player,
                 id_=0,
                 train_mode=False,
                 server="local",
                 **kwargs):
        self.config = self.config = loads(open("game/config/main_config.json", "r").read())
        self.player = player
        self.id = id_
        self.kwargs = kwargs

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
        pass

    def render(self, mode='console'):
        if self.server == "local":
            self.game.render()

    def reset(self, **kwargs):
        if self.server == "local":
            self.game = DiceAdventure(**self.kwargs)
        obs = self.get_state()
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

