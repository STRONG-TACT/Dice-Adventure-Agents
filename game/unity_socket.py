from json import loads
from websockets.sync.client import connect


class DiceAdventureWebSocket:
    def __init__(self, url):
        self.connection = connect(url, open_timeout=None, close_timeout=None)

    def execute_action(self, action):
        # Command to send to Game env
        action_command = {"command": "execute_action",
                          "action": action}
        # Planning phase: no inputs
        # Pinging phase: no inputs
        return self.send(str(action_command))

    def register(self, agent_id):
        # Command to send to Game env
        action_command = {"command": "register",
                          "agent_id": agent_id}
        # Planning phase: no inputs
        # Pinging phase: no inputs
        # print(f"Unity URL: {url} | Command: {action_command}")
        return self.send(str(action_command))

    def get_state(self, version):
        if version == "player":
            command = "get_state"
        elif version == "fow":
            command = "get_fow_state"
        else:
            command = "get_full_state"
        state = self.send(f'{{"command":"{command}"}}')
        return loads(state)

    def send(self, message):
        self.connection.send(message)
        return self.connection.recv()
