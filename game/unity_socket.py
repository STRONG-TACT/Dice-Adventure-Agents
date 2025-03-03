from json import loads
from json.decoder import JSONDecodeError
from time import sleep
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError, ConnectionClosed
from websockets.sync.client import connect


class UnityWebSocket:
    def __init__(self, url):
        self.url = url
        self.connection = connect(self.url, open_timeout=None, close_timeout=None)
        self.retry_sleep_time_seconds = .5

    def execute_action(self, action):
        # Command to send to Game env
        action_command = {"command": "execute_action", "action": action}
        # Planning phase: no inputs
        # Pinging phase: no inputs
        return self.send(str(action_command))

    def register(self, agent_id):
        # Command to send to Game env
        action_command = {"command": "register", "agent_id": agent_id}
        # Planning phase: no inputs
        # Pinging phase: no inputs
        return self.send(str(action_command))

    def get_state(self):
        state = self.send(f'{{"command":"{"get_fow_state"}"}}')
        try:
            return loads(state)
        except JSONDecodeError:
            print("Error decoding json in state:", state)

    def send(self, message, num_retries=5):
        for _ in range(num_retries):
            try:
                self.connection.send(message)
                return self.connection.recv()
            except ConnectionClosedOK:
                self.connection = connect(self.url, open_timeout=None, close_timeout=None)
            except ConnectionClosedError:
                self.connection = connect(self.url, open_timeout=None, close_timeout=None)
            except ConnectionClosed:
                self.connection = connect(self.url, open_timeout=None, close_timeout=None)
            sleep(self.retry_sleep_time_seconds)

