from json import loads
from websockets.sync.client import connect


def execute_action(url, action):
    # Command to send to Game env
    action_command = {"command": "execute_action",
                      "action": action}
    # Planning phase: no inputs
    # Pinging phase: no inputs
    return send(url, str(action_command))


def get_state(url):
    state = send(url, '{"command":"get_state"}')
    return loads(state)


def send(url, message):
    with connect(url) as websocket:
        websocket.send(message)
        return websocket.recv()

