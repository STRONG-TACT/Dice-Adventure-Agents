from json import loads
from websockets.sync.client import connect


def execute_action(url, action):
    # Command to send to Game env
    action_command = {"command": "execute_action",
                      "action": action}
    # Planning phase: no inputs
    # Pinging phase: no inputs
    return send(url, str(action_command))


def get_state(url, version):
    if version == "player":
        command = "get_state"
    elif version == "fog":
        command = "get_fog_state"
    else:
        command = "get_full_state"

    state = send(url, f'{"command":"{command}"}')
    return loads(state)


def send(url, message):
    with connect(url) as websocket:
        websocket.send(message)
        return websocket.recv()

