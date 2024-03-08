from game.env.dice_adventure_python_env import DiceAdventurePythonEnv
from agent_interface import DiceAdventureAgent

PLAYER = "Giant"
SERVER = "local"  # {local, unity}
AGENT_FILEPATH = "path/to/agent"
ACTION_LIST = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]


def main():
    # Load agent
    agent = DiceAdventureAgent()
    # Set up environment
    env = DiceAdventurePythonEnv(player=PLAYER, server=SERVER, train_mode=False)
    obs = env.reset()[0]

    while True:
        action = agent.take_action(state=obs, actions=ACTION_LIST)
        obs = env.step(action)
        # env.render()


if __name__ == "__main__":
    main()
