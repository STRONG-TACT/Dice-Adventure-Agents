from examples.random_agent.dice_adventure_python_env import DiceAdventurePythonEnv
from examples.random_agent.agent import DiceAdventureAgent

PLAYERS = ["Dwarf", "Giant", "Human"]
SERVER = "local"
ACTION_LIST = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]


def main():
    # Load agent
    agent = DiceAdventureAgent()
    # Set up environment
    env = DiceAdventurePythonEnv(server=SERVER)
    state = env.reset()[0]

    while True:
        for p in PLAYERS:
            action = agent.take_action(state=state, actions=ACTION_LIST)
            state = env.execute_action(player=p, game_action=action)
        # env.render()


if __name__ == "__main__":
    main()
