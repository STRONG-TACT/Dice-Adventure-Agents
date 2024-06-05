from time import sleep
from examples.production_agent.dice_adventure_python_env import DiceAdventurePythonEnv
from examples.production_agent.agent import DiceAdventureAgent


SERVER = "local"


def main():
    # Set up environment
    env = DiceAdventurePythonEnv(server=SERVER, state_version="full")
    state = env.reset()[0]
    # Load agents
    players = env.get_player_names()
    agents = [DiceAdventureAgent(p) for p in players]

    while True:
        for agent in agents:
            action = agent.take_action(state=state, actions=env.get_actions())
            state = env.execute_action(player=agent.character, game_action=action)
        env.render()
        sleep(.1)


if __name__ == "__main__":
    main()
