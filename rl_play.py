from time import sleep
from examples.RL_agent.dice_adventure_python_env import DiceAdventurePythonEnvRL
from examples.RL_agent.agent import DiceAdventureAgent

SERVER = "local"


def main():
    # Set up environment
    env = DiceAdventurePythonEnvRL(server=SERVER, state_version="fow")
    # Load agents
    players = env.get_player_names()
    agents = [DiceAdventureAgent(p, env.get_player_code(p)) for p in players]

    while True:
        for agent in agents:
            state = env.get_state(player=agent.character)
            action = agent.take_action(state=state, actions=env.get_actions())
            print(f"CHARACTER ({agent.character}) SUBMITTING ACTION: ({action})")
            env.execute_action(player=agent.character, game_action=action)
        env.render()
        # sleep(1)


if __name__ == "__main__":
    main()
