from time import sleep
from examples.random_agent.dice_adventure_gym_env import DiceAdventurePythonEnv
from examples.random_agent.agent import DiceAdventureAgent


SERVER = "local"


def main():
    # Set up environment
    env = DiceAdventurePythonEnv(player="dwarf", port='4649')
    # Load agents
    players = env.get_player_names()
    agents = [DiceAdventureAgent(p, env.get_player_code(player=p, port='4650')) for p in players]

    while True:
        for agent in agents:
            state = env.get_state()
            action = agent.take_action(state=state, actions=env.get_actions())
            env.execute_action(game_action=action)
        env.render()
        sleep(.1)


if __name__ == "__main__":
    main()
