from time import sleep
from examples.random_agent.dice_adventure_gym_env import DiceAdventureGymEnv
from examples.random_agent.dice_adventure_gym_env import get_player_id
from examples.random_agent.agent import DiceAdventureAgent


def main():
    players = ['dwarf', 'giant', 'human']
    # Set up environments
    envs = [DiceAdventurePythonEnv(player=p, port='4649') for p in players]
    # Create agents
    agents = [DiceAdventureAgent(p, get_player_id(player=p, port='4649')) for i, p in enumerate(players)]

    while True:
        for i in range(len(agents)):
            state = envs[i].get_state()
            action = agents[i].take_action(state=state, actions=env.get_actions())
            envs[i].execute_action(game_action=action)
        sleep(.1)


if __name__ == "__main__":
    main()
