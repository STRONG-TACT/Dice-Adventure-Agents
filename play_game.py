from time import sleep
from examples.random_agent.dice_adventure_gym_env import DiceAdventureGymEnv
from examples.random_agent.dice_adventure_gym_env import get_player_id
from examples.random_agent.agent import DiceAdventureAgent


def main():
    # Set up environment
    env = DiceAdventurePythonEnv(port='4649', train_mode=False)
    # Load agents
    players = env.get_player_names()
    agents = [DiceAdventureAgent(p, get_player_id(player=p)) for p in players]

    while True:
        for i in range(len(agents)):
            state = env.get_state(players[i])
            action = agent.take_action(state=state, actions=env.get_actions())
            env.execute_action(game_action=action)
        env.render()
        sleep(.1)


if __name__ == "__main__":
    main()
