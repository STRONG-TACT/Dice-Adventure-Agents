from agent_template.dice_adventure_gym_env import DiceAdventureGymEnv
from agent_template.dice_adventure_gym_env import get_player_id
from examples.random_agent.agent import DiceAdventureAgent


def main():
    # Set up environment
    env = DiceAdventureGymEnv(port='4649', game_executable_filepath='/DiceAdventure.exe')
    # sLoad agents
    agents = [('dwarf', DiceAdventureAgent('dwarf', get_player_id('dwarf'))),
              ('giant', DiceAdventureAgent('giant', get_player_id('giant'))),
              ('human', DiceAdventureAgent('human', get_player_id('human')))]
    env.play(agents)


if __name__ == "__main__":
    main()
