# PLEASE IMPORT ANY PACKAGES YOU NEED
from stable_baselines3 import MaskablePPO
from random import seed
from dice_adventure_python_env import DiceAdventureGymEnvRL
class DiceAdventureAgent:
    """
    An interface to connect agents to the Dice Adventure environment.
    Developers must implement the take_action() function.
    - init():         Initialize any needed variables.
    - take_action():  Determines which action to take given a state (list<dict>) and list of actions. Note that your
                      agent does not need to use the list of actions, it is just provided for convenience.
    """
    def __init__(self, character_name: str, character_id: str) -> None:
        """
        Initialize any needed variables. The character_name and character_id arguments specify the name and ID
        of the character this agent will play as.
        :param character_name: The character the agent will play as
        :param character_id: The character ID corresponding to the character

        Player ID mappings:
            dwarf : C11
            giant : C21
            human : C31
        """
        self.character = character_name
        self.character_id = character_id

        try:
            self.model_filename = "path/to/model.pth"
            self.model = MaskablePPO.load(self.model_filename)
        except Exception as e:
            print(f"Error loading model: {e}")

        self.env = DiceAdventureGymEnvRL(port="8000", game_executable_filepath="path/to/game.exe")

    def take_action(self, state: list[dict], actions: list[str]) -> str:
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   A 'Dice Adventure' game state
        :param actions: A list of string action names
        :return:        An action from the 'actions' list
        """
        try:
            seed(1)
            obs = self.env.get_observation(state)
            mask = self.env.action_masks(state)
            current_phase = state[0]["currentPhase"]
    

            
            # Use predict_with_mask for MaskablePPO
            action, _states = self.model.predict(
                observation=obs,
                action_masks=mask,
                deterministic=True
            )
            with open("action_log.txt", "w") as f:
                f.write(f"Mask={mask}, Phase={current_phase}, Action Predicted={action}\n")
        
            
            return self.env.action_map[int(action)]
        except Exception as e:
            print(f"Error in take_action: {e}")
            return 'wait'
