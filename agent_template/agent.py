# PLEASE IMPORT ANY PACKAGES YOU NEED

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

    def take_action(self, state: list[dict], actions: list[str]) -> str:
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   A 'Dice Adventure' game state
        :param actions: A list of string action names
        :return:        An action from the 'actions' list
        """
        raise NotImplementedError("Not implemented yet.")
