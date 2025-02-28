# PLEASE IMPORT ANY PACKAGES YOU NEED

class DiceAdventureAgent:
    """
    Provides a uniform interface to connect agents to Dice Adventure environment.
    Developers must implement the take_action() function.
    - init():         Initialize any needed variables.
    - take_action():  Determines which action to take given a state (dict) and list of actions. Note that your
                      agent does not need to use the list of actions, it is just provided for convenience.
    """
    def __init__(self, character_name, character_id):
        """
        Initialize any needed variables. The character_name and character_id arguments specify the name and ID
        of the character this agent will play as.
        :param character_name: (string) The character the agent will play as
        :param character_id: (string) The character ID corresponding to the character

        Player ID mappings:
            Dwarf : C11
            Giant : C21
            Human : C31
        """
        self.character = character_name
        self.character_id = character_id

    def take_action(self, state, actions):
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   (dict) A 'Dice Adventure' game state
        :param actions: (list) A list of string action names
        :return:        (string) An action from the 'actions' list
        """
        raise NotImplementedError("Not implemented yet.")
