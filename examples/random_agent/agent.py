from random import choice


class DiceAdventureAgent:
    """
    Provides a uniform interface to connect agents to Dice Adventure environment.
    Developers must implement the take_action() function.
    - init():         Initialize any needed variables.
    - take_action():  Determines which action to take given a state (dict) and list of actions. Note that your
                      agent does not need to use the list of actions, it is just provided for convenience.
    """
    def __init__(self, character_name, character_code):
        """
        Initialize any needed variables.
        :param character_name: (string) The character the agent will play as
        :param character_code: (string) The character code corresponding to the character

        Player code index:
        {
            "Dwarf" : "C1",
            "Giant" : "C2",
            "Human" : "C3"
        }
        """
        self.character = character_name
        self.character_code = character_code

    def take_action(self, state, actions):
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   (dict) A 'Dice Adventure' game state
        :param actions: (list) A list of string action names
        :return:        (string) An action from the 'actions' list
        """
        return choice(actions)
