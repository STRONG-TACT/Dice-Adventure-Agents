from random import choice
import json


class DiceAdventureAgent:
    """
    Provides a uniform interface to connect agents to Dice Adventure environment.
    Developers must implement the take_action() function.
    - init():         Initialize any needed variables.
    - take_action():  Determines which action to take given a state (dict) and list of actions. Note that your
                      agent does not need to use the list of actions, it is just provided for convenience.
    """
    def __init__(self, character):
        """
        Initialize any needed variables.
        :param character: (string) The character the agent will play as
        """
        self.character = character
        self.directions = ["up", "down", "left", "right"]

    def take_action(self, state, actions):
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   (dict) A 'Dice Adventure' game state
        :param actions: (list) A list of string action names
        :return:        (string) An action from the 'actions' list
        """
        # print(json.dumps(state, indent=2))
        scene = state["content"]["scene"]
        player, shrine, tower = self.get_objects(scene)
        # task = self.get_task(shrine, tower)
        task = "shrine" if shrine["reached"] else "tower"

        # No pinning
        if state["content"]["gameData"]["currentPhase"] == "Player_Pinning":
            return "submit"

        elif state["content"]["gameData"]["currentPhase"] == "Player_Planning":
            # Player is dead, do nothing (return wait)
            if player["dead"]:
                return "wait"
            # print(player)
            # print(task)
            # print(self.directions)
            if player["actionPoints"] > 0:

                if task == "shrine":
                    return self.get_direction(player, shrine, state)
                else:
                    return self.get_direction(player, tower, state)
            else:
                return "submit"

    def get_objects(self, scene):
        player = None
        shrine = None
        tower = None
        for obj in scene:
            # TODO CHANGE BACK TO ENTITY TYPE WHEN THEY ARE IN STATE FOR CHARACTERS
            if obj.get("name") == self.character:
                player = obj
            elif obj.get("entityType") == "Shrine" and obj.get("character") == self.character:
                shrine = obj
            elif obj.get("entityType") == "Goal":
                tower = obj
        return player, shrine, tower

    def get_direction(self, player, goal, state):
        """
        If x distance is larger than y distance, move along x-axis otherwise move along y-axis. If there is a wall
        blocking the chosen direction, randomly chose another direction.
        :param player: (dict) The player to take an action for
        :param goal: (dict) The goal the player is trying to get to
        :param state: (dict) The game state
        :return: (string) The directional action to take
        """
        x, y = self.get_x_y_cursor(player["x"], player["y"], player["actionPlan"])
        print(f"PLAYER: {player['name']} | AT ({x},{y})")

        if x == goal["x"] and y == goal["y"]:
            action = "submit"
        else:
            if abs(x - goal["x"]) >= abs(y - goal["y"]):
                action = "left" if goal["x"] < x else "right"
            else:
                action = "down" if goal["y"] < y else "up"

            directions = self.directions.copy()
            walls = [obj for obj in state["content"]["scene"] if obj.get("entityType") == "Wall"]
            while not self.check_valid_move(x, y, action, state, walls):
                directions.remove(action)
                action = choice(directions)

        return action

    def check_valid_move(self, x, y, action, state, walls):
        """
        Checks if the action will result in a valid move. A move is invalid if the action puts the character
        out of bounds of the grid or on a wall.
        :param x: (string) The player x position
        :param y: (string) The player y position
        :param action: (string) The action to take
        :param state: (dict) The game state
        :param walls: (list) A list of wall objects from the state
        :return: True/False
        """
        x, y = self.get_x_y_cursor(x, y, [action])

        return 0 <= x < state["content"]["gameData"]["boardWidth"] and \
            0 <= y < state["content"]["gameData"]["boardHeight"] and \
            not any([w for w in walls if w["x"] == x and w["y"] == y])

    @staticmethod
    def get_x_y_cursor(x, y, action_plan):
        for action in action_plan:
            if action == "up":
                y += 1
            elif action == "down":
                y -= 1
            elif action == "left":
                x -= 1
            elif action == "right":
                x += 1
            elif action == "wait":
                pass
        return x, y


    """
        def get_task(self, shrine, tower):
        if shrine:
            if not shrine["reached"]:
                task = "shrine"
            elif tower:
                task = "tower"
        return task
    """
