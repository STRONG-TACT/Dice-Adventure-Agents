from collections import Counter
from random import choice
from random import seed


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
        self.level = None
        self.visited = set()
        self.visited_counter = Counter()
        self.task = None
        self.tasks = ["shrine", "tower", "explore"]

    def take_action(self, state, actions):
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   (dict) A 'Dice Adventure' game state
        :param actions: (list) A list of string action names
        :return:        (string) An action from the 'actions' list
        """
        # seed(0)
        scene = state["content"]["scene"]
        player, shrine, tower = self.get_objects(scene)

        self.task = self.get_task(player, shrine, tower, scene)
        target = None
        if self.task not in self.tasks:
            target = self.find_object(scene, self.task)

        # No pinning
        if state["content"]["gameData"]["currentPhase"] == "Player_Pinning":
            return "submit"

        elif state["content"]["gameData"]["currentPhase"] == "Player_Planning":
            # Player is dead, do nothing (return wait)
            if player["dead"]:
                return "wait"

            if player["actionPoints"] > 0:

                if self.task == "shrine":
                    return self.get_direction(player, shrine, state)
                elif self.task == "tower":
                    return self.get_direction(player, tower, state)
                elif self.task == "explore":
                    return self.get_direction(player, tower, state, explore=True)
                else:
                    return self.get_direction(player, target, state)
            else:
                x, y = self.get_x_y_cursor(player["x"], player["y"], player["actionPlan"])
                self.visited.add((y,x))
                self.visited_counter[(y,x)] += 1
                return "submit"

    def get_objects(self, scene):
        player = None
        shrine = None
        tower = None
        for obj in scene:
            if obj.get("entityType") == self.character:
                player = obj
            elif obj.get("entityType") == "Shrine" and obj.get("character") == self.character:
                shrine = obj
            elif obj.get("entityType") == "Goal":
                tower = obj
        return player, shrine, tower

    @staticmethod
    def find_object(scene, obj_id):
        for obj in scene:
            if obj.get("id") == obj_id:
                return obj

    def get_direction(self, player, goal, state, explore=False):
        """
        If x distance is larger than y distance, move along x-axis otherwise move along y-axis. If there is a wall
        blocking the chosen direction, randomly chose another direction.
        :param player: (dict) The player to take an action for
        :param goal: (dict) The goal the player is trying to get to
        :param state: (dict) The game state
        :return: (string) The directional action to take
        """
        x, y = self.get_x_y_cursor(player["x"], player["y"], player["actionPlan"])

        # Explore when no shrine or tower in state
        if explore:
            action = choice(self.directions)
        else:
            if x == goal["x"] and y == goal["y"]:
                action = "submit"
            else:
                if abs(x - goal["x"]) >= abs(y - goal["y"]):
                    action = "left" if goal["x"] < x else "right"
                else:
                    action = "down" if goal["y"] < y else "up"

        directions = self.directions.copy()
        walls = [obj for obj in state["content"]["scene"] if obj.get("entityType") == "Wall"]
        while not self.check_valid_move(x, y, action, state, walls, explore):
            directions.remove(action)
            # If all locations are exhausted, take random action and clear visited list
            if not directions:
                action = choice(self.directions)
                self.visited = set()
                self.visited_counter = Counter()
                break
            action = choice(directions)

        return action

    def check_valid_move(self, x, y, action, state, walls, explore=False):
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

        valid_move = 0 <= x < state["content"]["gameData"]["boardWidth"] and \
            0 <= y < state["content"]["gameData"]["boardHeight"] and \
            not any([w for w in walls if w["x"] == x and w["y"] == y])

        if explore:
            valid_move = valid_move and (y,x) not in self.visited

        return valid_move

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

    def get_task(self, player, shrine, tower, scene):
        if shrine:
            # If agent has visited same spot too many times, give it new target
            if self.visited_counter[(player["y"], player["x"])] > 3 and self.task in self.tasks:
                return choice([i.get("id") for i in scene if i.get("entityType") not in ["Wall", "Open"]])

            if not shrine["reached"]:
                return "shrine"
            elif tower:
                return "tower"

        return "explore"

