from collections import Counter
from collections import defaultdict
from random import shuffle
import re
from tabulate import tabulate
from classes.game_objects import *


class Board:
    def __init__(self, width, height, object_positions, config):
        self.width = None
        self.height = None
        self.board = None
        self.objects = None
        self.config = config
        # Keeps track of object counts for indexing purposes
        self.obj_counts = None
        # Initialize board
        self.reset_board(width, height, object_positions)

    def reset_board(self, width, height, object_positions):

        self.board = defaultdict(dict)
        self.objects = {}
        # Keeps track of object counts for indexing purposes
        self.obj_counts = Counter()

        if width or height:
            self.width = width
            self.height = height

        for y in range(self.height):
            for x in range(self.width):
                if object_positions[y][x] == "##":
                    self.board[(y,x)] = None
                elif object_positions[y][x] == "..":
                    self.board[(y,x)] = {}
                else:
                    obj = self.create_object(x, y, object_positions[y][x])
                    self.board[(y,x)][obj.index] = obj
                    self.objects[obj.index] = obj

    def create_object(self, x_pos, y_pos, obj_code, placed_by=None):
        """

        :param x_pos:
        :param y_pos:
        :param obj_code:
        :param placed_by: Special parameter for pins. Determines which player placed pin
        :return:
        """
        self.obj_counts[obj_code] += 1
        index = obj_code
        if self.obj_counts[obj_code] > 1:
            index += f"({self.obj_counts[obj_code]})"
        # Player objects
        if re.match("\\dS", obj_code):
            obj = Player(obj_code=obj_code,
                         index=index,
                         name=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["NAME"],
                         x=x_pos,
                         y=y_pos,
                         action_points=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["ACTION_POINTS"],
                         health=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["HEALTH"],
                         sight_range=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["SIGHT_RANGE"],
                         dice_rolls=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["DICE_ROLLS"])
        # Enemy objects
        elif re.match("(M\\d|S\\d|T\\d)", obj_code):
            obj = Enemy(obj_code=obj_code,
                        index=index,
                        name=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["NAME"],
                        type_=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["TYPE"],
                        x=x_pos,
                        y=y_pos,
                        dice_rolls=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["DICE_ROLLS"],
                        action_points=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["ACTION_POINTS"]
                        if obj_code[0] == "M" else None)
        # Special objects
        else:
            # Walls and empty spaces don't get their own python objects
            if obj_code in ["##", ".."]:
                obj = None
            # Tower object
            elif obj_code == "**":
                obj = Tower(obj_code=obj_code,
                            index=index,
                            name=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["NAME"],
                            type_=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["TYPE"],
                            x=x_pos,
                            y=y_pos)
            # Shrine objects
            elif re.match("\\dG", obj_code):
                obj = Shrine(obj_code=obj_code,
                             index=index,
                             name=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["NAME"],
                             type_=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["TYPE"],
                             x=x_pos,
                             y=y_pos,
                             player_code=obj_code[0])
            # Pin objects
            else:
                obj = Pin(obj_code=obj_code,
                          index=index,
                          x=x_pos,
                          y=y_pos,
                          placed_by=placed_by,
                          type_=self.config["OBJECT_INFO"]["OBJECT_CODES"][obj_code]["TYPE"])

        return obj

    ##########################
    # POSITIONING & MOVEMENT #
    ##########################

    def check_valid_move(self, x, y, avoid=None, allow_wall=False):
        """
        Checks whether the given x,y position is a valid position for movement/placement.
        :param x: Specifies the x location to move/place to
        :param y: Specifies the y location to move/place to
        :return: True/False
        """
        if avoid is None:
            avoid = []
        # x,y position is within bounds
        # position does not contain a wall
        # no objects in the avoid list is at x,y position
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.board[(y,x)]:
                if not any([True for obj in self.board[(y,x)].values() if obj.name in avoid]):
                    return True
            elif allow_wall:
                return True
        return False

    def valid_move(self, x, y, action):
        mapping = {"left": (x-1, y), "right": (x+1, y), "up": (x, y+1), "down": (x, y-1), "wait": (x, y)}
        return self.check_valid_move(*mapping[action])

    def update_location_by_direction(self, action, x, y, avoid=None, allow_wall=False):
        """
        Updates character location based on given directional action. Note that when a 2D array is printed,
        moving "up" is equivalent to going down one index in the list of rows and vice versa for
        moving "down". Therefore, "up" decreases the x coordinate by 1 and "down" increases the
        x coordinate by 1.
        :param action:
        :param x:
        :param y:
        :return:
        """
        if action == "wait":
            pass
        elif action == "left" and self.check_valid_move(x-1, y, avoid=avoid, allow_wall=allow_wall):
            x -= 1
        elif action == "right" and self.check_valid_move(x+1, y, avoid=avoid, allow_wall=allow_wall):
            x += 1
        elif action == "up" and self.check_valid_move(x, y+1, avoid=avoid, allow_wall=allow_wall):
            y += 1
        elif action == "down" and self.check_valid_move(x, y-1, avoid=avoid, allow_wall=allow_wall):
            y -= 1

        return x, y

    def move_monster(self, m, directions):
        """
        Implements logic for monster movement. Currently, this is random walk.
        :param m: The monster to move
        :return: N/A
        """
        x = self.objects[m].x
        y = self.objects[m].y
        # old_pos = (x, y)

        shuffle(directions)
        while directions:
            op = directions.pop()
            new_x, new_y = self.update_location_by_direction(op, x, y, avoid=["Stone", "Trap"])
            if x != new_x or y != new_y:
                self.place(m, new_x, new_y, old_x=x, old_y=y)
                break

    def move(self, obj_index, action, x=None, y=None, old_pos=None, delete=False):
        """
        Moves the given object according to the given action
        :param obj_index: The object to move
        :param action: The cardinal action to take
        :param x: Specifies the x location of the player to move
        :param y: Specifies the y location of the player to move
        :param old_pos: Specifies the current x,y location of the player which becomes the previous location after the
        move
        :return: N/A
        """
        if action == "wait":
            # No-op
            return
        # x is None and y is None, need to get new position
        if x is None or y is None:
            x = self.objects[obj_index].x
            y = self.objects[obj_index].y
            old_pos = (x, y)

            x, y = self.update_location_by_direction(action, x, y)

        if x != old_pos[0] or y != old_pos[1]:
            self.place(obj_index, x, y, old_x=old_pos[0], old_y=old_pos[1], delete=delete)

    def place(self, obj_index, x, y, old_x=None, old_y=None, create=False, placed_by=None, delete=False):
        """
        Places the given object at the given x,y position
        :param obj_index: The object to place
        :param x: Specifies the x location to place on
        :param y: Specifies the y location to place on
        :param old_x: Specifies the current x location of the object which becomes the previous x location after the
        placement
        :param old_y: Specifies the current y location of the object which becomes the previous y location after the
        placement
        :param index: If provided, specifies that 'obj' should be placed on board with value of 'index' in parentheses
        (usually for identification purposes)
        :return:
        """
        # If 'index' parameter is provided, specifies that obj should be placed on board with value of
        # 'index' in parentheses (usually for identification purposes)
        # if index:
        #     obj = f"{obj}({index})"
        if create:
            new_obj = self.create_object(x, y, obj_index, placed_by=placed_by)
            self.board[(y,x)][obj_index] = new_obj
            self.objects[obj_index] = new_obj
        else:
            # Update location of object
            self.objects[obj_index].x = x
            self.objects[obj_index].y = y
            self.board[(y,x)][obj_index] = self.objects[obj_index]

            # Remove obj from old position if it was previously on the grid
            if old_x is not None:
                self.remove(obj_index, old_x, old_y, delete=delete)

    def remove(self, obj_index, x=None, y=None, delete=True):
        """
        Removes the given object from the grid. The x,y position of the object can be optionally supplied
        :param obj: The object to remove
        :param x: The x position of the object to remove
        :param y: The y position of the object to remove
        :param delete: If False, does not delete object from object list
        :return: N/A
        """

        if x is None or y is None:
            x = self.objects[obj_index].x
            y = self.objects[obj_index].y
        # Delete object from board
        self.board[(y,x)] = {k: v for k, v in self.board[(y,x)].items() if k != obj_index}
        # In this case, should delete object entirely (from game)
        if delete:
            del self.objects[obj_index]

    def multi_remove(self, objs):
        """
        Removes the given objects from the grid
        :param objs: A list of objects to remove
        :return: N/A
        """
        for o in objs:
            if isinstance(o, GameObject):
                obj_index = o.index
            else:
                obj_index = o
            self.remove(obj_index)

    ################
    # GOAL TESTING #
    ################

    def at(self, player, obj):
        """
        Checks whether the given player and object are co-located
        :param player: The player to check
        :param obj: The object to check
        :return: True/False
        """
        return self.objects[player].x == self.objects[obj].x \
            and self.objects[player].y == self.objects[obj].y

    def print_board(self, render_verbose=False, level=None, phase=None, num_rounds=None, round_cap=None):
        if render_verbose:
            info = [
                ["Level:", level],
                ["Phase:", phase],
                ["Number of Rounds Left:", round_cap - num_rounds],
                ["Subgoal Count:", self.objects["**"].subgoal_count]
            ]
            print("Game Info:")
            print(tabulate(info, tablefmt="grid"))

        table = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Wall
                if self.board[(y,x)] is None:
                    row.append("##")
                # Empty dict == empty space
                elif not self.board[(y,x)]:
                    row.append(" ")
                else:
                    codes = []
                    for o in self.board[(y,x)].values():
                        if isinstance(o, Player) and o.dead:
                            continue
                        codes.append(o.index)
                    string = "-".join(codes)
                    row.append(string)
            table.append(row)
        print("Grid:")
        print(tabulate(reversed(table), tablefmt="grid"), end="\n\n")
