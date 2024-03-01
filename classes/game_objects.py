from random import choice


class GameObject:
    def __init__(self, obj_code, index, x, y, type_):
        self.obj_code = obj_code
        self.index = index
        self.type = type_
        self.x = x
        self.y = y


class Goal(GameObject):
    def __init__(self, obj_code, index, name, type_, x, y):
        super().__init__(obj_code, index, x, y, type_)
        self.name = name
        self.reached = False


class Shrine(Goal):
    def __init__(self, obj_code, index, name, type_, x, y, player_code):
        super().__init__(obj_code, index, name, type_, x, y)
        self.player = self.get_player(player_code)

    @staticmethod
    def get_player(player_code):
        player_goal_map = {"1": "Dwarf", "2": "Giant", "3": "Human"}
        return player_goal_map[player_code]


class Tower(Goal):
    def __init__(self, obj_code, index, name, type_, x, y):
        super().__init__(obj_code, index, name, type_, x, y)
        self.subgoal_count = 0


class Enemy(GameObject):
    def __init__(self, obj_code, index, name, type_, x, y, dice_rolls, action_points=None):
        super().__init__(obj_code, index, x, y, type_)
        self.name = name
        self.dice_rolls = dice_rolls
        self.action_points = action_points

    def get_dice_roll(self):
        val = self.dice_rolls["VAL"]
        const = self.dice_rolls["CONST"]
        if val > 0:
            roll = choice(range(val))
        else:
            roll = 0
        return roll + const


class Player(GameObject):
    def __init__(self, obj_code, index, name, x, y, action_points, health, sight_range, dice_rolls):
        super().__init__(obj_code, index, x, y, type_=name)
        # Indexing
        self.name = name
        # Stats
        self.action_points = action_points
        self.max_action_points = action_points
        self.health = health
        self.max_health = health
        self.sight_range = sight_range
        self.dice_rolls = dice_rolls
        # Location
        self.start_x = x
        self.start_y = y
        self.prev_x = x
        self.prev_y = y
        # Status
        self.dead = False
        self.respawn_counter = None
        self.death_round = None
        self.goal_reached = False
        self.combat_success = False
        # Pinning
        self.pin_x = x
        self.pin_y = y
        self.placed_pin = False
        self.pin_finalized = False
        # Action planning
        self.action_plan = []
        self.action_positions = []
        self.action_plan_x = None
        self.action_path_y = None
        self.prev_action_plan_x = None
        self.prev_action_path_y = None
        self.action_plan_step = None
        self.action_plan_finalized = False

    def get_dice_roll(self, enemy_type):
        enemy_type = enemy_type.upper()
        val = self.dice_rolls[enemy_type]["VAL"]
        const = self.dice_rolls[enemy_type]["CONST"]
        if val > 0:
            roll = choice(range(val))
        else:
            roll = 0
        return roll + const

    def reset_phase_values(self):
        # Pinning
        self.pin_x = self.x
        self.pin_y = self.y
        self.placed_pin = False
        self.pin_finalized = False
        # Action planning
        self.action_plan = []
        self.action_positions = []
        self.action_plan_x = None
        self.action_path_y = None
        self.action_plan_step = None
        self.action_plan_finalized = False


class Pin(GameObject):
    def __init__(self, obj_code, index, x, y, placed_by, type_):
        super().__init__(obj_code, index, x, y, type_)
        self.name = obj_code
        self.placed_by = placed_by



