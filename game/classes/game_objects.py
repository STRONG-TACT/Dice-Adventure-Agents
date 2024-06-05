from random import choice


class GameObject:
    def __init__(self, obj_code, index, index_num, x, y, type_, entity_type):
        self.obj_code = obj_code
        self.index = index
        self.index_num = index_num
        self.type = type_
        self.entity_type = entity_type
        self.x = x
        self.y = y


class Goal(GameObject):
    def __init__(self, obj_code, index, index_num, name, type_, entity_type, x, y):
        super().__init__(obj_code, index, index_num, x, y, type_, entity_type)
        self.name = name
        self.reached = False


class Shrine(Goal):
    def __init__(self, obj_code, index, index_num, name, type_, entity_type, x, y, character):
        super().__init__(obj_code, index, index_num, name, type_, entity_type, x, y)
        self.player = character


class Tower(Goal):
    def __init__(self, obj_code, index, index_num, name, type_, entity_type, x, y):
        super().__init__(obj_code, index, index_num, name, type_, entity_type, x, y)
        self.subgoal_count = 0


class Enemy(GameObject):
    def __init__(self, obj_code, index, index_num, name, type_, entity_type, x, y, dice_rolls, action_points=None):
        super().__init__(obj_code, index, index_num, x, y, type_, entity_type)
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
    def __init__(self, obj_code, index, index_num, name, x, y, type_, entity_type, action_points, health, lives, sight_range, dice_rolls):
        super().__init__(obj_code, index, index_num, x, y, type_, entity_type)
        # Indexing
        self.name = name
        # Stats
        self.action_points = action_points
        self.max_action_points = action_points
        self.health = health
        self.max_health = health
        self.lives = lives
        self.max_lives = lives
        self.sight_range = sight_range
        self.dice_rolls = dice_rolls
        # Location
        self.start_x = x
        self.start_y = y
        self.prev_x = x
        self.prev_y = y
        self.seen_locations = set()
        # Update now
        self.update_seen_locations()
        # Status
        self.dead = False
        self.can_respawn = True
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

    def get_mask_radius(self):
        locations = []
        x_bound_upper = self.x + self.sight_range
        x_bound_lower = self.x - self.sight_range
        y_bound_upper = self.y + self.sight_range
        y_bound_lower = self.y - self.sight_range

        for j in range(y_bound_lower, y_bound_upper + 1):
            for i in range(x_bound_lower, x_bound_upper + 1):
                locations.append((j, i))
        return locations

    def update_seen_locations(self):
        self.seen_locations.update(self.get_mask_radius())


class Pin(GameObject):
    def __init__(self, obj_code, index, index_num, x, y, placed_by, type_, entity_type):
        super().__init__(obj_code, index, index_num, x, y, type_, entity_type)
        self.name = obj_code
        self.placed_by = placed_by



