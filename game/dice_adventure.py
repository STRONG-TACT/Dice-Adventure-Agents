from copy import deepcopy
from json import loads
from os import listdir
from time import time
from game.classes.board import Board
from game.classes.game_objects import *
from game.classes.metrics_tracker import GameMetricsTracker


class DiceAdventure:
    def __init__(self,
                 level=1,
                 limit_levels=None,
                 level_sampling=False,
                 model_number=1,
                 num_repeats=0,
                 render=False,
                 render_verbose=True,
                 restart_on_finish=False,
                 round_cap=0,
                 track_metrics=False):

        #################
        # GAME METADATA #
        #################
        self.config = loads(open("game/config/main_config.json", "r").read())
        self.terminated = False

        ##############
        # LEVEL VARS #
        ##############
        # Level Setup
        self.levels = {}
        self.levels_directory = "game/levels/"
        self.limit_levels = limit_levels if limit_levels \
            else [i+1 for i in range(len(listdir(self.levels_directory)))]
        self.get_levels()
        # Level Control
        self.curr_level_num = level if level in self.limit_levels else self.limit_levels[0]
        self.curr_level = deepcopy(self.levels[self.curr_level_num])
        self.num_repeats = num_repeats
        self.lvl_repeats = {lvl: self.num_repeats for lvl in self.levels}
        self.restart_on_finish = restart_on_finish
        self.restart_on_team_loss = False
        # Object codes
        self.empty = ".."
        self.tower = "**"
        self.wall = "##"
        self.player_code_mapping = self.config["OBJECT_INFO"]["OBJECT_CODE_MAPPINGS"]["PLAYERS"]

        # Places a cap on the number of rounds per level
        self.round_cap = round_cap
        # Determines whether levels should be randomly sampled
        self.level_sampling = level_sampling
        self.respawn_wait = 2

        ##########
        # BOARD #
        #########
        self.board = Board(width=len(self.curr_level[0]),
                           height=len(self.curr_level),
                           object_positions=self.curr_level,
                           config=self.config)

        ##############
        # PHASE VARS #
        ##############
        self.phase_num = 0
        self.phases = self.config["GAMEPLAY"]["PHASES"]["PHASE_LIST"]
        # Pin Planning
        self.pinning_phase_name = self.config["GAMEPLAY"]["PHASES"]["PINNING_PHASE_NAME"]
        self.valid_pin_actions = self.config["GAMEPLAY"]["ACTIONS"]["VALID_PIN_ACTIONS"]
        self.valid_pin_types = self.config["GAMEPLAY"]["ACTIONS"]["VALID_PIN_TYPES"]
        self.pin_code_mapping = self.config["OBJECT_INFO"]["OBJECT_CODE_MAPPINGS"]["PINS"]
        # Action Planning
        self.planning_phase_name = self.config["GAMEPLAY"]["PHASES"]["PLANNING_PHASE_NAME"]
        self.valid_move_actions = self.config["GAMEPLAY"]["ACTIONS"]["VALID_MOVE_ACTIONS"]
        # Actions
        self.directions = self.config["GAMEPLAY"]["ACTIONS"]["DIRECTIONS"]
        # Enemy Execution
        self.enemy_execution_phase_name = self.config["GAMEPLAY"]["PHASES"]["ENEMY_EXECUTION_PHASE_NAME"]
        # Countdown timer
        self.countdown_timer = time()
        self.countdown_max_time = self.config["GAMEPLAY"]["COUNTDOWN_TIMER_SECONDS"]
        #############
        # RENDERING #
        #############
        self.render_game = render
        self.render_verbose = render_verbose
        ###################
        # METRIC TRACKING #
        ###################
        self.track_metrics = track_metrics
        # Number of calls to execute_action() or get_state() functions
        self.num_calls = 0
        # Number of rounds completed
        self.num_rounds = 0
        # Metrics tracker
        if self.track_metrics:
            self.tracker = GameMetricsTracker(level=self.curr_level_num,
                                              metrics_config=self.config["GAMEPLAY"]["METRICS"],
                                              instance_id=model_number,
                                              model_number=model_number)

    #################
    # LEVEL CONTROL #
    #################
    def get_levels(self):
        levels = {int(filename.rstrip(".txt")): open(self.levels_directory + filename, "r").read()
                  for filename in listdir(self.levels_directory)}

        # This makes sure positions are indexed with origin at "bottom left"
        self.levels = {
            k: [[row[i:i + 2] for i in range(0, len(row), 2)] for row in reversed(v.strip().split("\n"))]
            for k, v in levels.items()
            if int(k) in self.limit_levels
        }

    def next_level(self):
        """
        Moves the game to the next level or repeats the same level
        :return:
        """
        # Don't change anything if restarting level due to whole team dying
        if not self.restart_on_team_loss:
            # print(self.lvl_repeats)
            eligible_levels = [k for k, v in self.lvl_repeats.items() if v >= 0]
            if not eligible_levels:
                self.terminated = True
                if self.track_metrics:
                    self.tracker.update(target="game", metric_name="game_over")
                return

            self.get_next_level(eligible_levels)

            # If finished final level and set to restart, go back to first level
            if self.curr_level_num > len(self.levels) and not self.level_sampling:
                if self.restart_on_finish:
                    # print(f"RESTARTING TO FIRST LEVEL!")
                    self.curr_level_num = self.limit_levels[0]
                else:
                    self.terminated = True
                    if self.track_metrics:
                        self.tracker.update(target="game", metric_name="game_over")
                    return
            self.restart_on_team_loss = False

        # Set current level
        self.curr_level = deepcopy(self.levels[self.curr_level_num])
        # Re-initialize values
        self.board = Board(width=len(self.curr_level[0]),
                           height=len(self.curr_level),
                           object_positions=self.curr_level,
                           config=self.config)
        self.phase_num = 0
        self.num_rounds = 0

    def get_next_level(self, eligible_levels):
        prev_level = int(str(self.curr_level_num))
        # If level sampling turned on, randomly sample for next level
        if self.level_sampling:
            self.curr_level_num = choice(list(eligible_levels))
        else:
            # Otherwise, move on to next level
            self.curr_level_num += 1
        self.lvl_repeats[self.curr_level_num] -= 1
        # Track whether moved to new level or repeated same level
        if self.track_metrics:
            if prev_level == self.curr_level_num:
                self.tracker.update(target="game", metric_name="num_repeats")
            else:
                self.tracker.update(target="game", metric_name="new_level", level=self.curr_level_num)

    ###########################
    # GET STATE & SEND ACTION #
    ###########################

    def get_state(self, player, version=None):
        """
        Constructs a state representation of the game.
        :return: Dict
        """
        self.check_countdown_timer()
        if self.terminated:
            return

        state = {
            "command": "get_state",
            "status": "OK" if not self.terminated else "Done",
            "message": "Full State",
            "content": {
                "gameData": {
                    "boardWidth": len(self.curr_level[0]),
                    "boardHeight": len(self.curr_level),
                    "currLevel": self.curr_level_num,
                    "currentPhase": self.phases[self.phase_num],
                    "timer": round(self.countdown_max_time - (time() - self.countdown_timer), 2)
                },
                "scene": []
            }
        }
        player_obj = self.board.objects[self.player_code_mapping[player]]
        # Defines the set of grid locations currently visible to the player
        sight_range = player_obj.get_mask_radius()
        if version == "player":
            visible_locations = sight_range
        elif version == "fow":
            visible_locations = player_obj.seen_locations
        else:
            visible_locations = None

        # Walls are not tracked as objects, need to count them here to make sure IDs are unique
        wall_count = 1
        for pos, obj_dict in self.board.board.items():
            if version in ["player", "fow"] and pos not in visible_locations:
                continue
            # Walls
            if obj_dict is None:
                wall_object_code = "##"

                state["content"]["scene"].append({"id": f"{wall_object_code}{wall_count}",
                                                  "objKey": wall_object_code,
                                                  "entityType": self.config["OBJECT_INFO"]["OBJECT_CODES"][wall_object_code]["ENTITY_TYPE"],
                                                  "x": int(pos[1]),
                                                  "y": int(pos[0])})
                wall_count += 1
            else:
                for o in obj_dict:
                    obj = obj_dict[o]
                    # Players are handled at the end of the function
                    if isinstance(obj, Player):
                        continue

                    ele = {"id": f"{obj.obj_code}{obj.index_num}",
                           "objKey": obj.obj_code,
                           "entityType": obj.entity_type,
                           "x": obj.x,
                           "y": obj.y}

                    # Goals
                    if isinstance(obj, Shrine):
                        ele.update({
                            "reached": obj.reached,
                            "character": obj.player
                        })

                    elif isinstance(obj, Tower):
                        ele.update({
                            "subgoalCount": obj.subgoal_count
                        })
                    # Enemies
                    elif isinstance(obj, Enemy):
                        # Monsters should ONLY be returned when in sight range of player, other enemy types can be
                        # returned even if not in sight range but have been seen before when 'version' is 'fow'
                        if version == "full" or obj.name != "Monster" or (obj.name == "Monster" and pos in sight_range):
                            ele.update({
                                "challengeDie": f"D{obj.dice_rolls['VAL']}+{obj.dice_rolls['CONST']}"
                            })
                            # Action points only apply to monsters
                            # if obj.name == "Monster":
                            #    ele["actionPoints"] = self.config["OBJECT_INFO"]["OBJECT_CODES"][obj.obj_code]["ACTION_POINTS"]
                    # Pins
                    elif isinstance(obj, Pin):
                        ele.update({
                            "placedBy": obj.placed_by
                        })
                    state["content"]["scene"].append(ele)

        state["content"]["scene"].extend(self._get_player_state(player_obj, version))

        return state

    def _get_player_state(self, player_obj, version):
        players = [self.board.objects[player_code] for player_code in self.player_code_mapping.values()]
        player_state = []
        sight_range = player_obj.get_mask_radius()
        pos = (player_obj.y, player_obj.x)
        for obj in players:
            ele = {"id": f"{obj.obj_code}{obj.index_num}",
                   "objKey": obj.obj_code,
                   "entityType": obj.entity_type,
                   "ready": obj.pin_finalized if self.phases[self.phase_num] == self.pinning_phase_name
                   else (obj.action_plan_finalized if self.phases[self.phase_num] == self.planning_phase_name
                         else True),
                   "lives": obj.lives
                   }

            # Can return player data ONLY if in current sight range or if giving full state or data relates
            # to self
            if version == "full" or pos in sight_range or obj.obj_code == player_obj.obj_code:
                ele.update({
                    "characterId": int(obj.obj_code[1]),
                    "health": obj.health,
                    "dead": obj.dead,
                    "x": obj.x,
                    "y": obj.y
                })
                # Only provide this information if state being provided is for given character or if giving
                # full state
                if obj.obj_code == player_obj.obj_code or version == "full":
                    ele.update({
                        "pinCursorX": obj.pin_x,
                        "pinCursorY": obj.pin_y,
                        "sightRange": obj.sight_range,
                        "monsterDie": f"D{obj.dice_rolls['MONSTER']['VAL']}+{obj.dice_rolls['MONSTER']['CONST']}",
                        "trapDie": f"D{obj.dice_rolls['TRAP']['VAL']}+{obj.dice_rolls['TRAP']['CONST']}",
                        "stoneDie": f"D{obj.dice_rolls['STONE']['VAL']}+{obj.dice_rolls['STONE']['CONST']}",
                        "actionPoints": obj.action_points,
                        "actionPlan": obj.action_plan
                    })
            player_state.append(ele)

        return player_state

    def execute_action(self, player, action):
        """
        Applies an action to the player given.
        :param player: The player to apply an action to
        :param action: The action to apply
        :return: N/A
        """
        if self.check_countdown_timer() or self.terminated:
            # No-op because phase timer ran out or game is terminated
            return

        player_code = self.player_code_mapping[player]
        # self.num_calls += 1.txt
        if self.track_metrics:
            # Track agent action
            self.tracker.update(target="game", metric_name="agent_action", player=player, agent_action=action,
                                phase=self.phases[self.phase_num])

        if self.phases[self.phase_num] == self.pinning_phase_name:
            self.pin_planning(player_code, action)
        elif self.phases[self.phase_num] == self.planning_phase_name:
            self.action_planning(player_code, action)

    def check_player_status(self):
        """
        Checks whether players are dead or alive and respawn players if enough game cycles have passed
        :return: N/A
        """
        dead = [p for p in self.player_code_mapping.values() if self.board.objects[p].dead]
        # If all players have died, reset level
        if len(dead) == 3:
            if self.track_metrics:
                self.tracker.update(target="game", metric_name="team_death")
            self.restart_on_team_loss = True
            self.next_level()
            return
        # Otherwise, check if players need respawning
        else:
            for p in dead:
                # Check if they've waited enough game cycles
                if self.board.objects[p].can_respawn and (self.num_rounds - self.board.objects[p].death_round
                                                          >= self.respawn_wait):
                    # Player has waited long enough
                    self.board.objects[p].dead = False
                    self.board.objects[p].death_round = None
                    self.board.objects[p].health = self.board.objects[p].max_health
                    self.board.objects[p].prev_x = self.board.objects[p].start_x
                    self.board.objects[p].prev_y = self.board.objects[p].start_y
                    self.board.place(p, x=self.board.objects[p].start_x, y=self.board.objects[p].start_y)

    def check_countdown_timer(self):
        """
        Checks if the max time for current round has passed. If so, auto-submits for all players and moves on to next
        phase.
        :return: N/A
        """
        if time() - self.countdown_timer >= self.countdown_max_time:
            for p in self.player_code_mapping.values():
                if self.phases[self.phase_num] == self.pinning_phase_name:
                    self.board.objects[p].pin_finalized = True
                elif self.phases[self.phase_num] == self.planning_phase_name:
                    self.board.objects[p].action_plan_finalized = True
            self.update_phase()
            return True
        return False

    ##############################
    # PHASE PLANNING & EXECUTION #
    ##############################

    def pin_planning(self, player, action):
        """
        Executes logic for the pin planning phase.
        :param player: The player to apply an action to
        :param action: The action to apply
        :return: N/A
        """
        # Player is dead, can not take actions at this time
        # Player supplied invalid action
        # Player has already finalized pin planning
        # Player out of action points
        if self.board.objects[player].dead or \
                action not in self.valid_pin_actions + self.valid_pin_types or \
                self.board.objects[player].pin_finalized:
            # No-op/invalid action
            return

        if action == "submit":
            self.board.objects[player].pin_finalized = True

        # Can only take action if player has enough action points
        if self.board.objects[player].action_points > 0:
            # elif self.players[player]["pin_type"] is None:
            if action in self.directions:
                x, y = self.board.update_location_by_direction(action,
                                                               self.board.objects[player].pin_x,
                                                               self.board.objects[player].pin_y)
                self.board.objects[player].pin_x = x
                self.board.objects[player].pin_y = y

            elif action in self.valid_pin_types:
                # Place new pin
                self.board.place(self.pin_code_mapping[action],
                                 self.board.objects[player].pin_x,
                                 self.board.objects[player].pin_y,
                                 create=True,
                                 placed_by=self.board.objects[player].name)
                self.board.objects[player].placed_pin = True
                self.board.objects[player].action_points -= 1
                # Reset pin_x and pin_y location to player position
                self.board.objects[player].pin_x = self.board.objects[player].x
                self.board.objects[player].pin_y = self.board.objects[player].y
                if self.track_metrics:
                    # Track pin placement
                    self.tracker.update(target="player", metric_name="pins",
                                        player=self.board.objects[player].name, pin_type=action)

        # If player is out of action points and has placed a pin, they are forced to submit
        if self.board.objects[player].placed_pin and self.board.objects[player].action_points <= 0:
            self.board.objects[player].pin_finalized = True

        self.check_phase()

    def action_planning(self, player, action):
        """
        Executes logic for the action planning phase.
        :param player: The player to apply an action to
        :param action: The action to apply
        :return: N/A
        """
        # Player is dead, can not take actions at this time
        if self.board.objects[player].dead \
                or action not in self.valid_move_actions \
                or self.board.objects[player].action_plan_finalized:
            # No-op/invalid action
            return

        # Set init values of action plan
        if not self.board.objects[player].action_plan:
            self.board.objects[player].action_path_x = self.board.objects[player].x
            self.board.objects[player].action_path_y = self.board.objects[player].y

        if action == "submit":
            self.board.objects[player].action_plan_finalized = True

        else:
            # Check for 'undo' action
            if action == "undo":
                self.undo(player)

            elif self.board.objects[player].action_points > 0:
                curr_x = self.board.objects[player].action_path_x
                curr_y = self.board.objects[player].action_path_y
                # print(f"{player} ACTION PATH: ({curr_x}, {curr_y})")
                # Test whether action supplied by agent was a valid move
                if not self.board.valid_move(curr_x, curr_y, action):
                    # No-op/invalid action
                    return
                else:
                    # get new cursor location
                    new_x, new_y = self.board.update_location_by_direction(action, curr_x, curr_y)
                    # Update player fields
                    self.board.objects[player].action_plan.append(action)
                    self.board.objects[player].action_positions.append((new_y, new_x))
                    self.board.objects[player].action_path_x = new_x
                    self.board.objects[player].action_path_y = new_y
                    self.board.objects[player].action_points -= 1

        self.check_phase()

    def check_phase(self):
        """
        Checks whether conditions have been met to end the current phase and apply the actions of the current phase.
        :return: N/A
        """
        curr_phase = self.phases[self.phase_num]

        # Need to end pinning phase, place pins, and begin planning phase
        if curr_phase == self.pinning_phase_name and all([self.board.objects[p].pin_finalized
                                                          for p in self.player_code_mapping.values()
                                                          if not self.board.objects[p].dead]):
            for p in self.player_code_mapping.values():
                # Reset values
                self.board.objects[p].pin_x = self.board.objects[p].x
                self.board.objects[p].pin_y = self.board.objects[p].y
                self.board.objects[p].pin_finalized = False
            # Change to action planning phase
            self.update_phase()
        elif curr_phase == self.planning_phase_name \
                and all([self.board.objects[p].action_plan_finalized
                         for p in self.player_code_mapping.values()
                         if not self.board.objects[p].dead]):
            # Update phase to player execution
            self.update_phase()
            # Reset action points
            for p in self.player_code_mapping.values():
                self.board.objects[p].action_points = self.board.objects[p].max_action_points
                self.board.objects[p].action_plan_finalized = False

            if self.execute_plans():
                self.next_level()
                return
            # Change to enemy execution phase
            self.update_phase()
            # Remove pins and reset player values
            objs = list(self.board.objects.keys())
            for o in objs:
                if isinstance(self.board.objects[o], Player):
                    self.board.objects[o].reset_phase_values()
                elif isinstance(self.board.objects[o], Pin):
                    self.board.remove(o)

    def update_phase(self):
        """
        Moves the game to the next phase.
        :return: N/A
        """
        # Track phases
        if self.track_metrics:
            self.tracker.update(target="game", metric_name="new_phase", phase=self.phases[self.phase_num])

        self.phase_num = (self.phase_num + 1) % len(self.phases)
        # Reset countdown timer
        self.countdown_timer = time()

        # Check if players need respawning
        self.check_player_status()
        # Trigger enemy movement
        if self.phases[self.phase_num] == self.enemy_execution_phase_name:
            self.execute_enemy_plans()
            self.num_rounds += 1
            # Track number of rounds
            if self.track_metrics:
                self.tracker.update(target="game", metric_name="new_round")
            # If a cap has been placed on the number of rounds per level and that cap has been exceeded,
            # move on to next level
            if self.round_cap and self.num_rounds > self.round_cap:
                self.next_level()

    def execute_plans(self):
        """
        Executes plans for each player by iterating over team until no actions remain in their plans.
        :return: N/A
        """
        # print("EXECUTING PLAYER MOVEMENT")
        max_moves = max([len(self.board.objects[p].action_plan) for p in self.player_code_mapping.values()])
        for i in range(max_moves):
            for p in self.player_code_mapping.values():
                # If player is alive, can move
                if not self.board.objects[p].dead and i < len(self.board.objects[p].action_plan):
                    # Get action to make and move
                    # action = self.players[p]["action_plan"][self.players[p]["action_plan_step"]]
                    action = self.board.objects[p].action_plan[i]
                    self.board.move(p, action, delete=False)

                    # Check if player has reached goal
                    goal_code = "K" + p[1]
                    if not self.board.objects[p].goal_reached and self.board.at(p, goal_code):
                        # Indicate goal reached
                        self.board.objects[p].goal_reached = True
                        self.board.objects[goal_code].reached = True
                        # Destroy goal
                        # self.board.remove(goal_code)
                        # Increment sub-goal counter
                        self.board.objects[self.tower].subgoal_count += 1
                    # Check if player has reached tower
                    if self.board.at(p, self.tower) and \
                            all([self.board.objects[i].goal_reached for i in self.player_code_mapping.values()]):
                        # self.update_phase()
                        return True
            # CHECK IF PLAYER AND MONSTER/TRAP/STONE IN SAME AREA AFTER
            # EACH PASS OF EACH CHARACTER MOVES
            self.check_combat(i)
        # self.update_phase()
        return False

    def execute_enemy_plans(self):
        """
        Executes plans for enemy monsters by iterating over enemy team until no actions remain in their plans.
        :return: N/A
        """
        # print("EXECUTING ENEMY MOVEMENT")
        monsters = [i for i in self.board.objects.values() if isinstance(i, Enemy) and i.name == "Monster"]
        if monsters:
            # for i in range(1.txt, max_moves + 1.txt):
            move_count = 0
            done = False
            while not done:
                done = True
                for m in monsters:
                    # Monster can move on this turn
                    if move_count < m.action_points:
                        done = False
                        self.board.move_monster(m.index, self.directions.copy())
                # Check to see if com at needs to be initiated
                self.check_combat()
                # Some monsters may have been defeated
                monsters = [i for i in self.board.objects.values() if isinstance(i, Enemy) and i.name == "Monster"]
                move_count += 1
        self.update_phase()

    def undo(self, p):
        """
        Implements an undo feature for the given player.
        :param p: The player to apply an undo action to
        :return: N/A
        """
        curr_phase = self.phases[self.phase_num]

        """
            # During pin planning, can only undo if pin is finalized, but player has not submitted plan
            if self.board.objects[p].pin_finalized 
                    and not self.players[p]["pin_plan_finalized"] 
                    and self.players[p]["pin_plan"]:
                last_action = self.players[p]["pin_plan"].pop()
                # Use the reverse of the last action selected to step the pin back to the previous position
                self.update_plan_location(p, self.reverse_actions[last_action], "pin_path")
        """
        if curr_phase == self.pinning_phase_name:
            return
        elif curr_phase == "action_planning":
            # Can only undo during action planning if there is an action in the action plan and user has not submitted
            if self.board.objects[p].action_plan:
                self.board.objects[p].action_plan.pop()
                last_position = self.board.objects[p].action_positions.pop()
                self.board.objects[p].action_plan_x = last_position[1]
                self.board.objects[p].action_plan_y = last_position[0]
                self.board.objects[p].action_points += 1
        else:
            # No-op/invalid action
            return

    ##########
    # COMBAT #
    ##########

    def check_combat(self, step_index=None):
        """
        Checks if players and enemies are co-located which would initiate combat
        :param step_index: Determines the position in the action sequence
        :return: N/A
        """
        # Gets location of all players. If multiple players end up on the same grid square, using a set will ensure
        # spot is only checked once for combat
        player_loc = set([(self.board.objects[p].y, self.board.objects[p].x)
                          for p in self.player_code_mapping.values()])
        # For each location where player is present, check if there are enemies. If so, initiate combat
        for loc in player_loc:
            players = [obj for obj in self.board.board[loc].values() if isinstance(obj, Player)]
            enemies = [obj for obj in self.board.board[loc].values() if isinstance(obj, Enemy)]
            # There are enemies at this position
            if enemies and players:
                self.combat(players, enemies, step_index)

    def combat(self, players, enemies, step_index):
        """
        Executes combat logic. Given the list of players at the x,y position, determine if any enemies are at
        that position and if so, initiate combat
        :param players: A list of players at the x,y position
        :param enemy_regex: A regex representing the enemy name
        :param enemy_type: The enemy type
        :param x: The x position of the grid where combat occurs
        :param y: The y position of the grid where combat occurs
        :return: N/A
        """
        # Enemies are always all the same type
        enemy_type = enemies[0].entity_type
        player_rolls = sum([p.get_dice_roll(enemy_type) for p in players])
        enemy_rolls = sum([e.get_dice_roll() for e in enemies])

        # Players win (players win ties)
        if player_rolls >= enemy_rolls:
            # print("PLAYERS WIN!")
            self.board.multi_remove(enemies)

            # Track player wins
            if self.track_metrics:
                for p in players:
                    sizes = set([e.type.split("_")[0] for e in enemies])
                    for size in sizes:
                        self.tracker.update(target="player", player=p.name, metric_name="combat",
                                            combat_outcome="win", enemy_type=enemy_type, enemy_size=size)
        # Players lose
        else:
            # Track player losses
            if self.track_metrics:
                # Track health lost and combat loss metrics
                self._metrics_combat_loss(players, enemies, enemy_type)
            for p in players:
                if enemy_type == "Monster":
                    # Lose a heart
                    p.health -= 1
                    # Go back a step if player has moved. Only applies to player movement phase, not enemy movement
                    # step_index is used to indicate if in player movement phase (could also check phase)
                    if step_index and p.action_plan:
                        # Truncate action plan
                        p.action_plan = []
                        # Get last position of player
                        prev_pos = p.action_positions[step_index - 1]
                        p.action_positions = []
                        # self.board.place(p.index, x=prev_pos[1], y=prev_pos[0])
                        self.board.move(p.index, x=prev_pos[1], y=prev_pos[0], old_pos=(p.y, p.x))
                elif enemy_type == "Trap":
                    # Lose a heart
                    p.health -= 1
                    if step_index:
                        # Truncate action plan
                        p.action_plan = []
                        p.action_positions = []
                elif enemy_type == "Stone":
                    if step_index:
                        # Truncate action plan
                        p.action_plan = []
                        p.action_positions = []

                # If player dies, remove from board
                if p.health <= 0:
                    self.board.remove(p.index, delete=False)
                    if self.track_metrics:
                        self.tracker.update(target="player", player=p.name, metric_name="death")
                    p.health = 0
                    # Player loses a life
                    p.lives -= 1
                    p.dead = True
                    p.death_round = self.num_rounds
                    if p.lives <= 0:
                        p.can_respawn = False
            # Traps are destroyed
            if enemy_type == "Trap":
                self.board.multi_remove(enemies)

    #############
    # RENDERING #
    #############

    def render(self):
        """
        Renders the game in the console as an ascii grid
        :return: N/A
        """
        self.board.print_board(self.render_verbose,
                               self.curr_level_num,
                               self.phases[self.phase_num])

    ###########
    # METRICS #
    ###########

    def _metrics_combat_loss(self, players, enemies, enemy_type):
        sizes = set([e.type.split("_")[0] for e in enemies])
        for p in players:
            for size in sizes:
                self.tracker.update(target="player", player=p.name, metric_name="combat",
                                    combat_outcome="lose", enemy_type=enemy_type, enemy_size=size)
            if enemy_type == "Monster":
                self.tracker.update(target="player", player=p.name, metric_name="health_loss")
            elif enemy_type == "Trap":
                self.tracker.update(target="player", player=p.name, metric_name="health_loss")
