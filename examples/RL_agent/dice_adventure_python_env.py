from game.dice_adventure import DiceAdventure
import game.unity_socket as unity_socket
from gymnasium import Env
from gymnasium import spaces
from json import loads
import numpy as np
from random import choice
from game.unity_socket import DiceAdventureWebSocket
import json
from math import ceil
import logging
import os


class DiceAdventurePythonEnvRL(Env):
    """
    Implements a custom gyn environment for the Dice Adventure game.
    """
    def __init__(self,
                 player="Dwarf",
                 id_=0,
                 train_mode=False,
                 server="local",
                 state_version="player",
                 **kwargs):
        """
        Init function for Dice Adventure gym environment.
        :param player:      (string) The player that will be used to play the game.
        :param id_:         (int) An optional ID parameter to distinguish this environment from others.
        :param train_mode:  (bool) A helper parameter to switch between training mode and play mode. When we test agents,
                                   we will use a "play" mode, where the step function simply takes an action and returns
                                   the next state.
        :param server:      (string) Determines which game version to use. Can be one of {local, unity}.
        :param kwargs:      (dict) Additional keyword arguments to pass into Dice Adventure game. Only applies when
                                   'server' is 'local'.
        """
        self.config = loads(open("game/config/main_config.json", "r").read())
        # Load observation config
        with open("examples/RL_agent/observation_config.json", "r") as f:
            self.obs_config = json.load(f)
        self.player = player
        self.id = id_
        self.kwargs = kwargs
        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit"]
        self.player_names = ["Dwarf", "Giant", "Human"]
    

        #################
        # GAME SETTINGS #
        #################
        self.state_version = state_version if state_version in ["full", "player", "fow"] else "player"
        self.object_positions = self.config["GYM_ENVIRONMENT"]["OBSERVATION"]["OBJECT_POSITIONS"]
        self.num_game_objects = len(self.object_positions)

        ##################
        # TRAIN SETTINGS #
        ##################
        self.train_mode = train_mode
        self.masks = {"Dwarf": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C1"]["SIGHT_RANGE"],
                      "Giant": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C2"]["SIGHT_RANGE"],
                      "Human": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C3"]["SIGHT_RANGE"]}
        self.max_mask_radius = max(self.masks.values())
        self.local_mask_radius = self.masks[self.player]
        self.action_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'wait',
                           5: 'submit', 6: 'undo'} #6: 'pinga', 7: 'pingb', 8: 'pingc', 9: 'pingd', 10: 'undo'}

        num_actions = len(self.action_map)
        self.action_space = spaces.Discrete(num_actions)
        # set number of actions taken to 0
        self.num_actions = 0
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        # self.mask_size = self.max_mask_radius * 2 + 1
        # vector_len = self.mask_size * self.mask_size * self.num_game_objects
        # self.observation_space = spaces.Box(low=-5, high=100,
        #                                     shape=(vector_len,), dtype=np.float32)

        # Define the full board dimensions and number of channels
        num_channels = self.obs_config["NUM_CHANNELS"]
        height = self.obs_config["TENSOR_SHAPE"][0]
        width = self.obs_config["TENSOR_SHAPE"][1]

        # If flattening the tensor:
        vector_len = num_channels * height * width  # 43 * 20 * 20 = 17200

        # Update observation space
        observation_space = spaces.Box(low=0, high=100, shape=(vector_len,), dtype=np.float32)
        self.observation_space = observation_space


        ###################
        # SERVER SETTINGS #
        ###################
        self.server = server
        self.unity_socket_url = self.config["GYM_ENVIRONMENT"]["UNITY"]["URL"]
        # self.websocket = DiceAdventureWebSocket(url=self.unity_socket_url.format(self.player.lower()))
        self.game = None
        if self.server == "local":
            try:
                self.game = DiceAdventure(**self.kwargs)
            except Exception as e:
                print(f"Error initializing DiceAdventure: {e}")
                raise

        
        
        # Update observation space to be channel-first
        shape = [self.obs_config["NUM_CHANNELS"]] + self.obs_config["TENSOR_SHAPE"]  # CHW format
        self.observation_space = spaces.Box(low=-5, high=100,
                                          shape=shape, dtype=np.float32)

        # Add tracking of reached shrines
        self.reached_shrines = set()  # Persist across steps

    def step(self, action) -> tuple:
        """
        :return: (observation, reward, terminated, truncated, info)
        """
        if self.train_mode:
            return self._step_train(action)
        # Wrap _step_play return in proper tuple format
        state = self._step_play(action)
        obs = self.get_observation(state)
        return obs, 0.0, False, False, {}

    def _get_2d_state(self, state):
        # create a 2d grid of the state with '-' for empty spaces
        grid = np.full((20, 20), '-', dtype=object)
        for obj in state["content"]["scene"]:
            x = obj["x"]
            y = obj["y"]
            grid[y, x] = obj.get("objKey", "")
        return grid

    def _save_state_log(self, grid):
        """ Saves the 2D state representation to a log file specific to each environment. """
        
        # Ensure logs directory exists
        log_dir = "state_logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file for this environment
        log_file = os.path.join(log_dir, f"env_{self.id}_log.txt")

        # Convert grid to string format
        grid_str = "\n".join("".join(str(cell) for cell in row) for row in grid)

        # Write grid to file
        with open(log_file, "a") as f:  # "a" mode appends to the file
            f.write(f"Step {self.num_actions}:\n{grid_str}\n\n")

        logging.debug(f"Env {self.id} - State saved to {log_file}")


    def _step_train(self, action):
        try:
            action = int(action)
            self.num_actions += 1
            logging.debug((f"Env {self.id} - {self.player} takes action: {action} ({self.action_map[action]})"))

            state = self.get_state()
            game_action = self.action_map[action]
            next_state = self.execute_action(self.player, game_action)

            reward = self._get_reward(state, next_state)
            logging.debug((f"Env {self.id} - {self.player} gets reward: {reward}"))
            

            # Simulate other players
            other_players = [p for p in self.player_names if p != self.player]
            for op in other_players:
                op_action = choice(self.actions)
                next_state = self.execute_action(player=op, game_action=op_action)
                logging.debug((f"Env {self.id} - {op} takes action: {op_action}"))
            

            terminated = next_state.get("status") == "GAME_OVER"
            truncated = False
            
            if terminated:
                new_obs, info = self.reset()
            else:
                new_obs = self.get_observation(next_state)
                info = {}

            return new_obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"Error in _step_train: {e}")
            # Return safe default values
            zero_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return zero_obs, 0, True, False, {}

    def get_observation(self, state):
        """
        Creates a 20x20xZ observation tensor where Z is the number of channels defined in observation_config.json.
        Each channel represents different game features like board boundaries, player positions, shrines, etc.
        """
        try:
            if state is None:
                return np.zeros((self.obs_config["NUM_CHANNELS"], 20, 20), dtype=np.float32)
            
            # Initialize observation tensor
            tensor = np.zeros((20, 20, self.obs_config["NUM_CHANNELS"]), dtype=np.float32)
            channels = self.obs_config["CHANNEL_MAPPING"]
            
            # Find player object and sight range
            player_obj = None
            for ele in state.get("content", {}).get("scene", []):
                if ele.get("objKey") == self.get_player_code(self.player):
                    player_obj = ele
                    break
                
            if player_obj is None:
                raise ValueError("Player object not found in scene")
            

            # set the board boundary
            board_width = state["content"]["gameData"]["boardWidth"]
            board_height = state["content"]["gameData"]["boardHeight"]
            # only use the first board_width x board_height of the tensor
            tensor[:board_height, :board_width, channels["BOARD_BOUNDARY"]] = 1

            # Get player position and sight range
            player_x = player_obj.get("x", 0)
            player_y = player_obj.get("y", 0)
            sight_range = self.local_mask_radius
            
            # Calculate visible positions
            visible_positions = set()

            for y in range(board_height):
                for x in range(board_width):
                    if max(abs(x - player_x), abs(y - player_y)) <= sight_range:
                        visible_positions.add((y, x))
            
            # Process each object in the scene
            for obj in state.get("content", {}).get("scene", []):
                x = obj.get("x", 0)
                y = obj.get("y", 0)
                entity_type = obj.get("entityType", "").lower()
                
                # Skip if position is not visible
                if (y, x) not in visible_positions:
                    continue
                
                obj_key = obj.get("objKey", "")
                

                if entity_type == "wall":  # Wall
                    tensor[y, x, channels["WALLS"]] = 1
                elif entity_type == "goal":  # Tower/Goal
                    tensor[y, x, channels["TOWER"]] = 1
                elif entity_type == "monster" or entity_type == "trap" or entity_type == "stone":
                    # Combat objects
                    combat_dice = obj.get("combatDice", "D0+0")
                    dice_value = float(combat_dice.split("+")[0][1:])
                    modifier = float(combat_dice.split("+")[1]) if "+" in combat_dice else 0
                    expected_value = (dice_value + 1)/2 + modifier if dice_value > 0 else modifier
                    tensor[y, x, channels[f"{obj_key}_DICE"]] += expected_value

                elif entity_type == "Shrine":
                    shrine_type = obj.get("character", "")
                    channel = f"{shrine_type.upper()}_SHRINE"
                    tensor[y, x, channels[channel]] = 2 if obj.get("reached") else 1
                
                elif obj_key.startswith("C"):  # Player
                    # add the expected skill dice to the corresponding skill channel
                    monster_skill = obj.get("monsterDice", "D0+0")
                    trap_skill = obj.get("trapDice", "D0+0")
                    rock_skill = obj.get("stoneDice", "D0+0")

                    monster_dice = float(monster_skill.split("+")[0][1:])
                    trap_dice = float(trap_skill.split("+")[0][1:])
                    rock_dice = float(rock_skill.split("+")[0][1:])
                    # note: no modifier for the skill dice

                    player_map = {"C1": "DWARF", "C2": "GIANT", "C3": "HUMAN"}
                    player_type = player_map[obj_key]
                    # add the expected skill dice to the corresponding skill channel
                    tensor[y, x, channels[f"{player_type}_MONSTER_SKILL"]] = (monster_dice + 1)/2
                    tensor[y, x, channels[f"{player_type}_TRAP_SKILL"]] = (trap_dice + 1)/2
                    tensor[y, x, channels[f"{player_type}_ROCK_SKILL"]] = (rock_dice + 1)/2

            # compute relative skill advantage for all locations in sight range AND where there are both characters and combat objects
            # for (y, x) in visible_positions:
            #     if tensor[y, x, channels["WALLS"]] == 1:
            #         continue
            #     trap_skill_sum = tensor[y, x, channels["DWARF_TRAP_SKILL"]] + tensor[y, x, channels["GIANT_TRAP_SKILL"]] + tensor[y, x, channels["HUMAN_TRAP_SKILL"]]
            #     rock_skill_sum = tensor[y, x, channels["DWARF_ROCK_SKILL"]] + tensor[y, x, channels["GIANT_ROCK_SKILL"]] + tensor[y, x, channels["HUMAN_ROCK_SKILL"]]
            #     monster_skill_sum = tensor[y, x, channels["DWARF_MONSTER_SKILL"]] + tensor[y, x, channels["GIANT_MONSTER_SKILL"]] + tensor[y, x, channels["HUMAN_MONSTER_SKILL"]]

            #     trap_presence = tensor[y, x, channels["T1_DICE"]]
            #     rock_presence = tensor[y, x, channels["R1_DICE"]]
            #     monster_presence = tensor[y, x, channels["M1_DICE"]] + tensor[y, x, channels["M2_DICE"]] + tensor[y, x, channels["M3_DICE"]]

            #     if trap_presence > 0 and trap_skill_sum > 0:
            #         tensor[y, x, channels["TRAP_COMBAT_ADVANTAGE"]] = trap_skill_sum - trap_presence
            #     if rock_presence > 0 and rock_skill_sum > 0:
            #         tensor[y, x, channels["ROCK_COMBAT_ADVANTAGE"]] = rock_skill_sum - rock_presence
            #     if monster_presence > 0 and monster_skill_sum > 0:
            #         tensor[y, x, channels["MONSTER_COMBAT_ADVANTAGE"]] = monster_skill_sum - monster_presence
                    

            # ---- Player identity channels ----
            # Set player identity channels
            player_channel = f"{self.player.upper()}_CHANNEL"
            tensor[:, :, channels[player_channel]] = 1
            
            # Add action plan and pins
            curr_phase = state["content"]["gameData"]["currentPhase"]

            # Set phase channels
            tensor[:, :, channels["PINNING_PHASE"]] = 1 if curr_phase == "Player_Pinning" else 0
            tensor[:, :, channels["ACTION_PLANNING_PHASE"]] = 0 if curr_phase == "Player_Pinning" else 1


        
            if curr_phase == "Player_Pinning":
                # Track pin cursor position
                cursor_x = player_obj.get("pinCursorX", player_x)
                cursor_y = player_obj.get("pinCursorY", player_y)
                tensor[cursor_y, cursor_x, channels["PIN_CURSOR"]] = 1
                
                # Track planned pin positions
                if hasattr(player_obj, "actionPlan"):
                    action_plan = player_obj.get("actionPlan", [])

                    # action_plan is a list of strings
                    num_actions = min(len(action_plan), 6)
                    
                    for i in range(num_actions):
                        action = action_plan[i]
                        # if action is a pin, it will probably be a pinga, pingb, pingc, pingd
                        if isinstance(action, str) and action.startswith('ping'):
                            # figure out which pin it is
                            pin_type = action[4].upper()
                            pin_value = self.obs_config["PIN_VALUES"][f"PING{pin_type}"]
                            tensor[cursor_y, cursor_x, channels[f'{self.player.upper()}_PINS']] = pin_value

            elif curr_phase == "Player_Planning":  # action_planning phase
                # print(f"Action planning phase: {player_obj.get('actionPlan', [])}")
                if hasattr(player_obj, "actionPlan"):
                    action_plan = player_obj.get("actionPlan", [])
                    num_actions = min(len(action_plan), 6)
                    
                    # Start from current position
                    current_x, current_y = player_x, player_y
                    
                    # Map each action to next position
                    for i in range(num_actions):
                        action = int(action_plan[i])  # Convert to int since actions are stored as integers
                        print(f"Action {i+1}: {action}")
                        action_channel = f"ACTION_{i+1}"
                        
                        # Mark current position in action channel
                        if action == "wait":  # wait
                            tensor[current_y, current_x, channels[action_channel]] = 1
                        elif action == "submit":  # submit
                            # how does submit work? do we submit after the action plan?
                            tensor[current_y, current_x, channels[action_channel]] = 1
                        elif i > 1 and action == "undo":  # undo (action 10)
                            if i > 2:  # If we have at least 2 previous actions
                                # Get the position from 2 actions ago, since i + 1 is the current action, we need to go back to i - 1
                                prev_action_channel = f"ACTION_{i-1}"
                                prev_position = np.where(tensor[:, :, channels[prev_action_channel]] == 1)
                                if len(prev_position[0]) > 0:  # Make sure we found a position
                                    current_x = int(prev_position[1][0])
                                    current_y = int(prev_position[0][0])
                            else:
                                # If it's only the second action, go back to start position
                                current_x = player_x
                                current_y = player_y
                            
                            # Mark current position in action channel
                            tensor[current_y, current_x, channels[action_channel]] = 1
                        else:  # movement actions
                            if action == "left":  # left
                                current_x = max(0, current_x - 1)
                            elif action == "right":  # right
                                current_x = min(board_width - 1, current_x + 1)
                            elif action == "up":  # up
                                current_y = min(board_height - 1, current_y + 1)
                            elif action == "down":  # down
                                current_y = max(0, current_y - 1)
                            
                            tensor[current_y, current_x, channels[action_channel]] = 1

            # Handle health and lives separately
            if not player_obj.get("dead", False):
                health = player_obj.get("health", 0)
        
                # Normalize health to [0,1]
                tensor[:, :, channels["HEALTH"]] = health
                # Normalize lives to [0,1] (assuming max lives is 3)
                tensor[:, :, channels["LIVES"]] = ceil(health / 3)
                
                tensor[:, :, channels["ACTION_POINTS"]] = player_obj.get("actionPoints", 0)
                

            # Reshape tensor to channel-first format CHW before returning
            tensor = np.transpose(tensor, (2, 0, 1))  
            return tensor

        except Exception as e:
            print(f"Error creating observation: {e}")
            # Make sure error case also returns channel-first format
            return np.zeros((self.obs_config["NUM_CHANNELS"], 20, 20), dtype=np.float32)


    def _step_play(self, action) -> dict:
        """Returns state dictionary"""
        state = self.execute_action(self.player, action)
        if state is None:
            return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}
        return state

    def close(self):
        """
        close() function from standard gym environment. Not implemented.
        :return: N/A
        """
        pass

    def render(self, mode='console'):
        if self.server == "local" and self.game is not None:
            self.game.render()

    def reset(self, seed=None):
        self.num_actions = 0
        # Clear reached shrines on reset
        self.reached_shrines = set()
        if seed:
            np.random.seed(seed)
        if self.server == "local":
            try:
                self.game = DiceAdventure(**self.kwargs)
            except Exception as e:
                print(f"Error resetting game: {e}")
                raise
        obs = self.get_state()
        return self.get_observation(obs), {}

    def execute_action(self, player, game_action):
        if self.server == "local":
            if self.game is None:
                raise RuntimeError("Game not initialized")
            self.game.execute_action(player, game_action)
            return self.get_state()
        else:
            url = self.unity_socket_url.format(player.lower())
            try:
                # Type checking for unity socket methods
                if not hasattr(unity_socket, 'execute_action') or not hasattr(unity_socket, 'get_state'):
                    raise AttributeError("Unity socket missing required methods")
                unity_socket.execute_action(url, game_action)  # type: ignore
                return unity_socket.get_state(url, self.state_version)  # type: ignore
            except Exception as e:
                print(f"Unity socket error: {e}")
                return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}

    def get_state(self, player=None, version=None, server=None):
        version = version if version is not None else self.state_version
        player = player if player is not None else self.player
        server = server if server else self.server

        try:
            if server == "local":
                if self.game is None:
                    raise RuntimeError("Game not initialized")
                return self.game.get_state(player, version)
            else:
                url = self.unity_socket_url.format(player.lower())
                return unity_socket.get_state(url, version)
        except Exception as e:
            print(f"Error getting state: {e}")
            # Return a minimal valid state
            return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}

    def get_actions(self):
        return self.actions

    def get_player_names(self):
        return self.player_names

    @staticmethod
    def get_player_code(player):
        codes = {"Dwarf": "C1", "Giant": "C2", "Human": "C3"}
        return codes[player]

    def _get_reward(self, state, next_state):
        """
        Calculates reward based on:
        1. Reaching shrines (+1)
        2. Reaching tower (+10)
        3. Health loss penalties (proportional to max health)
        4. Team-based penalties (if any character loses all health)
        5. Round-based decay (0.95^rounds)
        
        Returns: float - The calculated reward
        """
        try:
            reward = 0
            
            # Check level completion
            if state["content"]["gameData"]["currLevel"] != next_state["content"]["gameData"]["currLevel"]:
                print(f"[Env {self.id}-{self.player}] Level completed!")
                reward += 10
                return reward

            # Check for newly reached shrines
            for obj in next_state.get("content", {}).get("scene", []):
                if (obj.get("objKey", "").startswith("K") and 
                    obj.get("reached") and 
                    obj.get("objKey") not in self.reached_shrines):
                    
                    self.reached_shrines.add(obj.get("objKey"))
                    reward += 1
                    print(f"[Env {self.id}-{self.player}] New shrine reached: {obj.get('objKey')}, "
                          f"Total shrines: {len(self.reached_shrines)}")

            return reward

        except Exception as e:
            print(f"[Env {self.id}-{self.player}] Error calculating reward: {e}")
            return 0

    def _get_characters_from_state(self, state):
        chars = []
        for player in self.player_names:
            player_code = self.get_player_code(player)
            for ele in state["content"]["scene"]:
                if ele.get("objKey") == player_code:
                    chars.append(ele)
        return chars


