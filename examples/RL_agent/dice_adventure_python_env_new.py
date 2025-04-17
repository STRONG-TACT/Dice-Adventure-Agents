from gymnasium import Env
from gymnasium import spaces
from json import loads
import numpy as np
from random import choices
# from game.unity_socket import DiceAdventureWebSocket
from game.unity_socket_new import UnityWebSocket
import json
from math import ceil
import logging
import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import subprocess
import copy
from time import sleep
from threading import Thread
from typing import Any, Tuple, Union, Dict
from examples.random_agent.agent import DiceAdventureAgent
import psutil
import signal




class DiceAdventurePythonEnvRL(Env):
    """
    Implements a custom gyn environment for the Dice Adventure game.
    """
    def __init__(self,
                 port: str,
                 game_executable_filepath: str,
                 player: str = 'dwarf',
                 id_: int = 0,
                 train_mode=False
                #  server="local",
                #  state_version="player",
                #  **kwargs
                 ):
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
        self.id = id_
        self.train_mode = train_mode
        self.config = loads(open("game/config/main_config.json", "r").read())
        # # Load observation config
        with open("examples/RL_agent/observation_config_new.json", "r") as f:
            self.obs_config = json.load(f)
        self.player = player
        # self.id = id_
        # self.kwargs = kwargs
        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]
        # self.player_names = ["Dwarf", "Giant", "Human"]
        self.player_names = ["dwarf", "giant", "human"]
        self.sockets = {'dwarf': None, 'giant': None, 'human': None}
        self.port = port
        self.socket_url = "ws://localhost:{}/hmt/{}"
        self.game_executable_filepath = game_executable_filepath


        #################
        # GAME SETTINGS #
        #################
        # self.state_version = state_version if state_version in ["full", "player", "fow"] else "player"
        # self.object_positions = self.config["GYM_ENVIRONMENT"]["OBSERVATION"]["OBJECT_POSITIONS"]
        # self.num_game_objects = len(self.object_positions)

        ##################
        # TRAIN SETTINGS #
        ##################
        # self.train_mode = train_mode
        self.masks = {"dwarf": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C1"]["SIGHT_RANGE"],
                      "giant": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C2"]["SIGHT_RANGE"],
                      "human": self.config["OBJECT_INFO"]["OBJECT_CODES"]["C3"]["SIGHT_RANGE"]}
        self.max_mask_radius = max(self.masks.values())
        self.local_mask_radius = self.masks[self.player]
        self.action_map = {
            0: 'up',
            1: 'down', 
            2: 'left',
            3: 'right',
            4: 'wait',
            5: 'undo',
            6: 'submit',
            7: 'pinga',
            8: 'pingb',
            9: 'pingc',
            10: 'pingd'
        }

        # Define action space size
        self.action_space = spaces.Discrete(len(self.action_map))
        # set number of actions taken to 0
        self.num_actions = 0



        # Update observation space to be channel-first
        shape = [self.obs_config["NUM_CHANNELS"]] + self.obs_config["TENSOR_SHAPE"]  # CHW format
        self.observation_space = spaces.Box(low=0, high=30,
                                          shape=shape, dtype=np.float32)

        # Add tracking of reached shrines
        self.reached_shrines = set()  # Persist across steps

        # Add tracking of pins dropped in current phase
        self.pins_dropped_this_phase = 0

        # Add game process tracking
        self.game_process = None
        self.game_thread = None
        
        # Launch the game
        # if not self.launch_game(game_executable_filepath, port):
        #     raise RuntimeError("Failed to launch the game. Please check if the game executable exists and is accessible.")
        # self.launch_game(game_executable_filepath, port)

        # _launch_game(game_executable_filepath, port)



    
    def action_masks(self):
        """Returns a mask of valid actions for the current state"""
        try:
            state = self.get_state(self.player)
            if state is None:
                print("Warning: state is None, returning all-true mask")
                return np.ones(self.action_space.n, dtype=np.bool_)

            current_phase = state[0]["currentPhase"]
            mask = np.zeros(self.action_space.n, dtype=np.bool_)
            
            if current_phase == "Player_Pinning":
                # Allow ping actions and submit during pinning phase
                mask[6:] = True  # submit, undo, pinga, pingb, pingc, pingd
            else:  # Planning phase
                # Allow movement actions, wait, undo, and submit during planning phase
                mask[0:7] = True   # left, right, up, down, wait, submit, undo
            
            # Add debug print
            # print(f"Phase: {current_phase}, Mask shape: {mask.shape}, Mask: {mask}")
            return mask
        except Exception as e:
            print(f"Error creating action mask: {e}")
            # Return all actions as valid in case of error
            return np.ones(self.action_space.n, dtype=np.bool_)


    def step(self, action: str) -> tuple:
        """Execute one step in the environment"""
        try:
            if self.train_mode:
                print("Training mode")
                return self._step_train(action)
                
            logging.info(f"Executing action: {action}")
            state = self.get_state(self.player)
            if state is None:
                logging.error("Current state is None")
                return self.reset()[0], 0, True, False, {}
            
            next_state = self.execute_action(self.player, action)
            if next_state is None:
                logging.error("Next state is None after action")
                return self.reset()[0], 0, True, False, {}

            reward = self._get_reward(state, next_state)
            logging.info(f"Reward: {reward}")

            # Check termination conditions
            terminated = next_state[0]["currLevel"] != state[0]["currLevel"]
            game_over = next_state[0]["status"] == "GAME_OVER"
            truncated = False

            if terminated or game_over:
                logging.info("Episode terminated or game over, resetting...")
                new_obs, info = self.reset()
            else:
                new_obs = self.get_observation(next_state)
                info = {}

            return new_obs, reward, terminated, truncated, info

        except Exception as e:
            logging.error(f"Error in step: {str(e)}", exc_info=True)
            return self.reset()[0], 0, True, False, {"error": str(e)}

    # def step(self, action) -> tuple:
    #     """
    #     :return: (observation, reward, terminated, truncated, info)
    #     """
    #     if self.train_mode:
    #         return self._step_train(action)
    #     # Wrap _step_play return in proper tuple format
    #     state = self._step_play(action)
    #     obs = self.get_observation(state)
    #     return obs, 0.0, False, False, {}

    

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
            # print("Training mode")
            action = int(action)
            self.num_actions += 1
            logging.debug((f"Env {self.id} - {self.player} takes action: {action} ({self.action_map[action]})"))

            state = self.get_state(self.player)
            game_action = self.action_map[action]
            next_state = self.execute_action(self.player, game_action)

            reward = self._get_reward(state, next_state)
            logging.debug((f"Env {self.id} - {self.player} gets reward: {reward}"))
            

            # Simulate other players
            other_players = [p for p in self.player_names if p != self.player]
            for op in other_players:
                action_mask = list(self.action_masks().astype(np.float32))
                op_action = choices(self.actions, weights=action_mask, k=1)[0]
                next_state = self.execute_action(player=op, game_action=op_action)
                logging.debug((f"Env {self.id} - {op} takes action: {op_action}"))
            

            terminated = next_state[0]["currLevel"] != state[0]["currLevel"]
    
            game_over = next_state[0]["status"] == "GAME_OVER"
            truncated = False
            
            if terminated or game_over:
                new_obs, info = self.reset()
            else:
                new_obs = self.get_observation(next_state)
                info = {}

            return new_obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"Error in _step_train: {e}")
            # Return safe default values
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zero_obs, 0, True, False, {}

    def get_observation(self, state: list[dict]) -> np.ndarray:
        """Returns the observation tensor for the current state."""
        try:
            tensor = np.zeros((self.obs_config["NUM_CHANNELS"], 
                             self.obs_config["TENSOR_SHAPE"][0], 
                             self.obs_config["TENSOR_SHAPE"][1]), dtype=np.float32)
            channels = self.obs_config["CHANNEL_MAPPING"]

            # get the state
            state = self.get_state(self.player)
            if state is None:
                return tensor

            # Validate board dimensions
            board_width = state[0]["boardWidth"]
            board_height = state[0]["boardHeight"]

            # Set board boundary
            tensor[channels["BOARD_BOUNDARY"], :board_height, :board_width] = 1

            # Find player object
            player_obj = _find_player_obj(state, self.player)
            if player_obj is None:
                return tensor
            
            # 2. set player character on tensor
            player_type = get_player_from_id(player_obj["id"])
            # fill the character channel with 1s
            tensor[channels[f"PLAYER_IS_{player_type.upper()}"], :, :] = 1

            # 3. Set Health, Lives, and Action Points
            tensor[channels["HEALTH"], :, :] = player_obj["health"]
            tensor[channels["LIVES"], :, :] = ceil(player_obj["health"] / 3)
            tensor[channels["ACTION_POINTS"], :, :] = player_obj["actionPoints"]

            current_phase = state[0]["currentPhase"]
            if current_phase == "Player_Planning":
                # 6. Set the action planning phase
                tensor[channels["ACTION_PLANNING_PHASE"], :, :] = 1
                # 7. Iterate through the actions and set the positions of the player at each step (at most 6 actions)
                num_actions = min(len(player_obj["actionPlan"]), 6)
                # initialize action history for undo
                loc_history = []
                curr_y, curr_x = player_obj["y"], player_obj["x"]
                for i in range(num_actions):
                    action = player_obj["actionPlan"][i]
                    
            
                    if action == "up":
                        curr_y = max(0, curr_y - 1)
                    elif action == "down":
                        curr_y = min(curr_y + 1, board_height - 1)
                    elif action == "left":
                        curr_x = max(0, curr_x - 1)
                    elif action == "right":
                        curr_x = min(curr_x + 1, board_width - 1)
                    elif action == "wait":
                       continue
                    elif action == "undo":
                        if len(loc_history) > 0:
                            loc_history.pop()
                            curr_y, curr_x = loc_history[-1]
                    
                    loc_history.append((curr_y, curr_x))
                    tensor[channels[f"ACTION_{i+1}"], curr_y, curr_x] = 1

            # 8. iterate through the objects in the state and separate them into categories
            enemies = ["monster", "trap", "rock"]
            for obj in state[1:]:
                entity_type = obj.get("entityType", "").lower()
                obj_key = obj.get("objKey", "").lower()
                # TODO: sight status does not necessarily exist so check this
                sight_status = obj.get("sight_status", "").lower()
                if sight_status == "":
                    print(f"Sight status is empty for entity_type: {entity_type}, obj_key: {obj_key}")
                    print(f"obj: {obj}")
                    print(f"player_obj: {player_obj}")


                if "x" not in obj or "y" not in obj:
                    # FOW, cannot see this object
                    continue
               
                # set visibility of cell (unexplored, hidden, visible)
                tensor[channels[f"{sight_status.upper()}"], obj["y"], obj["x"]] = 1

                
                

                if entity_type in enemies:
                    if sight_status != "visible":
                        continue
                  
                    tensor[channels[f"{entity_type.upper()}_PRESCENCE"], obj["y"], obj["x"]] = 1
                    
                    # mark strength of enemy = expectation of dice roll + modifier
                    dice_faces, modifier = obj.get("challengeDie", "").split("+")
                    expected_strength = get_expected_strength(int(dice_faces[-1]), int(modifier))
                    # add to the total strength of the enemies
                    tensor[channels[f"{entity_type.upper()}_STRENGTH"], obj["y"], obj["x"]] += expected_strength

                elif entity_type == "character":
                    character_type = get_player_from_id(obj["id"])
                    tensor[channels[f"{character_type.upper()}_POSITION"], obj["y"], obj["x"]] = 1
                    # set player's skill for defeating monsters, traps, and rocks
                    monster_dice_faces, monster_modifier = self.obs_config["DICE_VALUES"][f"{character_type.upper()}_MONSTER"].split("+")
                    trap_dice_faces, trap_modifier = self.obs_config["DICE_VALUES"][f"{character_type.upper()}_TRAP"].split("+")
                    rock_dice_faces, rock_modifier = self.obs_config["DICE_VALUES"][f"{character_type.upper()}_ROCK"].split("+")
                    monster_skill = get_expected_strength(int(monster_dice_faces[-1]), int(monster_modifier))
                    trap_skill = get_expected_strength(int(trap_dice_faces[-1]), int(trap_modifier))
                    rock_skill = get_expected_strength(int(rock_dice_faces[-1]), int(rock_modifier))
                    tensor[channels[f"{character_type.upper()}_MONSTER_SKILL"], obj["y"], obj["x"]] = monster_skill
                    tensor[channels[f"{character_type.upper()}_TRAP_SKILL"], obj["y"], obj["x"]] = trap_skill
                    tensor[channels[f"{character_type.upper()}_ROCK_SKILL"], obj["y"], obj["x"]] = rock_skill
        
                
                elif entity_type == "shrine":
                    character_objKey = obj.get("character", "")
                    character = get_player_from_id(character_objKey)
                    if character:
                        tensor[channels[f"{character.upper()}_SHRINE"], obj["y"], obj["x"]] = 1
                elif entity_type == "wall":
                    tensor[channels["WALL"], obj["y"], obj["x"]] = 1
                elif entity_type == "goal":
                    tensor[channels["TOWER"], obj["y"], obj["x"]] = 1

                elif entity_type == 'pin':
                    pin_type = obj.get("objKey", "").upper()
                    placed_by = get_player_from_id(obj.get("placedBy", ""))
                    # placedBy is in {Dwarf, Giant, Human}
                    if placed_by:
                        tensor[channels[f"{placed_by.upper()}_PIN"], obj["y"], obj["x"]] = self.obs_config["PIN_VALUES"][f"{pin_type}"]
                
            
            return tensor

        except Exception as e:
            logging.error(f"Error in get_observation: {e}")
            return np.zeros((self.obs_config["NUM_CHANNELS"], 
                            self.obs_config["TENSOR_SHAPE"][0], 
                            self.obs_config["TENSOR_SHAPE"][1]), dtype=np.float32)


    def _step_play(self, action) -> list[dict]:
        """Returns state dictionary"""
        state = self.execute_action(self.player, action)
        if state is None:
            return [{"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}]
        return state

    def close(self):
        """Make sure to clean up when environment is closed"""
        self.kill_game()
        super().close()

    def render(self):
        pass
        

    def reset(self, seed=None):
        self.num_actions = 0
        self.reached_shrines = set()
        self.pins_dropped_this_phase = 0  # Reset pin counter on reset
        if seed:
            np.random.seed(seed)
        state = self.get_state(self.player)
        game_over = state[0]["status"] == "GAME_OVER"
        if game_over:
            self.restart_game(self.game_executable_filepath, self.port)
        obs = self.get_state(self.player)
        return self.get_observation(obs), {}
    
   

    # def execute_action(self, player, game_action):
    #     if self.server == "local":
    #         if self.game is None:
    #             raise RuntimeError("Game not initialized")
    #         self.game.execute_action(player, game_action)
    #         return self.get_state()
    #     else:
    #         url = self.unity_socket_url.format(player.lower())
    #         try:
    #             # Type checking for unity socket methods
    #             if not hasattr(unity_socket, 'execute_action') or not hasattr(unity_socket, 'get_state'):
    #                 raise AttributeError("Unity socket missing required methods")
    #             unity_socket.execute_action(url, game_action)  # type: ignore
    #             return unity_socket.get_state(url, self.state_version)  # type: ignore
    #         except Exception as e:
    #             print(f"Unity socket error: {e}")
    #             return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}

    def execute_action(self, player: str, game_action: str):
        """
        Executes the given action for the given player.
        :param player:      The player the action is being applied to
        :param game_action: The action to take
        :return:            The resulting state after taking the given action
        """
        # TODO CAPTURE RESPONSE AND RETURN TO USER
        self._get_socket(player).execute_action(game_action)
        return self.get_state(player)

    def _get_socket(self, player):
        # check if game is already running, otherwise sleep
        
        if self.sockets[player] is None:
            sleep(5)
            self.sockets[player] = UnityWebSocket(self.socket_url.format(self.port, player))
        return self.sockets[player]
    


    # def get_state(self, player=None, version=None, server=None):
    #     version = version if version is not None else self.state_version
    #     player = player if player is not None else self.player
    #     server = server if server else self.server

    #     try:
    #         if server == "local":
    #             if self.game is None:
    #                 raise RuntimeError("Game not initialized")
    #             return self.game.get_state(player, version)
    #         else:
    #             url = self.unity_socket_url.format(player.lower())
    #             return unity_socket.get_state(url, version)
    #     except Exception as e:
    #         print(f"Error getting state: {e}")
    #         # Return a minimal valid state
    #         return {"status": "ERROR", "content": {"scene": [], "gameData": {"currLevel": 0}}}

    def get_state(self, player: str):
        """
        Gets the current state of the game.

        The state is always given from the perspective of self.player, which defines how much of the environment (level)
        the agent can currently "see". The following describes how the Fog-Of-War mechanic limits the environment
        view for agents.
s
            In the Unity game, you can see a visibility mask for each character.
            Black positions have not been observed. Gray positions have been observed but are not
            currently in the player's view. This function returns all objects in the current sight range (view) of
            the player plus objects in positions that the player has seen before. Note that any object that can
            move (such as monsters and other players) are only returned when they are in the player's current
            view (i.e. not obscured by black or gray squares), but static objects such as walls, rocks, and traps,
            and shrines are returned if they've been previously observed.
        """
        
        return _simplify_state(self._get_socket(player).get_state(), player)

    def get_actions(self):
        # check which phase we are in 
        return self.actions

    def get_player_names(self):
        return self.player_names

    @staticmethod
    def get_player_code(player):
        codes = {"dwarf": "C1", "giant": "C2", "human": "C3"}
        return codes[player]

    def _get_reward(self, state, next_state):
        """
        Calculates reward based on:
        1. Reaching shrines (+1)
        2. Reaching tower (+10)
        3. Health loss penalties (proportional to max health)
        4. Team-based penalties (if any character loses all health)
        5. Round-based decay (0.95^rounds)
        6. Penalty for dropping multiple pins in same phase (-1)
        
        Returns: float - The calculated reward
        """
        try:
            reward = 0
            
            # Check level completion
            if state[0]["currLevel"] != next_state[0]["currLevel"]:
                print(f"[Env {self.id}-{self.player}] Level completed!")
                reward += 10
                return reward

            # Check for newly reached shrines
            for obj in next_state[1:]:
                if (obj.get("objKey", "").startswith("K") and 
                    obj.get("reached") and 
                    obj.get("objKey") not in self.reached_shrines):
                    
                    self.reached_shrines.add(obj.get("objKey"))
                    reward += 1
                    print(f"[Env {self.id}-{self.player}] New shrine reached: {obj.get('objKey')}, "
                          f"Total shrines: {len(self.reached_shrines)}")

            # Check for pin drops and apply penalty
            current_phase = state[0]["currentPhase"]
            next_phase = next_state[0]["currentPhase"]
            
            # If we're in pinning phase and dropped a pin
            if current_phase == "Player_Pinning":
                # Check if a pin was dropped by looking at the action plan
                player_obj = _find_player_obj(next_state, self.player)
                if player_obj and player_obj.get("actionPlan"):
                    last_action = player_obj["actionPlan"][-1]
                    if last_action.startswith("ping"):
                        self.pins_dropped_this_phase += 1
                        if self.pins_dropped_this_phase > 1:
                            reward -= 1  # Penalty for dropping multiple pins
                            print(f"[Env {self.id}-{self.player}] Penalty for dropping multiple pins in same phase")
            
            # Reset pin counter when phase changes
            if current_phase != next_phase:
                self.pins_dropped_this_phase = 0

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
    def kill_game(self):
        """Kill the current game process and all its children"""
        try:
            if self.game_process:
                # Get parent process
                parent = psutil.Process(self.game_process.pid)
                # Kill children first
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                # Kill parent
                parent.kill()
                print("Game process killed successfully")
        except (psutil.NoSuchProcess, AttributeError) as e:
            print(f"Process already terminated or not found: {e}")
        finally:
            self.game_process = None
            self.game_thread = None

    def restart_game(self, game_executable_filepath, port):
        """Kill current game and start a new one"""
        self.kill_game()
        self.launch_game(game_executable_filepath, port)

    def launch_game(self, game_executable_filepath, port):
        """Launch game with process tracking in headless mode"""
        try:
            # Use the exact command that works in the terminal
            command = f"open -n ./{game_executable_filepath} --args -localMode -stepTime 0 -hmtsocketurl ws://localhost -hmtsocketport {port}"
            
            # Use subprocess.Popen with shell=True to execute the command like in the terminal
            self.game_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print(f"Game launched with PID: {self.game_process.pid}")
            return True
        except Exception as e:
            print(f"Error launching game: {e}")
            return False

    
    # --- new functions ---

def _simplify_state(state: dict, player: str) -> list[dict]:
    """
    Simplifies the game state into a list of dictionary objects, each representing an object in the game.
    :param state: The game state
    :param player: The player the state is related to
    :return: The modified state as a list of dictionaries
    """

    # print(f"state: {state}")
    # breakpoint()
    state['content']['gameData']['id'] = 'gameData'
    state['content']['gameData']['status'] = state['status']
    game_data_obj = state['content']['gameData']
    scene = [game_data_obj] + state['content']['scene']

    return _add_sight_status(scene, player, game_data_obj)


def _launch_game(game_executable_filepath, port):
    game_thread = Thread(target=_launch_game_thread, args=(game_executable_filepath, port))
    game_thread.start()


def _launch_game_thread(game_executable_filepath, port):
    # Use the exact command that works in the terminal
    command = f"open -n ./{game_executable_filepath} --args -localMode -stepTime 0 -hmtsocketurl ws://localhost -hmtsocketport {port}"
    
    # Use subprocess.Popen with shell=True to execute the command like in the terminal
    process = subprocess.call(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for the game to start
    sleep(5)
    
    # Don't wait for the process to complete, as it will keep running
    return process



    
def _add_sight_status(scene: list[dict], player: str, game_data_obj: dict) -> list[dict]:
    """
    Modifies the state by adding whether objects are visible or hidden. It also adds 'Cell' objects with an 'unexplored'
    status to represent grid squares that have not yet been observed.
    :param scene: The list of objects in the state
    :param player: The name of the player
    :param game_data_obj: The state object containing high level information about the current state of the game
    :return: The modified scene list
    """
    player_obj = _find_player_obj(scene, player)
    x_lower = player_obj['x'] - player_obj['sightRange']
    x_upper = player_obj['x'] + player_obj['sightRange']
    y_lower = player_obj['y'] - player_obj['sightRange']
    y_upper = player_obj['y'] + player_obj['sightRange']

    all_cells = {(i, j) for i in range(game_data_obj['boardWidth']) for j in range(game_data_obj['boardHeight'])}
    scene_cells = {(obj.get('x'), obj.get('y')) for obj in scene}
    unexplored = all_cells - scene_cells

    for obj in scene:
        x, y = obj.get('x'), obj.get('y')
        if x is None or y is None:
            obj['sight_status'] = 'unexplored'
        elif x_lower <= x <= x_upper and y_lower <= y <= y_upper:
            obj['sight_status'] = 'visible'
        else:
            obj['sight_status'] = 'hidden'

    for i, pos in enumerate(unexplored):
        scene.append({'id': f'UE{i}', 'entityType': 'Cell',
                    'objKey': 'UE', 'sight_status': 'unexplored',
                    'x': pos[0], 'y': pos[1]})
    return scene

def _find_player_obj(scene: list[dict], player: str) -> [dict, None]:
    """
    Locates the player's object dictionary in the scene list.
    :param scene: The list of objects in the state
    :param player: The name of the player to be returned
    :return: The player dictionary object
    """
    pid = get_player_id(player)
    for obj in scene:
        if obj.get('id') == pid:
            return copy.copy(obj)


def get_player_id(player: str) -> str:
    """
    Gets the player ID.
    :param player: The player whose ID will be returned
    :return: The player's ID
    """
    ids = {"dwarf": "C11", "giant": "C21", "human": "C31"}
    return ids[player]



def play(self, agents: list[tuple[str, DiceAdventureAgent]]) -> None:
    """
    Plays the game with the provided agents.
    :param agents: A tuple containing the name of the player being controlled and the DiceAdventureAgent controlling
                    that player.
    :return:       None
    """
    for player_name, _ in agents:
        print(f"***Registering agent for: {player_name}***")
        # Register with game
        self._get_socket(player_name).register(player_name)
        print(f"***Agent for: ({player_name}) Registered.***")
    # Wait for unity to be 'truly' ready for actions
    sleep(1)
    while True:
        for player_name, agent in agents:
            state = self.get_state(player_name)
            action = agent.take_action(state=state, actions=self.get_actions())
            self.execute_action(player_name, action)
        sleep(.1)

def _get_socket(self, player):
    if self.sockets[player] is None:
        self.sockets[player] = UnityWebSocket(self.socket_url.format(self.port, player))
    return self.sockets[player]

def get_player_from_id(player_id: str) -> str:
    """
    Gets the player from the ID.
    :param player_id: The ID of the player
    :return: The player's name
    """
    ids = {"C11": "dwarf", "C21": "giant", "C31": "human"}
    return ids[player_id]

def get_expected_strength(dice_faces: int, modifier: int) -> float:
    """
    Gets the expected strength of a dice roll.
    :param dice_faces: The number of faces on the dice
    :param modifier: The modifier to the dice roll
    :return: The expected strength of the dice roll
    """
    expected_strength = (dice_faces + 1) / 2 + modifier if dice_faces > 0 else modifier
    return expected_strength



    

        

        


