from json import loads
import subprocess
import copy
import random
from time import sleep
from threading import Thread
from typing import Any, Tuple, Union, Dict, Optional
from gymnasium import Env
from examples.random_agent.agent import DiceAdventureAgent
from game.unity_socket import UnityWebSocket


class DiceAdventureGymEnv(Env):
    """
    Implements a custom gym environment for the Dice Adventure Unity game.
    """
    def __init__(self, port: str, game_executable_filepath: str, player: str = 'dwarf'):
        """
        Init function for Dice Adventure gym environment.
        :param player:      The player that will be used to train the agent. Defaults to the Dwarf.
        :param port:        The custom port that will be used to connect the agent to the game instance.
                            You should choose a port that is currently not in use by your system.
        :param game_executable_filepath:  The location of the game executable
        """
        self.player = player
        self.port = port
        self.socket_url = "ws://localhost:{}/hmt/{}"
        self.sockets = {'dwarf': None, 'giant': None, 'human': None}

        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit"]
        self.player_names = ["dwarf", "giant", "human"]
        
        # Store previous state for reward calculation
        self.prev_state = None
        self.prev_player_pos = None
        self.explored_cells = set()
        self.pinning_completed = {name: False for name in self.player_names}
        self.last_known_positions = {name: None for name in self.player_names}
        self.last_phase = {name: None for name in self.player_names}
        
        # Player IDs (C11, C21, C31) instead of names
        self.player_ids = ['C11', 'C21', 'C31']
        self.last_action = {pid: None for pid in self.player_ids}
        self.last_position = {pid: None for pid in self.player_ids}
        self.last_last_position = {pid: None for pid in self.player_ids}
        self.player_weaknesses = {
            'C11': ['Rock'],
            'C21': ['Trap'],
            'C31': ['Trap', 'Rock']
        }
        self.action_deltas = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        self.unexplored_adjacency_rewarded = set()

        _launch_game(game_executable_filepath, port)

    def step(self, action: str) -> tuple[Any,
                                         float,
                                         Union[bool, Any],
                                         bool,
                                         Union[dict[str, Any]]]:
        """
        Applies the given action to the game. Determines the next observation and reward, whether the training should
        terminate, whether training should be truncated, and additional info.
        :param action:  The action produced by the agent
        :return:        See below

        new_obs (dict) - The resulting game state after applying 'action' to the game
        reward (float) - The reward obtained from applying 'action' to the game. This must be defined by the user. A
                         helper function _get_reward() has been provided for convenience.
        terminated (bool) - Whether the game has terminated after applying 'action' to the game
        truncated (bool) - (See https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
        info (dict) - Additional information that should be passed back to model

        Note: Although this framework is usually used for RL models, users can develop any kind of model with this code.
        """
        prev_state = self.get_state(self.player) if self.prev_state is None else self.prev_state
        next_state = self.execute_action(self.player, action)

        # Get new observation
        terminated = self._is_game_done(next_state)
        if terminated:
            new_obs, info = self.reset()
        else:
            new_obs = self.get_state(self.player)
            new_obs = self._auto_submit_pinning_phase(new_obs, self.player)
            info = {}
        truncated = False

        # Calculate reward based on state transition
        reward = self._get_reward(prev_state, new_obs, action, terminated)
        
        # Update previous state
        self.prev_state = new_obs

        return new_obs, reward, terminated, truncated, info

    def close(self):
        """
        close() function from standard gym environment. Not implemented.
        :return: N/A
        """
        pass

    def render(self, mode='console'):
        """
        Not implemented.
        """
        pass

    def reset(self, **kwargs):
        """
        Reset the environment and clear previous state tracking.
        """
        self.prev_state = None
        self.prev_player_pos = None
        self.explored_cells = set()
        self.pinning_completed = {name: False for name in self.player_names}
        self.last_known_positions = {name: None for name in self.player_names}
        self.last_phase = {name: None for name in self.player_names}
        # Reset movement history
        self.last_action = {pid: None for pid in self.player_ids}
        self.last_position = {pid: None for pid in self.player_ids}
        self.last_last_position = {pid: None for pid in self.player_ids}
        self.unexplored_adjacency_rewarded.clear()
        state = self.get_state(self.player)
        state = self._auto_submit_pinning_phase(state, self.player)
        return state, {}

    def _get_reward(self, prev_state: list[dict], current_state: list[dict], 
                    action: str, terminated: bool) -> float:
        """
        Calculates the reward the agent should receive based on state transition.
        :param prev_state: Previous game state
        :param current_state: Current game state
        :param action: Action taken
        :param terminated: Whether episode terminated
        :return: Reward value
        """
        reward = 0.0
        
        # Get player object
        player_id = get_player_id(self.player)
        prev_player = None
        curr_player = None
        game_data = None
        
        for obj in prev_state:
            if obj.get('id') == player_id:
                prev_player = obj
            if obj.get('id') == 'gameData':
                game_data = obj
        
        for obj in current_state:
            if obj.get('id') == player_id:
                curr_player = obj
            if obj.get('id') == 'gameData':
                game_data = obj
        
        if prev_player is None or curr_player is None:
            return 0.0
        
        # Terminal rewards
        if terminated:
            # Check if won (reached goal)
            for obj in current_state:
                if obj.get('entityType') == 'Goal' and obj.get('id') == player_id:
                    reward += 100.0  # Big reward for reaching goal
                    break
            # Check if died
            if curr_player.get('dead', False):
                reward -= 50.0  # Penalty for death
        else:
            prev_pos = (prev_player.get('x', 0), prev_player.get('y', 0))
            curr_pos = (curr_player.get('x', 0), curr_player.get('y', 0))
            target_pos = self._get_target_position(prev_pos, action)

            position_penalty = self._get_obstacle_penalty(prev_state, target_pos, player_id)
            if position_penalty == 0.0:
                position_penalty = self._get_weakness_penalty(current_state, curr_pos, player_id)

            skip_exploration = position_penalty != 0.0
            reward += position_penalty

            if not skip_exploration:
                newly_explored = curr_pos not in self.explored_cells
                if newly_explored:
                    self.explored_cells.add(curr_pos)
                    reward += 30.0

                for obj in current_state:
                    if obj.get('entityType') == 'Cell' and obj.get('sight_status') == 'unexplored':
                        unexplored_x = obj.get('x', 0)
                        unexplored_y = obj.get('y', 0)
                        dist_to_unexplored = abs(curr_pos[0] - unexplored_x) + abs(curr_pos[1] - unexplored_y)
                        cell_coord = (unexplored_x, unexplored_y)
                        if cell_coord in self.unexplored_adjacency_rewarded:
                            continue
                        if dist_to_unexplored <= 1:
                            reward += 10.0
                            self.unexplored_adjacency_rewarded.add(cell_coord)
                            break
                        elif 2 <= dist_to_unexplored <= 3:
                            reward += 2.0
                            self.unexplored_adjacency_rewarded.add(cell_coord)
                            break

            if action == 'undo':
                reward -= 50.0

        return reward

    def _get_target_position(self, prev_pos: tuple, action: str) -> Optional[tuple]:
        delta = self.action_deltas.get(action)
        if delta is None:
            return None
        return (prev_pos[0] + delta[0], prev_pos[1] + delta[1])

    def _get_obstacle_penalty(self, state: list[dict], target_pos: Optional[tuple], player_id: str) -> float:
        if target_pos is None:
            return 0.0

        weaknesses = self.player_weaknesses.get(player_id, [])
        if not weaknesses:
            return 0.0

        for obj in state:
            if obj.get('entityType') not in weaknesses:
                continue
            obj_pos = (obj.get('x', 0), obj.get('y', 0))
            if obj_pos == target_pos:
                return -500.0

        return 0.0

    def _get_weakness_penalty(self, state: list[dict], curr_pos: tuple, player_id: str) -> float:
        weaknesses = self.player_weaknesses.get(player_id, [])
        if not weaknesses:
            return 0.0

        for obj in state:
            if obj.get('entityType') not in weaknesses:
                continue
            obj_pos = (obj.get('x', 0), obj.get('y', 0))
            if obj_pos == curr_pos:
                return -500.0

        return 0.0

    def _get_proficiency_reward(self, state: list[dict], player_pos: tuple, player_id: str) -> float:
        """
        Calculate reward based on character's proficiency (specialization).
        Based on dice stats:
        - Dwarf: trapDice D8+0 (highest) -> encourage Traps and Shrine 1G
        - Giant: stoneDice D8+0 (highest) -> encourage Rocks and Shrine 2G
        - Human: monsterDice D8+0 (highest) -> encourage Monsters and Shrine 3G
        """
        reward = 0.0
        player_x, player_y = player_pos
        
        # Character-specific mappings
        character_proficiencies = {
            'C11': {  # Dwarf
                'specialty': 'Trap',
                'weakness': 'Rock',  
                'shrine_key': '1G',
                'shrine_character': 'Dwarf'
            },
            'C21': {  # Giant
                'specialty': 'Rock',
                'weakness': 'Trap',  
                'shrine_key': '2G',
                'shrine_character': 'Giant'
            },
            'C31': {  # Human
                'specialty': 'monster',
                'weakness': ['Trap', 'Rock'],  
                'shrine_key': '3G',
                'shrine_character': 'Human'
            }
        }
        
        if player_id not in character_proficiencies:
            return 0.0
        
        prof = character_proficiencies[player_id]
        specialty_type = prof['specialty']
        weakness_type = prof.get('weakness', [])  
        shrine_key = prof['shrine_key']
        shrine_character = prof['shrine_character']
        
        # Convert weakness to list if it's a string
        if isinstance(weakness_type, str):
            weakness_type = [weakness_type]
        
        # Check for specialty objects at current position or nearby
        for obj in state:
            obj_x = obj.get('x')
            obj_y = obj.get('y')
            entity_type = obj.get('entityType', '')
            obj_key = obj.get('objKey', '')
            obj_character = obj.get('character', '')
            
            if obj_x is None or obj_y is None:
                continue
            
            # Calculate distance
            dist = abs(obj_x - player_x) + abs(obj_y - player_y)
            
            # Penalty for being at weakness object location (character-specific)
            # Very large penalty only when actually on the weakness object (0 distance)
            # No penalty for being adjacent/close, as characters may need to pass by to navigate
            if dist == 0 and entity_type in weakness_type:
                if player_id == 'C11':  # Dwarf - weak at Rock
                    reward -= 500.0  # Very large penalty for being at Rock
                elif player_id == 'C21':  # Giant - weak at Trap
                    reward -= 50.0  # Very large penalty for being at Trap
                elif player_id == 'C31':  # Human - weak at Trap and Rock
                    reward -= 50.0  # Very large penalty for being at Trap or Rock
            
            # Reward for being at specialty object location (character-specific)
            if dist == 0 and entity_type == specialty_type:
                if player_id == 'C11':  # Dwarf - Trap
                    reward += 200.0  # Very high reward for being at Trap
                elif player_id == 'C21':  # Giant - Rock
                    reward += 20.0  # Very high reward for being at Rock
                elif player_id == 'C31':  # Human - Monster
                    reward += 20.0  # Very high reward for being at Monster
                else:
                    reward += 1.5  # Big reward for being at specialty location
            # Reward for being adjacent to specialty object (character-specific)
            # elif dist == 1 and entity_type == specialty_type:
            #     if player_id == 'C11':  # Dwarf - Trap
            #         reward += 5.0  # High reward for being near Trap
            #     elif player_id == 'C21':  # Giant - Rock
            #         reward += 5.0  # High reward for being near Rock
            #     elif player_id == 'C31':  # Human - Monster
            #         reward += 5.0  # High reward for being near Monster
            #     else:
            #         reward += 0.8  # Medium reward for being near specialty object
            # Reward for being within 2 tiles of specialty object (character-specific)
            # elif dist == 2 and entity_type == specialty_type:
            #     if player_id == 'C11':  # Dwarf - Trap
            #         reward += 2.0  # Medium reward for approaching Trap
            #     elif player_id == 'C21':  # Giant - Rock
            #         reward += 2.0  # Medium reward for approaching Rock
            #     elif player_id == 'C31':  # Human - Monster
            #         reward += 2.0  # Medium reward for approaching Monster
            #     else:
            #         reward += 0.3  # Small reward for approaching specialty object
            # Reward for being within 3-5 tiles of specialty object (character-specific)
            # elif entity_type == specialty_type:
            #     if 3 <= dist <= 5:
            #         if player_id == 'C11':  # Dwarf - Trap
            #             reward += 0.5  # Small reward for being in range of Trap
            #         elif player_id == 'C21':  # Giant - Rock
            #             reward += 0.5  # Small reward for being in range of Rock
            #         elif player_id == 'C31':  # Human - Monster
            #             reward += 0.5  # Small reward for being in range of Monster
            
            # Reward for Shrine (gate) of matching character
            if entity_type == 'Shrine':
                # Check if it's the character's shrine by objKey or character attribute
                is_matching_shrine = (
                    obj_key == shrine_key or 
                    obj_character == shrine_character or
                    obj_character == player_id
                )
                
                if is_matching_shrine:
                    if dist == 0:
                        # Character-specific: Very high reward for reaching own Shrine
                        if player_id == 'C11':  # Dwarf - Shrine 1G
                            reward += 30.0  # Very high reward for reaching Shrine 1G
                        elif player_id == 'C21':  # Giant - Shrine 2G
                            reward += 30.0  # Very high reward for reaching Shrine 2G
                        elif player_id == 'C31':  # Human - Shrine 3G
                            reward += 30.0  # Very high reward for reaching Shrine 3G
                        else:
                            reward += 2.0  # Big reward for reaching own shrine
                    elif dist == 1:
                        # Character-specific: High reward for being near own Shrine
                        if player_id == 'C11':  # Dwarf - Shrine 1G
                            reward += 20.0  # High reward for being near Shrine 1G
                        elif player_id == 'C21':  # Giant - Shrine 2G
                            reward += 20.0  # High reward for being near Shrine 2G
                        elif player_id == 'C31':  # Human - Shrine 3G
                            reward += 20.0  # High reward for being near Shrine 3G
                        else:
                            reward += 1.0  # Medium reward for being near own shrine
                    elif dist == 2:
                        # Character-specific: Medium reward for approaching own Shrine
                        if player_id == 'C11':  # Dwarf - Shrine 1G
                            reward += 10.0  # Medium reward for approaching Shrine 1G
                        elif player_id == 'C21':  # Giant - Shrine 2G
                            reward += 10.0  # Medium reward for approaching Shrine 2G
                        elif player_id == 'C31':  # Human - Shrine 3G
                            reward += 10.0  # Medium reward for approaching Shrine 3G
                        else:
                            reward += 0.5  # Small reward for approaching own shrine
                    # Character-specific: Reward for being within 3-5 tiles of own Shrine
                    elif 3 <= dist <= 5:
                        if player_id == 'C11':  # Dwarf - Shrine 1G
                            reward += 2.0  # Small reward for being in range of Shrine 1G
                        elif player_id == 'C21':  # Giant - Shrine 2G
                            reward += 2.0  # Small reward for being in range of Shrine 2G
                        elif player_id == 'C31':  # Human - Shrine 3G
                            reward += 2.0  # Small reward for being in range of Shrine 3G
        
        return reward
    
    def _get_back_and_forth_penalty(self, action: str, prev_pos: tuple, curr_pos: tuple, player_id: str) -> float:
        """
        Penalize back-and-forth movement that wastes action points.
        Detects patterns like: right -> left (back to start) or up -> down (back to start)
        """
        penalty = 0.0
        
        # Only check for movement actions
        movement_actions = ['up', 'down', 'left', 'right']
        if action not in movement_actions:
            return 0.0
        
        # Check if we're back to the position from 2 steps ago (back-and-forth)
        last_last_pos = self.last_last_position.get(player_id)
        last_action = self.last_action.get(player_id)
        
        if last_last_pos is not None and last_action in movement_actions:
            # Check if current position equals position from 2 steps ago
            if curr_pos == last_last_pos:
                # Check if movements are opposite
                opposite_pairs = [
                    ('up', 'down'), ('down', 'up'),
                    ('left', 'right'), ('right', 'left')
                ]
                
                if (last_action, action) in opposite_pairs:
                    penalty = -5.5  # Significant penalty for wasting 2 action points
                    # Optional: print warning for debugging
                    # print(f"[Movement] {player_id} wasted AP: {last_action} -> {action} (back to start)")
        
        # Also check for immediate back-and-forth (if position didn't change)
        if prev_pos == curr_pos and action in movement_actions:
            # Movement action but didn't move (maybe blocked or invalid)
            penalty -= 5.5  # Smaller penalty for ineffective movement
        
        return penalty

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

    def get_state(self, player: str):
        """
        Gets the current state of the game.

        The state is always given from the perspective of self.player, which defines how much of the environment (level)
        the agent can currently "see". The following describes how the Fog-Of-War mechanic limits the environment
        view for agents.

            In the Unity game, you can see a visibility mask for each character.
            Black positions have not been observed. Gray positions have been observed but are not
            currently in the player's view. This function returns all objects in the current sight range (view) of
            the player plus objects in positions that the player has seen before. Note that any object that can
            move (such as monsters and other players) are only returned when they are in the player's current
            view (i.e. not obscured by black or gray squares), but static objects such as walls, stones, and traps,
            and shrines are returned if they've been previously observed.
        """
        simplified = _simplify_state(self._get_socket(player).get_state(),
                                     player,
                                     self.last_known_positions.get(player))
        self._update_last_known_position(simplified, player)
        return simplified

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
                state = self._auto_submit_pinning_phase(state, player_name)
                action = agent.take_action(state=state, actions=self.get_actions())
                self.execute_action(player_name, action)
            sleep(.01)  # Reduced sleep time for faster training

    def _get_socket(self, player):
        if self.sockets[player] is None:
            self.sockets[player] = UnityWebSocket(self.socket_url.format(self.port, player))
        return self.sockets[player]

    def _register(self, agent_id):
        # url = self.unity_socket_url.format(self.port, self.player.lower())
        # unity_socket.register(url, agent_id)
        self.websocket.register(agent_id)

    def get_actions(self):
        return self.actions

    def get_player_names(self):
        return self.player_names

    def _update_last_known_position(self, state: list[dict], player: str):
        player_obj = _find_player_obj(state, player)
        if not player_obj:
            return
        x = player_obj.get('x')
        y = player_obj.get('y')
        sight = player_obj.get('sightRange', 3)
        if x is not None and y is not None:
            self.last_known_positions[player] = (x, y, sight)

    @staticmethod
    def _is_game_done(state: list[dict]) -> bool:
        for obj in state:
            status = obj.get("status")
            if isinstance(status, str) and status.lower() == "done":
                return True
        return False

    @staticmethod
    def _get_current_phase_from_state(state: list[dict]) -> str:
        for obj in state:
            if obj.get('id') == 'gameData':
                return obj.get('currentPhase', '')
        return ''

    def _auto_submit_pinning_phase(self, state: list[dict], player: str) -> list[dict]:
        """
        Automatically submits the pinning phase if the game is waiting for a submit action.
        This prevents the Unity client from getting stuck after manual pinning.
        """
        phase = self._get_current_phase_from_state(state)
        prev_phase = self.last_phase.get(player)
        self.last_phase[player] = phase

        if not phase or 'Pinning' not in phase:
            self.pinning_completed[player] = True
            return state

        # New pinning phase detected, reset completion flag
        if prev_phase != phase:
            self.pinning_completed[player] = False

        if self.pinning_completed.get(player):
            return state

        print(f"[AutoSubmit] Detected pinning phase for {player}. Attempting automatic pinning...")
        latest_state = self._perform_random_pinning(player, state)

        if 'Pinning' in self._get_current_phase_from_state(latest_state):
            # It's normal to still be in Pinning phase until all players submit.
            print(f"[AutoSubmit] {player} submitted pin but phase still shows Pinning (likely waiting on other players).")
        
        # Regardless of overall phase, we've already sent ping+submit for this player.
        # Mark as completed so we don't spam additional submissions in the same phase.
        self.pinning_completed[player] = True
        print(f"[AutoSubmit] Pinning submission recorded for {player}.")

        return latest_state

    def _perform_random_pinning(self, player: str, state: list[dict]) -> list[dict]:
        """
        Performs a few random navigation commands before submitting to satisfy
        Unity's requirement that each player issues a pinning action.
        """
        socket = self._get_socket(player)
        pin_actions = ["pinga", "pingb", "pingc", "pingd"]

        # Choose exactly one ping action per pinning phase.
        action = random.choice(pin_actions)
        socket.execute_action(action)
        sleep(0.1)  # Increased wait time for ping action

        socket.execute_action('submit')
        sleep(0.2)  # Increased wait time for submit to process
        
        # Try to get state multiple times to ensure Unity has processed
        latest_state = self.get_state(player)
        max_retries = 3
        retry_count = 0
        
        # Verify we're actually out of pinning phase
        while retry_count < max_retries:
            phase = self._get_current_phase_from_state(latest_state)
            if 'Pinning' not in phase:
                break  # Successfully exited pinning phase
            retry_count += 1
            sleep(0.1)  # Wait a bit more
            latest_state = self.get_state(player)

        return latest_state

def _launch_game(game_executable_filepath, port):
    game_thread = Thread(target=_launch_game_thread, args=(game_executable_filepath, port))
    game_thread.start()

def _launch_game_thread(game_executable_filepath, port):
    command = [game_executable_filepath,
                "-localMode",
                "-stepTime", "0",  
                "-hmtsocketurl", "ws://localhost",
                "-hmtsocketport", "{}".format(port)]
    # subprocess.run(command)
    subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sleep(5)


#############
# UTILITIES #
#############
def _simplify_state(state: dict, player: str, last_known_pos=None) -> list[dict]:
    """
    Simplifies the game state into a list of dictionary objects, each representing an object in the game.
    :param state: The game state
    :param player: The player the state is related to
    :return: The modified state as a list of dictionaries
    """
    state['content']['gameData']['id'] = 'gameData'
    game_data_obj = state['content']['gameData']
    scene = [game_data_obj] + state['content']['scene']

    return _add_sight_status(scene, player, game_data_obj, last_known_pos)


def _add_sight_status(scene: list[dict], player: str, game_data_obj: dict, last_known_pos=None) -> list[dict]:
    """
    Modifies the state by adding whether objects are visible or hidden. It also adds 'Cell' objects with an 'unexplored'
    status to represent grid squares that have not yet been observed.
    :param scene: The list of objects in the state
    :param player: The name of the player
    :param game_data_obj: The state object containing high level information about the current state of the game
    :return: The modified scene list
    """
    player_obj = _find_player_obj(scene, player, last_known_pos)
    if not player_obj:
        return scene

    x = player_obj.get('x')
    y = player_obj.get('y')
    sight = player_obj.get('sightRange', 3)
    if x is None or y is None:
        return scene

    x_lower = x - sight
    x_upper = x + sight
    y_lower = y - sight
    y_upper = y + sight

    all_cells = {(i, j) for i in range(game_data_obj['boardWidth']) for j in range(game_data_obj['boardHeight'])}
    scene_cells = {(obj.get('x'), obj.get('y')) for obj in scene}
    unexplored = all_cells - scene_cells

    for obj in scene:
        x, y = obj.get('x'), obj.get('y')
        if x is None or y is None:
            continue
        if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
            obj['sight_status'] = 'visible'
        else:
            obj['sight_status'] = 'hidden'

    for i, pos in enumerate(unexplored):
        scene.append({'id': f'UE{i}', 'entityType': 'Cell',
                      'objKey': 'UE', 'sight_status': 'unexplored',
                      'x': pos[0], 'y': pos[1]})
    return scene


def _find_player_obj(scene: list[dict], player: str, last_known_pos=None) -> [dict, None]:
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
    if last_known_pos is not None:
        x, y, sight = last_known_pos
        return {'id': pid, 'x': x, 'y': y, 'sightRange': sight}


def get_player_id(player: str) -> str:
    """
    Gets the player ID.
    :param player: The player whose ID will be returned
    :return: The player's ID
    """
    ids = {"dwarf": "C11", "giant": "C21", "human": "C31"}
    return ids[player]
