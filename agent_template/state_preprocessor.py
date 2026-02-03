"""
State Preprocessor for Dice Adventure
Converts the game state (list of dicts) into a fixed-length feature vector for DQN
"""
import numpy as np
from typing import List, Dict, Tuple


class StatePreprocessor:
    """
    Preprocesses the Dice Adventure game state into a fixed-length feature vector.
    Uses grid-based encoding with a fixed-size grid around the player.
    """
    
    def __init__(self, grid_size: int = 15, max_objects: int = 50):
        """
        Initialize the state preprocessor.
        :param grid_size: Size of the grid around the player (grid_size x grid_size)
        :param max_objects: Maximum number of objects to consider
        """
        self.grid_size = grid_size
        self.max_objects = max_objects
        self.grid_center = grid_size // 2
        
        # Object type encodings
        self.object_types = {
            'Wall': 0,
            'Open': 1,
            'Goal': 2,
            'Shrine': 3,
            'monster': 4,
            'Dwarf': 5,
            'Giant': 6,
            'Human': 7,
            'Cell': 8,
            'gameData': 9
        }
        
    def preprocess(self, state: List[Dict], character_id: str) -> np.ndarray:
        """
        Convert game state to feature vector.
        :param state: List of game objects (dicts)
        :param character_id: ID of the character this agent controls
        :return: Fixed-length feature vector
        """
        # Find player object
        player_obj = None
        game_data = None
        for obj in state:
            if obj.get('id') == character_id:
                player_obj = obj
            if obj.get('id') == 'gameData':
                game_data = obj
        
        if player_obj is None or game_data is None:
            # Return zero vector if player not found
            return np.zeros(self._get_feature_size())
        
        player_x = player_obj.get('x', 0)
        player_y = player_obj.get('y', 0)
        
        # Create grid representation
        grid_features = self._create_grid(state, player_x, player_y, character_id)
        
        # Extract player features
        player_features = self._extract_player_features(player_obj)
        
        # Extract global features
        global_features = self._extract_global_features(state, game_data, character_id)
        
        # Combine all features
        feature_vector = np.concatenate([
            grid_features.flatten(),
            player_features,
            global_features
        ])
        
        return feature_vector.astype(np.float32)
    
    def _create_grid(self, state: List[Dict], player_x: int, player_y: int, 
                     character_id: str) -> np.ndarray:
        """
        Create a grid representation around the player.
        :return: Grid of shape (grid_size, grid_size, num_channels)
        """
        # Grid channels: [wall, open, goal, shrine, monster, other_player, unexplored, distance]
        grid = np.zeros((self.grid_size, self.grid_size, 8), dtype=np.float32)
        
        # Calculate offset to center player in grid
        offset_x = self.grid_center - player_x
        offset_y = self.grid_center - player_y
        
        for obj in state:
            obj_x = obj.get('x')
            obj_y = obj.get('y')
            
            if obj_x is None or obj_y is None:
                continue
            
            # Transform coordinates to grid space
            grid_x = obj_x + offset_x
            grid_y = obj_y + offset_y
            
            # Check if within grid bounds
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                entity_type = obj.get('entityType', '')
                obj_id = obj.get('id', '')
                sight_status = obj.get('sight_status', 'visible')
                
                # Skip if unexplored
                if sight_status == 'unexplored':
                    grid[grid_y, grid_x, 6] = 1.0  # unexplored channel
                    continue
                
                # Encode object type
                if entity_type == 'Wall':
                    grid[grid_y, grid_x, 0] = 1.0
                elif entity_type == 'Open':
                    grid[grid_y, grid_x, 1] = 1.0
                elif entity_type == 'Goal':
                    grid[grid_y, grid_x, 2] = 1.0
                elif entity_type == 'Shrine':
                    grid[grid_y, grid_x, 3] = 1.0
                elif entity_type == 'monster':
                    grid[grid_y, grid_x, 4] = 1.0
                elif obj_id != character_id and obj_id in ['C11', 'C21', 'C31']:
                    # Other players
                    grid[grid_y, grid_x, 5] = 1.0
                
                # Distance from player (normalized)
                distance = np.sqrt((grid_x - self.grid_center)**2 + (grid_y - self.grid_center)**2)
                grid[grid_y, grid_x, 7] = distance / (self.grid_size * np.sqrt(2))
        
        return grid
    
    def _extract_player_features(self, player_obj: Dict) -> np.ndarray:
        """
        Extract features from player object.
        """
        features = [
            player_obj.get('health', 0) / 10.0,  # Normalized health
            player_obj.get('actionPoints', 0) / 10.0,  # Normalized action points
            player_obj.get('sightRange', 0) / 10.0,  # Normalized sight range
            1.0 if player_obj.get('dead', False) else 0.0,  # Is dead
            len(player_obj.get('actionPlan', [])) / 10.0,  # Action plan length
        ]
        
        # Parse dice strings (e.g., "D6+0" -> 6)
        monster_dice = self._parse_dice(player_obj.get('monsterDice', 'D6+0'))
        trap_dice = self._parse_dice(player_obj.get('trapDice', 'D6+0'))
        stone_dice = self._parse_dice(player_obj.get('stoneDice', 'D6+0'))
        
        features.extend([
            monster_dice / 20.0,
            trap_dice / 20.0,
            stone_dice / 20.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_global_features(self, state: List[Dict], game_data: Dict, 
                                  character_id: str) -> np.ndarray:
        """
        Extract global game features.
        """
        # Count objects
        num_monsters = sum(1 for obj in state if obj.get('entityType') == 'monster')
        num_goals = sum(1 for obj in state if obj.get('entityType') == 'Goal')
        num_shrines = sum(1 for obj in state if obj.get('entityType') == 'Shrine')
        
        # Board dimensions (normalized)
        board_width = game_data.get('boardWidth', 20)
        board_height = game_data.get('boardHeight', 20)
        
        features = [
            game_data.get('level', 1) / 10.0,  # Level
            board_width / 50.0,  # Normalized width
            board_height / 50.0,  # Normalized height
            num_monsters / 20.0,  # Normalized monster count
            num_goals / 10.0,  # Normalized goal count
            num_shrines / 10.0,  # Normalized shrine count
        ]
        
        # Phase encoding (one-hot)
        phase = game_data.get('currentPhase', '')
        phase_features = [
            1.0 if 'Player' in phase else 0.0,
            1.0 if 'Pinning' in phase else 0.0,
            1.0 if 'Monster' in phase else 0.0,
        ]
        features.extend(phase_features)
        
        return np.array(features, dtype=np.float32)
    
    def _parse_dice(self, dice_str: str) -> float:
        """
        Parse dice string like "D6+0" to get the dice value.
        """
        try:
            if '+' in dice_str:
                parts = dice_str.split('+')
                dice_part = parts[0].replace('D', '')
                modifier = int(parts[1]) if len(parts) > 1 else 0
                return int(dice_part) + modifier
            else:
                return int(dice_str.replace('D', ''))
        except:
            return 6.0  # Default
    
    def _get_feature_size(self) -> int:
        """
        Calculate the total feature vector size.
        """
        grid_features = self.grid_size * self.grid_size * 8
        player_features = 8  # From _extract_player_features
        global_features = 9  # From _extract_global_features
        return grid_features + player_features + global_features

