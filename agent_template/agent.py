# PLEASE IMPORT ANY PACKAGES YOU NEED
import os
import numpy as np
from agent_template.dqn_agent import DQNAgent
from agent_template.state_preprocessor import StatePreprocessor


class DiceAdventureAgent:
    """
    DQN-based agent for Dice Adventure game.
    Uses a trained DQN model to make decisions.
    """
    def __init__(self, character_name: str, character_id: str, 
                 model_path: str = None) -> None:
        """
        Initialize the DQN agent.
        :param character_name: The character the agent will play as
        :param character_id: The character ID corresponding to the character
        :param model_path: Path to the trained model weights (optional)
        
        Player ID mappings:
            dwarf : C11
            giant : C21
            human : C31
        """
        self.character = character_name
        self.character_id = character_id
        
        # Action mapping
        self.actions = ["up", "down", "left", "right", "wait", "undo", "submit", 
                       "pinga", "pingb", "pingc", "pingd"]
        self.action_size = len(self.actions)
        
        # Initialize state preprocessor to get feature size
        preprocessor = StatePreprocessor()
        state_size = preprocessor._get_feature_size()
        
        # Initialize DQN agent
        self.dqn_agent = DQNAgent(
            state_size=state_size,
            action_size=self.action_size,
            character_id=character_id,
            epsilon=0.0,  # No exploration during inference
            epsilon_min=0.0
        )
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                self.dqn_agent.load(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using untrained model (random actions)")
        else:
            if model_path:
                print(f"Warning: Model path {model_path} does not exist. Using untrained model.")
            # Set epsilon to 1.0 for random actions if no model loaded
            self.dqn_agent.set_epsilon(1.0)

    def take_action(self, state: list[dict], actions: list[str]) -> str:
        """
        Given a game state and list of actions, the agent determines which action to take.
        :param state:   A 'Dice Adventure' game state
        :param actions: A list of string action names
        :return:        An action from the 'actions' list
        """
        # Get action index from DQN agent
        action_idx = self.dqn_agent.act(state, actions, training=False)
        
        # Map index to action string
        if 0 <= action_idx < len(self.actions):
            action = self.actions[action_idx]
            # Make sure the action is in the available actions list
            if action in actions:
                return action
        
        # Fallback: return first available action
        return actions[0] if actions else "wait"
