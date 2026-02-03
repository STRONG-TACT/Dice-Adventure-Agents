"""
DQN Agent with Experience Replay, Target Network, and Double DQN
"""
import numpy as np
import random
import tensorflow as tf
from collections import deque
from typing import Tuple, Optional, List
import os
import pickle

from agent_template.dqn_model import create_dqn_model
from agent_template.state_preprocessor import StatePreprocessor


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        :param capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences.
        :return: Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with all improvements: Experience Replay, Target Network, Double DQN, Dueling Architecture
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 character_id: str,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 hidden_units: List[int] = [256, 256]):
        """
        Initialize DQN Agent.
        :param state_size: Size of state feature vector
        :param action_size: Number of actions
        :param character_id: Character ID for this agent
        :param learning_rate: Learning rate for optimizer
        :param gamma: Discount factor
        :param epsilon: Initial exploration rate
        :param epsilon_min: Minimum exploration rate
        :param epsilon_decay: Epsilon decay rate
        :param memory_size: Replay buffer size
        :param batch_size: Training batch size
        :param target_update_freq: Frequency to update target network
        :param hidden_units: Hidden layer sizes
        """
        self.state_size = state_size
        self.action_size = action_size
        self.character_id = character_id
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Create networks
        self.q_network = create_dqn_model(state_size, action_size, hidden_units)
        self.target_network = create_dqn_model(state_size, action_size, hidden_units)
        
        # Initialize target network with same weights as main network
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # State preprocessor
        self.preprocessor = StatePreprocessor()
    
    def act(self, state: List[dict], actions: List[str], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        :param state: Game state (list of dicts)
        :param actions: List of available actions
        :param training: Whether in training mode (affects exploration)
        :return: Action index
        """
        # Preprocess state
        state_vector = self.preprocessor.preprocess(state, self.character_id)
        state_vector = np.expand_dims(state_vector, axis=0)
        
        # Epsilon-greedy action selection
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Get Q-values from network
        q_values = self.q_network(state_vector, training=False)
        return np.argmax(q_values[0])
    
    def remember(self, state: List[dict], action: int, reward: float, 
                 next_state: List[dict], done: bool):
        """
        Store experience in replay buffer.
        """
        state_vector = self.preprocessor.preprocess(state, self.character_id)
        next_state_vector = self.preprocessor.preprocess(next_state, self.character_id)
        
        self.memory.push(state_vector, action, reward, next_state_vector, done)
    
    def replay(self) -> Optional[float]:
        """
        Train the network on a batch of experiences from replay buffer.
        :return: Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Compute target Q-values using Double DQN
        mse_loss = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q_values = self.q_network(states, training=True)
            q_value = tf.reduce_sum(current_q_values * tf.one_hot(actions, self.action_size), axis=1)
            
            # Next Q-values using main network (for action selection)
            next_q_values_main = self.q_network(next_states, training=False)
            next_actions = tf.argmax(next_q_values_main, axis=1)
            
            # Target Q-values using target network (for value estimation)
            next_q_values_target = self.target_network(next_states, training=False)
            next_q_value = tf.reduce_sum(
                next_q_values_target * tf.one_hot(next_actions, self.action_size),
                axis=1
            )
            
            # Compute target
            target_q = rewards + (1 - dones) * self.gamma * next_q_value
            
            # Compute loss
            loss = mse_loss(target_q, q_value)
        
        # Update network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return float(loss.numpy())
    
    def save(self, filepath: str):
        """
        Save the model to file.
        :param filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.q_network.save_weights(filepath)
        
        # Save other parameters
        params = {
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'character_id': self.character_id
        }
        params_path = filepath.replace('.weights.h5', '_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, filepath: str):
        """
        Load the model from file.
        :param filepath: Path to load the model from
        """
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Load other parameters
        params_path = filepath.replace('.weights.h5', '_params.pkl')
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
                self.epsilon = params.get('epsilon', self.epsilon_min)
                self.update_counter = params.get('update_counter', 0)
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate."""
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))

