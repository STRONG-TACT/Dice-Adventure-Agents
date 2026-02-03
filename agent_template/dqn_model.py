"""
DQN Network Architecture
Implements Dueling DQN with Double DQN support
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


class DuelingDQN(keras.Model):
    """
    Dueling DQN architecture that separates state value and advantage estimation.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_units: list = [256, 256]):
        """
        Initialize Dueling DQN network.
        :param state_size: Size of the state feature vector
        :param action_size: Number of possible actions
        :param hidden_units: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        self.shared_layers = []
        for units in hidden_units:
            self.shared_layers.append(layers.Dense(units, activation='relu'))
            self.shared_layers.append(layers.BatchNormalization())
        
        # Value stream (estimates V(s))
        self.value_stream = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(1, name='value')
        ])
        
        # Advantage stream (estimates A(s,a))
        self.advantage_stream = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(action_size, name='advantage')
        ])
    
    def call(self, inputs):
        """
        Forward pass through the network.
        :param inputs: State tensor
        :return: Q-values for all actions
        """
        # Shared feature extraction
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)
        
        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values


def create_dqn_model(state_size: int, action_size: int, 
                     hidden_units: list = [256, 256]) -> DuelingDQN:
    """
    Factory function to create a DQN model.
    :param state_size: Size of the state feature vector
    :param action_size: Number of possible actions
    :param hidden_units: List of hidden layer sizes
    :return: DQN model
    """
    model = DuelingDQN(state_size, action_size, hidden_units)
    return model

