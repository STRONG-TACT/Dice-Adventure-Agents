import torch as th
import torch.nn as nn
import gym
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract shape information
        n_input_channels = observation_space.shape[0]  # First dimension (channels)
        
        # Define CNN architecture with smaller kernels and strides
        self.cnn = nn.Sequential(
            # First conv layer: input (43, 20, 20) -> output (32, 18, 18)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Second conv layer: (32, 18, 18) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Third conv layer: (64, 16, 16) -> (64, 14, 14)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Add max pooling to reduce dimensions: (64, 14, 14) -> (64, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
if __name__ == "__main__":
    # Test the CNN
    observation_space = spaces.Box(low=0, high=100, shape=(43, 20, 20), dtype=np.float32)
    cnn_extractor = CustomCNN(observation_space)
    print(cnn_extractor)
    
    # Test forward pass
    sample_input = th.randn(1, 43, 20, 20)
    output = cnn_extractor(sample_input)
    print(f"Output shape: {output.shape}")  # Should be (1, 256)
