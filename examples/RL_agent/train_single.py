import gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from cnn_extractor import CustomCNN
# Import the registration (this needs to happen before creating the env)
import gym
# from dice_adventure_python_env_simp import DiceAdventurePythonEnvRL
from dice_adventure_python_env_new import DiceAdventurePythonEnvRL
import json
import logging
import sys
import socket

# # Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

class RewardMonitorCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Get episode info from the monitor wrapper
        for info in self.locals['infos']:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                # Calculate statistics
                mean_reward = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
                mean_length = sum(self.episode_lengths[-100:]) / len(self.episode_lengths[-100:])
                
                # Log to tensorboard
                self.logger.record('rollout/ep_rew_mean', mean_reward)
                self.logger.record('rollout/ep_len_mean', mean_length)
                
                if self.n_calls % self.check_freq == 0:
                    print(f"Steps: {self.n_calls}")
                    print(f"Last episode reward: {self.episode_rewards[-1]:.2f}")
                    print(f"Mean 100 episode reward: {mean_reward:.2f}")
                    print(f"Mean 100 episode length: {mean_length:.2f}")
                    print("-" * 50)
        
        return True

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def train_single_env():
    # Check if port is in use
    # if is_port_in_use(8080):
    #     logging.error("Port 8080 is already in use. Please free the port first.")
    #     return
        
    # Register the environment
    # config = json.loads(open("examples/RL_agent/train_config.json").read())
    # Create the environment using the registered ID
    # kwargs = {**config["ENV_SETTINGS"], **config["GAME_SETTINGS"]}
    # env = DiceAdventurePythonEnvRL(
    #     **kwargs
    # )
    
    logging.info("Creating environment...")
    env = DiceAdventurePythonEnvRL(
        game_executable_filepath="/Users/thatavery/Documents/TAIL/new_folder/Dice-Adventure-Agents/DiceAdventure.app",
        port="8090",
        player="human",
        train_mode=True
    )
    logging.info("Environment created")
    env = Monitor(env, "./logs/monitor/")
    logging.info("Environment monitored")
    
    # Define policy kwargs with the custom CNN
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": 256
        }
    }
    logging.info("Policy kwargs defined")

    # Create the model with single environment
    model = MaskablePPO(
        "CnnPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=1,  # Reduce verbosity
        tensorboard_log="./logs/ppo_single_env/",
        learning_rate=3e-4,
        n_steps=2048,  # Reduce from 10240
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    logging.info("Model created")

    # Add checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,  # Save every 1000 steps
        save_path="./model_checkpoints/",
        name_prefix="dice_adventure_model",
        verbose=2
    )
    # RewardMonitorCallback()
    
    logging.info("Checkpoint callback created")
    logging.info("Starting training...")
    try:
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=1
        )
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        # Print environment state
        env_state = env.get_state("giant")
        logging.error(f"Final environment state: {env_state}")

if __name__ == "__main__":
    try:
        logging.info("Starting training script...")
        train_single_env()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        logging.info("Script ended")