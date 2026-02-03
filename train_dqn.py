"""
Training script for DQN agent on Dice Adventure game
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from time import sleep

from agent_template.dice_adventure_gym_env import DiceAdventureGymEnv, get_player_id
from agent_template.dqn_agent import DQNAgent
from agent_template.state_preprocessor import StatePreprocessor


def train_dqn(character_name: str = 'dwarf',
              port: str = '4649',
              game_executable: str = '/DiceAdventure.exe',
              episodes: int = 1000,
              max_steps: int = 500,
              save_dir: str = 'models',
              save_freq: int = 100,
              log_freq: int = 10,
              log_step_reward: bool = False):
    """
    Train a DQN agent on Dice Adventure.
    
    :param character_name: Character to train ('dwarf', 'giant', or 'human')
    :param port: Port for game connection
    :param game_executable: Path to game executable
    :param episodes: Number of training episodes
    :param max_steps: Maximum steps per episode
    :param save_dir: Directory to save models
    :param save_freq: Frequency to save model (in episodes)
    :param log_freq: Frequency to log progress (in episodes)
    :param log_step_reward: Whether to print each step's reward to the console
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    print(f"Initializing environment for {character_name}...")
    env = DiceAdventureGymEnv(port=port, game_executable_filepath=game_executable, player=character_name)
    
    # Get state and action sizes
    character_id = get_player_id(character_name)
    preprocessor = StatePreprocessor()
    state_size = preprocessor._get_feature_size()
    action_size = len(env.actions)
    
    # Initialize DQN agent
    print(f"Initializing DQN agent...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        character_id=character_id,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=100000,
        batch_size=64,
        target_update_freq=100
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    
    print(f"\nStarting training for {episodes} episodes...")
    print("=" * 60)
    
    try:
        for episode in range(episodes):
            # Reset environment
            state, _ = env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0
            
            # Run episode
            for step in range(max_steps):
                # Choose action
                action_idx = agent.act(state, env.actions, training=True)
                action = env.actions[action_idx]
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                if log_step_reward:
                    print(f"[Step Reward] Ep{episode + 1} Step{step + 1}: {reward:.2f}")

                # Let other players take random turns so game progresses
                if not terminated and not truncated:
                    _play_support_players(env, character_name)
                
                # Store experience
                agent.remember(state, action_idx, reward, next_state, terminated or truncated)
                
                # Train agent
                loss = agent.replay()
                if loss is not None:
                    total_loss += loss
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            recent_rewards.append(total_reward)
            recent_lengths.append(steps)
            
            avg_loss = total_loss / max(steps, 1)
            episode_losses.append(avg_loss)
            
            # Logging
            if (episode + 1) % log_freq == 0:
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Reward: {total_reward:.2f} (avg: {avg_reward:.2f}) | "
                      f"Length: {steps} (avg: {avg_length:.1f}) | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
            
            # Save model
            if (episode + 1) % save_freq == 0:
                model_path = os.path.join(save_dir, f"{character_name}_dqn_ep{episode+1}.weights.h5")
                agent.save(model_path)
                print(f"Model saved to {model_path}")
        
        # Save final model
        final_model_path = os.path.join(save_dir, f"{character_name}_dqn_final.weights.h5")
        agent.save(final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
        
        # Plot training curves
        plot_training_curves(episode_rewards, episode_lengths, episode_losses, save_dir, character_name)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        final_model_path = os.path.join(save_dir, f"{character_name}_dqn_interrupted.weights.h5")
        agent.save(final_model_path)
        print(f"Model saved to {final_model_path}")
    finally:
        env.close()


def _play_support_players(env: DiceAdventureGymEnv, main_player: str):
    """
    Sends random actions for players other than the one being trained.
    Prevents the game from stalling on their turns.
    """
    random_actions = env.get_actions()
    for player in env.get_player_names():
        if player == main_player:
            continue
        try:
            state = env.get_state(player)
            state = env._auto_submit_pinning_phase(state, player)
            action = random.choice(random_actions)
            env.execute_action(player, action)
            sleep(0.01)  # Reduced sleep time for faster training
        except Exception as exc:
            print(f"[SupportAgent] Warning: could not act for {player}: {exc}")


def plot_training_curves(rewards, lengths, losses, save_dir, character_name):
    """
    Plot and save training curves.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Rewards
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= 100:
        # Moving average
        window = 100
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2, label='Moving Average (100)')
        axes[0].legend()
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    
    # Episode lengths
    axes[1].plot(lengths, alpha=0.3, color='green')
    if len(lengths) >= 100:
        window = 100
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), moving_avg, color='red', linewidth=2, label='Moving Average (100)')
        axes[1].legend()
    axes[1].set_title('Episode Lengths')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True)
    
    # Losses
    axes[2].plot(losses, alpha=0.3, color='orange')
    if len(losses) >= 100:
        window = 100
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[2].plot(range(window-1, len(losses)), moving_avg, color='red', linewidth=2, label='Moving Average (100)')
        axes[2].legend()
    axes[2].set_title('Training Loss')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{character_name}_training_curves.png")
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Dice Adventure')
    parser.add_argument('--character', type=str, default='dwarf', 
                       choices=['dwarf', 'giant', 'human'],
                       help='Character to train')
    parser.add_argument('--port', type=str, default='4649',
                       help='Port for game connection')
    parser.add_argument('--game', type=str, default='/DiceAdventure.exe',
                       help='Path to game executable')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Frequency to save model (episodes)')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Frequency to log progress (episodes)')
    parser.add_argument('--log-step-reward', action='store_true',
                       help='Print reward value for every step')
    
    args = parser.parse_args()
    
    train_dqn(
        character_name=args.character,
        port=args.port,
        game_executable=args.game,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        log_step_reward=args.log_step_reward
    )


if __name__ == "__main__":
    main()

