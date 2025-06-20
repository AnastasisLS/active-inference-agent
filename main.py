"""
Advanced training script for the Active Inference Agent project.

This script provides a comprehensive command-line interface for training and comparing
different agents on the CartPole environment with advanced features:
- Sophisticated training loops with early stopping
- Comprehensive logging and visualization
- Model checkpointing and resuming
- Performance comparison and analysis
- Advanced hyperparameter tuning support
"""

import argparse
import os
import sys
import numpy as np
import torch
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.agent_config import AgentConfig, DQNConfig, ActiveInferenceConfig
from config.environment_config import EnvironmentConfig, CartPoleConfig
from config.training_config import TrainingConfig, ComparisonConfig
from environments.cartpole_env import create_cartpole_env
from agents.dqn_agent import create_dqn_agent
from agents.active_inference_agent import ActiveInferenceAgent
from utils.device_utils import print_device_info, get_global_device, safe_tensor_creation
from utils.data_logging import DataLogger
from utils.visualization import plot_training_curves, plot_agent_comparison
from utils.metrics import compute_performance_metrics
from utils.plotting import plot_belief_covariance_evolution

def setup_logging_and_directories(agent_type: str, experiment_name: str = None) -> Tuple[str, DataLogger]:
    """
    Setup logging and create necessary directories.
    
    Args:
        agent_type: Type of agent being trained
        experiment_name: Optional experiment name
        
    Returns:
        Tuple of (log_dir, data_logger)
    """
    if experiment_name is None:
        experiment_name = f"{agent_type}_{int(time.time())}"
    
    # Create directories
    log_dir = Path(f"data/logs/{experiment_name}")
    model_dir = Path(f"data/models/{experiment_name}")
    results_dir = Path(f"data/results/{experiment_name}")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data logger
    data_logger = DataLogger(log_dir)
    
    return str(log_dir), data_logger

def train_dqn_agent(episodes: int = 1000, eval_frequency: int = 50, 
                   save_model: bool = True, model_path: str = None,
                   experiment_name: str = None, resume_from: str = None):
    """
    Train a DQN agent on CartPole with advanced features.
    
    Args:
        episodes: Number of episodes to train for
        eval_frequency: Frequency of evaluation
        save_model: Whether to save the trained model
        model_path: Path to save the model
        experiment_name: Name for the experiment
        resume_from: Path to resume training from
    """
    print("=" * 60)
    print("TRAINING DQN AGENT")
    print("=" * 60)
    
    # Setup logging
    log_dir, data_logger = setup_logging_and_directories("dqn", experiment_name)
    
    # Show device information
    print_device_info()
    
    # Create environment
    env = create_cartpole_env(normalize_states=True, reward_shaping=False)
    print(f"Environment created: {env.get_environment_info()}")
    
    # Create agent
    agent = create_dqn_agent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[128, 128],
        learning_rate=0.001,
        batch_size=64,
        buffer_size=10000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    print(f"Agent created: {agent.get_agent_info()}")
    
    # Resume training if specified
    if resume_from and os.path.exists(resume_from):
        agent.load_model(resume_from)
        print(f"Resumed training from: {resume_from}")
    
    # Train agent with enhanced logging
    print(f"\nStarting training for {episodes} episodes...")
    start_time = time.time()
    
    training_stats = agent.train(env, episodes=episodes, eval_frequency=eval_frequency)
    
    training_time = time.time() - start_time
    
    # Log final statistics
    final_stats = agent.get_training_stats()
    final_stats['training_time'] = training_time
    final_stats['episodes'] = episodes
    
    data_logger.log_metrics(final_stats)
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Episodes: {final_stats['episode_count']}")
    print(f"Total Steps: {final_stats['total_steps']}")
    print(f"Average Reward: {final_stats['average_reward']:.2f}")
    print(f"Best Reward: {final_stats['best_reward']:.2f}")
    print(f"Recent Average Reward: {final_stats['recent_average_reward']:.2f}")
    print(f"Device Used: {final_stats['device']}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Get training info for additional details
    training_info = agent.get_training_info()
    print(f"Final Epsilon: {training_info['epsilon']:.3f}")
    print(f"Replay Buffer Size: {training_info['replay_buffer_size']}")
    
    # Save model if requested
    if save_model:
        if model_path is None:
            model_path = f"data/models/dqn_agent_{int(time.time())}.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save_model(model_path)
        print(f"Model saved to: {model_path}")
    
    # Create training plots
    plot_training_curves(training_stats, save_path=f"{log_dir}/training_curves.png")
    
    # Clean up
    env.close()
    
    return agent, final_stats

def train_active_inference_agent(episodes: int = 1000, eval_frequency: int = 50,
                                save_model: bool = True, model_path: str = None,
                                experiment_name: str = None, resume_from: str = None):
    """
    Train an Active Inference agent on CartPole with advanced features.
    
    Args:
        episodes: Number of episodes to train for
        eval_frequency: Frequency of evaluation
        save_model: Whether to save the trained model
        model_path: Path to save the model
        experiment_name: Name for the experiment
        resume_from: Path to resume training from
    """
    print("=" * 60)
    print("TRAINING ACTIVE INFERENCE AGENT")
    print("=" * 60)
    
    # Setup logging
    log_dir, data_logger = setup_logging_and_directories("active_inference", experiment_name)
    
    # Show device information
    print_device_info()
    
    # Create environment with partial observability
    env_config = CartPoleConfig(partial_observability=True)
    env = create_cartpole_env(config=env_config)
    env_info = env.get_environment_info()
    print(f"Environment created: {env_info}")
    
    # Create Active Inference agent
    device = get_global_device()
    agent = ActiveInferenceAgent(
        state_dim=env_info['state_dimension'],
        action_dim=env_info['action_dimension'],
        obs_dim=env_info['observation_dimension'],
        hidden_dim=128,
        learning_rate=0.001,
        belief_lr=0.1,
        temperature=1.0,
        planning_horizon=3,
        device=device
    )
    print(f"Active Inference Agent created on {device}")
    
    # Resume training if specified
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        print(f"Resumed training from: {resume_from}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    vfe_history = []
    efe_history = []
    belief_entropy_history = []
    action_entropy_history = []
    
    # Early stopping parameters
    best_reward = -float('inf')
    patience = 50
    patience_counter = 0
    
    print(f"\nStarting training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        obs, _ = env.reset()
        # No unsqueeze needed if obs is already 2D from env
        obs = safe_tensor_creation(obs, device, dtype=torch.float32)
        
        episode_reward = 0
        episode_length = 0
        episode_vfe = []
        episode_efe = []
        episode_belief_entropy = []
        episode_action_entropy = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(obs)
            
            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)
            next_obs = safe_tensor_creation(next_obs, device, dtype=torch.float32)
            
            # Store experience
            experience = {
                'obs': obs,
                'action': torch.tensor(action, dtype=torch.long, device=device),
                'reward': torch.tensor(reward, dtype=torch.float32, device=device),
                'next_obs': next_obs,
                'done': torch.tensor(done, dtype=torch.bool, device=device)
            }
            
            # Update agent
            update_stats = agent.update_torch(experience)
            
            # Record statistics
            episode_vfe.append(update_stats['vfe_loss'])
            episode_efe.append(update_stats['efe'])
            episode_belief_entropy.append(update_stats['belief_entropy'])
            episode_action_entropy.append(update_stats.get('action_entropy', 0.0))
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # Episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        vfe_history.append(np.mean(episode_vfe))
        efe_history.append(np.mean(episode_efe))
        belief_entropy_history.append(np.mean(episode_belief_entropy))
        action_entropy_history.append(np.mean(episode_action_entropy))
        
        # Early stopping check
        if episode_reward > best_reward:
            best_reward = episode_reward
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Evaluation and logging
        if (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            avg_length = np.mean(episode_lengths[-eval_frequency:])
            avg_vfe = np.mean(vfe_history[-eval_frequency:])
            avg_efe = np.mean(efe_history[-eval_frequency:])
            avg_belief_entropy = np.mean(belief_entropy_history[-eval_frequency:])
            avg_action_entropy = np.mean(action_entropy_history[-eval_frequency:])
            
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Avg VFE: {avg_vfe:8.3f} | "
                  f"Avg EFE: {avg_efe:8.3f} | "
                  f"Temp: {agent.temperature:.3f}")
            
            # Log metrics
            metrics = {
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'avg_vfe': avg_vfe,
                'avg_efe': avg_efe,
                'avg_belief_entropy': avg_belief_entropy,
                'avg_action_entropy': avg_action_entropy,
                'temperature': agent.temperature
            }
            data_logger.log_metrics(metrics)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at episode {episode + 1} (no improvement for {patience} episodes)")
            break
    
    training_time = time.time() - start_time
    
    # Final statistics
    final_stats = {
        'episode_count': len(episode_rewards),
        'total_steps': sum(episode_lengths),
        'average_reward': np.mean(episode_rewards),
        'best_reward': max(episode_rewards),
        'recent_average_reward': np.mean(episode_rewards[-100:]),
        'device': str(device),
        'training_time': training_time,
        'final_temperature': agent.temperature,
        'vfe_history': vfe_history,
        'efe_history': efe_history,
        'belief_entropy_history': belief_entropy_history,
        'action_entropy_history': action_entropy_history,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    # Add covariance trace and determinant histories if present
    if hasattr(agent, 'cov_trace_history') and hasattr(agent, 'cov_det_history'):
        final_stats['cov_trace_history'] = agent.cov_trace_history
        final_stats['cov_det_history'] = agent.cov_det_history
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Episodes: {final_stats['episode_count']}")
    print(f"Total Steps: {final_stats['total_steps']}")
    print(f"Average Reward: {final_stats['average_reward']:.2f}")
    print(f"Best Reward: {final_stats['best_reward']:.2f}")
    print(f"Recent Average Reward: {final_stats['recent_average_reward']:.2f}")
    print(f"Device Used: {final_stats['device']}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Final Temperature: {final_stats['final_temperature']:.3f}")
    
    # Save model if requested
    if save_model:
        if model_path is None:
            model_path = f"data/models/active_inference_agent_{int(time.time())}.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save(model_path)
        print(f"Model saved to: {model_path}")
    
    # Create training plots
    plot_training_curves(final_stats, save_path=f"{log_dir}/training_curves.png")
    # Plot belief covariance evolution if available
    if 'cov_trace_history' in final_stats and 'cov_det_history' in final_stats:
        plot_belief_covariance_evolution(final_stats, save_path=f"{log_dir}/belief_cov_evolution.png")
    
    # Clean up
    env.close()
    
    return agent, final_stats

def compare_agents(episodes: int = 500, num_runs: int = 3):
    """
    Compare DQN and Active Inference agents.
    
    Args:
        episodes: Number of episodes per run
        num_runs: Number of independent runs per agent
    """
    print("=" * 60)
    print("COMPARING AGENTS")
    print("=" * 60)
    
    # Show device information
    print_device_info()
    
    results = {
        'dqn': [],
        'active_inference': []
    }
    
    # Train DQN agent multiple times
    print(f"\nTraining DQN agent for {num_runs} runs...")
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        agent, stats = train_dqn_agent(episodes=episodes, eval_frequency=50, save_model=False)
        results['dqn'].append(stats)
    
    # Train Active Inference agent multiple times
    print(f"\nTraining Active Inference agent for {num_runs} runs...")
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        agent, stats = train_active_inference_agent(episodes=episodes, eval_frequency=50, save_model=False)
        results['active_inference'].append(stats)
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if results['dqn']:
        dqn_rewards = [stats['average_reward'] for stats in results['dqn']]
        dqn_best_rewards = [stats['best_reward'] for stats in results['dqn']]
        
        print(f"DQN Agent:")
        print(f"  Average Reward: {sum(dqn_rewards)/len(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
        print(f"  Best Reward: {sum(dqn_best_rewards)/len(dqn_best_rewards):.2f} ± {np.std(dqn_best_rewards):.2f}")
        print(f"  Device: {results['dqn'][0]['device']}")
    
    if results['active_inference']:
        ai_rewards = [stats['average_reward'] for stats in results['active_inference']]
        ai_best_rewards = [stats['best_reward'] for stats in results['active_inference']]
        
        print(f"Active Inference Agent:")
        print(f"  Average Reward: {sum(ai_rewards)/len(ai_rewards):.2f} ± {np.std(ai_rewards):.2f}")
        print(f"  Best Reward: {sum(ai_best_rewards)/len(ai_best_rewards):.2f} ± {np.std(ai_best_rewards):.2f}")
        print(f"  Device: {results['active_inference'][0]['device']}")
    
    return results

def main():
    """Main function to handle command-line arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train and compare agents on CartPole environment"
    )
    
    parser.add_argument(
        '--agent', 
        type=str, 
        choices=['dqn', 'active_inference', 'both'],
        default='dqn',
        help='Agent type to train'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=1000,
        help='Number of episodes to train for'
    )
    
    parser.add_argument(
        '--eval_frequency', 
        type=int, 
        default=50,
        help='Frequency of evaluation during training'
    )
    
    parser.add_argument(
        '--save_model', 
        action='store_true',
        help='Save the trained model'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None,
        help='Path to save/load the model'
    )
    
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Compare different agents'
    )
    
    parser.add_argument(
        '--num_runs', 
        type=int, 
        default=3,
        help='Number of runs for comparison'
    )
    
    args = parser.parse_args()
    
    # Show device information at startup
    print("=" * 60)
    print("ACTIVE INFERENCE AGENT PROJECT")
    print("=" * 60)
    print_device_info()
    
    if args.compare:
        # Compare agents
        compare_agents(episodes=args.episodes, num_runs=args.num_runs)
    elif args.agent == 'dqn':
        # Train DQN agent
        train_dqn_agent(
            episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            save_model=args.save_model,
            model_path=args.model_path
        )
    elif args.agent == 'active_inference':
        # Train Active Inference agent
        train_active_inference_agent(
            episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            save_model=args.save_model,
            model_path=args.model_path
        )
    elif args.agent == 'both':
        # Train both agents
        print("Training both agents...")
        train_dqn_agent(episodes=args.episodes, eval_frequency=args.eval_frequency)
        train_active_inference_agent(episodes=args.episodes, eval_frequency=args.eval_frequency)

if __name__ == "__main__":
    main() 