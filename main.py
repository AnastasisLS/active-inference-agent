"""
Main training script for the Active Inference Agent project.

This script provides a command-line interface for training and comparing
different agents on the CartPole environment.
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.agent_config import AgentConfig, DQNConfig, ActiveInferenceConfig
from config.environment_config import EnvironmentConfig, CartPoleConfig
from config.training_config import TrainingConfig, ComparisonConfig
from environments.cartpole_env import create_cartpole_env
from agents.dqn_agent import create_dqn_agent


def train_dqn_agent(episodes: int = 1000, eval_frequency: int = 50, 
                   save_model: bool = True, model_path: str = None):
    """
    Train a DQN agent on CartPole.
    
    Args:
        episodes: Number of episodes to train for
        eval_frequency: Frequency of evaluation
        save_model: Whether to save the trained model
        model_path: Path to save the model
    """
    print("=" * 60)
    print("TRAINING DQN AGENT")
    print("=" * 60)
    
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
    
    # Train agent
    print(f"\nStarting training for {episodes} episodes...")
    agent.train(env, episodes=episodes, eval_frequency=eval_frequency)
    
    # Print final statistics
    stats = agent.get_training_info()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Episodes: {stats['episode_count']}")
    print(f"Total Steps: {stats['total_steps']}")
    print(f"Average Reward: {stats['average_reward']:.2f}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Best Reward: {stats['best_reward']:.2f}")
    print(f"Recent Average Reward: {stats['recent_average_reward']:.2f}")
    print(f"Final Epsilon: {stats['epsilon']:.3f}")
    
    # Save model if requested
    if save_model:
        if model_path is None:
            model_path = "data/models/dqn_agent.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save_model(model_path)
        print(f"Model saved to: {model_path}")
    
    # Clean up
    env.close()
    
    return agent, stats


def train_active_inference_agent(episodes: int = 1000, eval_frequency: int = 50,
                                save_model: bool = True, model_path: str = None):
    """
    Train an Active Inference agent on CartPole.
    
    Args:
        episodes: Number of episodes to train for
        eval_frequency: Frequency of evaluation
        save_model: Whether to save the trained model
        model_path: Path to save the model
    """
    print("=" * 60)
    print("TRAINING ACTIVE INFERENCE AGENT")
    print("=" * 60)
    
    # Create environment
    env = create_cartpole_env(normalize_states=True, reward_shaping=False)
    print(f"Environment created: {env.get_environment_info()}")
    
    # TODO: Implement Active Inference agent
    print("Active Inference agent not yet implemented.")
    print("This will be implemented in Week 2-3 of the project.")
    
    # For now, return None
    return None, {}


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
    
    # TODO: Train Active Inference agent multiple times
    print(f"\nTraining Active Inference agent for {num_runs} runs...")
    print("Active Inference agent not yet implemented.")
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if results['dqn']:
        dqn_rewards = [stats['average_reward'] for stats in results['dqn']]
        dqn_success_rates = [stats['success_rate'] for stats in results['dqn']]
        
        print(f"DQN Agent:")
        print(f"  Average Reward: {sum(dqn_rewards)/len(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
        print(f"  Success Rate: {sum(dqn_success_rates)/len(dqn_success_rates):.2%} ± {np.std(dqn_success_rates):.2%}")
    
    if results['active_inference']:
        ai_rewards = [stats['average_reward'] for stats in results['active_inference']]
        ai_success_rates = [stats['success_rate'] for stats in results['active_inference']]
        
        print(f"Active Inference Agent:")
        print(f"  Average Reward: {sum(ai_rewards)/len(ai_rewards):.2f} ± {np.std(ai_rewards):.2f}")
        print(f"  Success Rate: {sum(ai_success_rates)/len(ai_success_rates):.2%} ± {np.std(ai_success_rates):.2%}")
    
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
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run setup tests'
    )
    
    args = parser.parse_args()
    
    # Run setup tests if requested
    if args.test:
        print("Running setup tests...")
        import test_setup
        success = test_setup.main()
        if not success:
            print("Setup tests failed. Please fix the issues before proceeding.")
            return
    
    # Create data directories
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Run training based on arguments
    if args.compare:
        compare_agents(episodes=args.episodes, num_runs=args.num_runs)
    
    elif args.agent == 'dqn':
        train_dqn_agent(
            episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            save_model=args.save_model,
            model_path=args.model_path
        )
    
    elif args.agent == 'active_inference':
        train_active_inference_agent(
            episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            save_model=args.save_model,
            model_path=args.model_path
        )
    
    elif args.agent == 'both':
        print("Training both agents...")
        dqn_agent, dqn_stats = train_dqn_agent(
            episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            save_model=args.save_model,
            model_path=args.model_path
        )
        
        ai_agent, ai_stats = train_active_inference_agent(
            episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            save_model=args.save_model,
            model_path=args.model_path
        )
    
    else:
        print(f"Unknown agent type: {args.agent}")
        parser.print_help()


if __name__ == "__main__":
    main() 