"""
Plotting utilities for the Active Inference Agent project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd


def plot_training_curves(episode_rewards: List[float], episode_lengths: List[float], 
                        training_losses: Optional[List[float]] = None,
                        save_path: Optional[str] = None):
    """
    Plot training curves for an agent.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        training_losses: List of training losses (optional)
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add moving average
    window = min(50, len(episode_rewards) // 10)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                        color='red', linewidth=2, label=f'{window}-episode moving average')
        axes[0, 0].legend()
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6, color='green')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=195, color='red', linestyle='--', alpha=0.7, label='Success threshold')
    axes[0, 1].legend()
    
    # Training losses
    if training_losses:
        axes[1, 0].plot(training_losses, alpha=0.6, color='orange')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(training_losses) >= window:
            loss_moving_avg = np.convolve(training_losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(training_losses)), loss_moving_avg, 
                            color='red', linewidth=2, label=f'{window}-step moving average')
            axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No training losses recorded', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Losses')
    
    # Success rate over time
    success_rate = []
    window_size = 100
    for i in range(window_size, len(episode_lengths) + 1):
        recent_lengths = episode_lengths[i-window_size:i]
        success_rate.append(sum(1 for length in recent_lengths if length >= 195) / len(recent_lengths))
    
    if success_rate:
        axes[1, 1].plot(range(window_size, len(episode_lengths) + 1), success_rate, 
                       color='purple', linewidth=2)
        axes[1, 1].set_title('Success Rate (100-episode window)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough episodes for success rate', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Success Rate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_episode_rewards(episode_rewards: List[float], agent_name: str = "Agent",
                        save_path: Optional[str] = None):
    """
    Plot episode rewards for a single agent.
    
    Args:
        episode_rewards: List of episode rewards
        agent_name: Name of the agent for the plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(episode_rewards, alpha=0.6, color='blue', label='Episode Rewards')
    
    # Add moving average
    window = min(50, len(episode_rewards) // 10)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label=f'{window}-episode moving average')
    
    plt.title(f'{agent_name} - Episode Rewards', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_agent_comparison(agent_results: Dict[str, List[float]], 
                         metric: str = 'rewards',
                         save_path: Optional[str] = None):
    """
    Plot comparison between different agents.
    
    Args:
        agent_results: Dictionary with agent names as keys and metric lists as values
        metric: Name of the metric being plotted
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (agent_name, values) in enumerate(agent_results.items()):
        color = colors[i % len(colors)]
        plt.plot(values, alpha=0.6, color=color, label=agent_name)
        
        # Add moving average
        window = min(50, len(values) // 10)
        if len(values) >= window:
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(values)), moving_avg, 
                    color=color, linewidth=2, alpha=0.8)
    
    plt.title(f'Agent Comparison - {metric.title()}', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel(metric.title())
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 