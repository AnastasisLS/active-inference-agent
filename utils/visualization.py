"""
Visualization utilities for the Active Inference Agent project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd


def create_agent_comparison_plots(agent_results: Dict[str, Dict[str, Any]],
                                 save_path: Optional[str] = None):
    """
    Create comprehensive comparison plots for multiple agents.
    
    Args:
        agent_results: Dictionary with agent names as keys and results as values
        save_path: Path to save the plots (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Agent Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    agent_names = list(agent_results.keys())
    success_rates = [results['success_rate'] for results in agent_results.values()]
    avg_rewards = [results['average_reward'] for results in agent_results.values()]
    avg_lengths = [results['average_length'] for results in agent_results.values()]
    
    # 1. Success Rate Comparison
    bars1 = axes[0, 0].bar(agent_names, success_rates, color=['blue', 'red', 'green', 'orange'])
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, success_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2%}', ha='center', va='bottom')
    
    # 2. Average Reward Comparison
    bars2 = axes[0, 1].bar(agent_names, avg_rewards, color=['blue', 'red', 'green', 'orange'])
    axes[0, 1].set_title('Average Reward Comparison')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, avg_rewards):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 3. Average Length Comparison
    bars3 = axes[1, 0].bar(agent_names, avg_lengths, color=['blue', 'red', 'green', 'orange'])
    axes[1, 0].set_title('Average Episode Length Comparison')
    axes[1, 0].set_ylabel('Average Length')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=195, color='red', linestyle='--', alpha=0.7, label='Success threshold')
    axes[1, 0].legend()
    
    # Add value labels on bars
    for bar, value in zip(bars3, avg_lengths):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 4. Performance Radar Chart (if we have multiple metrics)
    if len(agent_names) >= 2:
        # Normalize metrics for radar chart
        max_reward = max(avg_rewards)
        max_length = max(avg_lengths)
        
        normalized_rewards = [r/max_reward for r in avg_rewards]
        normalized_lengths = [l/max_length for l in avg_lengths]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, agent_name in enumerate(agent_names):
            values = [success_rates[i], normalized_rewards[i], normalized_lengths[i]]
            values += values[:1]  # Complete the circle
            
            axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=agent_name)
            axes[1, 1].fill(angles, values, alpha=0.25)
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(['Success Rate', 'Normalized Reward', 'Normalized Length'])
        axes[1, 1].set_title('Performance Radar Chart')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need at least 2 agents for radar chart', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Radar Chart')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_belief_states(belief_history: List[np.ndarray], 
                           save_path: Optional[str] = None):
    """
    Visualize belief state evolution over time.
    
    Args:
        belief_history: List of belief state arrays
        save_path: Path to save the plot (optional)
    """
    if not belief_history:
        print("No belief history provided")
        return
    
    belief_array = np.array(belief_history)
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    plt.imshow(belief_array.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Belief Probability')
    plt.title('Belief State Evolution Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('State Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_uncertainty_analysis(belief_entropies: List[float], 
                            free_energies: List[float] = None,
                            save_path: Optional[str] = None):
    """
    Plot uncertainty analysis for Active Inference agents.
    
    Args:
        belief_entropies: List of belief state entropies
        free_energies: List of free energy values (optional)
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Uncertainty Analysis', fontsize=16, fontweight='bold')
    
    # Plot belief entropies
    axes[0].plot(belief_entropies, color='blue', alpha=0.7)
    axes[0].set_title('Belief State Entropy Over Time')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Entropy')
    axes[0].grid(True, alpha=0.3)
    
    # Add moving average
    window = min(50, len(belief_entropies) // 10)
    if len(belief_entropies) >= window:
        moving_avg = np.convolve(belief_entropies, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(belief_entropies)), moving_avg, 
                    color='red', linewidth=2, label=f'{window}-step moving average')
        axes[0].legend()
    
    # Plot free energies if provided
    if free_energies:
        axes[1].plot(free_energies, color='green', alpha=0.7)
        axes[1].set_title('Free Energy Over Time')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Free Energy')
        axes[1].grid(True, alpha=0.3)
        
        # Add moving average for free energy
        if len(free_energies) >= window:
            fe_moving_avg = np.convolve(free_energies, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(free_energies)), fe_moving_avg, 
                        color='red', linewidth=2, label=f'{window}-step moving average')
            axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No free energy data provided', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Free Energy Over Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_action_distributions(action_distributions: List[np.ndarray],
                            agent_name: str = "Agent",
                            save_path: Optional[str] = None):
    """
    Plot action distributions over time.
    
    Args:
        action_distributions: List of action probability arrays
        agent_name: Name of the agent
        save_path: Path to save the plot (optional)
    """
    if not action_distributions:
        print("No action distributions provided")
        return
    
    action_array = np.array(action_distributions)
    num_actions = action_array.shape[1]
    
    plt.figure(figsize=(12, 6))
    
    for action_idx in range(num_actions):
        plt.plot(action_array[:, action_idx], 
                label=f'Action {action_idx}', 
                alpha=0.7, linewidth=2)
    
    plt.title(f'{agent_name} - Action Distribution Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('Action Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_summary_plot(agent_results: Dict[str, Dict[str, Any]],
                               save_path: Optional[str] = None):
    """
    Create a comprehensive training summary plot.
    
    Args:
        agent_results: Dictionary with agent names as keys and results as values
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Summary', fontsize=16, fontweight='bold')
    
    agent_names = list(agent_results.keys())
    
    # 1. Success Rate vs Average Reward
    success_rates = [results['success_rate'] for results in agent_results.values()]
    avg_rewards = [results['average_reward'] for results in agent_results.values()]
    
    scatter = axes[0, 0].scatter(avg_rewards, success_rates, 
                                c=range(len(agent_names)), 
                                cmap='viridis', s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Average Reward')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate vs Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add agent labels
    for i, name in enumerate(agent_names):
        axes[0, 0].annotate(name, (avg_rewards[i], success_rates[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    # 2. Performance Metrics Bar Chart
    metrics = ['success_rate', 'average_reward', 'average_length']
    metric_labels = ['Success Rate', 'Avg Reward', 'Avg Length']
    
    x = np.arange(len(agent_names))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[metric] for results in agent_results.values()]
        # Normalize values for better visualization
        if metric == 'average_reward':
            values = [v/500 for v in values]  # Normalize by max reward
        elif metric == 'average_length':
            values = [v/500 for v in values]  # Normalize by max length
        
        axes[0, 1].bar(x + i*width, values, width, label=label, alpha=0.7)
    
    axes[0, 1].set_xlabel('Agents')
    axes[0, 1].set_ylabel('Normalized Performance')
    axes[0, 1].set_title('Performance Metrics Comparison')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(agent_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training Efficiency (if available)
    if all('convergence_episode' in results for results in agent_results.values()):
        convergence_episodes = [results.get('convergence_episode', float('inf')) 
                              for results in agent_results.values()]
        valid_convergence = [ep for ep in convergence_episodes if ep != float('inf')]
        
        if valid_convergence:
            axes[1, 0].bar(agent_names, convergence_episodes, alpha=0.7)
            axes[1, 0].set_xlabel('Agents')
            axes[1, 0].set_ylabel('Convergence Episode')
            axes[1, 0].set_title('Training Efficiency (Lower is Better)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No convergence data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Efficiency')
    else:
        axes[1, 0].text(0.5, 0.5, 'No convergence data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Efficiency')
    
    # 4. Overall Performance Score
    overall_scores = []
    for results in agent_results.values():
        score = results['success_rate'] * 0.6 + (results['average_reward'] / 500) * 0.4
        overall_scores.append(score)
    
    bars = axes[1, 1].bar(agent_names, overall_scores, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Agents')
    axes[1, 1].set_ylabel('Overall Score')
    axes[1, 1].set_title('Overall Performance Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, overall_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 