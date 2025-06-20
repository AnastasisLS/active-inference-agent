"""
Metrics utilities for the Active Inference Agent project.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import torch


def compute_success_rate(episode_lengths: List[int], threshold: int = 195) -> float:
    """
    Compute the success rate based on episode lengths.
    
    Args:
        episode_lengths: List of episode lengths
        threshold: Minimum length for a successful episode
        
    Returns:
        Success rate as a float between 0 and 1
    """
    if not episode_lengths:
        return 0.0
    
    successful_episodes = sum(1 for length in episode_lengths if length >= threshold)
    return successful_episodes / len(episode_lengths)


def compute_learning_curves(episode_rewards: List[float], window_size: int = 100) -> List[float]:
    """
    Compute learning curves using moving averages.
    
    Args:
        episode_rewards: List of episode rewards
        window_size: Size of the moving average window
        
    Returns:
        List of moving average values
    """
    if len(episode_rewards) < window_size:
        return []
    
    learning_curves = []
    for i in range(window_size, len(episode_rewards) + 1):
        recent_rewards = episode_rewards[i-window_size:i]
        learning_curves.append(np.mean(recent_rewards))
    
    return learning_curves


def compute_agent_statistics(episode_rewards: List[float], episode_lengths: List[int],
                           training_losses: List[float] = None) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for an agent.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        training_losses: List of training losses (optional)
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {}
    
    # Basic statistics
    stats['num_episodes'] = len(episode_rewards)
    stats['total_steps'] = sum(episode_lengths)
    
    # Reward statistics
    stats['average_reward'] = np.mean(episode_rewards)
    stats['std_reward'] = np.std(episode_rewards)
    stats['min_reward'] = np.min(episode_rewards)
    stats['max_reward'] = np.max(episode_rewards)
    stats['median_reward'] = np.median(episode_rewards)
    
    # Length statistics
    stats['average_length'] = np.mean(episode_lengths)
    stats['std_length'] = np.std(episode_lengths)
    stats['min_length'] = np.min(episode_lengths)
    stats['max_length'] = np.max(episode_lengths)
    stats['median_length'] = np.median(episode_lengths)
    
    # Success rate
    stats['success_rate'] = compute_success_rate(episode_lengths)
    
    # Recent performance (last 100 episodes)
    if len(episode_rewards) >= 100:
        recent_rewards = episode_rewards[-100:]
        recent_lengths = episode_lengths[-100:]
        stats['recent_average_reward'] = np.mean(recent_rewards)
        stats['recent_success_rate'] = compute_success_rate(recent_lengths)
    else:
        stats['recent_average_reward'] = stats['average_reward']
        stats['recent_success_rate'] = stats['success_rate']
    
    # Training loss statistics
    if training_losses:
        stats['average_loss'] = np.mean(training_losses)
        stats['std_loss'] = np.std(training_losses)
        stats['min_loss'] = np.min(training_losses)
        stats['max_loss'] = np.max(training_losses)
        stats['final_loss'] = training_losses[-1] if training_losses else None
    
    return stats


def compute_convergence_metrics(episode_rewards: List[float], 
                              convergence_threshold: float = 0.95,
                              window_size: int = 100) -> Dict[str, Any]:
    """
    Compute convergence metrics for training.
    
    Args:
        episode_rewards: List of episode rewards
        convergence_threshold: Threshold for considering convergence (fraction of max reward)
        window_size: Size of the moving average window
        
    Returns:
        Dictionary containing convergence metrics
    """
    if len(episode_rewards) < window_size:
        return {'converged': False, 'convergence_episode': None}
    
    # Assume max reward is 500 for CartPole
    max_reward = 500
    target_reward = max_reward * convergence_threshold
    
    # Compute moving average
    moving_avg = compute_learning_curves(episode_rewards, window_size)
    
    # Find convergence point
    convergence_episode = None
    for i, avg_reward in enumerate(moving_avg):
        if avg_reward >= target_reward:
            convergence_episode = i + window_size
            break
    
    return {
        'converged': convergence_episode is not None,
        'convergence_episode': convergence_episode,
        'final_average_reward': moving_avg[-1] if moving_avg else np.mean(episode_rewards),
        'target_reward': target_reward
    }


def compute_uncertainty_metrics(belief_entropies: List[float] = None,
                              free_energies: List[float] = None) -> Dict[str, Any]:
    """
    Compute uncertainty-related metrics for Active Inference agents.
    
    Args:
        belief_entropies: List of belief state entropies
        free_energies: List of free energy values
        
    Returns:
        Dictionary containing uncertainty metrics
    """
    metrics = {}
    
    if belief_entropies:
        metrics['average_belief_entropy'] = np.mean(belief_entropies)
        metrics['std_belief_entropy'] = np.std(belief_entropies)
        metrics['max_belief_entropy'] = np.max(belief_entropies)
        metrics['min_belief_entropy'] = np.min(belief_entropies)
    
    if free_energies:
        metrics['average_free_energy'] = np.mean(free_energies)
        metrics['std_free_energy'] = np.std(free_energies)
        metrics['max_free_energy'] = np.max(free_energies)
        metrics['min_free_energy'] = np.min(free_energies)
    
    return metrics


def compare_agents(agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple agents based on their statistics.
    
    Args:
        agent_results: Dictionary with agent names as keys and statistics as values
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {}
    
    # Compare success rates
    success_rates = {name: stats['success_rate'] for name, stats in agent_results.items()}
    comparison['success_rate_ranking'] = sorted(success_rates.items(), 
                                              key=lambda x: x[1], reverse=True)
    
    # Compare average rewards
    avg_rewards = {name: stats['average_reward'] for name, stats in agent_results.items()}
    comparison['reward_ranking'] = sorted(avg_rewards.items(), 
                                        key=lambda x: x[1], reverse=True)
    
    # Compare convergence
    if all('convergence_episode' in stats for stats in agent_results.values()):
        convergence_episodes = {name: stats['convergence_episode'] 
                              for name, stats in agent_results.items()}
        # Filter out None values (non-converged agents)
        converged_agents = {name: episode for name, episode in convergence_episodes.items() 
                          if episode is not None}
        if converged_agents:
            comparison['convergence_ranking'] = sorted(converged_agents.items(), 
                                                     key=lambda x: x[1])
    
    # Overall ranking (based on success rate and average reward)
    overall_scores = {}
    for name, stats in agent_results.items():
        score = stats['success_rate'] * 0.6 + (stats['average_reward'] / 500) * 0.4
        overall_scores[name] = score
    
    comparison['overall_ranking'] = sorted(overall_scores.items(), 
                                         key=lambda x: x[1], reverse=True)
    
    return comparison 


def compute_performance_metrics(episode_rewards: List[float], 
                               episode_lengths: List[int],
                               training_losses: List[float] = None,
                               belief_entropies: List[float] = None,
                               free_energies: List[float] = None) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics for an agent.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        training_losses: List of training losses (optional)
        belief_entropies: List of belief entropies (optional)
        free_energies: List of free energies (optional)
        
    Returns:
        Dictionary containing comprehensive performance metrics
    """
    # Basic agent statistics
    basic_stats = compute_agent_statistics(episode_rewards, episode_lengths, training_losses)
    
    # Convergence metrics
    convergence_metrics = compute_convergence_metrics(episode_rewards)
    
    # Uncertainty metrics
    uncertainty_metrics = compute_uncertainty_metrics(belief_entropies, free_energies)
    
    # Combine all metrics
    performance_metrics = {
        **basic_stats,
        **convergence_metrics,
        **uncertainty_metrics
    }
    
    # Additional performance indicators
    if episode_rewards:
        # Learning efficiency
        performance_metrics['learning_efficiency'] = basic_stats['recent_average_reward'] / basic_stats['average_reward'] if basic_stats['average_reward'] > 0 else 0
        
        # Stability (coefficient of variation)
        performance_metrics['reward_stability'] = basic_stats['std_reward'] / basic_stats['average_reward'] if basic_stats['average_reward'] > 0 else float('inf')
        
        # Performance trend (comparing first and last 100 episodes)
        if len(episode_rewards) >= 200:
            first_100 = episode_rewards[:100]
            last_100 = episode_rewards[-100:]
            performance_metrics['performance_improvement'] = np.mean(last_100) - np.mean(first_100)
        else:
            performance_metrics['performance_improvement'] = 0
    
    return performance_metrics 


def batch_cov_trace(cov_batch):
    """Compute the trace of each covariance matrix in a batch."""
    return cov_batch.diagonal(dim1=-2, dim2=-1).sum(-1)

def batch_cov_det(cov_batch):
    """Compute the determinant of each covariance matrix in a batch."""
    return torch.det(cov_batch) 