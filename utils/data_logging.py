"""
Data logging utilities for the Active Inference Agent project.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def setup_logging(log_dir: str = "logs") -> str:
    """
    Set up logging directory and return the path.
    
    Args:
        log_dir: Directory name for logs
        
    Returns:
        Path to the logging directory
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(log_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir)
    
    return experiment_dir


def log_training_metrics(episode_rewards: List[float], 
                        episode_lengths: List[int],
                        training_losses: Optional[List[float]] = None,
                        agent_name: str = "agent",
                        log_dir: str = "logs",
                        save_plots: bool = True):
    """
    Log training metrics to files.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        training_losses: List of training losses (optional)
        agent_name: Name of the agent
        log_dir: Directory to save logs
        save_plots: Whether to save plots
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to CSV
    csv_filename = os.path.join(log_dir, f"{agent_name}_metrics_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'reward', 'length']
        if training_losses:
            fieldnames.append('loss')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            row = {'episode': i, 'reward': reward, 'length': length}
            if training_losses and i < len(training_losses):
                row['loss'] = training_losses[i]
            writer.writerow(row)
    
    # Save summary statistics
    summary = {
        'agent_name': agent_name,
        'timestamp': timestamp,
        'num_episodes': len(episode_rewards),
        'total_steps': sum(episode_lengths),
        'average_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'average_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_length': np.min(episode_lengths),
        'max_length': np.max(episode_lengths),
        'success_rate': sum(1 for length in episode_lengths if length >= 195) / len(episode_lengths)
    }
    
    if training_losses:
        summary.update({
            'average_loss': np.mean(training_losses),
            'std_loss': np.std(training_losses),
            'min_loss': np.min(training_losses),
            'max_loss': np.max(training_losses),
            'final_loss': training_losses[-1] if training_losses else None
        })
    
    summary_filename = os.path.join(log_dir, f"{agent_name}_summary_{timestamp}.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training metrics saved to {csv_filename}")
    print(f"Summary saved to {summary_filename}")
    
    return csv_filename, summary_filename


def save_experiment_results(agent_results: Dict[str, Dict[str, Any]],
                           experiment_name: str = "experiment",
                           log_dir: str = "logs"):
    """
    Save comprehensive experiment results.
    
    Args:
        agent_results: Dictionary with agent names as keys and results as values
        experiment_name: Name of the experiment
        log_dir: Directory to save results
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir)
    
    # Save agent results
    for agent_name, results in agent_results.items():
        agent_file = os.path.join(experiment_dir, f"{agent_name}_results.json")
        with open(agent_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Create comparison summary
    comparison_summary = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'num_agents': len(agent_results),
        'agent_names': list(agent_results.keys()),
        'comparison': {}
    }
    
    # Compare success rates
    success_rates = {name: results['success_rate'] for name, results in agent_results.items()}
    comparison_summary['comparison']['success_rates'] = success_rates
    comparison_summary['comparison']['success_rate_ranking'] = sorted(
        success_rates.items(), key=lambda x: x[1], reverse=True
    )
    
    # Compare average rewards
    avg_rewards = {name: results['average_reward'] for name, results in agent_results.items()}
    comparison_summary['comparison']['average_rewards'] = avg_rewards
    comparison_summary['comparison']['reward_ranking'] = sorted(
        avg_rewards.items(), key=lambda x: x[1], reverse=True
    )
    
    # Overall ranking
    overall_scores = {}
    for name, results in agent_results.items():
        score = results['success_rate'] * 0.6 + (results['average_reward'] / 500) * 0.4
        overall_scores[name] = score
    
    comparison_summary['comparison']['overall_scores'] = overall_scores
    comparison_summary['comparison']['overall_ranking'] = sorted(
        overall_scores.items(), key=lambda x: x[1], reverse=True
    )
    
    # Save comparison summary
    comparison_file = os.path.join(experiment_dir, "comparison_summary.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    # Create CSV summary
    csv_data = []
    for agent_name, results in agent_results.items():
        row = {
            'agent_name': agent_name,
            'success_rate': results['success_rate'],
            'average_reward': results['average_reward'],
            'average_length': results['average_length'],
            'total_episodes': results['num_episodes'],
            'total_steps': results['total_steps']
        }
        csv_data.append(row)
    
    csv_file = os.path.join(experiment_dir, "agent_comparison.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print(f"Experiment results saved to {experiment_dir}")
    print(f"Comparison summary: {comparison_file}")
    print(f"CSV summary: {csv_file}")
    
    return experiment_dir


def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """
    Load experiment results from a directory.
    
    Args:
        experiment_dir: Directory containing experiment results
        
    Returns:
        Dictionary containing loaded results
    """
    results = {}
    
    # Load comparison summary
    comparison_file = os.path.join(experiment_dir, "comparison_summary.json")
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            results['comparison'] = json.load(f)
    
    # Load individual agent results
    for filename in os.listdir(experiment_dir):
        if filename.endswith('_results.json'):
            agent_name = filename.replace('_results.json', '')
            agent_file = os.path.join(experiment_dir, filename)
            with open(agent_file, 'r') as f:
                results[agent_name] = json.load(f)
    
    return results


def log_active_inference_metrics(belief_entropies: List[float],
                               free_energies: List[float],
                               action_distributions: List[np.ndarray],
                               agent_name: str = "active_inference_agent",
                               log_dir: str = "logs"):
    """
    Log Active Inference specific metrics.
    
    Args:
        belief_entropies: List of belief state entropies
        free_energies: List of free energy values
        action_distributions: List of action probability arrays
        agent_name: Name of the agent
        log_dir: Directory to save logs
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uncertainty metrics
    uncertainty_data = {
        'belief_entropies': belief_entropies,
        'free_energies': free_energies,
        'average_belief_entropy': np.mean(belief_entropies),
        'std_belief_entropy': np.std(belief_entropies),
        'average_free_energy': np.mean(free_energies),
        'std_free_energy': np.std(free_energies)
    }
    
    uncertainty_file = os.path.join(log_dir, f"{agent_name}_uncertainty_{timestamp}.json")
    with open(uncertainty_file, 'w') as f:
        json.dump(uncertainty_data, f, indent=2)
    
    # Save action distributions
    action_file = os.path.join(log_dir, f"{agent_name}_actions_{timestamp}.csv")
    with open(action_file, 'w', newline='') as csvfile:
        num_actions = action_distributions[0].shape[0] if action_distributions else 0
        fieldnames = ['timestep'] + [f'action_{i}_prob' for i in range(num_actions)]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for t, action_dist in enumerate(action_distributions):
            row = {'timestep': t}
            for i, prob in enumerate(action_dist):
                row[f'action_{i}_prob'] = prob
            writer.writerow(row)
    
    print(f"Active Inference metrics saved:")
    print(f"  Uncertainty: {uncertainty_file}")
    print(f"  Actions: {action_file}")
    
    return uncertainty_file, action_file


def create_experiment_report(experiment_dir: str, output_file: str = None) -> str:
    """
    Create a comprehensive experiment report.
    
    Args:
        experiment_dir: Directory containing experiment results
        output_file: Output file path (optional)
        
    Returns:
        Path to the generated report
    """
    results = load_experiment_results(experiment_dir)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(experiment_dir, f"experiment_report_{timestamp}.md")
    
    with open(output_file, 'w') as f:
        f.write("# Experiment Report\n\n")
        
        if 'comparison' in results:
            comparison = results['comparison']
            f.write(f"## Experiment: {comparison.get('experiment_name', 'Unknown')}\n")
            f.write(f"**Date:** {comparison.get('timestamp', 'Unknown')}\n")
            f.write(f"**Number of Agents:** {comparison.get('num_agents', 0)}\n\n")
            
            # Overall ranking
            f.write("## Overall Performance Ranking\n\n")
            overall_ranking = comparison.get('comparison', {}).get('overall_ranking', [])
            for i, (agent_name, score) in enumerate(overall_ranking, 1):
                f.write(f"{i}. **{agent_name}**: {score:.3f}\n")
            f.write("\n")
            
            # Success rate ranking
            f.write("## Success Rate Ranking\n\n")
            success_ranking = comparison.get('comparison', {}).get('success_rate_ranking', [])
            for i, (agent_name, rate) in enumerate(success_ranking, 1):
                f.write(f"{i}. **{agent_name}**: {rate:.2%}\n")
            f.write("\n")
            
            # Reward ranking
            f.write("## Average Reward Ranking\n\n")
            reward_ranking = comparison.get('comparison', {}).get('reward_ranking', [])
            for i, (agent_name, reward) in enumerate(reward_ranking, 1):
                f.write(f"{i}. **{agent_name}**: {reward:.1f}\n")
            f.write("\n")
        
        # Individual agent details
        f.write("## Individual Agent Details\n\n")
        for agent_name, agent_results in results.items():
            if agent_name == 'comparison':
                continue
                
            f.write(f"### {agent_name}\n\n")
            f.write(f"- **Success Rate:** {agent_results.get('success_rate', 0):.2%}\n")
            f.write(f"- **Average Reward:** {agent_results.get('average_reward', 0):.1f}\n")
            f.write(f"- **Average Length:** {agent_results.get('average_length', 0):.1f}\n")
            f.write(f"- **Total Episodes:** {agent_results.get('num_episodes', 0)}\n")
            f.write(f"- **Total Steps:** {agent_results.get('total_steps', 0)}\n\n")
    
    print(f"Experiment report saved to {output_file}")
    return output_file 