"""
Training configuration classes for agent training and evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os


@dataclass
class TrainingConfig:
    """Configuration for training agents."""
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_frequency: int = 50  # Evaluate every N episodes
    save_frequency: int = 100  # Save model every N episodes
    
    # Logging and monitoring
    log_dir: str = 'data/logs'
    model_dir: str = 'data/models'
    results_dir: str = 'data/results'
    use_wandb: bool = False
    wandb_project: str = 'active-inference-cartpole'
    wandb_entity: Optional[str] = None
    
    # Evaluation
    eval_episodes: int = 50
    eval_max_steps: int = 500
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 100  # Episodes without improvement
    min_improvement: float = 0.01
    
    # Checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    max_checkpoints: int = 5
    
    # Performance tracking
    track_metrics: List[str] = None
    plot_frequency: int = 100
    
    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default metrics to track
        if self.track_metrics is None:
            self.track_metrics = [
                'episode_reward',
                'episode_length',
                'success_rate',
                'average_reward',
                'loss',
                'epsilon',  # For DQN
                'belief_entropy',  # For Active Inference
                'free_energy'  # For Active Inference
            ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        return {
            'num_episodes': self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'eval_frequency': self.eval_frequency,
            'save_frequency': self.save_frequency,
            'log_dir': self.log_dir,
            'model_dir': self.model_dir,
            'results_dir': self.results_dir,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'eval_episodes': self.eval_episodes,
            'eval_max_steps': self.eval_max_steps,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_improvement': self.min_improvement,
            'save_best_model': self.save_best_model,
            'save_last_model': self.save_last_model,
            'max_checkpoints': self.max_checkpoints,
            'track_metrics': self.track_metrics,
            'plot_frequency': self.plot_frequency
        }


@dataclass
class ComparisonConfig:
    """Configuration for comparing different agents."""
    
    # Agents to compare
    agents: List[str] = None  # ['dqn', 'active_inference']
    
    # Comparison parameters
    num_episodes: int = 500
    num_runs: int = 5  # Number of independent runs per agent
    confidence_level: float = 0.95
    
    # Metrics to compare
    comparison_metrics: List[str] = None
    
    # Visualization
    plot_comparison: bool = True
    save_plots: bool = True
    plot_format: str = 'png'  # 'png', 'pdf', 'svg'
    
    def __post_init__(self):
        if self.agents is None:
            self.agents = ['dqn', 'active_inference']
        
        if self.comparison_metrics is None:
            self.comparison_metrics = [
                'episode_reward',
                'episode_length',
                'success_rate',
                'average_reward',
                'learning_speed',
                'stability',
                'uncertainty_handling'
            ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get comparison configuration as dictionary."""
        return {
            'agents': self.agents,
            'num_episodes': self.num_episodes,
            'num_runs': self.num_runs,
            'confidence_level': self.confidence_level,
            'comparison_metrics': self.comparison_metrics,
            'plot_comparison': self.plot_comparison,
            'save_plots': self.save_plots,
            'plot_format': self.plot_format
        }


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    
    # Experiment settings
    experiment_name: str = 'active_inference_experiment'
    description: str = 'Active Inference vs DQN comparison on CartPole'
    
    # Training configuration
    training_config: TrainingConfig = None
    
    # Comparison configuration
    comparison_config: ComparisonConfig = None
    
    # Agent configurations
    agent_configs: Dict[str, Any] = None
    
    # Environment configuration
    env_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.training_config is None:
            self.training_config = TrainingConfig()
        
        if self.comparison_config is None:
            self.comparison_config = ComparisonConfig()
        
        if self.agent_configs is None:
            self.agent_configs = {}
        
        if self.env_config is None:
            self.env_config = {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete experiment configuration as dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'training_config': self.training_config.get_config(),
            'comparison_config': self.comparison_config.get_config(),
            'agent_configs': self.agent_configs,
            'env_config': self.env_config
        }


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_COMPARISON_CONFIG = ComparisonConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig() 