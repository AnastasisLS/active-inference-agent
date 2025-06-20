"""
Utility functions for the Active Inference Agent project.

This package contains utility functions for plotting, metrics, visualization, and data logging.
"""

from .plotting import *
from .metrics import *
from .visualization import *
from .data_logging import *
from .device_utils import *

__all__ = [
    # Plotting utilities
    'plot_training_curves',
    'plot_episode_rewards',
    'plot_agent_comparison',
    
    # Metrics utilities
    'compute_success_rate',
    'compute_learning_curves',
    'compute_agent_statistics',
    
    # Visualization utilities
    'create_agent_comparison_plots',
    'visualize_belief_states',
    'plot_uncertainty_analysis',
    
    # Data logging utilities
    'setup_logging',
    'log_training_metrics',
    'save_experiment_results',
    
    # Device utilities
    'get_device',
    'get_device_info',
    'print_device_info',
    'to_device',
    'create_tensor_on_device',
    'get_device_memory_info',
    'print_memory_info',
    'get_global_device',
    'set_global_device',
    'DEVICE'
] 