"""
Agents package for the Active Inference Agent project.

This package contains agent implementations including DQN and Active Inference agents.
"""

from .base_agent import BaseAgent
from .dqn_agent import DQNAgent

# Import Active Inference agent if available (will be implemented in Week 2-3)
try:
    from .active_inference_agent import ActiveInferenceAgent
    __all__ = ['BaseAgent', 'DQNAgent', 'ActiveInferenceAgent']
except ImportError:
    # Active Inference agent not yet implemented
    __all__ = ['BaseAgent', 'DQNAgent'] 