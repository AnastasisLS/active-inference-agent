"""
Configuration package for the Active Inference Agent project.

This package contains configuration classes for agents, environments, and training.
"""

from .agent_config import AgentConfig
from .environment_config import EnvironmentConfig
from .training_config import TrainingConfig

__all__ = ['AgentConfig', 'EnvironmentConfig', 'TrainingConfig'] 