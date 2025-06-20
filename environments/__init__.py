"""
Environment package for the Active Inference Agent project.

This package contains environment wrappers and implementations.
"""

from .cartpole_env import CartPoleEnv
from .environment_wrapper import EnvironmentWrapper

__all__ = ['CartPoleEnv', 'EnvironmentWrapper'] 