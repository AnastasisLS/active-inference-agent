"""
Base environment wrapper class for the Active Inference Agent project.
"""

import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from config.environment_config import EnvironmentConfig


class EnvironmentWrapper(ABC):
    """
    Base class for environment wrappers.
    
    This class provides a common interface for different environments
    and handles common functionality like state normalization,
    reward shaping, and episode tracking.
    """
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the environment wrapper.
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.env = None
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # State normalization parameters
        self.state_mean = None
        self.state_std = None
        self.normalize_states = config.get_config().get('normalize_states', True)
        
        # Initialize the environment
        self._initialize_environment()
        
        # Set up state normalization if enabled
        if self.normalize_states:
            self._setup_state_normalization()
    
    @abstractmethod
    def _initialize_environment(self):
        """Initialize the specific environment."""
        pass
    
    def _setup_state_normalization(self):
        """Set up state normalization parameters."""
        config = self.config.get_config()
        self.state_mean = config.get('state_mean', np.zeros(self.observation_space.shape[0]))
        self.state_std = config.get('state_std', np.ones(self.observation_space.shape[0]))
        
        # Avoid division by zero
        self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self.env.reset(seed=seed)
        
        observation, info = self.env.reset()
        
        # Normalize state if enabled
        if self.normalize_states:
            observation = self._normalize_state(observation)
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Normalize state if enabled
        if self.normalize_states:
            observation = self._normalize_state(observation)
        
        # Apply reward shaping if enabled
        if self.config.get_config().get('reward_shaping', False):
            reward = self._shape_reward(observation, reward)
        
        # Update episode tracking
        self.total_steps += 1
        
        # Check if episode is done
        done = terminated or truncated
        if done:
            self.episode_count += 1
        
        return observation, reward, terminated, truncated, info
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize the state vector.
        
        Args:
            state: Raw state vector
            
        Returns:
            Normalized state vector
        """
        if not self.normalize_states:
            return state
        
        return (state - self.state_mean) / self.state_std
    
    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Denormalize the state vector.
        
        Args:
            normalized_state: Normalized state vector
            
        Returns:
            Denormalized state vector
        """
        if not self.normalize_states:
            return normalized_state
        
        return normalized_state * self.state_std + self.state_mean
    
    def _shape_reward(self, observation: np.ndarray, reward: float) -> float:
        """
        Apply reward shaping.
        
        Args:
            observation: Current observation
            reward: Original reward
            
        Returns:
            Shaped reward
        """
        # Default implementation: no shaping
        return reward * self.config.get_config().get('reward_scale', 1.0)
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        if self.env:
            self.env.close()
    
    @property
    def observation_space(self) -> gym.Space:
        """Get the observation space."""
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.Space:
        """Get the action space."""
        return self.env.action_space
    
    def get_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get state space bounds."""
        return self.config.get_state_bounds()
    
    def get_action_space(self) -> Tuple[int, int]:
        """Get action space."""
        return self.config.get_action_space()
    
    def get_success_threshold(self) -> int:
        """Get success threshold for episodes."""
        return self.config.get_config().get('success_threshold', 195)
    
    def is_successful_episode(self, episode_length: int) -> bool:
        """
        Check if an episode was successful.
        
        Args:
            episode_length: Length of the episode
            
        Returns:
            True if episode was successful
        """
        return episode_length >= self.get_success_threshold()
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.episode_rewards:
            return {
                'episode_count': 0,
                'total_steps': 0,
                'average_reward': 0.0,
                'average_length': 0.0,
                'success_rate': 0.0
            }
        
        successful_episodes = sum(
            1 for length in self.episode_lengths 
            if self.is_successful_episode(length)
        )
        
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'average_reward': np.mean(self.episode_rewards),
            'average_length': np.mean(self.episode_lengths),
            'success_rate': successful_episodes / len(self.episode_lengths)
        }
    
    def record_episode(self, episode_reward: float, episode_length: int):
        """
        Record episode statistics.
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 