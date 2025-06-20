"""
Base agent class for the Active Inference Agent project.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import random

from utils.device_utils import get_device


class BaseAgent(ABC):
    """
    Base class for all agents in the Active Inference Agent project.
    
    This class provides common functionality for agent training, evaluation,
    and interaction with environments.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Agent configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Set device using device utilities
        device_name = config.get('device', 'auto')
        self.device = get_device(device_name)
        
        # Set random seed
        self.seed = config.get('seed', 42)
        self._set_seed()
        
        # Training state
        self.training = True
        self.episode_count = 0
        self.total_steps = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.evaluation_results = []
        
        # Initialize the agent
        self._initialize_agent()
    
    @abstractmethod
    def _initialize_agent(self):
        """Initialize the specific agent implementation."""
        pass
    
    def _set_seed(self):
        """Set random seed for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update the agent with experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass
    
    def train(self, env, episodes: int = 1000, eval_frequency: int = 50):
        """
        Train the agent on the given environment.
        
        Args:
            env: Environment to train on
            episodes: Number of episodes to train for
            eval_frequency: Frequency of evaluation
        """
        self.training = True
        
        for episode in range(episodes):
            episode_reward, episode_length = self._train_episode(env)
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Episode Length: {episode_length}")
            
            # Evaluate periodically
            if (episode + 1) % eval_frequency == 0:
                eval_reward = self.evaluate(env, episodes=10)
                self.evaluation_results.append({
                    'episode': episode + 1,
                    'eval_reward': eval_reward
                })
                print(f"Evaluation at episode {episode + 1}: "
                      f"Average reward = {eval_reward:.2f}")
    
    def _train_episode(self, env) -> Tuple[float, int]:
        """
        Train for a single episode.
        
        Args:
            env: Environment to train on
            
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action = self.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update agent
            self.update(state, action, reward, next_state, done)
            
            # Update episode tracking
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Move to next state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        return episode_reward, episode_length
    
    def evaluate(self, env, episodes: int = 10) -> float:
        """
        Evaluate the agent on the given environment.
        
        Args:
            env: Environment to evaluate on
            episodes: Number of episodes for evaluation
            
        Returns:
            Average reward across evaluation episodes
        """
        self.training = False
        eval_rewards = []
        
        for _ in range(episodes):
            episode_reward, _ = self._evaluate_episode(env)
            eval_rewards.append(episode_reward)
        
        self.training = True
        return np.mean(eval_rewards)
    
    def _evaluate_episode(self, env) -> Tuple[float, int]:
        """
        Evaluate for a single episode.
        
        Args:
            env: Environment to evaluate on
            
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action = self.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update episode tracking
            episode_reward += reward
            episode_length += 1
            
            # Move to next state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        return episode_reward, episode_length
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        if not self.episode_rewards:
            return {
                'episode_count': 0,
                'total_steps': 0,
                'average_reward': 0.0,
                'average_length': 0.0,
                'best_reward': 0.0,
                'recent_average_reward': 0.0
            }
        
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'average_reward': np.mean(self.episode_rewards),
            'average_length': np.mean(self.episode_lengths),
            'best_reward': np.max(self.episode_rewards),
            'recent_average_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'device': str(self.device)
        }
    
    def save_model(self, filepath: str):
        """
        Save the agent model to a file.
        
        Args:
            filepath: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, filepath: str):
        """
        Load the agent model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        raise NotImplementedError("Subclasses must implement load_model")
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Action probabilities
        """
        raise NotImplementedError("Subclasses must implement get_action_distribution")
    
    def get_value_estimate(self, state: np.ndarray) -> float:
        """
        Get the value estimate for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Value estimate
        """
        raise NotImplementedError("Subclasses must implement get_value_estimate")
    
    def reset_training_stats(self):
        """Reset all training statistics."""
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.evaluation_results = [] 