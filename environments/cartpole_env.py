"""
CartPole environment wrapper for the Active Inference Agent project.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .environment_wrapper import EnvironmentWrapper
from config.environment_config import EnvironmentConfig, CartPoleConfig


class CartPoleEnv(EnvironmentWrapper):
    """
    CartPole environment wrapper.
    
    This wrapper provides a standardized interface for the CartPole environment
    with additional features like state normalization, reward shaping, and
    episode tracking.
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the CartPole environment wrapper.
        
        Args:
            config: Environment configuration (optional)
        """
        if config is None:
            config = EnvironmentConfig(env_type='cartpole')
        
        super().__init__(config)
        self.partial_observability = self.config.cartpole_config.partial_observability
    
    def _initialize_environment(self):
        """Initialize the CartPole environment."""
        config = self.config.get_config()
        env_name = config.get('env_name', 'CartPole-v1')
        render_mode = config.get('render_mode')
        seed = config.get('seed', 42)
        
        # Create the environment
        self.env = gym.make(env_name, render_mode=render_mode)
        
        # Set the seed for reproducibility
        if seed is not None:
            self.env.reset(seed=seed)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.partial_observability:
            # Return only position and angle
            partial_obs = np.array([observation[0], observation[2]])
            return partial_obs, reward, terminated, truncated, info
            
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        observation, info = self.env.reset(seed=seed, options=options)
        
        if self.partial_observability:
            # Return only position and angle
            partial_obs = np.array([observation[0], observation[2]])
            return partial_obs, info
            
        return observation, info
    
    def _shape_reward(self, observation: np.ndarray, reward: float) -> float:
        """
        Apply reward shaping for CartPole.
        
        This implementation provides additional reward signals based on
        the state of the cart and pole to encourage better behavior.
        
        Args:
            observation: Current observation [cart_pos, cart_vel, pole_angle, pole_vel]
            reward: Original reward (1.0 for each step)
            
        Returns:
            Shaped reward
        """
        if not self.config.get_config().get('reward_shaping', False):
            return reward
        
        # Extract state components
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        # Base reward
        shaped_reward = reward
        
        # Penalty for being far from center
        center_penalty = -0.1 * abs(cart_pos)
        shaped_reward += center_penalty
        
        # Penalty for high cart velocity
        velocity_penalty = -0.05 * abs(cart_vel)
        shaped_reward += velocity_penalty
        
        # Penalty for pole angle deviation from upright
        angle_penalty = -0.1 * abs(pole_angle)
        shaped_reward += angle_penalty
        
        # Penalty for high pole angular velocity
        angular_velocity_penalty = -0.05 * abs(pole_vel)
        shaped_reward += angular_velocity_penalty
        
        # Apply reward scale
        reward_scale = self.config.get_config().get('reward_scale', 1.0)
        shaped_reward *= reward_scale
        
        return shaped_reward
    
    def get_state_info(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Get detailed information about the current state.
        
        Args:
            observation: Current observation
            
        Returns:
            Dictionary with state information
        """
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        return {
            'cart_position': cart_pos,
            'cart_velocity': cart_vel,
            'pole_angle': pole_angle,
            'pole_angular_velocity': pole_vel,
            'distance_from_center': abs(cart_pos),
            'pole_deviation': abs(pole_angle),
            'total_energy': abs(cart_vel) + abs(pole_vel)
        }
    
    def is_balanced_state(self, observation: np.ndarray, tolerance: float = 0.1) -> bool:
        """
        Check if the cart-pole system is in a balanced state.
        
        Args:
            observation: Current observation
            tolerance: Tolerance for considering the state balanced
            
        Returns:
            True if the system is balanced
        """
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        # Check if cart is near center
        cart_centered = abs(cart_pos) < tolerance
        
        # Check if pole is near upright
        pole_upright = abs(pole_angle) < tolerance
        
        # Check if velocities are low
        low_velocities = abs(cart_vel) < tolerance and abs(pole_vel) < tolerance
        
        return cart_centered and pole_upright and low_velocities
    
    def get_episode_quality_score(self, episode_observations: list) -> float:
        """
        Calculate a quality score for an episode based on how well
        the cart-pole system was balanced.
        
        Args:
            episode_observations: List of observations from the episode
            
        Returns:
            Quality score between 0 and 1
        """
        if not episode_observations:
            return 0.0
        
        balanced_steps = 0
        total_steps = len(episode_observations)
        
        for obs in episode_observations:
            if self.is_balanced_state(obs):
                balanced_steps += 1
        
        return balanced_steps / total_steps
    
    def get_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get state space bounds for CartPole."""
        return {
            'cart_position': (-4.8, 4.8),
            'cart_velocity': (-np.inf, np.inf),
            'pole_angle': (-0.418, 0.418),  # radians
            'pole_angular_velocity': (-np.inf, np.inf)
        }
    
    def get_action_space(self) -> Tuple[int, int]:
        """Get action space for CartPole."""
        return (0, 1)  # left, right
    
    def get_success_threshold(self) -> int:
        """Get success threshold for CartPole episodes."""
        return 195  # Episodes lasting > 195 steps are considered successful
    
    def render_episode(self, agent, max_steps: int = 500, render_mode: str = 'human'):
        """
        Render an episode with the given agent.
        
        Args:
            agent: Agent to use for the episode
            max_steps: Maximum number of steps
            render_mode: Rendering mode
        """
        observation, info = self.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Render the current state
            self.render(mode=render_mode)
            
            # Agent selects action
            action = agent.select_action(observation)
            
            # Take step in environment
            observation, reward, terminated, truncated, info = self.step(action)
            total_reward += reward
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        print(f"Episode finished after {step + 1} steps with total reward: {total_reward}")
        self.close()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the CartPole environment.
        
        Returns:
            Dictionary with environment information
        """
        return {
            'environment_name': 'CartPole-v1',
            'state_dimension': 4, # Full state dimension
            'observation_dimension': 2 if self.partial_observability else 4,
            'action_dimension': 2,
            'state_components': ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'],
            'observation_components': ['cart_position', 'pole_angle'] if self.partial_observability else ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'],
            'actions': ['left', 'right'],
            'success_threshold': self.get_success_threshold(),
            'max_steps': self.config.get_config().get('max_steps', 500),
            'state_bounds': self.get_state_bounds(),
            'normalize_states': self.normalize_states,
            'reward_shaping': self.config.get_config().get('reward_shaping', False)
        }


# Convenience function for creating CartPole environment
def create_cartpole_env(config: Optional[CartPoleConfig] = None, **kwargs) -> CartPoleEnv:
    """
    Create a CartPole environment with specified settings.
    
    Args:
        config: A CartPoleConfig object.
        **kwargs: Overrides for config (e.g., render_mode, seed).
        
    Returns:
        CartPole environment wrapper
    """
    if config is None:
        config = CartPoleConfig()

    # Apply kwargs to config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    env_config = EnvironmentConfig(
        env_type='cartpole',
        cartpole_config=config,
        render_mode=kwargs.get('render_mode'),
        seed=kwargs.get('seed', 42)
    )
    
    return CartPoleEnv(env_config) 