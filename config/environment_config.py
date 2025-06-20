"""
Environment configuration classes for different environments.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class CartPoleConfig:
    """Configuration for CartPole environment."""
    
    # Environment name
    env_name: str = 'CartPole-v1'
    
    # State space bounds
    cart_position_bounds: Tuple[float, float] = (-4.8, 4.8)
    cart_velocity_bounds: Tuple[float, float] = (-np.inf, np.inf)
    pole_angle_bounds: Tuple[float, float] = (-0.418, 0.418)  # radians
    pole_angular_velocity_bounds: Tuple[float, float] = (-np.inf, np.inf)
    
    # Action space
    action_space: Tuple[int, int] = (0, 1)  # left, right
    
    # Episode settings
    max_steps: int = 500
    success_threshold: int = 195  # Episodes lasting > 195 steps are considered successful
    
    # State normalization
    normalize_states: bool = True
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    
    # Partial observability
    partial_observability: bool = True
    
    # Reward shaping
    reward_shaping: bool = False
    reward_scale: float = 1.0
    
    # State discretization (for active inference)
    discretize_states: bool = False
    discretization_levels: int = 10
    
    def __post_init__(self):
        # Set default state normalization parameters
        if self.state_mean is None:
            # Approximate means for CartPole states
            self.state_mean = np.array([0.0, 0.0, 0.0, 0.0])
        
        if self.state_std is None:
            # Approximate standard deviations for CartPole states
            self.state_std = np.array([2.0, 2.0, 0.2, 2.0])


@dataclass
class EnvironmentConfig:
    """Main environment configuration class."""
    
    # Environment type
    env_type: str = 'cartpole'  # 'cartpole', 'mountaincar', etc.
    
    # Environment-specific configs
    cartpole_config: CartPoleConfig = None
    
    # Common parameters
    render_mode: Optional[str] = None  # 'human', 'rgb_array', None
    seed: int = 42
    
    def __post_init__(self):
        if self.cartpole_config is None:
            self.cartpole_config = CartPoleConfig()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for the specified environment type."""
        if self.env_type == 'cartpole':
            return {
                'env_name': self.cartpole_config.env_name,
                'cart_position_bounds': self.cartpole_config.cart_position_bounds,
                'cart_velocity_bounds': self.cartpole_config.cart_velocity_bounds,
                'pole_angle_bounds': self.cartpole_config.pole_angle_bounds,
                'pole_angular_velocity_bounds': self.cartpole_config.pole_angular_velocity_bounds,
                'action_space': self.cartpole_config.action_space,
                'max_steps': self.cartpole_config.max_steps,
                'success_threshold': self.cartpole_config.success_threshold,
                'normalize_states': self.cartpole_config.normalize_states,
                'state_mean': self.cartpole_config.state_mean,
                'state_std': self.cartpole_config.state_std,
                'reward_shaping': self.cartpole_config.reward_shaping,
                'reward_scale': self.cartpole_config.reward_scale,
                'discretize_states': self.cartpole_config.discretize_states,
                'discretization_levels': self.cartpole_config.discretization_levels,
                'render_mode': self.render_mode,
                'seed': self.seed
            }
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")
    
    def get_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get state space bounds for the environment."""
        if self.env_type == 'cartpole':
            return {
                'cart_position': self.cartpole_config.cart_position_bounds,
                'cart_velocity': self.cartpole_config.cart_velocity_bounds,
                'pole_angle': self.cartpole_config.pole_angle_bounds,
                'pole_angular_velocity': self.cartpole_config.pole_angular_velocity_bounds
            }
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")
    
    def get_action_space(self) -> Tuple[int, int]:
        """Get action space for the environment."""
        if self.env_type == 'cartpole':
            return self.cartpole_config.action_space
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")


# Default configurations
DEFAULT_CARTPOLE_CONFIG = CartPoleConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig() 