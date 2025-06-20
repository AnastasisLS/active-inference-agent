"""
Agent configuration classes for DQN and Active Inference agents.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    
    # Network architecture
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    activation: str = 'relu'
    
    # Training parameters
    batch_size: int = 64
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate
    target_update_freq: int = 1000
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]
        
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference agent."""
    
    # Generative model
    state_dim: int = 4
    action_dim: int = 2
    obs_dim: int = 4
    hidden_dim: int = 64
    
    # Belief state
    belief_update_rate: float = 0.1
    belief_entropy_threshold: float = 0.5
    
    # Action planning
    planning_horizon: int = 5
    num_action_samples: int = 10
    temperature: float = 1.0
    
    # Free energy computation
    epistemic_weight: float = 1.0
    pragmatic_weight: float = 1.0
    complexity_weight: float = 1.0
    
    # Learning rates
    transition_lr: float = 0.001
    observation_lr: float = 0.001
    belief_lr: float = 0.01
    
    # Preferred states (for CartPole: balanced state)
    preferred_states: Optional[torch.Tensor] = None
    
    # Device
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set preferred states for CartPole (balanced position)
        if self.preferred_states is None:
            # Preferred state: cart at center, pole upright, low velocities
            self.preferred_states = torch.tensor([0.0, 0.0, 0.0, 0.0], 
                                               device=self.device)


@dataclass
class AgentConfig:
    """Main agent configuration class."""
    
    # Agent type
    agent_type: str = 'dqn'  # 'dqn' or 'active_inference'
    
    # Agent-specific configs
    dqn_config: DQNConfig = None
    active_inference_config: ActiveInferenceConfig = None
    
    # Common parameters
    state_dim: int = 4
    action_dim: int = 2
    seed: int = 42
    
    def __post_init__(self):
        if self.dqn_config is None:
            self.dqn_config = DQNConfig()
        
        if self.active_inference_config is None:
            self.active_inference_config = ActiveInferenceConfig()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for the specified agent type."""
        if self.agent_type == 'dqn':
            return {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dims': self.dqn_config.hidden_dims,
                'learning_rate': self.dqn_config.learning_rate,
                'batch_size': self.dqn_config.batch_size,
                'buffer_size': self.dqn_config.buffer_size,
                'gamma': self.dqn_config.gamma,
                'tau': self.dqn_config.tau,
                'target_update_freq': self.dqn_config.target_update_freq,
                'epsilon_start': self.dqn_config.epsilon_start,
                'epsilon_end': self.dqn_config.epsilon_end,
                'epsilon_decay': self.dqn_config.epsilon_decay,
                'device': self.dqn_config.device,
                'seed': self.seed
            }
        elif self.agent_type == 'active_inference':
            return {
                'state_dim': self.active_inference_config.state_dim,
                'action_dim': self.active_inference_config.action_dim,
                'obs_dim': self.active_inference_config.obs_dim,
                'hidden_dim': self.active_inference_config.hidden_dim,
                'belief_update_rate': self.active_inference_config.belief_update_rate,
                'belief_entropy_threshold': self.active_inference_config.belief_entropy_threshold,
                'planning_horizon': self.active_inference_config.planning_horizon,
                'num_action_samples': self.active_inference_config.num_action_samples,
                'temperature': self.active_inference_config.temperature,
                'epistemic_weight': self.active_inference_config.epistemic_weight,
                'pragmatic_weight': self.active_inference_config.pragmatic_weight,
                'complexity_weight': self.active_inference_config.complexity_weight,
                'transition_lr': self.active_inference_config.transition_lr,
                'observation_lr': self.active_inference_config.observation_lr,
                'belief_lr': self.active_inference_config.belief_lr,
                'preferred_states': self.active_inference_config.preferred_states,
                'device': self.active_inference_config.device,
                'seed': self.seed
            }
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")


# Default configurations
DEFAULT_DQN_CONFIG = DQNConfig()
DEFAULT_ACTIVE_INFERENCE_CONFIG = ActiveInferenceConfig()
DEFAULT_AGENT_CONFIG = AgentConfig() 