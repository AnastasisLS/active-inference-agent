"""
Active Inference Agent

Implements an advanced active inference agent that minimizes free energy through:
1. Perception: Bayesian belief update with Kalman filtering
2. Action: Expected free energy minimization with planning
3. Learning: Generative model adaptation and preference learning

References:
- Friston et al., 2015, "Active Inference and Learning"
- Millidge et al., 2021, "On the Relationship Between Active Inference and Control as Inference"
- Friston et al., 2017, "Active Inference: A Process Theory"
- Parr et al., 2019, "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings

from agents.base_agent import BaseAgent
from models.generative_model import GenerativeModel
from utils.active_inference_utils import (
    compute_variational_free_energy,
    compute_expected_free_energy,
    update_belief_bayesian,
    select_action_active_inference,
    PreferenceModel
)
from utils.device_utils import get_device, safe_tensor_creation

logger = logging.getLogger(__name__)

class BeliefEncoder(nn.Module):
    """
    Neural network for encoding observations into belief states.
    """
    def __init__(self, obs_dim: int, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim * 2)  # Mean and log variance
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation into belief parameters.
        
        Args:
            obs: (batch, obs_dim) Observation tensor
            
        Returns:
            mean: (batch, state_dim) Belief mean
            log_var: (batch, state_dim) Belief log variance
        """
        encoded = self.encoder(obs)
        mean = encoded[:, :self.state_dim]
        log_var = encoded[:, self.state_dim:]
        return mean, log_var

class ActionNetwork(nn.Module):
    """
    Neural network for action selection based on belief state.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.action_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits from belief state.
        
        Args:
            belief: (batch, state_dim) Belief state
            
        Returns:
            logits: (batch, action_dim) Action logits
        """
        return self.action_net(belief)

class ActiveInferenceAgent(BaseAgent):
    """
    Advanced Active Inference Agent for CartPole control.
    
    The agent maintains:
    - Belief state: q(s) - approximate posterior over states (Gaussian)
    - Generative model: p(o,s) - observation and state model
    - Preference model: p(o) - desired observations
    - Neural networks for belief encoding and action selection
    
    Features:
    - Kalman filter belief update
    - Multi-step planning with expected free energy
    - Adaptive temperature for exploration
    - Belief state regularization
    - Comprehensive logging and statistics
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        obs_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        belief_lr: float = 0.1,
        temperature: float = 1.0,
        planning_horizon: int = 3,
        device: Optional[str] = None,
        **kwargs
    ):
        # Create config dict for base class
        config = {
            'device': device,
            'seed': kwargs.get('seed', 42),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'obs_dim': obs_dim,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'belief_lr': belief_lr,
            'temperature': temperature,
            'planning_horizon': planning_horizon,
            **kwargs
        }
        
        super().__init__(state_dim, action_dim, config)
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.belief_lr = belief_lr
        self.temperature = temperature
        self.planning_horizon = planning_horizon
        
        # Initialize components
        self.device = get_device(device)
        
        # Generative model
        self.generative_model = GenerativeModel(
            state_dim, action_dim, obs_dim, self.device
        )
        
        # Preference model
        self.preference_model = PreferenceModel(obs_dim, self.device)
        
        # Belief state (mean, cov)
        self.belief = None  # (mean, cov)
        self.belief_history = []
        
        # Neural networks for belief encoding and action selection
        self.belief_encoder = BeliefEncoder(obs_dim, state_dim, hidden_dim).to(self.device)
        self.action_network = ActionNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.belief_optimizer = torch.optim.Adam(
            self.belief_encoder.parameters(), lr=belief_lr, weight_decay=1e-4
        )
        self.action_optimizer = torch.optim.Adam(
            self.action_network.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.belief_scheduler = torch.optim.lr_scheduler.StepLR(
            self.belief_optimizer, step_size=100, gamma=0.95
        )
        self.action_scheduler = torch.optim.lr_scheduler.StepLR(
            self.action_optimizer, step_size=100, gamma=0.95
        )
        
        # Training statistics
        self.vfe_history = []
        self.efe_history = []
        self.belief_entropy_history = []
        self.action_entropy_history = []
        self.reward_history = []
        
        # Adaptive temperature for exploration
        self.temperature_decay = 0.995
        self.min_temperature = 0.1
        
        # Experience buffer for belief learning
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        logger.info(f"Active Inference Agent initialized on {self.device}")
    
    def _initialize_agent(self):
        """Initialize the active inference agent components."""
        # This is called by the base class constructor
        # Most initialization is done in __init__, but we can add any additional setup here
        pass
    
    def initialize_belief(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the belief state (mean, covariance) given the first partial observation.
        Args:
            obs: (batch, obs_dim) Partial observation tensor (pos, angle)
        Returns:
            belief: Tuple (mean, cov)
        """
        batch_size = obs.shape[0]
        # Initialize belief mean with observations, setting unobserved velocities to zero
        mean = torch.zeros(batch_size, self.state_dim, device=self.device)
        if obs.shape[1] == self.obs_dim: # obs_dim should be 2
            mean[:, 0] = obs[:, 0]  # cart_position
            mean[:, 2] = obs[:, 1]  # pole_angle
        
        # Set initial covariance to identity * 1.0 (high uncertainty)
        cov = torch.eye(self.state_dim, device=self.device).unsqueeze(0).expand(batch_size, self.state_dim, self.state_dim) * 1.0
        
        print(f"[DEBUG] Initial belief mean: {mean.cpu().numpy()}")
        print(f"[DEBUG] Initial belief covariance: {cov[0].cpu().numpy()}")
        self.belief = (mean, cov)
        logger.debug(f"Initialized Gaussian belief: mean shape {mean.shape}, cov shape {cov.shape}")
        return self.belief
    
    def update_belief(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update belief state using Kalman filter update.
        
        Args:
            obs: (batch, obs_dim) Observation tensor
            action: (batch,) Action tensor
            
        Returns:
            belief: Updated belief tuple (mean, cov)
        """
        if self.belief is None:
            self.belief = self.initialize_belief(obs)
        
        # Update belief using Kalman filter
        mean, cov = update_belief_bayesian(self.belief, obs, action, self.generative_model)
        
        # Clamp covariance diagonal for stability (MPS-compatible)
        min_var = 1e-8
        for i in range(cov.shape[0]):
            diag = torch.diag(cov[i])
            # Only clamp if non-finite or negative
            diag = torch.where(torch.isfinite(diag) & (diag > min_var), diag, torch.full_like(diag, min_var))
            cov[i] = torch.diag(diag)
        
        # Check for NaNs in mean or covariance
        if not torch.isfinite(mean).all() or not torch.isfinite(cov).all():
            mean = torch.zeros_like(mean)
            cov = torch.eye(self.state_dim, device=self.device).unsqueeze(0).expand_as(cov).clone()
        
        self.belief = (mean, cov)
        
        # Store belief history
        self.belief_history.append((
            mean.detach().cpu().numpy(), 
            cov.detach().cpu().numpy()
        ))
        # Store trace and determinant for plotting
        cov_trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu().numpy()
        try:
            cov_det = torch.det(cov).detach().cpu().numpy()
        except Exception:
            cov_det = np.full((cov.shape[0],), np.nan)
        if not hasattr(self, 'cov_trace_history'):
            self.cov_trace_history = []
        if not hasattr(self, 'cov_det_history'):
            self.cov_det_history = []
        self.cov_trace_history.append(cov_trace)
        self.cov_det_history.append(cov_det)
        # Debug print
        print(f"[DEBUG] Cov trace: {cov_trace}, Cov det: {cov_det}")
        return self.belief
    
    def compute_vfe(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute Variational Free Energy for current belief and observation.
        
        Args:
            obs: (batch, obs_dim) Observation tensor
            
        Returns:
            vfe: (batch,) Variational free energy
        """
        if self.belief is None:
            self.belief = self.initialize_belief(obs)
        
        mean, _ = self.belief
        vfe = compute_variational_free_energy(mean, obs, self.generative_model)
        self.vfe_history.append(vfe.detach().cpu().numpy())
        return vfe
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action given the current state (base class interface).
        
        Args:
            state: Current state observation as numpy array
            
        Returns:
            Selected action
        """
        # Convert numpy array to torch tensor safely
        obs = safe_tensor_creation(state, self.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
        
        # Use the existing select_action method
        action = self.select_action_torch(obs)
        
        return action
    
    def select_action_torch(self, obs: torch.Tensor, available_actions: Optional[List[int]] = None) -> int:
        """
        Select action by minimizing expected free energy (torch interface).
        
        Args:
            obs: (batch, obs_dim) Observation tensor
            available_actions: List of available actions (if None, use all actions)
            
        Returns:
            action: Selected action index
        """
        # Use previous action if available, else zeros
        if not hasattr(self, 'last_action') or self.last_action is None:
            self.last_action = torch.zeros((obs.shape[0] if obs.dim() > 1 else 1,), 
                                         dtype=torch.long, device=self.device)
        
        # Update belief
        self.belief = self.update_belief(obs, self.last_action)
        
        if available_actions is None:
            available_actions = list(range(self.action_dim))
        
        mean, _ = self.belief
        
        # Select action using expected free energy
        action, efe_values = select_action_active_inference(
            mean,
            self.generative_model,
            self.preference_model,
            available_actions,
            temperature=self.temperature,
            horizon=self.planning_horizon
        )
        
        # Store last action for next update
        self.last_action = torch.tensor([action], device=self.device)
        
        # Store EFE for statistics
        self.efe_history.append(efe_values.detach().cpu().numpy())
        
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update the agent with experience (base class interface).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert numpy arrays to torch tensors
        obs = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([done], dtype=torch.bool, device=self.device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
            reward_tensor = reward_tensor.unsqueeze(0)
            done_tensor = done_tensor.unsqueeze(0)
        
        # Create experience dict
        experience = {
            'obs': obs,
            'action': action_tensor,
            'reward': reward_tensor,
            'next_obs': next_obs,
            'done': done_tensor
        }
        
        # Use the existing update method
        self.update_torch(experience)
    
    def gaussian_entropy(self, cov: torch.Tensor) -> float:
        """
        Compute the entropy of a multivariate Gaussian with covariance cov.
        Args:
            cov: (batch, state_dim, state_dim) covariance matrix
        Returns:
            entropy: float (mean over batch)
        """
        state_dim = cov.shape[-1]
        # Clamp only if non-finite or negative
        for i in range(cov.shape[0]):
            diag = torch.diag(cov[i])
            diag = torch.where(torch.isfinite(diag) & (diag > 1e-8), diag, torch.full_like(diag, 1e-8))
            cov[i] = torch.diag(diag)
        cov = cov + torch.eye(state_dim, device=cov.device) * 1e-8
        logdet = torch.logdet(cov)
        if not torch.isfinite(logdet).all():
            print(f"[WARNING] Non-finite logdet in entropy calculation: {logdet}")
            logdet = torch.where(torch.isfinite(logdet), logdet, torch.tensor(0.0, device=logdet.device))
        entropy = 0.5 * (state_dim * (1.0 + np.log(2 * np.pi)) + logdet)
        print(f"[DEBUG] Belief entropy: {entropy.mean().item():.4f}, Cov diag: {torch.diag(cov[0]).cpu().numpy()}, Mean: {cov.mean().item():.4f}")
        return entropy.mean().item()

    def update_torch(self, experience: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent parameters based on experience (torch interface).
        
        Args:
            experience: Dictionary containing experience data
            
        Returns:
            Dict containing training statistics
        """
        obs = experience['obs'].to(self.device)
        action = experience['action'].to(self.device)
        reward = experience['reward'].to(self.device)
        next_obs = experience['next_obs'].to(self.device)
        done = experience['done'].to(self.device)
        
        # Store experience in buffer
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Update belief with current observation and action
        self.belief = self.update_belief(obs, action)
        
        # Detach belief state after update to prevent autograd from tracking across steps
        self.belief = (self.belief[0].detach(), self.belief[1].detach())
        
        # Compute VFE for current belief and observation
        vfe = self.compute_vfe(obs)
        
        mean, cov = self.belief
        action_logits = self.action_network(mean)
        action_probs = F.softmax(action_logits / self.temperature, dim=-1)
        
        # Compute action entropy for regularization
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        # Compute belief entropy (Gaussian)
        belief_entropy = self.gaussian_entropy(cov)
        self.belief_entropy_history.append(belief_entropy)
        self.action_entropy_history.append(action_entropy.detach().cpu().item())
        self.reward_history.append(reward.detach().cpu().numpy())
        
        # Action loss based on EFE (simplified)
        efe = compute_expected_free_energy(
            mean, action, self.generative_model, self.preference_model, horizon=1
        )
        action_loss = efe.mean() - 0.01 * action_entropy  # Encourage exploration
        
        # Combine losses and do a single backward pass
        total_loss = vfe.mean() + action_loss
        self.belief_optimizer.zero_grad()
        self.action_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.action_network.parameters(), max_norm=1.0)
        self.belief_optimizer.step()
        self.action_optimizer.step()
        
        # Update learning rate schedulers
        self.belief_scheduler.step()
        self.action_scheduler.step()
        
        # Update temperature for exploration
        self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)
        
        return {
            'vfe_loss': vfe.mean().item(),
            'action_loss': action_loss.item(),
            'belief_entropy': belief_entropy,
            'action_entropy': action_entropy.item(),
            'efe': efe.mean().item(),
            'temperature': self.temperature
        }
    
    def get_belief_state(self) -> torch.Tensor:
        """
        Get current belief state.
        
        Returns:
            Current belief mean
        """
        if self.belief is None:
            return torch.zeros(self.state_dim, device=self.device)
        return self.belief[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'vfe_history': self.vfe_history,
            'efe_history': self.efe_history,
            'belief_entropy_history': self.belief_entropy_history,
            'action_entropy_history': self.action_entropy_history,
            'belief_history': self.belief_history,
            'current_belief': self.belief,
            'temperature': self.temperature,
            'experience_buffer_size': len(self.experience_buffer)
        }
        
        # Compute recent averages
        if len(self.vfe_history) > 0:
            stats['recent_vfe'] = np.mean(self.vfe_history[-100:])
        if len(self.efe_history) > 0:
            stats['recent_efe'] = np.mean(self.efe_history[-100:])
        if len(self.reward_history) > 0:
            stats['recent_reward'] = np.mean(self.reward_history[-100:])
        
        return stats
    
    def save(self, path: str):
        """
        Save agent state to file.
        
        Args:
            path: Path to save the agent
        """
        state_dict = {
            'belief_encoder_state_dict': self.belief_encoder.state_dict(),
            'action_network_state_dict': self.action_network.state_dict(),
            'belief_optimizer_state_dict': self.belief_optimizer.state_dict(),
            'action_optimizer_state_dict': self.action_optimizer.state_dict(),
            'belief_scheduler_state_dict': self.belief_scheduler.state_dict(),
            'action_scheduler_state_dict': self.action_scheduler.state_dict(),
            'belief': self.belief,
            'temperature': self.temperature,
            'vfe_history': self.vfe_history,
            'efe_history': self.efe_history,
            'belief_entropy_history': self.belief_entropy_history,
            'action_entropy_history': self.action_entropy_history,
            'belief_history': self.belief_history,
            'reward_history': self.reward_history,
            'experience_buffer': self.experience_buffer,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'obs_dim': self.obs_dim,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'belief_lr': self.belief_lr,
                'temperature': self.temperature,
                'planning_horizon': self.planning_horizon
            }
        }
        
        torch.save(state_dict, path)
        logger.info(f"Active Inference Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent state from file.
        
        Args:
            path: Path to load the agent from
        """
        state_dict = torch.load(path, map_location=self.device)
        
        self.belief_encoder.load_state_dict(state_dict['belief_encoder_state_dict'])
        self.action_network.load_state_dict(state_dict['action_network_state_dict'])
        self.belief_optimizer.load_state_dict(state_dict['belief_optimizer_state_dict'])
        self.action_optimizer.load_state_dict(state_dict['action_optimizer_state_dict'])
        self.belief_scheduler.load_state_dict(state_dict['belief_scheduler_state_dict'])
        self.action_scheduler.load_state_dict(state_dict['action_scheduler_state_dict'])
        
        self.belief = state_dict['belief']
        self.temperature = state_dict['temperature']
        self.vfe_history = state_dict['vfe_history']
        self.efe_history = state_dict['efe_history']
        self.belief_entropy_history = state_dict['belief_entropy_history']
        self.action_entropy_history = state_dict['action_entropy_history']
        self.belief_history = state_dict['belief_history']
        self.reward_history = state_dict['reward_history']
        self.experience_buffer = state_dict['experience_buffer']
        
        logger.info(f"Active Inference Agent loaded from {path}")
    
    def reset(self):
        """
        Reset agent state for new episode.
        """
        self.belief = None
        self.last_action = None
        logger.debug("Active Inference Agent reset for new episode")
    
    def get_agent_info(self) -> str:
        """
        Get information about the agent.
        
        Returns:
            String containing agent information
        """
        return (f"ActiveInferenceAgent(state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, obs_dim={self.obs_dim}, "
                f"hidden_dim={self.hidden_dim}, device={self.device})")
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get current training information.
        
        Returns:
            Dictionary containing training information
        """
        return {
            'temperature': self.temperature,
            'belief_lr': self.belief_scheduler.get_last_lr()[0],
            'action_lr': self.action_scheduler.get_last_lr()[0],
            'experience_buffer_size': len(self.experience_buffer),
            'belief_history_length': len(self.belief_history)
        } 