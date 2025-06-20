"""
Advanced Generative Model for Active Inference (CartPole)

Implements a sophisticated probabilistic generative model:
- State transition: p(s_{t+1} | s_t, a_t) with learnable parameters
- Observation: p(o_t | s_t) with noise modeling
- Parameter learning and adaptation
- Numerical stability improvements

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
from torch.distributions import MultivariateNormal, Normal
from typing import Any, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    """
    Advanced Linear-Gaussian generative model for CartPole.
    State: s (4D), Action: a (discrete, 2), Observation: o (4D)
    
    Features:
    - Learnable transition and observation parameters
    - Adaptive noise modeling
    - Numerical stability improvements
    - Parameter regularization
    """
    
    def __init__(self, state_dim: int, action_dim: int, obs_dim: int, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        
        # Learnable state transition parameters
        self.A = nn.Parameter(torch.eye(state_dim, device=self.device))
        self.B = nn.Parameter(torch.zeros((state_dim, action_dim), device=self.device))
        
        # Initialize B with CartPole-specific dynamics
        with torch.no_grad():
            self.B[1, 0] = -1.0  # Left action affects cart velocity
            self.B[1, 1] = 1.0   # Right action affects cart velocity
            self.B[3, 0] = -0.5  # Left action affects pole angular velocity
            self.B[3, 1] = 0.5   # Right action affects pole angular velocity
        
        # Learnable noise parameters - increased for partial observability
        self.log_Q_diag = nn.Parameter(torch.log(torch.ones(state_dim, device=self.device) * 0.1))
        self.log_R_diag = nn.Parameter(torch.log(torch.ones(obs_dim, device=self.device) * 0.1))
        
        # Observation model (partial observability)
        # C is now fixed, not learnable
        self.C = torch.zeros((obs_dim, state_dim), device=self.device)
        self.C[0, 0] = 1.0  # Observe cart position
        self.C[1, 2] = 1.0  # Observe pole angle
        
        # Parameter regularization
        self.param_reg_weight = 0.01
        
        logger.info(f"Generative model initialized: state_dim={state_dim}, action_dim={action_dim}, obs_dim={obs_dim}")
        print(f"[DEBUG] Initial Q: {torch.exp(self.log_Q_diag).detach().cpu().numpy()}")
        print(f"[DEBUG] Initial R: {torch.exp(self.log_R_diag).detach().cpu().numpy()}")
    
    @property
    def Q(self) -> torch.Tensor:
        """State transition noise covariance matrix."""
        q_diag = torch.exp(self.log_Q_diag).clamp(min=1e-6)
        print(f"[DEBUG] Q diag: {q_diag.detach().cpu().numpy()}")
        return torch.diag(q_diag)
    
    @property
    def R(self) -> torch.Tensor:
        """Observation noise covariance matrix."""
        r_diag = torch.exp(self.log_R_diag).clamp(min=1e-6)
        print(f"[DEBUG] R diag: {r_diag.detach().cpu().numpy()}")
        return torch.diag(r_diag)
    
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        State transition model: p(s_{t+1} | s_t, a_t) ~ N(mean, Q)
        
        Args:
            state: (batch, state_dim) Current state
            action: (batch,) Action indices
            
        Returns:
            mean: (batch, state_dim) Predicted state mean
            cov: (state_dim, state_dim) State transition covariance
        """
        # One-hot encode actions
        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        
        # Linear transition: s_{t+1} = A s_t + B a_t
        mean = state @ self.A.T + action_onehot @ self.B.T
        
        # Add small regularization to ensure positive definiteness
        cov = self.Q + torch.eye(self.state_dim, device=self.device) * 1e-6
        
        return mean, cov
    
    def observation(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Observation model: p(o_t | s_t) ~ N(mean, R)
        
        Args:
            state: (batch, state_dim) State
            
        Returns:
            mean: (batch, obs_dim) Predicted observation mean
            cov: (obs_dim, obs_dim) Observation covariance
        """
        # Linear observation: o_t = C s_t
        mean = state @ self.C.T
        
        # Add small regularization to ensure positive definiteness
        cov = self.R + torch.eye(self.obs_dim, device=self.device) * 1e-6
        
        return mean, cov
    
    def sample_transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Sample next state from transition model.
        
        Args:
            state: (batch, state_dim) Current state
            action: (batch,) Action indices
            
        Returns:
            next_state: (batch, state_dim) Sampled next state
        """
        mean, cov = self.transition(state, action)
        dist = MultivariateNormal(mean, cov)
        return dist.sample()
    
    def sample_observation(self, state: torch.Tensor) -> torch.Tensor:
        """
        Sample observation from observation model.
        
        Args:
            state: (batch, state_dim) State
            
        Returns:
            obs: (batch, obs_dim) Sampled observation
        """
        mean, cov = self.observation(state)
        dist = MultivariateNormal(mean, cov)
        return dist.sample()
    
    def log_likelihood(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(o_t | s_t) for observation likelihood.
        
        Args:
            obs: (batch, obs_dim) Observations
            state: (batch, state_dim) States
            
        Returns:
            log_prob: (batch,) Log likelihoods
        """
        mean, cov = self.observation(state)
        dist = MultivariateNormal(mean, cov)
        return dist.log_prob(obs)
    
    def transition_log_likelihood(self, next_state: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(s_{t+1} | s_t, a_t) for state transition likelihood.
        
        Args:
            next_state: (batch, state_dim) Next states
            state: (batch, state_dim) Current states
            action: (batch,) Action indices
            
        Returns:
            log_prob: (batch,) Log likelihoods
        """
        mean, cov = self.transition(state, action)
        dist = MultivariateNormal(mean, cov)
        return dist.log_prob(next_state)
    
    def parameter_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for model parameters.
        
        Returns:
            reg_loss: Regularization loss
        """
        # Regularize A to be close to identity
        A_reg = torch.sum((self.A - torch.eye(self.state_dim, device=self.device))**2)
        
        # Regularize C to be close to identity
        C_reg = torch.sum((self.C - torch.eye(self.obs_dim, device=self.device))**2)
        
        # Regularize noise parameters to be reasonable
        Q_reg = torch.sum(self.log_Q_diag**2)
        R_reg = torch.sum(self.log_R_diag**2)
        
        total_reg = A_reg + C_reg + 0.1 * (Q_reg + R_reg)
        return self.param_reg_weight * total_reg
    
    def update_parameters(self, obs_sequence: torch.Tensor, action_sequence: torch.Tensor, 
                         learning_rate: float = 0.001) -> Dict[str, float]:
        """
        Update model parameters using observed data.
        
        Args:
            obs_sequence: (batch, seq_len, obs_dim) Observation sequence
            action_sequence: (batch, seq_len) Action sequence
            learning_rate: Learning rate for parameter updates
            
        Returns:
            Dict containing training statistics
        """
        self.train()
        
        # Create optimizer for model parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        batch_size, seq_len, _ = obs_sequence.shape
        
        # Initialize belief
        belief_mean = torch.zeros(batch_size, self.state_dim, device=self.device)
        belief_cov = torch.eye(self.state_dim, device=self.device).unsqueeze(0).expand(batch_size, self.state_dim, self.state_dim)
        
        total_loss = 0.0
        total_vfe = 0.0
        
        for t in range(seq_len - 1):
            obs = obs_sequence[:, t]
            next_obs = obs_sequence[:, t + 1]
            action = action_sequence[:, t]
            
            # Predict next state
            pred_mean, pred_cov = self.transition(belief_mean, action)
            
            # Predict next observation
            obs_mean, obs_cov = self.observation(pred_mean)
            
            # Compute VFE
            vfe = -self.log_likelihood(next_obs, pred_mean)
            total_vfe += vfe.mean()
            
            # Update belief (simplified)
            belief_mean = pred_mean
            belief_cov = pred_cov
        
        # Total loss including regularization
        total_loss = total_vfe + self.parameter_regularization_loss()
        
        # Update parameters
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'vfe_loss': total_vfe.item(),
            'reg_loss': self.parameter_regularization_loss().item()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model parameters.
        
        Returns:
            Dict containing model information
        """
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'obs_dim': self.obs_dim,
            'A_norm': torch.norm(self.A).item(),
            'B_norm': torch.norm(self.B).item(),
            'C_norm': torch.norm(self.C).item(),
            'Q_diag_mean': torch.exp(self.log_Q_diag).mean().item(),
            'R_diag_mean': torch.exp(self.log_R_diag).mean().item()
        } 