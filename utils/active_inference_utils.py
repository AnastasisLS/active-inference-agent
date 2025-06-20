"""
Active Inference Utilities

Implements core active inference computations:
- Variational Free Energy (VFE)
- Expected Free Energy (EFE)
- Bayesian belief update with Kalman filtering
- Action selection with planning
- Advanced preference models

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
from typing import Tuple, List, Optional, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

def compute_variational_free_energy(
    belief: torch.Tensor,
    obs: torch.Tensor,
    generative_model,
    prior_belief: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Variational Free Energy (VFE) = -log p(o) + KL(q(s) || p(s|o))
    
    Args:
        belief: (batch, state_dim) Current belief about state
        obs: (batch, obs_dim) Current observation
        generative_model: GenerativeModel instance
        prior_belief: (batch, state_dim) Prior belief (if None, use uniform)
    
    Returns:
        vfe: (batch,) Variational free energy
    """
    batch_size = belief.shape[0]
    
    # Prior belief (uniform if not provided)
    if prior_belief is None:
        prior_belief = torch.ones_like(belief) / belief.shape[-1]
    
    # Likelihood term: -log p(o|s)
    log_likelihood = generative_model.log_likelihood(obs, belief)
    
    # KL divergence: KL(q(s) || p(s))
    # For discrete beliefs, KL = sum(q * log(q/p))
    safe_belief = belief.clamp(min=1e-8)
    safe_prior = prior_belief.clamp(min=1e-8)
    kl_div = torch.sum(safe_belief * torch.log(safe_belief / safe_prior), dim=-1)
    
    # VFE = -log p(o|s) + KL(q(s) || p(s))
    vfe = -log_likelihood + kl_div
    
    return vfe

def compute_expected_free_energy(
    belief: torch.Tensor,
    action: torch.Tensor,
    generative_model,
    preference_model,
    horizon: int = 1
) -> torch.Tensor:
    """
    Compute Expected Free Energy (EFE) for action selection.
    EFE = E_q(s)[log q(s) - log p(o,s|a)]
    
    Args:
        belief: (batch, state_dim) Current belief about state
        action: (batch,) Action to evaluate
        generative_model: GenerativeModel instance
        preference_model: Preference model for desired observations
        horizon: Planning horizon
    
    Returns:
        efe: (batch,) Expected free energy
    """
    batch_size = belief.shape[0]
    state_dim = belief.shape[-1]
    
    # Initialize EFE with proper shape
    efe = torch.zeros(batch_size, device=belief.device, dtype=belief.dtype)
    
    # Current belief as starting point
    current_belief = belief.clone()
    
    for t in range(horizon):
        # Predict next state distribution
        next_state_mean, next_state_cov = generative_model.transition(
            current_belief, action
        )
        
        # Sample from predicted state distribution
        next_state_dist = MultivariateNormal(next_state_mean, next_state_cov)
        sampled_states = next_state_dist.sample()
        
        # Predict observations
        obs_mean, obs_cov = generative_model.observation(sampled_states)
        obs_dist = MultivariateNormal(obs_mean, obs_cov)
        predicted_obs = obs_dist.sample()
        
        # Compute preference term: log p(o) (desired observations)
        log_preference = preference_model.log_prob(predicted_obs)
        
        # Compute ambiguity term: -log p(o|s)
        log_likelihood = generative_model.log_likelihood(predicted_obs, sampled_states)
        
        # EFE components - ensure compatible shapes
        ambiguity = -log_likelihood  # Should be (batch,)
        risk = -log_preference       # Should be (batch,)
        
        # Ensure all tensors are at least 1D
        if ambiguity.dim() == 0:
            ambiguity = ambiguity.unsqueeze(0)
        if risk.dim() == 0:
            risk = risk.unsqueeze(0)
        if efe.dim() == 0:
            efe = efe.unsqueeze(0)
        
        # Ensure all tensors are 1D (but don't squeeze if it would make them 0D)
        if ambiguity.dim() > 1:
            ambiguity = ambiguity.flatten()
        if risk.dim() > 1:
            risk = risk.flatten()
        if efe.dim() > 1:
            efe = efe.flatten()
        
        # Ensure all have the same batch size
        target_batch_size = max(ambiguity.shape[0], risk.shape[0], efe.shape[0])
        
        if ambiguity.shape[0] != target_batch_size:
            if ambiguity.shape[0] == 1:
                ambiguity = ambiguity.expand(target_batch_size)
            else:
                ambiguity = ambiguity[:target_batch_size]
        
        if risk.shape[0] != target_batch_size:
            if risk.shape[0] == 1:
                risk = risk.expand(target_batch_size)
            else:
                risk = risk[:target_batch_size]
        
        if efe.shape[0] != target_batch_size:
            if efe.shape[0] == 1:
                efe = efe.expand(target_batch_size)
            else:
                efe = efe[:target_batch_size]
        
        # Accumulate EFE
        efe += ambiguity + risk
        
        # Update belief for next step
        current_belief = next_state_mean
    
    return efe

def update_belief_bayesian(
    belief: Tuple[torch.Tensor, torch.Tensor],
    obs: torch.Tensor,
    action: torch.Tensor,
    generative_model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update belief using Kalman filter equations for linear-Gaussian systems.
    
    Args:
        belief: Tuple (mean, cov) of current belief
        obs: (batch, obs_dim) New observation
        action: (batch,) Action taken
        generative_model: GenerativeModel instance
        
    Returns:
        updated_belief: Tuple (mean, cov) of updated belief
    """
    mean, cov = belief  # (batch, state_dim), (batch, state_dim, state_dim)
    
    # Ensure mean has the right shape
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)  # Add batch dimension
    batch_size = mean.shape[0]
    state_dim = mean.shape[1]
    
    # Ensure obs has the right shape
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)  # Add batch dimension
    obs_dim = obs.shape[1]

    # Prediction step
    A = generative_model.A  # (state_dim, state_dim)
    B = generative_model.B  # (state_dim, action_dim)
    Q = generative_model.Q  # (state_dim, state_dim)
    C = generative_model.C  # (obs_dim, state_dim)
    R = generative_model.R  # (obs_dim, obs_dim)

    # One-hot encode actions
    action_onehot = F.one_hot(action.long(), num_classes=B.shape[1]).float()
    
    # Predict mean: μ̂ = Aμ + Ba
    predicted_mean = mean @ A.T + action_onehot @ B.T
    
    # Predict covariance: Σ̂ = AΣA^T + Q
    # Handle batch dimension properly
    if cov.dim() == 2:
        # Single covariance matrix, expand to batch
        cov = cov.unsqueeze(0).expand(batch_size, state_dim, state_dim)
    
    predicted_cov = torch.zeros_like(cov)
    for i in range(batch_size):
        predicted_cov[i] = A @ cov[i] @ A.T + Q
    
    # Update step (Kalman gain)
    # For simplicity, use the first batch element's covariance for Kalman gain
    # This is a reasonable approximation for small batch sizes
    cov_for_gain = predicted_cov[0] if batch_size > 1 else predicted_cov.squeeze(0)
    
    # K = Σ̂C^T(CΣ̂C^T + R)^(-1)
    innovation_cov = C @ cov_for_gain @ C.T + R
    kalman_gain = cov_for_gain @ C.T @ torch.inverse(innovation_cov)
    
    # Update mean: μ = μ̂ + K(y - Cμ̂)
    predicted_obs = predicted_mean @ C.T
    innovation = obs - predicted_obs
    
    # Apply Kalman gain to innovation
    if batch_size == 1:
        updated_mean = predicted_mean + innovation @ kalman_gain.T
    else:
        # For batch processing, use the same Kalman gain for all batch elements
        updated_mean = predicted_mean + innovation @ kalman_gain.T
    
    # Update covariance: Σ = (I - KC)Σ̂
    identity = torch.eye(state_dim, device=mean.device)
    updated_cov = torch.zeros_like(predicted_cov)
    for i in range(batch_size):
        updated_cov[i] = (identity - kalman_gain @ C) @ predicted_cov[i]
    
    return updated_mean, updated_cov

def select_action_active_inference(
    belief: torch.Tensor,
    generative_model,
    preference_model,
    available_actions: List[int],
    temperature: float = 1.0,
    horizon: int = 1
) -> Tuple[int, torch.Tensor]:
    """
    Select action by minimizing expected free energy.
    
    Args:
        belief: (batch, state_dim) Current belief
        generative_model: GenerativeModel instance
        preference_model: Preference model
        available_actions: List of available actions
        temperature: Temperature for action selection
        horizon: Planning horizon
        
    Returns:
        selected_action: Index of selected action
        efe_values: EFE values for all actions
    """
    batch_size = belief.shape[0]
    
    if batch_size > 1:
        warnings.warn("Batch size > 1 in select_action_active_inference; returning first action only.")
        belief = belief[:1]  # Use first belief only
    
    efe_values = []
    
    # Compute EFE for each available action
    for action_idx in available_actions:
        action = torch.tensor([action_idx], device=belief.device)
        efe = compute_expected_free_energy(
            belief, action, generative_model, preference_model, horizon
        )
        efe_values.append(efe.item())
    
    efe_tensor = torch.tensor(efe_values, device=belief.device)
    
    # Convert EFE to action probabilities (lower EFE = higher probability)
    # Use softmax with negative EFE (since we want to minimize EFE)
    action_probs = F.softmax(-efe_tensor / temperature, dim=0)
    
    # Sample action from probability distribution
    action_dist = Categorical(action_probs)
    selected_action_idx = action_dist.sample()
    selected_action = available_actions[selected_action_idx.item()]
    
    return selected_action, efe_tensor

class PreferenceModel:
    """
    Advanced preference model for desired observations.
    Implements both simple Gaussian preferences and learned preferences.
    """
    
    def __init__(self, obs_dim: int, device: str = 'cpu', preference_type: str = 'gaussian'):
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.preference_type = preference_type
        
        if preference_type == 'gaussian':
            # Simple Gaussian preferences centered at zero
            self.preference_mean = torch.zeros(obs_dim, device=self.device)
            self.preference_cov = torch.eye(obs_dim, device=self.device)
        elif preference_type == 'learned':
            # Learnable preference network
            self.preference_net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown preference type: {preference_type}")
    
    def log_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of observations under preference model.
        
        Args:
            obs: (batch, obs_dim) Observations
            
        Returns:
            log_prob: (batch,) Log probabilities
        """
        if self.preference_type == 'gaussian':
            dist = MultivariateNormal(self.preference_mean, self.preference_cov)
            return dist.log_prob(obs)
        elif self.preference_type == 'learned':
            return self.preference_net(obs).squeeze(-1)
        else:
            raise ValueError(f"Unknown preference type: {self.preference_type}")

class AdvancedPreferenceModel(nn.Module):
    """
    Advanced preference model with multiple preference types and learning capabilities.
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 64, device: str = 'cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # Preference network
        self.preference_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output preference score between 0 and 1
        ).to(self.device)
        
        # Target preferences (learnable)
        self.target_preferences = nn.Parameter(
            torch.randn(obs_dim, device=self.device) * 0.1
        )
        
        # Preference strength
        self.preference_strength = nn.Parameter(torch.tensor(1.0, device=self.device))
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute preference scores for observations.
        
        Args:
            obs: (batch, obs_dim) Observations
            
        Returns:
            preference_scores: (batch,) Preference scores
        """
        # Learned preference component
        learned_pref = self.preference_net(obs).squeeze(-1)
        
        # Target-based preference component
        target_dist = Normal(self.target_preferences, self.preference_strength)
        target_pref = target_dist.log_prob(obs).sum(dim=-1)
        
        # Combine preferences
        combined_pref = learned_pref + 0.1 * target_pref
        
        return combined_pref
    
    def log_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for compatibility with existing code.
        
        Args:
            obs: (batch, obs_dim) Observations
            
        Returns:
            log_prob: (batch,) Log probabilities
        """
        return self.forward(obs) 