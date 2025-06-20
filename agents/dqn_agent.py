"""
Deep Q-Network (DQN) agent implementation for the Active Inference Agent project.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, Any, List, Tuple, Optional

from .base_agent import BaseAgent
from utils.device_utils import get_device, to_device


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture for CartPole.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize the DQN network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Build the network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        """Get the number of experiences in the buffer."""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent implementation.
    
    This agent uses a neural network to approximate Q-values and includes
    experience replay and target network for stable training.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Agent configuration dictionary
        """
        super().__init__(state_dim, action_dim, config)
    
    def _initialize_agent(self):
        """Initialize the DQN agent components."""
        # Extract configuration parameters
        hidden_dims = self.config.get('hidden_dims', [128, 128])
        learning_rate = self.config.get('learning_rate', 0.001)
        batch_size = self.config.get('batch_size', 64)
        buffer_size = self.config.get('buffer_size', 10000)
        gamma = self.config.get('gamma', 0.99)
        tau = self.config.get('tau', 0.005)
        target_update_freq = self.config.get('target_update_freq', 1000)
        epsilon_start = self.config.get('epsilon_start', 1.0)
        epsilon_end = self.config.get('epsilon_end', 0.01)
        epsilon_decay = self.config.get('epsilon_decay', 0.995)
        
        # Store configuration
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Get device (auto-select best available)
        self.device = get_device()
        print(f"DQN Agent using device: {self.device}")
        
        # Initialize networks
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        
        # Initialize target network with same weights as main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training state
        self.epsilon = epsilon_start
        self.steps_since_target_update = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        if self.training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_dim)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
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
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()
            if loss is not None:
                self.training_losses.append(loss)
        
        # Update target network periodically
        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_freq:
            self._update_target_network()
            self.steps_since_target_update = 0
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _train_step(self):
        """
        Perform one training step.
        
        Returns:
            Loss value if training was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _update_target_network(self):
        """Update target network using soft update."""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Action probabilities
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            # Convert Q-values to probabilities using softmax
            probabilities = F.softmax(q_values, dim=1)
            return probabilities.cpu().numpy().squeeze()
    
    def get_value_estimate(self, state: np.ndarray) -> float:
        """
        Get the maximum Q-value for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Maximum Q-value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max().item()
    
    def save_model(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            'agent_type': 'DQN',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'device': str(self.device),
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'config': self.config
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get training information.
        
        Returns:
            Dictionary containing training information
        """
        return {
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'steps_since_target_update': self.steps_since_target_update,
            'training_losses': self.training_losses[-100:] if self.training_losses else []
        }


def create_dqn_agent(
    state_dim: int = 4,
    action_dim: int = 2,
    hidden_dims: List[int] = None,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    buffer_size: int = 10000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    device: str = 'auto',
    seed: int = 42
) -> DQNAgent:
    """
    Create a DQN agent with the specified configuration.
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        buffer_size: Size of the replay buffer
        gamma: Discount factor
        epsilon_start: Initial epsilon value
        epsilon_end: Final epsilon value
        epsilon_decay: Epsilon decay rate
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        seed: Random seed
        
    Returns:
        DQNAgent: Configured DQN agent
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create configuration
    config = {
        'hidden_dims': hidden_dims or [128, 128],
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'gamma': gamma,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'device': device
    }
    
    return DQNAgent(state_dim, action_dim, config) 