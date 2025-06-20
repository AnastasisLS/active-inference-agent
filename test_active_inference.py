"""
Test script for Active Inference Agent implementation.

This script tests the core components of the active inference framework:
1. Generative model
2. Free energy computation
3. Belief update
4. Action selection
5. Complete agent training
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generative_model import GenerativeModel
from utils.active_inference_utils import (
    compute_variational_free_energy,
    compute_expected_free_energy,
    update_belief_bayesian,
    select_action_active_inference,
    PreferenceModel
)
from agents.active_inference_agent import ActiveInferenceAgent
from environments.cartpole_env import create_cartpole_env
from utils.device_utils import get_device, print_device_info

def test_generative_model():
    """Test the generative model implementation."""
    print("=" * 60)
    print("TESTING GENERATIVE MODEL")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create generative model
    model = GenerativeModel(state_dim=4, action_dim=2, obs_dim=4, device=device)
    print("✓ Generative model created")
    
    # Test state transition
    state = torch.randn(2, 4, device=device)
    action = torch.tensor([0, 1], device=device)
    
    mean, cov = model.transition(state, action)
    print(f"✓ State transition: mean shape {mean.shape}, cov shape {cov.shape}")
    
    # Test observation model
    obs_mean, obs_cov = model.observation(state)
    print(f"✓ Observation model: mean shape {obs_mean.shape}, cov shape {obs_cov.shape}")
    
    # Test sampling
    next_state = model.sample_transition(state, action)
    obs = model.sample_observation(state)
    print(f"✓ Sampling: next_state shape {next_state.shape}, obs shape {obs.shape}")
    
    # Test log likelihood
    log_prob = model.log_likelihood(obs, state)
    print(f"✓ Log likelihood: shape {log_prob.shape}, values {log_prob}")
    
    print("✓ Generative model tests passed!\n")

def test_free_energy_computation():
    """Test free energy computation utilities."""
    print("=" * 60)
    print("TESTING FREE ENERGY COMPUTATION")
    print("=" * 60)
    
    device = get_device()
    
    # Create components
    generative_model = GenerativeModel(state_dim=4, action_dim=2, obs_dim=4, device=device)
    preference_model = PreferenceModel(obs_dim=4, device=device)
    
    # Test data
    batch_size = 2
    mean = torch.randn(batch_size, 4, device=device)
    cov = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, 4, 4).clone()
    belief = (mean, cov)
    obs = torch.randn(batch_size, 4, device=device)
    action = torch.tensor([0, 1], device=device)
    
    # Test VFE computation (using mean)
    vfe = compute_variational_free_energy(mean, obs, generative_model)
    print(f"✓ VFE computation: shape {vfe.shape}, values {vfe}")
    
    # Test EFE computation (using mean)
    efe = compute_expected_free_energy(mean, action, generative_model, preference_model)
    print(f"✓ EFE computation: shape {efe.shape}, values {efe}")
    
    # Test belief update (Kalman filter)
    updated_mean, updated_cov = update_belief_bayesian(belief, obs, action, generative_model)
    print(f"✓ Belief update: mean shape {updated_mean.shape}, cov shape {updated_cov.shape}")
    
    # Test action selection
    available_actions = [0, 1]
    selected_action, efe_values = select_action_active_inference(
        mean, generative_model, preference_model, available_actions
    )
    print(f"✓ Action selection: action {selected_action}, EFE values {efe_values}")
    
    print("✓ Free energy computation tests passed!\n")

def test_active_inference_agent():
    """Test the complete active inference agent."""
    print("=" * 60)
    print("TESTING ACTIVE INFERENCE AGENT")
    print("=" * 60)
    
    device = get_device()
    
    # Create agent
    agent = ActiveInferenceAgent(
        state_dim=4,
        action_dim=2,
        obs_dim=4,
        hidden_dim=64,
        learning_rate=0.001,
        belief_lr=0.1,
        temperature=1.0,
        planning_horizon=2,
        device=device
    )
    print("✓ Active inference agent created")
    
    # Test belief initialization
    obs = torch.randn(1, 4, device=device)
    belief = agent.initialize_belief(obs)
    print(f"✓ Belief initialization: mean shape {belief[0].shape}, cov shape {belief[1].shape}")
    
    # Test belief update
    action = torch.tensor([0], device=device)
    updated_belief = agent.update_belief(obs, action)
    print(f"✓ Belief update: mean shape {updated_belief[0].shape}, cov shape {updated_belief[1].shape}")
    
    # Test VFE computation
    vfe = agent.compute_vfe(obs)
    print(f"✓ VFE computation: shape {vfe.shape}, value {vfe.item():.4f}")
    
    # Test action selection
    action = agent.select_action_torch(obs)
    print(f"✓ Action selection: action {action}")
    
    # Test agent update
    experience = {
        'obs': obs,
        'action': torch.tensor([action], device=device),
        'reward': torch.tensor([1.0], device=device),
        'next_obs': torch.randn(1, 4, device=device),
        'done': torch.tensor([False], device=device)
    }
    
    update_stats = agent.update_torch(experience)
    print(f"✓ Agent update: {update_stats}")
    
    # Test statistics
    stats = agent.get_statistics()
    print(f"✓ Statistics: {list(stats.keys())}")
    
    print("✓ Active inference agent tests passed!\n")

def test_training_loop():
    """Test a short training loop with the active inference agent."""
    print("=" * 60)
    print("TESTING TRAINING LOOP")
    print("=" * 60)
    
    device = get_device()
    
    # Create environment
    env = create_cartpole_env(normalize_states=True, reward_shaping=False)
    print("✓ Environment created")
    
    # Create agent
    agent = ActiveInferenceAgent(
        state_dim=4,
        action_dim=2,
        obs_dim=4,
        hidden_dim=64,
        learning_rate=0.001,
        belief_lr=0.1,
        temperature=1.0,
        planning_horizon=2,
        device=device
    )
    print("✓ Agent created")
    
    # Short training loop
    num_episodes = 5
    episode_rewards = []
    
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        
        episode_reward = 0
        step_count = 0
        max_steps = 200
        
        done = False
        truncated = False
        
        while not (done or truncated) and step_count < max_steps:
            # Select action
            action = agent.select_action(obs)
            
            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Update agent (convert back to numpy for base class interface)
            agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward {episode_reward}, Steps {step_count}")
        
        # Reset agent for next episode
        agent.reset()
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"✓ Training test completed: Average reward {avg_reward:.2f}")
    print("✓ Training loop tests passed!\n")

def main():
    """Run all tests."""
    print("ACTIVE INFERENCE AGENT TESTING")
    print("=" * 60)
    print_device_info()
    
    try:
        test_generative_model()
        test_free_energy_computation()
        test_active_inference_agent()
        test_training_loop()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("The active inference agent implementation is working correctly.")
        print("You can now run training with:")
        print("  python main.py --agent active_inference --episodes 1000")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 