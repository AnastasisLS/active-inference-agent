"""
Test setup script to verify the installation and basic functionality.
"""

import sys
import importlib
from typing import List, Dict, Any


def test_imports() -> Dict[str, bool]:
    """
    Test if all required packages can be imported.
    
    Returns:
        Dictionary with import test results
    """
    required_packages = [
        'torch',
        'torchvision',
        'gymnasium',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        'jupyter',
        'pymc',
        'arviz',
        'tqdm'
    ]
    
    results = {}
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            results[package] = True
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            results[package] = False
            print(f"✗ {package} import failed: {e}")
    
    return results


def test_project_modules() -> Dict[str, bool]:
    """
    Test if project modules can be imported.
    
    Returns:
        Dictionary with module test results
    """
    project_modules = [
        'config',
        'config.agent_config',
        'config.environment_config',
        'config.training_config',
        'environments',
        'environments.cartpole_env',
        'environments.environment_wrapper',
        'agents',
        'agents.base_agent',
        'agents.dqn_agent'
    ]
    
    results = {}
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            results[module] = True
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            results[module] = False
            print(f"✗ {module} import failed: {e}")
    
    return results


def test_basic_functionality() -> Dict[str, bool]:
    """
    Test basic functionality of the project components.
    
    Returns:
        Dictionary with functionality test results
    """
    results = {}
    
    try:
        # Test configuration
        from config.agent_config import DQNConfig, ActiveInferenceConfig, AgentConfig
        from config.environment_config import CartPoleConfig, EnvironmentConfig
        from config.training_config import TrainingConfig
        
        # Test environment
        from environments.cartpole_env import CartPoleEnv, create_cartpole_env
        
        # Test agents
        from agents.dqn_agent import DQNAgent, create_dqn_agent
        
        results['config_creation'] = True
        print("✓ Configuration classes created successfully")
        
        # Test environment creation
        env = create_cartpole_env()
        results['environment_creation'] = True
        print("✓ CartPole environment created successfully")
        
        # Test agent creation
        agent = create_dqn_agent()
        results['agent_creation'] = True
        print("✓ DQN agent created successfully")
        
        # Test basic interaction
        state, info = env.reset()
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        results['basic_interaction'] = True
        print("✓ Basic environment-agent interaction successful")
        
        # Clean up
        env.close()
        
    except Exception as e:
        results['basic_functionality'] = False
        print(f"✗ Basic functionality test failed: {e}")
        return results
    
    results['basic_functionality'] = True
    return results


def test_pytorch_functionality() -> Dict[str, bool]:
    """
    Test PyTorch functionality.
    
    Returns:
        Dictionary with PyTorch test results
    """
    results = {}
    
    try:
        import torch
        import torch.nn as nn
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        
        results['tensor_operations'] = True
        print("✓ PyTorch tensor operations successful")
        
        # Test neural network
        model = nn.Linear(3, 2)
        output = model(x)
        
        results['neural_network'] = True
        print("✓ PyTorch neural network operations successful")
        
        # Test device availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x_gpu = x.to(device)
            results['gpu_available'] = True
            print("✓ GPU available and working")
        else:
            results['gpu_available'] = False
            print("ℹ GPU not available, using CPU")
        
    except Exception as e:
        results['pytorch_functionality'] = False
        print(f"✗ PyTorch functionality test failed: {e}")
        return results
    
    results['pytorch_functionality'] = True
    return results


def test_gymnasium_functionality() -> Dict[str, bool]:
    """
    Test Gymnasium functionality.
    
    Returns:
        Dictionary with Gymnasium test results
    """
    results = {}
    
    try:
        import gymnasium as gym
        
        # Test environment creation
        env = gym.make('CartPole-v1')
        
        results['env_creation'] = True
        print("✓ Gymnasium CartPole environment created successfully")
        
        # Test basic interaction
        observation, info = env.reset()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        results['env_interaction'] = True
        print("✓ Gymnasium environment interaction successful")
        
        # Clean up
        env.close()
        
    except Exception as e:
        results['gymnasium_functionality'] = False
        print(f"✗ Gymnasium functionality test failed: {e}")
        return results
    
    results['gymnasium_functionality'] = True
    return results


def main():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("ACTIVE INFERENCE AGENT PROJECT - SETUP TEST")
    print("=" * 60)
    
    print("\n1. Testing package imports...")
    import_results = test_imports()
    
    print("\n2. Testing project modules...")
    module_results = test_project_modules()
    
    print("\n3. Testing PyTorch functionality...")
    pytorch_results = test_pytorch_functionality()
    
    print("\n4. Testing Gymnasium functionality...")
    gymnasium_results = test_gymnasium_functionality()
    
    print("\n5. Testing basic functionality...")
    functionality_results = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_results = {
        'imports': import_results,
        'modules': module_results,
        'pytorch': pytorch_results,
        'gymnasium': gymnasium_results,
        'functionality': functionality_results
    }
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        category_total = len(results)
        category_passed = sum(results.values())
        
        total_tests += category_total
        passed_tests += category_passed
        
        print(f"\n{category.upper()}:")
        print(f"  Passed: {category_passed}/{category_total}")
        if category_passed < category_total:
            failed_tests = [k for k, v in results.items() if not v]
            print(f"  Failed: {', '.join(failed_tests)}")
    
    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The setup is ready for development.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please check the installation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 