#!/usr/bin/env python3
"""
Test script to demonstrate device utilities and GPU acceleration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.device_utils import (
    get_device, get_device_info, print_device_info, 
    to_device, create_tensor_on_device, get_global_device
)


def test_device_selection():
    """Test device selection functionality."""
    print("=" * 60)
    print("DEVICE SELECTION TEST")
    print("=" * 60)
    
    # Test automatic device selection
    device = get_device()
    print(f"Auto-selected device: {device}")
    
    # Test specific device selection
    cpu_device = get_device('cpu')
    print(f"CPU device: {cpu_device}")
    
    # Test device info
    info = get_device_info()
    print(f"Device info: {info}")
    
    # Print detailed device information
    print_device_info()


def test_tensor_operations():
    """Test tensor operations on the selected device."""
    print("\n" + "=" * 60)
    print("TENSOR OPERATIONS TEST")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create tensors on device
    x = create_tensor_on_device([1, 2, 3, 4], device=device)
    y = create_tensor_on_device([5, 6, 7, 8], device=device)
    
    print(f"Tensor x: {x}")
    print(f"Tensor x device: {x.device}")
    print(f"Tensor y: {y}")
    print(f"Tensor y device: {y.device}")
    
    # Perform operations
    z = x + y
    print(f"x + y = {z}")
    print(f"Result device: {z.device}")
    
    # Matrix multiplication
    A = create_tensor_on_device([[1, 2], [3, 4]], device=device)
    B = create_tensor_on_device([[5, 6], [7, 8]], device=device)
    C = torch.mm(A, B)
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"A × B:\n{C}")


def test_neural_network():
    """Test neural network operations on the selected device."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK TEST")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create a simple neural network
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = torch.nn.Linear(4, 10)
            self.fc2 = torch.nn.Linear(10, 2)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create network and move to device
    net = SimpleNet()
    net = to_device(net, device)
    
    print(f"Network device: {next(net.parameters()).device}")
    
    # Create input data
    input_data = create_tensor_on_device([[1.0, 2.0, 3.0, 4.0]], device=device)
    print(f"Input data: {input_data}")
    print(f"Input device: {input_data.device}")
    
    # Forward pass
    with torch.no_grad():
        output = net(input_data)
        print(f"Output: {output}")
        print(f"Output device: {output.device}")
    
    # Test training step
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Create target
    target = create_tensor_on_device([[0.5, 0.8]], device=device)
    
    # Training step
    optimizer.zero_grad()
    output = net(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print(f"Loss device: {loss.device}")


def test_performance_comparison():
    """Test performance difference between CPU and GPU."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    
    # Test on CPU
    cpu_device = torch.device('cpu')
    print(f"Testing on CPU: {cpu_device}")
    
    # Create large tensors
    size = 1000
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time:
        start_time.record()
    
    A_cpu = torch.randn(size, size, device=cpu_device)
    B_cpu = torch.randn(size, size, device=cpu_device)
    C_cpu = torch.mm(A_cpu, B_cpu)
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        cpu_time = start_time.elapsed_time(end_time)
        print(f"CPU matrix multiplication time: {cpu_time:.2f} ms")
    
    # Test on best available device
    device = get_device()
    if device.type != 'cpu':
        print(f"Testing on {device}")
        
        if start_time:
            start_time.record()
        
        A_gpu = torch.randn(size, size, device=device)
        B_gpu = torch.randn(size, size, device=device)
        C_gpu = torch.mm(A_gpu, B_gpu)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            gpu_time = start_time.elapsed_time(end_time)
            print(f"{device.type.upper()} matrix multiplication time: {gpu_time:.2f} ms")
            
            if device.type != 'cpu':
                speedup = cpu_time / gpu_time
                print(f"Speedup: {speedup:.2f}x faster on {device.type.upper()}")
    else:
        print("No GPU available for comparison")


def main():
    """Run all device tests."""
    print("ACTIVE INFERENCE AGENT - DEVICE UTILITIES TEST")
    print("=" * 60)
    
    try:
        # Test device selection
        test_device_selection()
        
        # Test tensor operations
        test_tensor_operations()
        
        # Test neural network
        test_neural_network()
        
        # Test performance comparison
        test_performance_comparison()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 