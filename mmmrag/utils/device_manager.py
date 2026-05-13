#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Device Management Utility for MMMRAG System
Provides device allocation and management functions for multi-GPU setups
"""

import torch
from typing import List, Optional, Union

from mmmrag.config.config import Config


class DeviceManager:
    """
    Device management class for handling multi-GPU setups
    Provides functionality to distribute models and operations across multiple GPUs
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize device manager
        
        Args:
            config: Configuration object containing GPU device settings
        """
        self.config = config or Config()
        self.gpu_devices = getattr(self.config, 'GPU_DEVICES', ["cuda:0", "cuda:1"])
        self.default_gpu = getattr(self.config, 'DEFAULT_GPU', "cuda:0")
        
        # Add debug logging
        print(f"DEBUG: Initializing DeviceManager with GPU_DEVICES: {self.gpu_devices}")
        print(f"DEBUG: Default GPU: {self.default_gpu}")
        
        self.available_devices = self._detect_available_devices()
        self.device_count = len(self.available_devices)
        
        print(f"DEBUG: Detected available devices: {self.available_devices}")
        print(f"DEBUG: Device count: {self.device_count}")
        
        # Debug memory usage on initialization
        print(f"DEBUG: Initial memory usage check:")
        for i, device_str in enumerate(self.available_devices):
            try:
                memory_info = self.get_device_memory_usage(i)
                print(f"DEBUG: Device {device_str}: free={memory_info['free']/1e9:.2f}GB, used={memory_info['used']/1e9:.2f}GB")
            except Exception as e:
                print(f"DEBUG: Error checking memory for device {i}: {e}")
        
    def _detect_available_devices(self) -> List[str]:
        """
        Detect available GPU devices
        
        Returns:
            List of available GPU device strings
        """
        available = []
        for device in self.gpu_devices:
            try:
                device_obj = torch.device(device)
                # Test if device is available
                torch.tensor([0], device=device_obj)
                available.append(device)
            except Exception:
                continue
        return available
    
    def get_device(self, device_index: Optional[int] = None) -> torch.device:
        """
        Get a specific device or the default device
        
        Args:
            device_index: Index of the device to get (0-based)
            
        Returns:
            torch.device object
        """
        if not self.available_devices:
            return torch.device("cpu")
        
        if device_index is None:
            return torch.device(self.default_gpu)
        
        # Wrap around if index exceeds available devices
        device_index = device_index % len(self.available_devices)
        return torch.device(self.available_devices[device_index])
    
    def distribute_model(self, model: torch.nn.Module, device_index: int = 0) -> torch.nn.Module:
        """
        Distribute a model to a specific device
        
        Args:
            model: PyTorch model to distribute
            device_index: Index of the device to use
            
        Returns:
            Model moved to the specified device
        """
        device = self.get_device(device_index)
        return model.to(device)
    
    def distribute_tensor(self, tensor: torch.Tensor, device_index: int = 0) -> torch.Tensor:
        """
        Move a tensor to a specific device
        
        Args:
            tensor: PyTorch tensor to move
            device_index: Index of the device to use
            
        Returns:
            Tensor moved to the specified device
        """
        device = self.get_device(device_index)
        return tensor.to(device)
    
    def sync_devices(self):
        """
        Synchronize operations across all devices
        """
        for device_str in self.available_devices:
            device = torch.device(device_str)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
    
    def get_device_memory_usage(self, device_index: int = 0) -> dict:
        """
        Get memory usage for a specific device
        
        Args:
            device_index: Index of the device to check
            
        Returns:
            Dictionary with memory usage information
        """
        device = self.get_device(device_index)
        if device.type != 'cuda':
            return {"total": 0, "used": 0, "free": 0}
        
        device_idx = int(str(device).split(':')[-1])
        mem_total = torch.cuda.get_device_properties(device_idx).total_memory
        mem_allocated = torch.cuda.memory_allocated(device_idx)
        mem_reserved = torch.cuda.memory_reserved(device_idx)
        # Use the maximum of allocated and reserved to get actual used memory
        mem_used = max(mem_allocated, mem_reserved)
        mem_free = mem_total - mem_used
        
        return {
            "total": mem_total,
            "used": mem_used,
            "free": mem_free
        }
    
    def get_best_device(self) -> torch.device:
        """
        Get the device with the most free memory
        
        Returns:
            torch.device object with the most free memory
        """
        if not self.available_devices:
            return torch.device("cpu")
        
        best_device = self.get_device(0)
        max_free_memory = 0
        
        for i, device_str in enumerate(self.available_devices):
            try:
                memory_info = self.get_device_memory_usage(i)
                # Add debug logging
                print(f"DEBUG: Device {device_str} memory usage: free={memory_info['free']/1e9:.2f}GB, used={memory_info['used']/1e9:.2f}GB, total={memory_info['total']/1e9:.2f}GB")
                
                if memory_info["free"] > max_free_memory:
                    max_free_memory = memory_info["free"]
                    best_device = self.get_device(i)
                    print(f"DEBUG: New best device: {device_str} with {memory_info['free']/1e9:.2f}GB free")
            except Exception as e:
                print(f"DEBUG: Error checking memory for device {i}: {e}")
                continue
        
        print(f"DEBUG: Final best device selected: {best_device}")
        return best_device
    
    def ensure_device_consistency(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Ensure all tensors are on the same device
        
        Args:
            tensors: List of tensors to check
            
        Returns:
            List of tensors all on the same device
        """
        if not tensors:
            return []
        
        # Use the device of the first tensor
        target_device = tensors[0].device
        
        # Move all tensors to the target device
        return [tensor.to(target_device) for tensor in tensors]
    
    def get_device_str(self, device: torch.device) -> str:
        """
        Get string representation of a device
        
        Args:
            device: torch.device object
            
        Returns:
            String representation of the device
        """
        return str(device)
    
    def is_device_available(self, device_str: str) -> bool:
        """
        Check if a device is available
        
        Args:
            device_str: Device string to check
            
        Returns:
            Boolean indicating if the device is available
        """
        return device_str in self.available_devices


# Global device manager instance
device_manager = DeviceManager()


def get_device(device_index: Optional[int] = None) -> torch.device:
    """
    Convenience function to get a device
    
    Args:
        device_index: Index of the device to get
        
    Returns:
        torch.device object
    """
    return device_manager.get_device(device_index)


def distribute_model(model: torch.nn.Module, device_index: int = 0) -> torch.nn.Module:
    """
    Convenience function to distribute a model to a device
    
    Args:
        model: PyTorch model to distribute
        device_index: Index of the device to use
        
    Returns:
        Model moved to the specified device
    """
    return device_manager.distribute_model(model, device_index)


def ensure_device_consistency(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Convenience function to ensure all tensors are on the same device
    
    Args:
        tensors: List of tensors to check
        
    Returns:
        List of tensors all on the same device
    """
    return device_manager.ensure_device_consistency(tensors)