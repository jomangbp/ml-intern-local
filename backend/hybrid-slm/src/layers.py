"""
RMSNorm Implementation
Fast and memory-efficient normalization layer
"""

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    RMSNorm is simpler than LayerNorm and achieves similar performance:
    - No mean centering (reduces computation)
    - Normalizes by RMS of hidden states
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [..., hidden_size]
        
        Returns:
            normalized: [..., hidden_size]
        """
        # Compute RMS: sqrt(mean(x^2))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        
        # Scale by learnable weight
        return normalized * self.weight


def rmsnorm_forward(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Functional RMSNorm for use in fused operations
    
    Args:
        hidden_states: [batch, seq_len, hidden_size]
        weight: [hidden_size]
        eps: epsilon for numerical stability
    
    Returns:
        normalized: [batch, seq_len, hidden_size]
    """
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + eps)
    return normalized * weight


class RMSNormBias(nn.Module):
    """
    RMSNorm with optional bias (used in some architectures)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        
        if self.bias is not None:
            normalized = normalized + self.bias
        
        return normalized * self.weight
