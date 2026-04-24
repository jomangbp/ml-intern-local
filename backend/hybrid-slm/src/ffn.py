"""
Feed-Forward Networks with SwiGLU Activation
Memory-efficient implementation for RTX 3060
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network
    
    SwiGLU = Swish(x) * Gate(x) where:
    - Swish(x) = x * sigmoid(x)
    - Gate is a linear projection
    
    Formula: FFN(x) = Swish(x @ W1) * (x @ W2) @ W3
    
    Benefits:
    - Better performance than standard ReLU/Mish
    - Widely used in modern LLMs (LLaMA, Qwen, Mistral)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Project down for efficiency (intermediate is 4x hidden typically)
        # We'll use 3 separate projections for clarity
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)  # Swish gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)    # Output
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias)  # Input transformation
        
        self.dropout = nn.Dropout(hidden_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following GPT-2 / LLaMA style"""
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
        if self.w3.bias is not None:
            nn.init.zeros_(self.w3.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # SwiGLU: Swish(W1(x)) * W3(x)
        # Using SiLU (Sigmoid Linear Unit) which is equivalent to Swish
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def swiglu_forward(x: torch.Tensor, w1: nn.Linear, w2: nn.Linear, w3: nn.Linear) -> torch.Tensor:
    """
    Functional SwiGLU for use in fused operations
    
    Args:
        x: [batch, seq_len, hidden_size]
        w1: first linear layer
        w2: output linear layer  
        w3: gate linear layer
    
    Returns:
        output: [batch, seq_len, hidden_size]
    """
    return F.silu(w1(x)) * w3(x) @ w2.weight.t()


class GatedLinearUnit(nn.Module):
    """
    Simple Gated Linear Unit (GLU) without Swish
    FFN(x) = (x @ W1) * (x @ W2)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward (optional for future extension)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        bias: bool = False
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=bias)
        
        # Expert weights
        self.w1 = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=bias)
            for _ in range(num_experts)
        ])
        self.w2 = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=bias)
            for _ in range(num_experts)
        ])
        self.w3 = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=bias)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        B, L, H = x.shape
        
        # Route to experts
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each token with top-k experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # [B, L]
            prob = top_k_probs[..., i:i+1]    # [B, L, 1]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # [B, L]
                if mask.any():
                    expert_input = x[mask]
                    expert_output = F.silu(self.w1[e](expert_input)) * self.w3[e](expert_input)
                    expert_output = self.w2[e](expert_output)
                    
                    # Add to output with probability weighting
                    output[mask] += expert_output * prob[mask].squeeze(-1)
        
        return output
