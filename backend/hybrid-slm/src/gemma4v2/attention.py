"""
Gemma 4 V3 Attention Layer

Based on actual Gemma 4 architecture (google/gemma-4-26B-A4B-it):
- Sliding window attention (1024 tokens)
- Full attention layers every 6th layer
- GQA with num_key_value_heads=8
- GeGLU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.layers import RMSNorm
from src.rotary import RotaryEmbedding


class Gemma4QKNorm(nn.Module):
    """QK-Norm as used in Gemma 4"""
    
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(head_dim, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., head_dim]
        return self.norm(x)


class Gemma4SlidingAttention(nn.Module):
    """
    Gemma 4 Sliding Window Attention
    
    Based on actual Gemma 4 architecture:
    - Sliding window: 1024 tokens
    - GQA: num_kv_heads=8, num_heads=16
    - RoPE theta: 10000
    - QK-Norm before attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.scale = head_dim ** -0.5
        
        self.n_rep = num_heads // num_kv_heads
        
        # QK-Norm (Gemma 4 innovation)
        self.q_norm = Gemma4QKNorm(head_dim)
        self.k_norm = Gemma4QKNorm(head_dim)
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rotary_emb = None
        self.dropout = dropout
    
    def set_rotary_emb(self, rotary_emb):
        self.rotary_emb = rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        B, L, H = hidden_states.shape
        
        # Project Q, K, V -> [B, L, H, D]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply QK-Norm (Gemma 4)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose to [B, H, L, D]
        q = q.transpose(1, 2)  # [B, num_heads, L, D]
        k = k.transpose(1, 2)  # [B, num_kv_heads, L, D]
        v = v.transpose(1, 2)  # [B, num_kv_heads, L, D]
        
        # Expand KV heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
        
        # Apply RoPE
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)
        
        # Compute attention with sliding window
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sliding window mask
        if L > self.sliding_window:
            # Create mask: only attend to last sliding_window tokens
            mask = torch.zeros(L, L, device=attn_weights.device, dtype=torch.bool)
            mask[:, -self.sliding_window:] = True
            # Make it causal (upper triangle is -inf)
            causal_mask = torch.triu(torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1)
            mask = mask & ~causal_mask  # Only keep within sliding window
            
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Full causal mask
            causal_mask = torch.triu(
                torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.o_proj(output)
        
        return output, None


class Gemma4FullAttention(nn.Module):
    """
    Gemma 4 Full Attention (Global attention)
    
    Used every 6th layer (full attention layers)
    RoPE theta: 1000000 (1M)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.n_rep = num_heads // num_kv_heads
        
        # QK-Norm (Gemma 4)
        self.q_norm = Gemma4QKNorm(head_dim)
        self.k_norm = Gemma4QKNorm(head_dim)
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rotary_emb = None
        self.dropout = dropout
    
    def set_rotary_emb(self, rotary_emb):
        self.rotary_emb = rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        B, L, H = hidden_states.shape
        
        # Project Q, K, V -> [B, L, H, D]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply QK-Norm (Gemma 4)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose to [B, H, L, D]
        q = q.transpose(1, 2)  # [B, num_heads, L, D]
        k = k.transpose(1, 2)  # [B, num_kv_heads, L, D]
        v = v.transpose(1, 2)  # [B, num_kv_heads, L, D]
        
        # Expand KV heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
        
        # Apply RoPE
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)
        
        # Full causal attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.o_proj(output)
        
        return output, None
