"""
Gemma 4-Style Full Attention Layer

Key innovations:
1. Dual head dimensions (wider heads for global layers)
2. K=V shared projections (attention_k_eq_v)
3. QK-norm (L2 normalization on Q and K)
4. RoPE with configurable theta per layer type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.layers import RMSNorm


class Gemma4FullAttention(nn.Module):
    """
    Gemma 4 Full Attention with:
    - GQA with dual head dimensions
    - K=V shared projections
    - QK-norm (L2 normalization)
    - RoPE
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,  # This is the GLOBAL head dim (wider)
        use_qk_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm
        
        assert num_heads % num_kv_heads == 0
        self.n_rep = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        # V projection is shared with K (Gemma 4: attention_k_eq_v)
        # We still store it as a separate parameter for clarity but could share
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        
        # QK-norm (Gemma 4): L2 normalize Q and K before attention
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        
        # RoPE (set externally)
        self.rotary_emb = None
    
    def set_rotary_emb(self, rotary_emb):
        self.rotary_emb = rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, L, H = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        
        # QK-norm: L2 normalize Q and K (Gemma 4 innovation)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Expand KV heads for GQA (before RoPE so KV has same dim as Q)
        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, L, self.n_rep, self.num_kv_heads, self.head_dim).reshape(
                B, L, self.num_kv_heads * self.n_rep, self.head_dim
            )
            v = v[:, :, None, :, :].expand(B, L, self.n_rep, self.num_kv_heads, self.head_dim).reshape(
                B, L, self.num_kv_heads * self.n_rep, self.head_dim
            )
        
        # Transpose to [B, H, L, D] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE in [B, H, L, D] format
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)
        
        # Standard scaled dot-product attention (already in [B, H, L, D] format)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)  # [B, H, L, D]
        output = output.transpose(1, 2).reshape(B, L, -1)  # [B, L, H*D]
        output = self.o_proj(output)
        
        present = None
        return output, present
