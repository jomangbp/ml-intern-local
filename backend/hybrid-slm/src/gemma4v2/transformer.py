"""
Gemma 4 V4 Transformer Blocks (DENSE - No MoE!)

Based on actual Gemma 4 31B architecture (google/gemma-4-31B-it):
- enable_moe_block: false (DENSE!)
- 5:1 interleaved local/global pattern
- QK-Norm + sandwich norm
- Standard GeGLU FFN (NO MoE!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layers import RMSNorm
from src.rotary import RotaryEmbedding
from src.gemma4v2.attention import Gemma4SlidingAttention, Gemma4FullAttention


class GeGLU(nn.Module):
    """GeGLU activation as used in Gemma 4"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma4SlidingAttentionBlock(nn.Module):
    """
    Gemma 4 Sliding Attention Block (DENSE)
    
    Pattern: input_norm → sliding_attention → post_attn_norm → +x
           → pre_ffn_norm → ffn (GeGLU) → post_ffn_norm → +x → * layer_scalar
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        sliding_window: int = 1024,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        use_sandwich_norm: bool = True,
        use_layer_scalar: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_sandwich_norm = use_sandwich_norm
        
        # Input norm (pre-norm)
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Sliding attention
        self.attn = Gemma4SlidingAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
            dropout=dropout,
        )
        
        # RoPE (theta=10000 for sliding)
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
        )
        self.attn.set_rotary_emb(self.rotary_emb)
        
        # Sandwich norm
        if use_sandwich_norm:
            self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.pre_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # FFN with GeGLU (Gemma 4 DENSE - no MoE!)
        self.ffn = GeGLU(hidden_size, intermediate_size)
        
        # Layer scalar
        if use_layer_scalar:
            self.layer_scalar = nn.Parameter(torch.ones(1) * math.sqrt(1.0 / 8))
        else:
            self.layer_scalar = None
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        # Pre-norm + attention
        h = self.input_norm(x)
        attn_out, _ = self.attn(h, attention_mask=attention_mask, position_ids=position_ids)
        
        # Post-attention residual
        if self.use_sandwich_norm:
            h = x + self.post_attn_norm(attn_out)
        else:
            h = x + attn_out
        
        # Pre-FFN + FFN (GeGLU) + post-FFN + residual
        if self.use_sandwich_norm:
            ffn_input = self.pre_ffn_norm(h)
            ffn_out = self.ffn(ffn_input)
            h = h + self.post_ffn_norm(ffn_out)
        else:
            h = h + self.ffn(h)
        
        # Layer scalar
        if self.layer_scalar is not None:
            h = h * self.layer_scalar
        
        return h, None


class Gemma4FullAttentionBlock(nn.Module):
    """
    Gemma 4 Full Attention Block (DENSE - NO MoE!)
    
    Used every 6th layer
    Pattern: input_norm → full_attention → post_attn_norm → +x
           → pre_ffn_norm → ffn (GeGLU) → post_ffn_norm → +x → * layer_scalar
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        use_sandwich_norm: bool = True,
        use_layer_scalar: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_sandwich_norm = use_sandwich_norm
        
        # Input norm
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Full attention
        self.attn = Gemma4FullAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        
        # RoPE (theta=1000000 for full attention)
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
        )
        self.attn.set_rotary_emb(self.rotary_emb)
        
        # Sandwich norm
        if use_sandwich_norm:
            self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.pre_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # FFN with GeGLU (Gemma 4 DENSE - no MoE!)
        self.mlp = GeGLU(hidden_size, intermediate_size)
        
        # Layer scalar
        if use_layer_scalar:
            self.layer_scalar = nn.Parameter(torch.ones(1) * math.sqrt(1.0 / 8))
        else:
            self.layer_scalar = None
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        # Pre-norm + attention
        h = self.input_norm(x)
        attn_out, _ = self.attn(h, attention_mask=attention_mask, position_ids=position_ids)
        
        # Post-attention residual
        if self.use_sandwich_norm:
            h = x + self.post_attn_norm(attn_out)
        else:
            h = x + attn_out
        
        # Pre-FFN + FFN (GeGLU) + post-FFN + residual
        if self.use_sandwich_norm:
            ffn_input = self.pre_ffn_norm(h)
            ffn_out = self.mlp(ffn_input)
            h = h + self.post_ffn_norm(ffn_out)
        else:
            h = h + self.mlp(h)
        
        # Layer scalar
        if self.layer_scalar is not None:
            h = h * self.layer_scalar
        
        return h, None
