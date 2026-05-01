"""
IMU-1 Style Transformer Block - FIXED implementation

Changes from broken version:
1. LayerNorm Scaling applied to BOTH input_norm AND post_attn_norm (not just input_norm)
2. Value residuals threaded through all layers from layer 0
3. No double-scaling of pre_ffn_norm
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
from src.gemma4v2.hybrid5_attention import Hybrid5SlidingAttention, Hybrid5Attention


class Hybrid5SwiGLU(nn.Module):
    """SwiGLU activation as used in IMU-1"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Hybrid5TransformerBlock(nn.Module):
    """IMU-1 Style Transformer Block - FIXED
    
    LayerNorm Scaling: 1/sqrt(layer+1) applied to BOTH input_norm and post_attn_norm
    Value Residuals: threaded from layer 0 through all layers
    """
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
        intermediate_size: int, layer_idx: int, attention_type: str = "sliding",
        sliding_window: int = 1024, max_seq_len: int = 2048, rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6, use_per_head_gating: bool = True,
        use_layer_norm_scaling: bool = True, use_qk_norm: bool = True,
        use_value_residuals: bool = True, dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.attention_type = attention_type
        self.use_layer_norm_scaling = use_layer_norm_scaling
        self.use_value_residuals = use_value_residuals
        
        # LayerNorm Scaling: 1/sqrt(layer_idx+1) - 1-indexed as per IMU-1
        # Layer 0 -> scale = 1/sqrt(1) = 1.0 (no change)
        # Layer 7 -> scale = 1/sqrt(8) = 0.354
        self.layer_scale = 1.0 / math.sqrt(layer_idx + 1) if use_layer_norm_scaling else 1.0
        
        # Input norm (pre-norm) - SCALED
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Attention
        if attention_type == "sliding":
            self.attn = Hybrid5SlidingAttention(
                hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, sliding_window=sliding_window,
                use_qk_norm=use_qk_norm, use_per_head_gating=use_per_head_gating,
                use_value_residuals=use_value_residuals, dropout=dropout,
            )
        else:
            self.attn = Hybrid5Attention(
                hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, use_qk_norm=use_qk_norm,
                use_per_head_gating=use_per_head_gating,
                use_value_residuals=use_value_residuals, dropout=dropout,
            )
        
        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, base=rope_theta)
        self.attn.set_rotary_emb(self.rotary_emb)
        
        # Sandwich Norm - post_attn_norm is SCALED (per IMU-1)
        self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # FFN with SwiGLU
        self.ffn = Hybrid5SwiGLU(hidden_size, intermediate_size)
        
        # Layer scalar
        self.layer_scalar = nn.Parameter(torch.ones(1) * math.sqrt(1.0 / 8))
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                value_residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm with LayerNorm Scaling (applied to input_norm)
        h = self.input_norm(x) * self.layer_scale
        
        # Attention (with value residual threading)
        attn_out, new_value_residual = self.attn(
            h, attention_mask=attention_mask, position_ids=position_ids,
            value_residual=value_residual,
        )
        
        # Post-attention residual (post_attn_norm also scaled per IMU-1)
        h = x + self.post_attn_norm(attn_out) * self.layer_scale
        
        # Pre-FFN norm (NOT scaled - matches Gemma4 pattern, avoids double scaling)
        ffn_input = self.pre_ffn_norm(h)
        
        # FFN
        ffn_out = self.ffn(ffn_input)
        
        # Post-FFN residual (NOT scaled - only input_norm and post_attn_norm get the scale)
        h = h + self.post_ffn_norm(ffn_out)
        
        # Layer scalar
        h = h * self.layer_scalar
        
        return h, new_value_residual
