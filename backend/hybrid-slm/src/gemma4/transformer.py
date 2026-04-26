"""
Gemma 4 Transformer Block

Key innovations:
1. Sandwich norm: 4 RMSNorms per layer (pre-attention, post-attention, pre-FFN, post-FFN)
2. Learnable layer scalar per layer
3. Residual connections after post-norms (Gemma 2/4 style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layers import RMSNorm
from src.rotary import RotaryEmbedding


class GeLUFFN(nn.Module):
    """Gated MLP with GeLU(tanh) activation (Gemma 4 uses this instead of SwiGLU/SiLU)
    
    Formula: FFN(x) = down(gelu(gate(x)) * up(x))
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GeLU(tanh) activation as in Gemma 4
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma4FullAttentionBlock(nn.Module):
    """
    Gemma 4 Full Attention Block with sandwich norm and layer scalar.
    
    Pattern:
    x -> input_norm -> attention -> post_attn_norm -> +x (residual)
      -> pre_ffn_norm -> ffn -> post_ffn_norm -> +x (residual)
      -> * layer_scalar
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        use_sandwich_norm: bool = True,
        use_layer_scalar: bool = True,
        use_qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_sandwich_norm = use_sandwich_norm
        
        # Import here to avoid circular imports
        from src.gemma4.attention import Gemma4FullAttention
        
        # Attention
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = Gemma4FullAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            dropout=dropout,
        )
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
        )
        self.attn.set_rotary_emb(self.rotary_emb)
        
        # Post-attention norm (sandwich)
        if use_sandwich_norm:
            self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.pre_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        else:
            self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.pre_ffn_norm = None
            self.post_ffn_norm = None
        
        # FFN
        self.ffn = GeLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Layer scalar (Gemma 4 / Gemma 2)
        if use_layer_scalar:
            self.layer_scalar = nn.Parameter(torch.ones(1) * math.sqrt(1.0 / 8))
        else:
            self.layer_scalar = None
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Pre-norm + attention
        h = self.input_norm(x)
        attn_out, present = self.attn(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        
        # Post-attention norm + residual
        if self.use_sandwich_norm:
            h = x + self.post_attn_norm(attn_out)
        else:
            h = x + attn_out
        
        # Pre-FFN norm + FFN + post-FFN norm + residual
        if self.use_sandwich_norm:
            ffn_input = self.pre_ffn_norm(h)
            ffn_out = self.ffn(ffn_input)
            h = h + self.post_ffn_norm(ffn_out)
        else:
            h = h + self.ffn(h)
        
        # Layer scalar
        if self.layer_scalar is not None:
            h = h * self.layer_scalar
        
        return h, present


class Gemma4LinearAttentionBlock(nn.Module):
    """
    Linear Attention Block (GatedDeltaNet) with Gemma 4 sandwich norm.
    
    Uses the existing ChunkwiseDeltaAttention but wraps it with:
    - Sandwich norm (4 norms per layer)
    - Layer scalar
    - GeLU FFN
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_linear_heads: int = 4,
        linear_head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        use_sandwich_norm: bool = True,
        use_layer_scalar: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_sandwich_norm = use_sandwich_norm
        
        from src.linear_attention import ChunkwiseDeltaAttention
        
        # Attention
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = ChunkwiseDeltaAttention(
            hidden_size=hidden_size,
            num_heads=num_linear_heads,
            head_dim=linear_head_dim,
            chunk_size=128,
            dropout=dropout,
        )
        
        # Post-attention norm (sandwich)
        if use_sandwich_norm:
            self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.pre_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        else:
            self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.pre_ffn_norm = None
            self.post_ffn_norm = None
        
        # FFN (GeLU as in Gemma 4)
        self.ffn = GeLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Layer scalar
        if use_layer_scalar:
            self.layer_scalar = nn.Parameter(torch.ones(1) * math.sqrt(1.0 / 8))
        else:
            self.layer_scalar = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        # Pre-norm + attention
        h = self.input_norm(x)
        attn_out = self.attn(h, attention_mask)
        
        # Post-attention norm + residual
        if self.use_sandwich_norm:
            h = x + self.post_attn_norm(attn_out)
        else:
            h = x + attn_out
        
        # Pre-FFN norm + FFN + post-FFN norm + residual
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
