"""
Hybrid Transformer Block
Combines Linear Attention (GatedDeltaNet) + Full Attention
Following Qwen3.6 pattern: 3× Linear + 1× Full per block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import RMSNorm
from src.attention import FullAttention
from src.linear_attention import GatedDeltaNetAttention, ChunkwiseDeltaAttention
from src.ffn import SwiGLUFeedForward
from src.rotary import RotaryEmbedding, precompute_freqs_cis


class LinearAttentionBlock(nn.Module):
    """
    Linear Attention Block using Gated DeltaNet
    
    Uses chunkwise processing for memory efficiency on RTX 3060.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 2048,
        conv_kernel_dim: int = 4,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        num_linear_heads: int = 8,
        linear_head_dim: int = 64
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pre-norm
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Gated DeltaNet Attention - simplified interface
        self.attn = ChunkwiseDeltaAttention(
            hidden_size=hidden_size,
            num_heads=num_linear_heads,
            head_dim=linear_head_dim,
            chunk_size=128,  # Larger chunks for parallelized matrix ops
            dropout=dropout
        )
        
        # Post-attention norm
        self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # FFN
        self.ffn = SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections"""
        # Pre-norm + attention + residual
        h = x + self.attn(self.input_norm(x), attention_mask)
        
        # Pre-norm + FFN + residual
        h = h + self.ffn(self.post_attn_norm(h))
        
        return h


class FullAttentionBlock(nn.Module):
    """
    Full Attention Block with GQA
    
    Standard Transformer attention with rotary embeddings.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 13824,
        max_seq_len: int = 4096,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pre-norm
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Full Attention with GQA
        self.attn = FullAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta
        )
        self.attn.set_rotary_emb(self.rotary_emb)
        
        # Post-attention norm
        self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # FFN
        self.ffn = SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass"""
        # Pre-norm + attention + residual
        h, present = self.attn(
            self.input_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        h = h + x
        
        # Pre-norm + FFN + residual
        h = h + self.ffn(self.post_attn_norm(h))
        
        return h, present


class HybridBlock(nn.Module):
    """
    Hybrid Block: 3× Linear Attention + 1× Full Attention
    
    Pattern from Qwen3.6:
    - 3 Gated DeltaNet layers (efficient, O(n) attention)
    - 1 Full Attention layer (resets state, enables long-range dependencies)
    - Each followed by SwiGLU FFN
    
    Total: 4 layers per block, repeated N times
    """
    
    def __init__(
        self,
        layer_idx: int,
        config,
        use_linear: bool = True
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_linear = use_linear
        
        if use_linear:
            self.linear_attn = LinearAttentionBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_linear_heads=config.linear_num_key_heads,
                linear_head_dim=config.linear_key_head_dim,
                conv_kernel_dim=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps
            )
        else:
            self.full_attn = FullAttentionBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                max_seq_len=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps
            )
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass"""
        if self.use_linear:
            return self.linear_attn(x, attention_mask), None
        else:
            return self.full_attn(
                x,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )


class HybridTransformer(nn.Module):
    """
    Hybrid Transformer with mixed Linear + Full Attention
    
    Architecture:
    - 8 layers total (6 linear + 2 full attention for 6GB)
    - Each block: 3× Linear + 1× Full
    - GQA in full attention layers
    - RMSNorm + pre-norm throughout
    - SwiGLU FFN
    - Supports gradient checkpointing for memory saving
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(0.0)  # No dropout during training
        
        # Build layers with pattern: Linear, Linear, Linear, Full
        self.layers = nn.ModuleList()
        
        for layer_idx in range(config.num_hidden_layers):
            # Every 4th layer (1-indexed: 4, 8, 12, ...) uses full attention
            is_full_attention = (layer_idx + 1) % config.full_attention_interval == 0
            self.layers.append(
                HybridBlock(layer_idx, config, use_linear=not is_full_attention)
            )
        
        # Final norm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Tie weights between embed and lm_head (optional)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will be tied
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        # For HuggingFace compatibility, we can also use torch.utils.checkpoint
        for layer in self.layers:
            layer.gradient_checkpointing_enable = True if hasattr(layer, 'gradient_checkpointing_enable') else None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False
    ):
        """
        Forward pass with optional gradient checkpointing
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: optional mask
            position_ids: optional positions
            use_cache: whether to return KV cache
            return_hidden_states: whether to return all layer outputs
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            (optional) hidden_states: list of layer outputs
        """
        B, L = input_ids.shape
        
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Collect hidden states if needed
        all_hidden_states = [] if return_hidden_states else None
        
        # Process layers with optional gradient checkpointing
        for layer_idx, layer in enumerate(self.layers):
            # Check if this is a full attention layer
            if hasattr(layer, 'full_attn'):
                h, present = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_value=None  # For training, no KV cache
                )
            else:
                h, present = layer(
                    hidden_states,
                    attention_mask=attention_mask
                )
            
            hidden_states = h
            
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final norm
        hidden_states = self.final_norm(hidden_states)
        
        # Output
        if self.lm_head is None:
            # Tie weights: use matrix multiplication equivalent
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())
        else:
            logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': all_hidden_states
        }


def tie_weights(model: HybridTransformer):
    """Tie embedding and output weights"""
    model.lm_head = None
