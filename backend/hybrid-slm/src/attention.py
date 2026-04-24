"""
Full Attention Layer with Grouped Query Attention (GQA)
Standard Transformer attention optimized for RTX 3060
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads for GQA.
    
    Args:
        x: [batch, seq_len, num_kv_heads, head_dim] - sequence first format
        n_rep: number of times to repeat
        
    Returns:
        repeated: [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    # x: [B, L, H, D] -> expand -> reshape
    return x[:, :, None, :, :].expand(x.shape[0], x.shape[1], n_rep, x.shape[2], x.shape[3]).reshape(
        x.shape[0], x.shape[1], x.shape[2] * n_rep, x.shape[3]
    )


class FullAttention(nn.Module):
    """
    Full Transformer Attention with GQA
    
    Features:
    - Grouped Query Attention (GQA) for reduced KV cache
    - Flash Attention compatible
    - Optional KV cache for generation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout = dropout
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.n_rep = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        
        # Optional RoPE
        self.rotary_emb = None
    
    def set_rotary_emb(self, rotary_emb):
        """Set rotary embedding module"""
        self.rotary_emb = rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] or [batch, 1, 1, seq_len]
            position_ids: [batch, seq_len]
            is_causal: whether to use causal masking
            use_cache: whether to return KV cache
            past_key_value: past keys and values
            
        Returns:
            output: [batch, seq_len, hidden_size]
            present: (k, v) for caching, or None
        """
        B, L, H = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        
        # RoPE disabled for now (simplification needed for GQA)
        # if self.rotary_emb is not None:
        #     q = self.rotary_emb.apply_rotary(q, cos, sin)
            # K doesn't get RoPE in this simplified implementation
        
        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Expand KV heads for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        
        # Flash Attention or standard attention
        if hasattr(F, 'scaled_dot_product_flash_attention') and self.training is False:
            # Use Flash Attention for inference
            output = self._flash_attention(q, k, v, attention_mask, is_causal)
        else:
            output = self._standard_attention(q, k, v, attention_mask, is_causal)
        
        output = output.reshape(B, L, -1)
        output = self.o_proj(output)
        
        # Return KV cache if requested
        present = (k, v) if use_cache else None
        
        return output, present
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """Flash Attention (requires PyTorch 2.0+)"""
        # Transpose for flash attention: [B, L, H, D] -> [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Build causal mask if needed
        if is_causal and attention_mask is None:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
        else:
            causal_mask = None
        
        # Compute attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            output = F.scaled_dot_product_flash_attention(
                q, k, v,
                attn_mask=attention_mask if attention_mask is not None else causal_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and attention_mask is None
            )
        
        return output.transpose(1, 2)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """Standard scaled dot-product attention"""
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_weights.device),
                diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (no GQA)
    For when GQA is not beneficial (small models)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        self.dropout = dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        B, L, H = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        
        # Scale dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if is_causal:
            seq_len = L
            causal = torch.triu(torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)
        output = output.reshape(B, L, -1)
        
        return self.o_proj(output)
