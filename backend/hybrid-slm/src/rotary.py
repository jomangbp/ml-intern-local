"""
Rotary Position Embeddings (RoPE)
Efficient position encoding for extended context
"""

import torch
import torch.nn as nn
import math


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    Args:
        x: [..., dim] where dim is even
        
    Returns:
        rotated: [..., dim]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key.
    
    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        cos: [seq_len, head_dim/2] or [batch, seq_len, head_dim/2]
        sin: [seq_len, head_dim/2] or [batch, seq_len, head_dim/2]
        position_ids: [batch, seq_len] - optional explicit positions
        
    Returns:
        q: rotated query
        k: rotated key
    """
    # Handle multi-dimensional cos/sin
    if cos.dim() == 2:
        # [seq_len, dim] -> expand to [1, 1, seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # [batch, seq_len, dim] -> [batch, 1, seq_len, dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequencies for RoPE.
    
    Args:
        seq_len: maximum sequence length
        n_elem: dimension of the embeddings (must be even)
        theta: base for the exponential decay
        device: device to store tensors
        dtype: data type
        
    Returns:
        cos: [seq_len, n_elem/2]
        sin: [seq_len, n_elem/2]
    """
    # Compute angles
    freqs = 1.0 / (theta ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, n_elem/2]
    
    # Compute cos and sin
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    
    return cos, sin


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding layer
    
    This implementation is optimized for inference with caching of frequencies.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.dtype = dtype
        
        # Precompute frequencies
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # [max_seq_len, dim/2]
        
        self.register_buffer("freqs", freqs, persistent=False)
        
        # Cache for dynamic sequence lengths
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        position_ids: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            q: [batch, heads, seq_len, head_dim]
            k: [batch, heads, seq_len, head_dim]
            position_ids: optional position indices
            
        Returns:
            q: rotated query
            k: rotated key
        """
        seq_len = q.shape[2]
        
        # Recompute or retrieve from cache
        if seq_len > self._seq_len_cached:
            # Need to recompute
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.freqs.device, dtype=self.freqs.dtype)
            freqs = torch.outer(t, self.freqs[0]) if self.freqs.dim() == 1 else self.freqs[:seq_len]
            
            cos = freqs.cos().to(self.dtype)
            sin = freqs.sin().to(self.dtype)
            
            self._cos_cached = cos
            self._sin_cached = sin
        
        cos = self._cos_cached
        sin = self._sin_cached
        
        # Expand dimensions for broadcasting
        cos_expanded = cos[None, None, :, :]  # [1, 1, seq_len, rope_dim]
        sin_expanded = sin[None, None, :, :]  # [1, 1, seq_len, rope_dim]
        rope_len = cos.shape[-1]  # dim/2 = 32
        
        # For key (2 heads, head_dim=64)
        k_rope = k[..., :rope_len]  # [B, L, num_kv_heads, rope_len]
        k_embed = k.clone()
        k_embed[..., :rope_len] = (k_rope * cos_expanded) + (rotate_half(k_rope) * sin_expanded)
        
        # For query (14 heads, head_dim=64)
        q_rope = q[..., :rope_len]  # [B, L, num_heads, rope_len]
        q_embed = q.clone()
        q_embed[..., :rope_len] = (q_rope * cos_expanded) + (rotate_half(q_rope) * sin_expanded)
        
        return q_embed, k_embed
    
    @staticmethod
    def apply_rotary(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a single tensor (typically query)"""
        # q: [B, L, H, D], cos/sin: [1, 1, seq_len, rope_dim]
        rope_len = cos.shape[-1]
        
        q_rope = q[..., :rope_len]
        q_embed = q.clone()
        q_embed[..., :rope_len] = (q_rope * cos) + (rotate_half(q_rope) * sin)
        
        return q_embed
    
    def clear_cache(self):
        """Clear frequency cache"""
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None


class LinearRotaryEmbedding(nn.Module):
    """
    Optimized RoPE for linear attention (smaller head dimensions)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: torch.device = None
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Precompute inverse frequencies
        inv_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)
        
        # Build cache
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Build frequency cache"""
        t = torch.arange(seq_len, device=self.inv_freqs.device)
        freqs = torch.outer(t, self.inv_freqs)
        
        self.register_buffer("cos_cos", freqs.cos(), persistent=False)
        self.register_buffer("cos_sin", freqs.sin(), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE"""
        seq_len = q.shape[2]
        
        cos = self.cos_cos[:seq_len]
        sin = self.cos_sin[:seq_len]
        
        q_embed = torch.cat([
            q[..., :self.dim//2] * cos - rotate_half(q[..., :self.dim//2]) * sin,
            q[..., self.dim//2:] * cos - rotate_half(q[..., self.dim//2:]) * sin
        ], dim=-1)
        
        k_embed = torch.cat([
            k[..., :self.dim//2] * cos - rotate_half(k[..., :self.dim//2]) * sin,
            k[..., self.dim//2:] * cos - rotate_half(k[..., self.dim//2:]) * sin
        ], dim=-1)
        
        return q_embed, k_embed
