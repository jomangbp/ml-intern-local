"""
IMU-1 Style Attention - FIXED implementation based on reference code (thepowerfuldeez/imu1_base)

Techniques:
1. QK-Norm - RMS normalize Q, scalar gain on Q only (applied AFTER RoPE, AFTER KV repeat)
2. Per-Head Gating - 2*sigma(g_h) * Attention_h, default init for gate weights
3. Value Residuals - Mix current V with first layer V, threaded through all layers

Reference: arXiv:2602.02522 (IMU-1)
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


class Imu1QKNorm(nn.Module):
    """IMU-1 QK-Norm: RMS normalize Q and K, learnable scalar gain on Q ONLY.
    
    Applied AFTER RoPE and AFTER KV repetition.
    Gain is a single scalar (not per-dim), initialized to 1.0.
    """
    
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Single learnable scalar gain, initialized to 1.0
        self.gain = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [..., head_dim] - query tensors
            k: [..., head_dim] - key tensors
        Returns:
            Normalized (q, k) with gain applied only to q
        """
        # Compute in float32 for stability
        qf = q.float()
        kf = k.float()
        
        # RMS normalize: x * rsqrt(mean(x^2) + eps)
        qf = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Apply gain ONLY to Q, not K
        qf = qf * self.gain
        
        return qf.to(q.dtype), kf.to(k.dtype)


class PerHeadGating(nn.Module):
    """
    IMU-1 Per-Head Gating: out_h = 2 * sigma(g_h) * Attention_h
    
    g = W_g @ x, where W_g in R^{d x n_h}
    At init: sigma(0)=0.5, so 2*sigma(0)=1.0 (neutral)
    
    Uses default PyTorch init for gate weights (NOT zero init).
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
        # Projection: hidden_size -> num_heads (one scalar gate per head)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # Use default PyTorch init (fan-in: std = 1/sqrt(hidden_size))
        # NOT zero init — zero init kills gradient diversity
    
    def forward(self, x: torch.Tensor, attention_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, H] - input hidden states (pre-attention, for computing gates)
            attention_out: [B, L, num_heads * head_dim] - attention output
        Returns:
            gated_output: [B, L, num_heads * head_dim]
        """
        B, L, _ = x.shape
        
        # Compute per-head gates: 2 * sigmoid(g)
        gates = self.gate_proj(x).sigmoid() * 2.0  # [B, L, num_heads]
        
        # Reshape attention output and apply gating per head
        head_dim = attention_out.shape[-1] // self.num_heads
        attn_reshaped = attention_out.view(B, L, self.num_heads, head_dim)
        
        # Apply gating: broadcast gate_values to match head_dim
        gated = attn_reshaped * gates.unsqueeze(-1)  # [B, L, num_heads, head_dim]
        
        return gated.view(B, L, -1)


class ValueResidual(nn.Module):
    """
    IMU-1 Value Residual: V^(l) = s * (alpha1*V_local + alpha2*V^(1)) / sqrt(alpha1^2 + alpha2^2)
    
    Init: (s, alpha1, alpha2) = (1, 1, 0) -> starts as standard attention
    """
    
    def __init__(self, hidden_size: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        # Learnable parameters
        self.alpha1 = nn.Parameter(torch.tensor([1.0]))
        self.alpha2 = nn.Parameter(torch.tensor([0.0]))
        self.value_scale = nn.Parameter(torch.tensor([1.0]))
        self.value_norm_eps = 1e-8
    
    def forward(self, v_local: torch.Tensor, value_residual: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_local: [B, num_kv_heads, L, head_dim] - current layer's value projection (AFTER repeat_kv)
            value_residual: [B, num_heads, L, head_dim] or None - first layer's value (AFTER repeat_kv)
        Returns:
            mixed_v: mixed values for attention
            value_residual_out: to pass to next layer
        """
        v1 = v_local if value_residual is None else value_residual
        
        denom = torch.rsqrt(self.alpha1.square() + self.alpha2.square() + self.value_norm_eps)
        mixed = self.value_scale * (self.alpha1 * v_local + self.alpha2 * v1) * denom
        
        return mixed, mixed


class Hybrid5Attention(nn.Module):
    """IMU-1 Full Attention (fixed) with QK-Norm, Per-Head Gating, Value Residuals"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_qk_norm: bool = True,
        use_per_head_gating: bool = True,
        use_value_residuals: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.use_per_head_gating = use_per_head_gating
        self.use_value_residuals = use_value_residuals
        
        self.n_rep = num_heads // num_kv_heads
        
        # QK-Norm: single scalar gain, applied after RoPE
        if use_qk_norm:
            self.qk_norm = Imu1QKNorm(head_dim)
        
        # Value residuals (per-layer parameters)
        if use_value_residuals:
            self.value_residual_module = ValueResidual(hidden_size, num_heads, head_dim)
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Per-Head Gating
        if use_per_head_gating:
            self.head_gating = PerHeadGating(hidden_size, num_heads)
        
        self.rotary_emb = None
        self.dropout = dropout
    
    def set_rotary_emb(self, rotary_emb):
        self.rotary_emb = rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H = hidden_states.shape
        
        # Project Q, K, V -> [B, L, H, D]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Transpose to [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand KV heads (GQA) - BEFORE RoPE and QK-Norm
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
        
        # Apply RoPE FIRST
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)
        
        # QK-Norm AFTER RoPE and AFTER KV repetition (IMU-1 correct order)
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)
        
        # Value Residuals (after KV repeat)
        new_value_residual = None
        if self.use_value_residuals:
            v, new_value_residual = self.value_residual_module(v, value_residual)
        
        # Scaled dot-product attention (NO 1/sqrt(d) when QK-Norm is active)
        if self.use_qk_norm:
            # QK-Norm already bounds magnitudes, no additional scaling needed
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)  # [B, num_heads, L, D]
        output = output.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)  # [B, L, H]
        
        # Per-Head Gating (before o_proj)
        if self.use_per_head_gating:
            output = self.head_gating(hidden_states, output)
        
        output = self.o_proj(output)
        
        return output, new_value_residual


class Hybrid5SlidingAttention(nn.Module):
    """IMU-1 Sliding Window Attention (fixed) with QK-Norm, Per-Head Gating, Value Residuals"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int = 1024,
        use_qk_norm: bool = True,
        use_per_head_gating: bool = True,
        use_value_residuals: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.use_qk_norm = use_qk_norm
        self.use_per_head_gating = use_per_head_gating
        self.use_value_residuals = use_value_residuals
        
        self.n_rep = num_heads // num_kv_heads
        
        # QK-Norm
        if use_qk_norm:
            self.qk_norm = Imu1QKNorm(head_dim)
        
        # Value residuals (per-layer parameters)
        if use_value_residuals:
            self.value_residual_module = ValueResidual(hidden_size, num_heads, head_dim)
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Per-Head Gating
        if use_per_head_gating:
            self.head_gating = PerHeadGating(hidden_size, num_heads)
        
        self.rotary_emb = None
        self.dropout = dropout
    
    def set_rotary_emb(self, rotary_emb):
        self.rotary_emb = rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H = hidden_states.shape
        
        # Project Q, K, V -> [B, L, H, D]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Transpose to [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand KV heads (GQA) - BEFORE RoPE and QK-Norm
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.head_dim).reshape(
                B, self.num_heads, L, self.head_dim
            )
        
        # Apply RoPE FIRST
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)
        
        # QK-Norm AFTER RoPE and AFTER KV repetition
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)
        
        # Value Residuals (after KV repeat)
        new_value_residual = None
        if self.use_value_residuals:
            v, new_value_residual = self.value_residual_module(v, value_residual)
        
        # Scaled dot-product attention
        if self.use_qk_norm:
            # QK-Norm already bounds magnitudes
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Sliding window mask
        if L > self.sliding_window:
            mask = torch.zeros(L, L, device=attn_weights.device, dtype=torch.bool)
            mask[:, -self.sliding_window:] = True
            causal_mask = torch.triu(torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1)
            mask = mask & ~causal_mask
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal_mask = torch.triu(
                torch.ones(L, L, device=attn_weights.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        
        # Per-Head Gating (before o_proj)
        if self.use_per_head_gating:
            output = self.head_gating(hidden_states, output)
        
        output = self.o_proj(output)
        
        return output, new_value_residual
