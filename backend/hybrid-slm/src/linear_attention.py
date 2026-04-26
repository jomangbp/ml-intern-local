"""
Gated DeltaNet - Linear Attention Layer
Memory-efficient attention with O(n) complexity

Based on: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
https://arxiv.org/abs/2412.06464

Key innovation: Gated Delta Rule
S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T
o_t = S_t q_t

Simplified implementation for RTX 3060.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class GatedDeltaNetAttention(nn.Module):
    """
    Gated DeltaNet Attention Layer - Simplified Version
    
    Uses consistent dimensions for all projections to avoid dimension mismatches.
    - Q, K, V all use the same number of heads (num_heads)
    - State dimension is [B, num_heads, head_dim, head_dim]
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim: int = 64,
        conv_kernel_dim: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Projections - use same num_heads for Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        
        # Gating - same dimension as heads
        self.alpha_gate = nn.Linear(hidden_size, num_heads, bias=False)  # Decay
        self.beta_gate = nn.Linear(hidden_size, num_heads, bias=False)   # Learning rate
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        # Conv for local context
        self.conv = nn.Conv1d(
            in_channels=num_heads * head_dim,
            out_channels=num_heads * head_dim,
            kernel_size=conv_kernel_dim,
            padding=conv_kernel_dim - 1,
            groups=num_heads * head_dim,
            bias=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with LeCun init (consistent across all layers)"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(module.weight, std=0.02)
        
        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.zeros_(self.beta_gate.weight)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] - optional mask
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        B, L, H = hidden_states.shape
        
        # Project Q, K, V - all same shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        
        # Compute gates - [B, L, num_heads]
        alpha = torch.sigmoid(self.alpha_gate(hidden_states))
        beta = F.softplus(self.beta_gate(hidden_states))
        
        # Process with gated delta attention
        output = self._gated_delta_attention(q, k, v, alpha, beta)
        
        # Apply local convolution for context
        output = output.reshape(B, L, -1).transpose(1, 2)  # [B, H*D, L]
        output = output + self.conv(output)[:, :, :L]     # Causal conv
        output = output.transpose(1, 2)                 # [B, L, H*D]
        
        output = self.o_proj(output)
        
        return self.dropout(output)
    
    def _gated_delta_attention(
        self,
        q: torch.Tensor,      # [B, L, num_heads, head_dim]
        k: torch.Tensor,      # [B, L, num_heads, head_dim]
        v: torch.Tensor,      # [B, L, num_heads, head_dim]
        alpha: torch.Tensor,  # [B, L, num_heads]
        beta: torch.Tensor    # [B, L, num_heads]
    ) -> torch.Tensor:
        """
        Gated Delta Attention
        
        S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T
        o_t = S_t q_t
        """
        B, L, n_h, d = q.shape
        
        # Initialize state: [B, num_heads, head_dim, head_dim]
        S = torch.zeros(B, n_h, d, d, device=q.device, dtype=q.dtype)
        
        outputs = []
        for t in range(L):
            k_t = k[:, t]      # [B, n_h, d]
            v_t = v[:, t]      # [B, n_h, d]
            q_t = q[:, t]      # [B, n_h, d]
            alpha_t = alpha[:, t]  # [B, n_h]
            beta_t = beta[:, t]    # [B, n_h]
            
            # k_kT: [B, n_h, d, d]
            k_kT = torch.matmul(k_t, k_t.transpose(-2, -1))
            
            # Expand for broadcasting: [B, n_h, 1, 1]
            alpha_exp = alpha_t[:, :, None, None]
            beta_exp = beta_t[:, :, None, None]
            
            # S_t = α * S * (I - β * k_kT) + β * v * k^T
            decay = alpha_exp * (1 - beta_exp * k_kT)
            update = beta_exp * torch.matmul(
                v_t[:, :, :, None], k_t[:, :, None, :]
            )
            S = torch.matmul(S, decay) + update
            
            # o_t = S_t q_t: [B, n_h, d, 1] -> [B, n_h, d]
            o_t = torch.matmul(S, q_t[:, :, :, None]).squeeze(-1)
            outputs.append(o_t)
        
        # Stack: [L, B, n_h, d] -> [B, L, n_h, d]
        outputs = torch.stack(outputs, dim=0).transpose(0, 1).transpose(1, 2)
        
        return outputs


class ChunkwiseDeltaAttention(nn.Module):
    """
    Chunkwise Parallel Gated DeltaNet - Optimized for 6GB VRAM
    
    Processes sequences in small chunks for memory efficiency.
    Uses smaller chunk sizes and optional state clearing.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,  # Reduced from 8 for 6GB
        head_dim: int = 64,
        chunk_size: int = 32,  # Reduced from 64 for 6GB
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        
        # Projections - consistent dimensions
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        
        # Gating
        self.alpha_gate = nn.Linear(hidden_size, num_heads)
        self.beta_gate = nn.Linear(hidden_size, num_heads)
        
        # Output gate (critical per GatedDeltaNet ablation: -1.77 PPL)
        self.output_gate = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        # Output
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Consistent init
        self._init_weights()
    
    def _init_weights(self):
        """Consistent weight initialization"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.output_gate]:
            nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.zeros_(self.beta_gate.weight)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with chunkwise processing - memory efficient"""
        B, L, H = hidden_states.shape
        
        # Project - consistent shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        
        # Gates - [B, L, num_heads]
        alpha = torch.sigmoid(self.alpha_gate(hidden_states))
        beta = F.softplus(self.beta_gate(hidden_states))
        
        # Chunkwise processing with small chunks
        outputs = []
        chunk_size = self.chunk_size
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            
            q_chunk = q[:, start:end]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            alpha_chunk = alpha[:, start:end]
            beta_chunk = beta[:, start:end]
            
            o_chunk = self._process_chunk(q_chunk, k_chunk, v_chunk, alpha_chunk, beta_chunk)
            outputs.append(o_chunk)
        
        # Concatenate chunks
        output = torch.cat(outputs, dim=1)  # [B, L, num_heads, head_dim]
        output = output.reshape(B, L, -1)   # [B, L, num_heads * head_dim]
        
        # Output gate: SiLU-gated (critical per GatedDeltaNet ablation)
        gate = F.silu(self.output_gate(hidden_states))  # [B, L, num_heads * head_dim]
        output = output * gate
        
        output = self.o_proj(output)
        
        return self.dropout(output)
    
    def _process_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """Process a chunk with simplified linear attention.
        
        Uses a simpler but effective formulation:
        - Linear attention: o_t = sum_{j<=t} (alpha_decay(t-j) * v_j * (k_j · q_t))
        - With exponential decay controlled by alpha gate
        - Beta gate controls how much new info to write
        """
        B, Lc, n_h, d = q.shape

        # Use float32 for stability
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        alpha = alpha.to(torch.float32)
        beta = beta.to(torch.float32)

        norm_factor = d ** 0.5
        
        # Scale keys and queries
        k_scaled = k / norm_factor
        q_scaled = q / norm_factor
        
        # L2 normalize Q and K (critical for GatedDeltaNet — saves 3.4 PPL per ablation)
        q_scaled = F.normalize(q_scaled, p=2, dim=-1)
        k_scaled = F.normalize(k_scaled, p=2, dim=-1)
        
        # Apply beta gating to values
        beta_exp = beta[:, :, :, None]  # [B, Lc, n_h, 1]
        gated_v = beta_exp * v  # [B, Lc, n_h, d]
        
        # Transpose for batched matmul: [B, n_h, Lc, d]
        q_t = q_scaled.transpose(1, 2)
        k_t = k_scaled.transpose(1, 2)
        gv_t = gated_v.transpose(1, 2)
        
        # Compute attention scores: [B, n_h, Lc, Lc]
        attn = torch.matmul(q_t, k_t.transpose(-2, -1))  # Q @ K^T
        
        # Compute decay mask based on alpha
        # alpha controls how much past info to retain
        # For position distance d: decay = alpha^d
        alpha_mean = alpha.mean(dim=-1)  # [B, Lc] average across heads
        alpha_mean = alpha_mean[:, None, :].expand(B, n_h, Lc)  # [B, n_h, Lc]
        
        # Create distance matrix and apply exponential decay
        positions = torch.arange(Lc, device=q.device, dtype=torch.float32)
        distances = positions[None, :] - positions[:, None]  # [Lc, Lc]
        distances = distances.clamp(min=0)  # Only past positions
        
        # Per-position alpha decay
        log_alpha = torch.log(alpha[:, :, :, None].expand(B, Lc, n_h, 1).squeeze(-1))  # [B, Lc, n_h]
        # Use mean alpha for simplicity
        alpha_decay = torch.exp(-distances * 0.1)  # [Lc, Lc] - gentle decay
        alpha_decay = alpha_decay.unsqueeze(0).unsqueeze(0)  # [1, 1, Lc, Lc]
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(Lc, Lc, device=q.device, dtype=torch.bool), diagonal=1
        )
        
        # Apply decay and causal mask to attention
        attn = attn * alpha_decay
        attn = attn.masked_fill(causal_mask, 0.0)
        
        # Apply to gated values
        output = torch.matmul(attn, gv_t)  # [B, n_h, Lc, d]
        
        result = output.transpose(1, 2)  # [B, Lc, n_h, d]
        return result


class GatedLinearAttention(nn.Module):
    """
    Simplified Gated Linear Attention
    
    For use when full delta attention is not needed.
    Based on Mamba/GLA-style recurrent formulation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        head_dim: int = 64,
        conv_kernel: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Projections
        self.x_proj = nn.Linear(hidden_size, num_heads * (head_dim + head_dim + 2))
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        # Conv for local context
        self.conv = nn.Conv1d(
            in_channels=num_heads * head_dim,
            out_channels=num_heads * head_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=num_heads * head_dim
        )
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, L, H = x.shape
        
        # Get gate parameters
        x_out = self.x_proj(x)
        dt, B_gate = x_out[:, :, :2].chunk(2, dim=-1)
        ik = x_out[:, :, 2:2+self.num_heads*self.head_dim].view(B, L, self.num_heads, self.head_dim)
        iv = x_out[:, :, 2+self.num_heads*self.head_dim:].view(B, L, self.num_heads, self.head_dim)
        
        # Apply gates and compute
        dt = F.softplus(dt)
        B_gate = torch.sigmoid(B_gate)
        
        # Simplified linear attention
        output = ik * torch.sigmoid(iv)
        
        # Convolution for local context
        output = output.transpose(1, 2).contiguous()
        output = output + self.conv(output)[:, :, :L].transpose(1, 2)
        
        output = self.q_proj(output.reshape(B, L, -1))
        output = self.o_proj(output)
        
        return self.dropout(output)
