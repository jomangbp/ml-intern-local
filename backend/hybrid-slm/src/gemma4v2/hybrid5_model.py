"""
Hybrid5 SLM Model - FIXED implementation

Changes:
1. Value residuals threaded from layer 0 through all layers
2. LayerNorm Scaling on both input_norm and post_attn_norm
3. Correct QK-Norm (scalar gain, Q-only, after RoPE)
4. Per-Head Gating with proper init
5. No 1/sqrt(d) scaling when QK-Norm is active
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layers import RMSNorm
from src.gemma4v2.hybrid5_config import Hybrid5Config
from src.gemma4v2.hybrid5_transformer import Hybrid5TransformerBlock


class Hybrid5Model(nn.Module):
    """Hybrid5 Small Language Model - Fixed IMU-1 implementation"""
    
    def __init__(self, config: Hybrid5Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        layer_pattern = config.get_layer_pattern()
        self.layers = nn.ModuleList()
        
        for layer_idx, layer_type in enumerate(layer_pattern):
            if layer_type == "full":
                layer = Hybrid5TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_global_key_value_heads,
                    head_dim=config.global_head_dim,
                    intermediate_size=config.intermediate_size,
                    layer_idx=layer_idx,
                    attention_type="full",
                    max_seq_len=config.max_position_embeddings,
                    rope_theta=config.rope_theta_full,
                    rms_norm_eps=config.rms_norm_eps,
                    use_per_head_gating=config.use_per_head_gating,
                    use_layer_norm_scaling=config.use_layer_norm_scaling,
                    use_qk_norm=config.use_qk_norm,
                    use_value_residuals=config.use_value_residuals,
                )
            else:
                layer = Hybrid5TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    layer_idx=layer_idx,
                    attention_type="sliding",
                    sliding_window=config.sliding_window,
                    max_seq_len=config.max_position_embeddings,
                    rope_theta=config.rope_theta_sliding,
                    rms_norm_eps=config.rms_norm_eps,
                    use_per_head_gating=config.use_per_head_gating,
                    use_layer_norm_scaling=config.use_layer_norm_scaling,
                    use_qk_norm=config.use_qk_norm,
                    use_value_residuals=config.use_value_residuals,
                )
            self.layers.append(layer)
        
        # Final norm - NO LayerNorm Scaling (per IMU-1)
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = None
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
    
    def _move_rotary_to_device(self, device):
        """Move rotary embedding buffers to the correct device"""
        for layer in self.layers:
            layer.rotary_emb.freqs = layer.rotary_emb.freqs.to(device)
            if hasattr(layer.rotary_emb, '_cos_cached') and layer.rotary_emb._cos_cached is not None:
                layer.rotary_emb._cos_cached = layer.rotary_emb._cos_cached.to(device)
                layer.rotary_emb._sin_cached = layer.rotary_emb._sin_cached.to(device)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        B, L = input_ids.shape
        
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        
        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Thread value_residual through all layers
        # Layer 0 produces the first value_residual, all subsequent layers use it
        value_residual = None
        
        for layer in self.layers:
            hidden_states, value_residual = layer(
                hidden_states, 
                position_ids=position_ids, 
                attention_mask=attention_mask,
                value_residual=value_residual,
            )
        
        hidden_states = self.final_norm(hidden_states)
        logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())
        
        cap = self.config.final_logit_softcapping
        logits = cap * torch.tanh(logits / cap)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
        
        return {'loss': loss, 'logits': logits}
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, do_sample=True):
        self.eval()
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self(input_ids=generated)
            logits = outputs['logits'][:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated


def create_hybrid5_model(config=None):
    if config is None:
        config = Hybrid5Config()
    
    model = Hybrid5Model(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_pattern = config.get_layer_pattern()
    num_full = sum(1 for t in layer_pattern if t == "full")
    num_sliding = len(layer_pattern) - num_full
    
    print(f"\n{'='*60}")
    print(f"Hybrid5 SLM Model Created (FIXED)")
    print(f"{'='*60}")
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"\nFixed Techniques (IMU-1):")
    print(f"  1. QK-Norm: scalar gain on Q only, after RoPE")
    print(f"  2. Per-Head Gating: default init, 2*sigma gate")
    print(f"  3. Value Residuals: threaded from layer 0")
    print(f"  4. LayerNorm Scaling: 1/sqrt(l+1) on input+post_attn norms")
    print(f"  5. No 1/sqrt(d) when QK-Norm active")
    print(f"\nArchitecture (Gemma 4 pattern):")
    print(f"  Layers: {config.num_hidden_layers} ({num_sliding} sliding + {num_full} full)")
    print(f"  Pattern: {layer_pattern}")
    print(f"  Sliding window: {config.sliding_window} tokens")
    print(f"{'='*60}")
    
    return model
