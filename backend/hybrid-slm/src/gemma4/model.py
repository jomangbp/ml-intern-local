"""
Gemma 4-Inspired SLM Model
~77M parameters, optimized for 6GB VRAM

Top-level model that combines:
- Embedding with tied weights
- Gemma 4-style transformer blocks (sandwich norm, layer scalar, dual head dim)
- Final norm
- Logit softcapping (Gemma 4 innovation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layers import RMSNorm
from src.gemma4.config import Gemma4SLMConfig
from src.gemma4.transformer import (
    Gemma4FullAttentionBlock,
    Gemma4LinearAttentionBlock,
)


class Gemma4SLMModel(nn.Module):
    """
    Gemma 4-Inspired Small Language Model
    
    Architecture:
    - 8 layers (6 linear GatedDeltaNet + 2 full attention)
    - Full attention uses Gemma 4 innovations: dual head dim, K=V shared, QK-norm
    - Sandwich norm (4 RMSNorms per layer + layer scalar)
    - GeLU(tanh) activation
    - Logit softcapping to [-30, 30]
    - Tied embeddings
    """
    
    def __init__(self, config: Gemma4SLMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Build layers
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            is_full_attention = (layer_idx + 1) % config.full_attention_interval == 0
            
            if is_full_attention:
                layer = Gemma4FullAttentionBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_global_key_value_heads,
                    head_dim=config.global_head_dim,  # Dual: wider heads for global
                    intermediate_size=config.intermediate_size,
                    max_seq_len=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                    use_layer_scalar=config.use_layer_scalar,
                    use_qk_norm=config.use_qk_norm,
                )
            else:
                layer = Gemma4LinearAttentionBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_linear_heads=config.linear_num_key_heads,
                    linear_head_dim=config.linear_key_head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                    use_layer_scalar=config.use_layer_scalar,
                )
            
            self.layers.append(layer)
        
        # Final norm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # No separate lm_head when tied (use embed_tokens.weight)
        self.lm_head = None
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Consistent initialization"""
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        
        # Scale embeddings by sqrt(hidden_size) (Gemma 4 does this)
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)
        
        # Position ids
        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Process layers
        for layer in self.layers:
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'rotary_emb'):
                # Full attention block
                hidden_states, _ = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                )
            else:
                # Linear attention block
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )
        
        # Final norm
        hidden_states = self.final_norm(hidden_states)
        
        # Compute logits (tied embeddings)
        logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())
        
        # Logit softcapping (Gemma 4 innovation)
        # Cap logits to [-30, 30] range for training stability
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        
        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self(input_ids=generated)
            
            logits = outputs['logits'][:, -1, :]
            
            # Undo softcapping for generation
            if self.config.final_logit_softcapping is not None:
                cap = self.config.final_logit_softcapping
                # Already capped, but temperature scaling works on capped values
            
            logits = logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated
    
    @classmethod
    def from_pretrained(cls, path: str, config: Gemma4SLMConfig = None):
        if config is None:
            config = Gemma4SLMConfig()
        model = cls(config)
        checkpoint_path = f"{path}/pytorch_model.bin"
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}")
        return model


def create_gemma4_model(config: Gemma4SLMConfig = None) -> Gemma4SLMModel:
    """Factory function to create Gemma 4 SLM model"""
    if config is None:
        config = Gemma4SLMConfig()
    
    model = Gemma4SLMModel(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nGemma 4 SLM created:")
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  Architecture: Gemma 4-inspired hybrid (Linear + Full Attention)")
    full_layers = config.num_hidden_layers // config.full_attention_interval
    linear_layers = config.num_hidden_layers - full_layers
    print(f"  Layers: {config.num_hidden_layers} ({linear_layers} linear + {full_layers} full)")
    print(f"  Local head dim: {config.head_dim}")
    print(f"  Global head dim: {config.global_head_dim}")
    print(f"  Sandwich norm: {config.use_sandwich_norm}")
    print(f"  Layer scalar: {config.use_layer_scalar}")
    print(f"  Logit softcapping: {config.final_logit_softcapping}")
    print(f"  Activation: {config.hidden_act}")
    
    return model
