"""
Gemma 4 V4 SLM Model (DENSE - No MoE!)
~77M parameters

Based on actual Gemma 4 architecture from google/gemma-4-31B-it:
- DENSE model: enable_moe_block=false
- 5:1 interleaved local/global pattern
- Sliding window attention (1024 tokens)
- QK-Norm
- GeGLU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layers import RMSNorm
from src.gemma4v2.config import Gemma4Config, Gemma4TrainingConfig
from src.gemma4v2.transformer import Gemma4SlidingAttentionBlock, Gemma4FullAttentionBlock


class Gemma4Model(nn.Module):
    """
    Gemma 4-Inspired Small Language Model v4 (DENSE)
    
    Based on actual Gemma 4 31B architecture (google/gemma-4-31B-it):
    - DENSE: enable_moe_block=false
    - 5:1 interleaved local/global attention pattern
    - Sliding window attention (1024 tokens)
    - Standard FFN (NO MoE!)
    - QK-Norm + sandwich norm
    - GeGLU activation
    - Final logit soft-capping: 30.0
    """
    
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Build layers with 5:1 interleaved pattern (Gemma 4 DENSE)
        layer_pattern = config.get_layer_pattern()
        self.layers = nn.ModuleList()
        
        for layer_idx, layer_type in enumerate(layer_pattern):
            if layer_type == "full":
                # Full attention layer with standard FFN
                layer = Gemma4FullAttentionBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_global_key_value_heads,
                    head_dim=config.global_head_dim,
                    intermediate_size=config.intermediate_size,
                    max_seq_len=config.max_position_embeddings,
                    rope_theta=config.rope_theta_full,  # 1M for full attention
                    rms_norm_eps=config.rms_norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                    use_layer_scalar=config.use_layer_scalar,
                )
            else:
                # Sliding attention layer with standard FFN
                layer = Gemma4SlidingAttentionBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    sliding_window=config.sliding_window,
                    max_seq_len=config.max_position_embeddings,
                    rope_theta=config.rope_theta_sliding,  # 10k for sliding
                    rms_norm_eps=config.rms_norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                    use_layer_scalar=config.use_layer_scalar,
                )
            
            self.layers.append(layer)
        
        # Final norm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # No separate lm_head when tied
        self.lm_head = None
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count layer types
        num_full = sum(1 for t in layer_pattern if t == "full")
        num_sliding = len(layer_pattern) - num_full
        self.num_full_layers = num_full
        self.num_sliding_layers = num_sliding
    
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
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        
        # Embed (with Gemma scaling)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)
        
        # Position ids
        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Process layers with 5:1 interleaved pattern
        for layer in self.layers:
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'rotary_emb'):
                # Full attention block
                hidden_states, _ = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
            else:
                # Sliding attention block
                hidden_states, _ = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
        
        # Final norm
        hidden_states = self.final_norm(hidden_states)
        
        # Compute logits (tied embeddings)
        logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())
        
        # Apply final logit soft-capping (Gemma 4: cap=30.0)
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
    def from_pretrained(cls, path: str, config: Gemma4Config = None):
        if config is None:
            config = Gemma4Config()
        model = cls(config)
        checkpoint_path = f"{path}/pytorch_model.bin"
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}")
        return model


def create_gemma4_model(config: Gemma4Config = None) -> Gemma4Model:
    """Factory function to create Gemma 4 SLM model (DENSE)"""
    if config is None:
        config = Gemma4Config()
    
    model = Gemma4Model(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_pattern = config.get_layer_pattern()
    num_full = sum(1 for t in layer_pattern if t == "full")
    num_sliding = len(layer_pattern) - num_full
    
    print(f"\nGemma 4 V4 SLM (DENSE) created:")
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  Architecture: Gemma 4-style DENSE (5:1 interleaved, NO MoE)")
    print(f"  Layers: {config.num_hidden_layers} ({num_sliding} sliding + {num_full} full)")
    print(f"  Pattern: {layer_pattern}")
    print(f"  Sliding window: {config.sliding_window} tokens")
    print(f"  GQA: {config.num_key_value_heads} KV heads, {config.num_attention_heads} Q heads")
    print(f"  Final logit softcap: {config.final_logit_softcapping}")
    print(f"  Activation: {config.hidden_act}")
    print(f"  Vocab size: {config.vocab_size}")
    
    return model
