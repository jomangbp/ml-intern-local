"""
Hybrid SLM Model
Small Language Model with Hybrid Linear + Full Attention

150M parameters targeting SOTA performance for models under 500M
Optimized for RTX 3060 Mobile (12GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer import HybridTransformer, tie_weights
from src.layers import RMSNorm
from configs.model_config import HybridSLMConfig


class HybridSLMModel(nn.Module):
    """
    Hybrid Small Language Model
    
    Architecture: Hybrid (GatedDeltaNet + Full Attention)
    Parameters: 150M
    Context: 4096 tokens
    Precision: bfloat16
    """
    
    def __init__(self, config: HybridSLMConfig):
        super().__init__()
        self.config = config
        
        # Main transformer
        self.model = HybridTransformer(config)
        
        # Tie weights (LLaMA style)
        tie_weights(self.model)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Delegate to the inner model
        if hasattr(self, 'model') and hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        self._gradient_checkpointing_enabled = True
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following stable initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding
            position_ids: [batch, seq_len]
            labels: [batch, seq_len] - for training
            use_cache: whether to return KV cache
            
        Returns:
            Dictionary with:
            - loss: cross-entropy loss (if labels provided)
            - logits: [batch, seq_len, vocab_size]
            - hidden_states: list of layer outputs (optional)
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache
        )
        
        logits = outputs['logits']
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.get('hidden_states')
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Simple text generation
        
        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus filtering
            
        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self(input_ids=generated, use_cache=True)
            
            logits = outputs['logits'][:, -1, :]  # [batch, vocab]
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop if EOS token
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated
    
    @classmethod
    def from_pretrained(cls, path: str, config: HybridSLMConfig = None):
        """Load pretrained model"""
        if config is None:
            config = HybridSLMConfig()
        
        model = cls(config)
        
        # Load weights if checkpoint exists
        checkpoint_path = f"{path}/pytorch_model.bin"
        if path:
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                print(f"No checkpoint found at {checkpoint_path}, using random initialization")
        
        return model


class HybridSLMForConditionalGeneration(nn.Module):
    """
    Model wrapper for chat/instruction following
    """
    
    def __init__(self, config: HybridSLMConfig):
        super().__init__()
        self.config = config
        self.model = HybridSLMModel(config)
        
        # Optional: use pad token as eos
        if config.pad_token_id is None:
            self.config.pad_token_id = config.eos_token_id
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )


def create_model(config: HybridSLMConfig = None) -> HybridSLMModel:
    """Factory function to create model"""
    if config is None:
        config = HybridSLMConfig()
    
    model = HybridSLMModel(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Architecture: Hybrid (GatedDeltaNet + Full Attention)")
    print(f"  Layers: {config.num_hidden_layers} ({config.num_hidden_layers // config.full_attention_interval} full, "
          f"{config.num_hidden_layers - config.num_hidden_layers // config.full_attention_interval} linear)")
    
    return model


# Export
__all__ = [
    'HybridSLMModel',
    'HybridSLMForConditionalGeneration',
    'HybridTransformer',
    'create_model',
    'HybridSLMConfig'
]
