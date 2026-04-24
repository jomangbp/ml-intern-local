"""
Hybrid SLM Configuration for 150M parameters
Target: SOTA for models under 500M parameters

Architecture: 3× GatedDeltaNet + 1× FullAttention blocks
Hardware: RTX 3060 Mobile (12GB VRAM) / 6GB VRAM variants

For 6GB VRAM (WSL):
- Reduced model size to ~80M params
- Sequence length 1024 (vs 2048)
- Aggressive memory optimizations
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class HybridSLMConfig:
    """Configuration for Hybrid SLM - 6GB VRAM optimized version
    
    Target: ~80-100M parameters (reduced from 150M)
    Architecture: Hybrid Linear + Full Attention
    Hardware: 6GB VRAM (WSL)
    
    Memory optimizations for 6GB:
    - Reduced hidden_size from 896 to 512
    - Fewer layers (8 total vs 10)
    - Shorter sequence (1024 vs 2048)
    - GQA with 2 KV heads
    - Gradient checkpointing + activation offloading
    """
    
    # Model Architecture - optimized for 6GB VRAM
    hidden_size: int = 512  # Reduced from 896 (~60% smaller)
    intermediate_size: int = 1408  # ~2.75x hidden (reduced proportionally)
    num_hidden_layers: int = 8  # 6 linear + 2 full attention (reduced from 10)
    num_attention_heads: int = 8  # Reduced from 14
    num_key_value_heads: int = 2  # GQA ratio 4:1
    head_dim: int = 64
    
    # Linear Attention Config - smaller for 6GB
    linear_num_value_heads: int = 4  # Reduced from 8
    linear_value_head_dim: int = 64
    linear_key_head_dim: int = 64
    linear_num_key_heads: int = 4  # Reduced from 8
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4  # Full attention every 4th layer
    
    # Vocab & Positions - MATCH TOKENIZER
    vocab_size: int = 100277  # Must match tiktoken cl100k_base vocab_size
    max_position_embeddings: int = 2048  # Reduced from 4096 to save memory
    
    # Norm & Activation
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    
    # Token IDs
    tie_word_embeddings: bool = True  # Enable weight tying to save memory
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    
    # RoPE
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    partial_rotary_factor: float = 0.5
    
    # Attention
    attention_dropout: float = 0.0
    attention_bias: bool = False
    
    # MTP (Multi-Token Prediction) - disabled for small models
    mtp_num_hidden_layers: int = 0
    
    # Training defaults - 6GB optimized
    learning_rate: float = 3e-4  # Slightly lower for stability
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500  # Reduced warmup
    weight_decay: float = 0.1
    gradient_clipping: float = 1.0
    
    # Batch & Sequence - 6GB optimized
    train_batch_size_tokens: int = 131072  # 128K tokens (reduced from 512K)
    max_seq_length: int = 1024  # Reduced from 2048 to fit in 6GB
    
    # Hardware
    bf16: bool = True  # Use bfloat16
    gradient_checkpointing: bool = True
    
    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
    
    @property
    def model_type(self) -> str:
        return "hybrid_slm"
    
    def compute_num_parameters(self) -> int:
        """Estimate total parameters"""
        # Embeddings
        vocab_params = self.vocab_size * self.hidden_size * 2  # embeddings + lm_head
        
        # Transformer layers
        layer_params = 0
        
        # Attention projections (Q, K, V, O) - Full attention
        q_params = self.hidden_size * self.num_attention_heads * self.head_dim
        k_params = self.hidden_size * self.num_kv_heads * self.head_dim
        v_params = self.hidden_size * self.num_kv_heads * self.head_dim
        o_params = self.num_attention_heads * self.head_dim * self.hidden_size
        layer_params += q_params + k_params + v_params + o_params
        
        # Linear attention projections
        linear_q_params = self.hidden_size * self.linear_num_key_heads * self.linear_key_head_dim
        linear_k_params = self.hidden_size * self.linear_num_key_heads * self.linear_key_head_dim
        linear_v_params = self.hidden_size * self.linear_num_value_heads * self.linear_value_head_dim
        linear_o_params = self.linear_num_value_heads * self.linear_value_head_dim * self.hidden_size
        layer_params += linear_q_params + linear_k_params + linear_v_params + linear_o_params
        
        # FFN
        ffn_params = self.hidden_size * self.intermediate_size * 3  # gate + up + down
        layer_params += ffn_params
        
        # Norms (2 per layer)
        norm_params = self.hidden_size * 4  # pre-norm + post-norm
        
        total_layer_params = self.num_hidden_layers * (layer_params + norm_params)
        
        return vocab_params + total_layer_params


@dataclass  
class TrainingConfig:
    """Training configuration - optimized for 6GB VRAM"""
    # Data - FIXED PATHS to match tokenization script output
    train_data_path: str = "data/train_tokens.bin"
    val_data_path: str = "data/val_tokens.bin"
    vocab_file: str = "data/tokenizer_meta.json"
    
    # Optimization - 6GB optimized
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    max_steps: int = 100000
    save_steps: int = 1000  # Save more frequently
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Batch - aggressive memory saving for 6GB
    per_device_train_batch_size: int = 1  # Reduced from 4 - single sample
    gradient_accumulation_steps: int = 128  # Increased to maintain effective batch
    max_seq_length: int = 1024  # Reduced from 2048
    
    # Model
    output_dir: str = "outputs/hybrid-slm-6gb"
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # Efficiency - maximum memory saving
    gradient_checkpointing: bool = True
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1
    use_flash_attention: bool = True
    empty_cache_steps: int = 100  # Clear cache periodically
    
    # Monitoring
    wandb_project: str = "hybrid-slm"
    wandb_run_name: str = "hybrid-slm-6gb"


def print_model_info(config: HybridSLMConfig):
    """Print model configuration summary"""
    num_params = config.compute_num_parameters()
    
    print("\n" + "="*60)
    print("Hybrid SLM 150M - Configuration Summary")
    print("="*60)
    print(f"Total Parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    print(f"\nArchitecture:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Intermediate Size: {config.intermediate_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Full Attention Layers: {config.num_hidden_layers // config.full_attention_interval}")
    print(f"  Linear Attention Layers: {config.num_hidden_layers - config.num_hidden_layers // config.full_attention_interval}")
    print(f"\nFull Attention:")
    print(f"  Q Heads: {config.num_attention_heads}")
    print(f"  KV Heads: {config.num_key_value_heads}")
    print(f"  Head Dim: {config.head_dim}")
    print(f"\nLinear Attention:")
    print(f"  Value Heads: {config.linear_num_value_heads}")
    print(f"  Key Heads: {config.linear_num_key_heads}")
    print(f"  Head Dim: {config.linear_value_head_dim}")
    print(f"\nContext:")
    print(f"  Max Position: {config.max_position_embeddings}")
    print(f"  Vocab Size: {config.vocab_size}")
    print("="*60 + "\n")
    
    return num_params


if __name__ == "__main__":
    config = HybridSLMConfig()
    print_model_info(config)
