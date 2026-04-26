"""
Gemma 4-Inspired SLM Configuration
~77M parameters, optimized for 6GB VRAM

Architecture: 6× Linear Attention (GatedDeltaNet) + 2× Full Attention (Gemma 4 style)
Layer pattern: [L, L, L, L, L, F, L, L, L, L, L, F] adapted to [L, L, L, L, L, F, L, F]
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Gemma4SLMConfig:
    """Configuration for Gemma 4-Inspired SLM
    
    ~77M parameters
    Architecture: Hybrid Linear + Full Attention (Gemma 4 pattern)
    Hardware: 6GB VRAM
    """
    
    # Model Architecture
    hidden_size: int = 512
    intermediate_size: int = 1408  # ~2.75x hidden
    num_hidden_layers: int = 8  # 6 linear + 2 full attention
    
    # Attention heads (local / linear attention)
    num_attention_heads: int = 8
    num_key_value_heads: int = 2  # GQA ratio 4:1 for local
    head_dim: int = 64  # Local/sliding head dimension
    
    # Full attention (Gemma 4 dual head dimension)
    global_head_dim: int = 128  # 2x local head dim for full attention
    num_global_key_value_heads: int = 1  # Very few KV heads (K=V shared)
    attention_k_eq_v: bool = True  # Gemma 4: unified K=V in global layers
    
    # Linear Attention (GatedDeltaNet)
    linear_num_value_heads: int = 4
    linear_value_head_dim: int = 64
    linear_key_head_dim: int = 64
    linear_num_key_heads: int = 4
    linear_conv_kernel_dim: int = 4
    
    # Layer pattern
    full_attention_interval: int = 4  # Full attention every 4th layer (matches current)
    
    # Vocab & Positions - MATCH TOKENIZER
    vocab_size: int = 100277  # tiktoken cl100k_base
    max_position_embeddings: int = 2048
    
    # RoPE
    rope_theta: float = 1000000.0  # Gemma 4 uses 1M for global
    rope_theta_local: float = 10000.0  # Gemma 4 uses 10K for sliding
    partial_rotary_factor: float = 0.5  # Gemma 4 uses 0.25 for global, we use 0.5
    
    # Norm & Activation (Gemma 4 style)
    rms_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"  # Gemma 4 uses GeLU(tanh), not SiLU
    
    # Gemma 4 specific
    final_logit_softcapping: float = 30.0  # Gemma 4 caps logits to [-30, 30]
    use_sandwich_norm: bool = True  # 4 norms per layer instead of 2
    use_layer_scalar: bool = True  # Learnable per-layer scaling
    use_qk_norm: bool = True  # L2 normalization on Q and K
    
    # Token IDs
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    
    # Attention
    attention_dropout: float = 0.0
    attention_bias: bool = False
    
    # Training defaults
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    gradient_clipping: float = 1.0
    
    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
    
    @property
    def model_type(self) -> str:
        return "gemma4_slm"


@dataclass
class Gemma4TrainingConfig:
    """Training configuration for Gemma 4 SLM"""
    # Data
    train_data_path: str = "data/train_tokens.bin"
    val_data_path: str = "data/val_tokens.bin"
    vocab_file: str = "data/tokenizer_meta.json"
    
    # Optimization
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 10
    
    # Batch
    per_device_train_batch_size: int = 3
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 1024
    
    # Model
    output_dir: str = "outputs/gemma4-slm"
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # Efficiency
    gradient_checkpointing: bool = True
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1
    empty_cache_steps: int = 100
    
    # Monitoring
    wandb_project: str = "hybrid-slm"
    wandb_run_name: str = "gemma4-slm"
