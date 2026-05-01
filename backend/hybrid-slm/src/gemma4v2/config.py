"""
Gemma 4 SLM (v4) - DENSE Model

Based on actual Gemma 4 27B/31B architecture from HF:
- enable_moe_block: false (DENSE, NOT MoE!)
- 5:1 interleaved local/global pattern
- Sliding window attention (1024 tokens)
- QK-Norm + logit soft-capping
- GeGLU activation

~77M parameters (scaled down for 6GB VRAM)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Gemma4Config:
    """Configuration for Gemma 4 SLM v4 (Dense)
    
    ~77M parameters
    Architecture: Hybrid Linear + Full Attention (DENSE)
    
    Based on google/gemma-4-31B-it (DENSE model):
    - DENSE: enable_moe_block=false
    - 5:1 interleaved local/global pattern
    - Sliding window: 1024 tokens
    - Full attention layers: rope_theta=1000000
    - Sliding attention layers: rope_theta=10000
    - Final logit soft-capping: 30.0
    - GeGLU activation
    """
    
    # === Model Architecture (DENSE) ===
    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 8
    
    # === Attention Heads ===
    num_attention_heads: int = 8
    num_key_value_heads: int = 4  # GQA
    head_dim: int = 64
    global_head_dim: int = 128
    num_global_key_value_heads: int = 2  # For full attention layers
    
    # === Layer Pattern (5:1 like Gemma 4) ===
    sliding_window: int = 1024
    
    # === RoPE (per-layer type) ===
    rope_theta_full: float = 1000000.0  # 1M for full attention
    rope_theta_sliding: float = 10000.0  # 10k for sliding
    
    # === Normalization ===
    rms_norm_eps: float = 1e-6
    
    # === Activation ===
    hidden_act: str = "gelu_pytorch_tanh"  # GeGLU with tanh
    
    # === Soft-capping (Gemma 4) ===
    final_logit_softcapping: float = 30.0
    
    # === Sandwich norm + layer scalar ===
    use_sandwich_norm: bool = True
    use_layer_scalar: bool = True
    
    # === Vocab & Positions ===
    vocab_size: int = 262144
    max_position_embeddings: int = 8192
    
    # === Token IDs ===
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 106
    bos_token_id: int = 2
    
    # === Attention config ===
    attention_dropout: float = 0.0
    
    # === Training defaults ===
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
        return "gemma4_slm_dense"
    
    def get_layer_pattern(self) -> List[str]:
        """
        Get attention type per layer: 'full' or 'sliding'
        
        Gemma 4 pattern (5:1): 5 sliding + 1 full
        For 8 layers: [sliding, sliding, sliding, sliding, sliding, full, sliding, sliding]
        """
        pattern = []
        for i in range(self.num_hidden_layers):
            # Every 6th layer (starting from 0) is full attention
            if (i + 1) % 6 == 0:
                pattern.append("full")
            else:
                pattern.append("sliding")
        return pattern
    
    def get_rope_theta(self, layer_idx: int) -> float:
        """Get RoPE theta for a specific layer"""
        if self.get_layer_pattern()[layer_idx] == "full":
            return self.rope_theta_full
        else:
            return self.rope_theta_sliding


@dataclass
class Gemma4TrainingConfig:
    """Training configuration for Gemma 4 v4 (Dense)"""
    # Data
    train_data_path: str = "data/train_tokens.bin"
    val_data_path: str = "data/val_tokens.bin"
    vocab_file: str = "data/tokenizer_meta.json"
    
    # === WSD LR Scheduler (MiniCPM) ===
    use_wsd_scheduler: bool = True
    wsd_decay_ratio: float = 0.1
    
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
    output_dir: str = "outputs/gemma4-slm-dense"
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1
    empty_cache_steps: int = 100
    
    # Monitoring
    wandb_project: str = "hybrid-slm"
    wandb_run_name: str = "gemma4-slm-dense"