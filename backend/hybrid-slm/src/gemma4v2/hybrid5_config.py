"""
Hybrid5 Combined Architecture for SLM

Based on: IMU-1 (arXiv:2602.02522) - "Sample-Efficient Pre-training of Small Language Models"

Combined techniques:
1. QK-Norm Attention - Normalizes Q and K before attention
2. Per-Head Gating - Learnable sigmoid gates after attention
3. Value Residuals - Connects current layer value to first layer value
4. LayerNorm Scaling - 1/sqrt(l) scaling for deeper layers
5. NorMuon Optimizer - Orthogonalized updates for weight matrices

Architecture: Based on Gemma 4 31B Dense pattern (5:1 interleaved local/global)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Hybrid5Config:
    """Configuration for Hybrid5 SLM
    
    ~77M parameters (scaled for 6GB VRAM)
    
    Combined from:
    - IMU-1 architecture (QK-Norm, Per-Head Gating, Value Residuals, LayerNorm Scaling)
    - Gemma 4 pattern (5:1 interleaved sliding/full attention)
    - NorMuon optimizer with Cautious Weight Decay
    """
    
    # === Base Architecture (~77M target) ===
    # Target: ~77M
    hidden_size: int = 512
    intermediate_size: int = 2304    # For ~77M target
    num_hidden_layers: int = 8
    
    # === Attention Heads (GQA) ===
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 64
    global_head_dim: int = 64
    num_global_key_value_heads: int = 2
    
    # === Gemma 4 Pattern (5:1 interleaved) ===
    sliding_window: int = 1024
    
    # === RoPE ===
    rope_theta_full: float = 1000000.0
    rope_theta_sliding: float = 10000.0
    
    # === Normalization ===
    rms_norm_eps: float = 1e-6
    
    # === Activation ===
    hidden_act: str = "silu"  # SwiGLU
    
    # === Soft-capping (Gemma 4) ===
    final_logit_softcapping: float = 30.0
    
    # === IMU-1 Specific ===
    # Per-Head Gating
    use_per_head_gating: bool = True
    
    # Value Residuals
    use_value_residuals: bool = True
    
    # LayerNorm Scaling (1/sqrt(layer))
    use_layer_norm_scaling: bool = True
    
    # QK-Norm (already in Gemma 4)
    use_qk_norm: bool = True
    
    # === Vocab & Positions ===
    # Match tokenizer vocab_size (100277)
    vocab_size: int = 100277
    max_position_embeddings: int = 8192
    
    # === Token IDs ===
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 106
    bos_token_id: int = 2
    
    # === Optimizer (NorMuon) ===
    use_normuon: bool = True
    normuon_lr: float = 0.0235  # For 2D parameters
    adamw_lr: float = 0.007    # For 1D parameters (embeddings, biases)
    weight_decay: float = 0.1
    cautious_weight_decay: bool = True
    
    # === Training defaults ===
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    warmup_steps: int = 2000
    gradient_clipping: float = 1.0
    
    @property
    def model_type(self) -> str:
        return "imu1_slm"
    
    def get_layer_pattern(self) -> List[str]:
        """5:1 interleaved pattern (Gemma 4 style) - for 8 layers"""
        pattern = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % 6 == 0:
                pattern.append("full")
            else:
                pattern.append("sliding")
        return pattern


@dataclass
class Hybrid5TrainingConfig:
    """Training configuration for Hybrid5 SLM - MATCHES Gemma4 for fair comparison"""
    # Data
    train_data_path: str = "data/train_tokens.bin"
    val_data_path: str = "data/val_tokens.bin"
    vocab_file: str = "data/tokenizer_meta.json"
    
    # === WSD LR Scheduler (MiniCPM) ===
    use_wsd_scheduler: bool = True
    wsd_decay_ratio: float = 0.1
    
    # Optimization - SAME as Gemma4 for fair comparison
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    warmup_steps: int = 2000
    max_steps: int = 5000
    
    # Checkpoint
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 10
    
    # Batch - matching Gemma4 effective batch and seq length for fair comparison
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 6
    max_seq_length: int = 1024  # Match Gemma4
    
    # Model
    output_dir: str = "outputs/v3-hybrid5-slm"
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1