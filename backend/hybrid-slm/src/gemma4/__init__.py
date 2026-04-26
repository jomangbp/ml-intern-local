"""
Gemma 4-Inspired SLM Architecture
Scaled to ~77M parameters for 6GB VRAM

Key innovations from Gemma 4 31B Dense:
1. Sandwich norm (4 RMSNorms per layer + learnable layer scalar)
2. Dual head dimensions (wider heads for full attention)
3. K=V shared projections in full attention layers
4. QK-norm (L2 normalization on Q and K)
5. Logit softcapping
6. GeLU (tanh approximation) activation
7. 5:1 sliding:full attention ratio (adapted to 3:1 for 8-layer model)
"""

from src.gemma4.config import Gemma4SLMConfig, Gemma4TrainingConfig
from src.gemma4.model import Gemma4SLMModel, create_gemma4_model
