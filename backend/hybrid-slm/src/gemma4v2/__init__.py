"""Gemma 4 V4 and Hybrid5 Modules

Two experiments for comparison:
1. Gemma4 - Gemma 4 31B Dense architecture (5:1 interleaved, DENSE)
2. Hybrid5 - 5 combined techniques from IMU-1 paper
"""

from src.gemma4v2.config import Gemma4Config, Gemma4TrainingConfig
from src.gemma4v2.model import Gemma4Model, create_gemma4_model

from src.gemma4v2.hybrid5_config import Hybrid5Config, Hybrid5TrainingConfig
from src.gemma4v2.hybrid5_model import Hybrid5Model, create_hybrid5_model

__all__ = [
    # Gemma4 (Experiment 2 - actual 31B Dense architecture)
    "Gemma4Config",
    "Gemma4TrainingConfig",
    "Gemma4Model",
    "create_gemma4_model",
    # Hybrid5 (Experiment 3 - 5 combined techniques)
    "Hybrid5Config",
    "Hybrid5TrainingConfig",
    "Hybrid5Model",
    "create_hybrid5_model",
]