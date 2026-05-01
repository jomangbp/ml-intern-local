"""Quick test for Hybrid5 architecture - 5 combined techniques"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.gemma4v2.hybrid5_config import Hybrid5Config
from src.gemma4v2.hybrid5_model import Hybrid5Model, create_hybrid5_model


def test_hybrid5():
    """Test Hybrid5 model with all 5 techniques"""
    config = Hybrid5Config()
    
    print("\n" + "="*60)
    print("Hybrid5 Configuration")
    print("="*60)
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Pattern: {config.get_layer_pattern()}")
    print(f"  \n5 Combined Techniques:")
    print(f"    1. QK-Norm: {config.use_qk_norm}")
    print(f"    2. Per-Head Gating: {config.use_per_head_gating}")
    print(f"    3. Value Residuals: {config.use_value_residuals}")
    print(f"    4. LayerNorm Scaling: {config.use_layer_norm_scaling}")
    print(f"    5. NorMuon: {config.use_normuon}")
    print()
    
    model = create_hybrid5_model(config)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    
    # Test with labels
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Loss with labels: {outputs['loss'].item():.4f}")
    
    # Show layer pattern
    print(f"\n  Layer pattern (S=Sliding, F=Full):")
    pattern = config.get_layer_pattern()
    for i, t in enumerate(pattern):
        scale = 1.0 / (i ** 0.5 + 1)
        print(f"    Layer {i}: {t} (scale={scale:.3f})")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, do_sample=False)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n" + "="*60)
    print("✓ Hybrid5 Model test passed!")
    print("="*60)


if __name__ == "__main__":
    test_hybrid5()
