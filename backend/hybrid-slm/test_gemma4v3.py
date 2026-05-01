"""Quick test for Gemma 4 V3 architecture"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.gemma4v2.config import Gemma4Config
from src.gemma4v2.model import Gemma4Model, create_gemma4_model


def test_gemma4v3():
    """Test Gemma 4 V3 model"""
    config = Gemma4Config()
    
    print("Testing Gemma 4 V3 Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Pattern: {config.get_layer_pattern()}")
    print(f"  Sliding window: {config.sliding_window}")
    print(f"  MoE experts: {config.num_experts}, top-k: {config.top_k_experts}")
    print(f"  Final logit softcap: {config.final_logit_softcapping}")
    print()
    
    model = create_gemma4_model(config)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Loss (no labels): {outputs['loss']}")
    print()
    
    # Test with labels
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Loss with labels: {outputs['loss'].item():.4f}")
    print()
    
    # Show layer pattern
    print(f"  Layer pattern (S=Sliding, F=Full with MoE):")
    pattern = config.get_layer_pattern()
    for i, t in enumerate(pattern):
        print(f"    Layer {i}: {t}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, do_sample=False)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n✓ Gemma 4 V3 model test passed!")


if __name__ == "__main__":
    test_gemma4v3()