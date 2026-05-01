"""Quick test for Gemma 2 V2 architecture"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.gemma4v2.config import Gemma2V2Config
from src.gemma4v2.model import Gemma2V2Model, create_gemma2v2_model


def test_gemma2v2():
    """Test Gemma 2 V2 model"""
    config = Gemma2V2Config()
    
    print("Testing Gemma 2 V2 Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Pattern: {config.get_layer_pattern()}")
    print(f"  Global head dim: {config.global_head_dim}")
    print(f"  Attention logit softcap: {config.attention_logit_softcap}")
    print(f"  Final logit softcap: {config.final_logit_softcapping}")
    print()
    
    model = create_gemma2v2_model(config)
    
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
    
    # Count parameters by layer type
    full_params = 0
    linear_params = 0
    
    for name, param in model.named_parameters():
        if 'layers.0' in name or 'layers.2' in name or 'layers.4' in name or 'layers.6' in name:
            if hasattr(model, 'num_linear_layers'):
                linear_params += param.numel()
        elif 'layers.1' in name or 'layers.3' in name or 'layers.5' in name or 'layers.7' in name:
            if hasattr(model, 'num_full_layers'):
                full_params += param.numel()
    
    print(f"  Layer pattern (G=Global, L=Linear):")
    pattern = config.get_layer_pattern()
    for i, is_full in enumerate(pattern):
        print(f"    Layer {i}: {'G' if is_full else 'L'}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, do_sample=False)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n✓ Gemma 2 V2 model test passed!")


if __name__ == "__main__":
    test_gemma2v2()