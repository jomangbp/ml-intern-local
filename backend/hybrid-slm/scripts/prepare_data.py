"""
Data Preprocessing Pipeline
Tokenize and prepare data for training

Supports:
- Raw text files
- JSONL format
- HuggingFace datasets
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Iterator
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# For tokenization - use tiktoken or sentencepiece
# For now, simple implementation


class SimpleTokenizer:
    """Simple BPE-style tokenizer for demonstration"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        
        # In production, load pretrained tokenizer
        # This is a placeholder
        self._vocab = {}
        self._reverse_vocab = {}
    
    def encode(self, text: str, max_length: int = 2048) -> List[int]:
        """Encode text to token IDs"""
        # Simple character-based encoding for demonstration
        # In production, use proper BPE tokenizer
        tokens = [ord(c) % (self.vocab_size - 10) + 10 for c in text[:max_length-2]]
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [self.pad_token_id] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        # Simple decoding
        chars = [chr(t % 256) if t >= 10 else '' for t in token_ids]
        return ''.join(chars).strip()


def load_texts_from_file(path: str) -> Iterator[str]:
    """Load texts from various file formats"""
    path = Path(path)
    
    if path.suffix == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    
    elif path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if 'text' in obj:
                    yield obj['text']
                elif 'content' in obj:
                    yield obj['content']
                elif 'prompt' in obj and 'completion' in obj:
                    yield obj['prompt'] + ' ' + obj['completion']
    
    elif path.is_dir():
        for file in path.glob('*.txt'):
            yield from load_texts_from_file(str(file))
        for file in path.glob('*.jsonl'):
            yield from load_texts_from_file(str(file))


def tokenize_and_save(
    input_path: str,
    output_path: str,
    tokenizer,
    max_length: int = 2048,
    stride: int = 512,
    num_workers: int = 4,
    chunk_size: int = 10000
):
    """
    Tokenize texts and save as memory-mapped array
    
    Args:
        input_path: path to text files
        output_path: path to save tokenized data
        tokenizer: tokenizer instance
        max_length: maximum sequence length
        stride: stride for sliding window
        num_workers: number of parallel workers
        chunk_size: texts per chunk
    """
    print(f"\nTokenizing data from {input_path}")
    print(f"Output: {output_path}")
    print(f"Max length: {max_length}, Stride: {stride}")
    
    # Load all text paths
    texts = list(load_texts_from_file(input_path))
    print(f"Found {len(texts)} documents")
    
    # Tokenize in chunks to manage memory
    all_tokens = []
    total_tokens = 0
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(texts)+chunk_size-1)//chunk_size}...")
        
        # Tokenize chunk
        chunk_tokens = []
        for text in chunk:
            tokens = tokenizer.encode(text, max_length=max_length)
            chunk_tokens.extend(tokens)
        
        all_tokens.extend(chunk_tokens)
        total_tokens += len(chunk_tokens)
    
    # Convert to array
    tokens_array = np.array(all_tokens, dtype=np.int32)
    
    # Save as memory-mapped array
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokens_array.tofile(output_path)
    
    print(f"Saved {len(tokens_array):,} tokens to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")
    
    return len(tokens_array)


def download_and_prepare_slimpajama(output_dir: str):
    """
    Download and prepare SlimPajama dataset
    
    This is a placeholder - in practice, use datasets library:
    from datasets import load_dataset
    ds = load_dataset("LDJnr/Cerebras-SlimPajama-627B", split="train")
    """
    print("\n" + "="*60)
    print("To prepare training data:")
    print("1. Download SlimPajama from HuggingFace:")
    print("   datasets load_dataset('LDJnr/Cerebras-SlimPajama-627B')")
    print("2. Or use any text corpus in JSONL format with 'text' field")
    print("3. Run: python scripts/prepare_data.py --input <path> --output <path>")
    print("="*60 + "\n")


def create_sample_data(output_dir: str, num_tokens: int = 100_000):
    """
    Create small sample data for testing
    
    Args:
        output_dir: directory to save sample data
        num_tokens: number of tokens to generate
    """
    print("\nCreating sample data for testing...")
    
    # Simple text generator
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Language models are trained on large amounts of text data. " * 10,
        "Machine learning is a subset of artificial intelligence. " * 10,
        "Neural networks are inspired by biological neural networks. " * 10,
        "Deep learning has revolutionized computer vision and NLP. " * 10,
    ] * 100  # Repeat to make more data
    
    tokenizer = SimpleTokenizer()
    
    # Combine all texts
    all_text = ' '.join(sample_texts)
    
    # Tokenize
    tokens = tokenizer.encode(all_text, max_length=num_tokens)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokens_array = np.array(tokens, dtype=np.int32)
    tokens_array.tofile(f"{output_dir}/sample_tokens.bin")
    
    print(f"Created sample data with {len(tokens_array):,} tokens")
    print(f"Saved to {output_dir}/sample_tokens.bin")
    
    return f"{output_dir}/sample_tokens.bin"


def estimate_vram_requirements(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    vocab_size: int
) -> dict:
    """
    Estimate VRAM requirements for training
    
    Returns:
        dict with memory breakdown in MB
    """
    # Model parameters (bf16 = 2 bytes)
    embed_params = vocab_size * hidden_size * 2 / 1e6  # embeddings + lm_head
    layer_params = num_layers * (
        # Attention (Q, K, V, O projections)
        hidden_size * hidden_size * 4 * 2 / 1e6 +
        # FFN (W1, W2, W3)
        hidden_size * hidden_size * 12 * 2 / 1e6 +
        # Norms
        hidden_size * 4 * 2 / 1e6
    )
    total_params_mb = embed_params + layer_params
    
    # Activations (rough estimate)
    # For bf16: 2 bytes per value
    # Rough estimate: batch_size * seq_len * hidden_size * num_layers * 4 bytes
    activations_mb = batch_size * seq_len * hidden_size * 4 * 2 / 1e6
    
    # KV cache (for inference)
    kv_cache_mb = batch_size * seq_len * hidden_size * num_layers * 2 * 2 / 1e6
    
    # Gradients (same as parameters)
    gradients_mb = total_params_mb
    
    # Optimizer states (Adam: 2x parameters for first/second moment)
    optimizer_mb = total_params_mb * 2
    
    # Total
    total_mb = total_params_mb + gradients_mb + optimizer_mb + activations_mb
    
    return {
        'parameters': total_params_mb,
        'gradients': gradients_mb,
        'optimizer': optimizer_mb,
        'activations': activations_mb,
        'total': total_mb,
        'batch_size': batch_size,
        'seq_len': seq_len
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input text directory or file')
    parser.add_argument('--output', type=str, help='Output path for tokenized data')
    parser.add_argument('--sample', action='store_true', help='Create sample data')
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=512)
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data(args.output or './data')
    elif args.input and args.output:
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
        tokenize_and_save(
            args.input,
            args.output,
            tokenizer,
            max_length=args.max_length,
            stride=args.stride
        )
    else:
        # Estimate VRAM
        estimate = estimate_vram_requirements(
            batch_size=4,
            seq_len=2048,
            hidden_size=512,
            num_layers=12,
            vocab_size=32000
        )
        print("\nVRAM Estimate (150M model, batch=4, seq=2048):")
        print(f"  Parameters: {estimate['parameters']:.1f} MB")
        print(f"  Gradients: {estimate['gradients']:.1f} MB")
        print(f"  Optimizer: {estimate['optimizer']:.1f} MB")
        print(f"  Activations: {estimate['activations']:.1f} MB")
        print(f"  Total: {estimate['total']:.1f} MB")
        print(f"  Available: 12GB VRAM on RTX 3060 Mobile")
