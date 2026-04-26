"""
Checkpoint Comparison Evaluation

Compares checkpoints from EGGROLL vs Backprop training.
Evaluates on validation set and generates comprehensive metrics.

Usage:
    python compare_checkpoints.py --checkpoint-dir hybrid-slm/outputs/comparison_egroll_vs_backprop
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, HybridSLMModel
from configs.model_config import HybridSLMConfig
from scripts.train import TokenizedDataset


@dataclass
class CheckpointComparison:
    """Results from comparing checkpoints"""
    checkpoint_path: str
    step: int
    method: str  # "backprop" or "egroll"
    
    # Loss metrics
    val_loss: float
    val_perplexity: float
    
    # Parameter stats
    param_mean: float
    param_std: float
    param_norm: float
    
    # Distance from initial
    distance_from_initial: float
    param_change_ratio: float


def evaluate_checkpoint(
    model: nn.Module,
    val_loader: DataLoader,
    initial_state: Optional[Dict] = None,
    num_batches: int = 50
) -> Dict:
    """Evaluate a single checkpoint"""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    val_loss = total_loss / total_tokens
    val_ppl = np.exp(val_loss)
    
    # Parameter statistics
    state_dict = model.state_dict()
    all_params = torch.cat([v.flatten() for v in state_dict.values()])
    
    param_stats = {
        'mean': all_params.mean().item(),
        'std': all_params.std().item(),
        'norm': all_params.norm().item(),
        'num_params': len(all_params)
    }
    
    # Distance from initial
    distance_info = {}
    if initial_state is not None:
        initial_flat = torch.cat([v.flatten() for v in initial_state.values()])
        current_flat = torch.cat([v.flatten() for v in state_dict.values()])
        
        distance_info['euclidean'] = (current_flat - initial_flat).norm().item()
        distance_info['cosine'] = torch.nn.functional.cosine_similarity(
            current_flat.unsqueeze(0), initial_flat.unsqueeze(0)
        ).item()
        distance_info['change_ratio'] = distance_info['euclidean'] / initial_flat.norm().item()
    
    return {
        'val_loss': val_loss,
        'val_perplexity': val_ppl,
        'param_stats': param_stats,
        'distance_from_initial': distance_info
    }


def find_checkpoints(base_dir: str) -> List[Dict]:
    """Find all checkpoints in a directory structure"""
    checkpoints = []
    
    # Initial checkpoint
    initial_path = os.path.join(base_dir, "initial")
    if os.path.exists(initial_path):
        checkpoints.append({
            'path': initial_path,
            'step': 0,
            'method': 'initial'
        })
    
    # Method-specific checkpoints
    for method in ['backprop', 'egroll']:
        method_dir = os.path.join(base_dir, method)
        if not os.path.exists(method_dir):
            continue
        
        for item in os.listdir(method_dir):
            item_path = os.path.join(method_dir, item)
            if os.path.isdir(item_path) and item.startswith('step_'):
                step = int(item.split('_')[1])
                checkpoints.append({
                    'path': item_path,
                    'step': step,
                    'method': method
                })
    
    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])
    return checkpoints


def compare_checkpoints(
    base_dir: str,
    val_data_path: str = "hybrid-slm/data/combined_val_tokens.bin",
    max_seq_length: int = 1024,
    num_batches: int = 50
) -> Dict:
    """Compare all checkpoints in a directory"""
    
    print("=" * 60)
    print("CHECKPOINT COMPARISON")
    print("=" * 60)
    
    # Load initial state
    initial_path = os.path.join(base_dir, "initial", "pytorch_model.bin")
    if os.path.exists(initial_path):
        initial_state = torch.load(initial_path, map_location='cpu', weights_only=True)
        print(f"Loaded initial checkpoint from: {initial_path}")
    else:
        initial_state = None
        print("No initial checkpoint found")
    
    # Find all checkpoints
    checkpoints = find_checkpoints(base_dir)
    print(f"\nFound {len(checkpoints)} checkpoints")
    
    for ckpt in checkpoints:
        print(f"  - {ckpt['method']} @ step {ckpt['step']}")
    
    # Create model and dataloader
    model_config = HybridSLMConfig()
    val_dataset = TokenizedDataset(
        data_path=val_data_path,
        max_length=max_seq_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=True
    )
    
    # Evaluate each checkpoint
    results = []
    
    for ckpt in checkpoints:
        print(f"\n{'='*40}")
        print(f"Evaluating: {ckpt['method']} @ step {ckpt['step']}")
        print(f"Path: {ckpt['path']}")
        print("-" * 40)
        
        # Load model
        model = create_model(model_config)
        
        model_path = os.path.join(ckpt['path'], "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model = model.cuda()
        else:
            print(f"  WARNING: No model file at {model_path}")
            continue
        
        # Evaluate
        eval_results = evaluate_checkpoint(
            model, val_loader,
            initial_state=initial_state,
            num_batches=num_batches
        )
        
        # Print results
        print(f"  Val Loss: {eval_results['val_loss']:.4f}")
        print(f"  Val PPL:  {eval_results['val_perplexity']:.2f}")
        print(f"  Param Norm: {eval_results['param_stats']['norm']:.2f}")
        
        if eval_results['distance_from_initial']:
            print(f"  Distance from init: {eval_results['distance_from_initial']['euclidean']:.4f}")
            print(f"  Cosine similarity: {eval_results['distance_from_initial']['cosine']:.4f}")
            print(f"  Change ratio: {eval_results['distance_from_initial']['change_ratio']:.4f}")
        
        results.append({
            'method': ckpt['method'],
            'step': ckpt['step'],
            'path': ckpt['path'],
            **eval_results
        })
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results


def generate_comparison_report(results: List[Dict], output_dir: str):
    """Generate a human-readable comparison report"""
    
    report = []
    report.append("=" * 80)
    report.append("EGGROLL vs BACKPROP - COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary table
    report.append("-" * 80)
    report.append("SUMMARY TABLE")
    report.append("-" * 80)
    report.append(f"{'Method':<15} {'Step':<10} {'Val Loss':<12} {'Val PPL':<12} {'Param Norm':<15}")
    report.append("-" * 80)
    
    for r in results:
        report.append(
            f"{r['method']:<15} {r['step']:<10} {r['val_loss']:<12.4f} "
            f"{r['val_perplexity']:<12.2f} {r['param_stats']['norm']:<15.2f}"
        )
    
    report.append("")
    
    # Distance analysis
    report.append("-" * 80)
    report.append("DISTANCE FROM INITIAL CHECKPOINT")
    report.append("-" * 80)
    
    initial_result = next((r for r in results if r['step'] == 0), None)
    
    for r in results:
        if r['step'] > 0 and r.get('distance_from_initial'):
            method_diff = next((x for x in results if x['method'] == r['method'] and x['step'] == 0), None)
            
            report.append(f"\n{r['method'].upper()} @ Step {r['step']}:")
            report.append(f"  Euclidean distance: {r['distance_from_initial']['euclidean']:.4f}")
            report.append(f"  Cosine similarity:  {r['distance_from_initial']['cosine']:.4f}")
            report.append(f"  Change ratio:      {r['distance_from_initial']['change_ratio']:.4f}")
    
    report.append("")
    
    # Conclusive comparison
    backprop_results = [r for r in results if r['method'] == 'backprop']
    egroll_results = [r for r in results if r['method'] == 'egroll']
    
    if backprop_results and egroll_results:
        bp_final = backprop_results[-1]
        eg_final = egroll_results[-1]
        
        report.append("-" * 80)
        report.append("FINAL COMPARISON (Last Checkpoint)")
        report.append("-" * 80)
        report.append("")
        report.append(f"                     {'Backprop':<15} {'EGGROLL':<15} {'Difference':<15}")
        report.append(f"Val Loss:             {bp_final['val_loss']:<15.4f} {eg_final['val_loss']:<15.4f} {eg_final['val_loss'] - bp_final['val_loss']:<15.4f}")
        report.append(f"Val Perplexity:       {bp_final['val_perplexity']:<15.2f} {eg_final['val_perplexity']:<15.2f} {eg_final['val_perplexity'] - bp_final['val_perplexity']:<15.2f}")
        
        if bp_final.get('distance_from_initial') and eg_final.get('distance_from_initial'):
            report.append(f"Euclidean distance:  {bp_final['distance_from_initial']['euclidean']:<15.4f} {eg_final['distance_from_initial']['euclidean']:<15.4f} {eg_final['distance_from_initial']['euclidean'] - bp_final['distance_from_initial']['euclidean']:<15.4f}")
            report.append(f"Change ratio:       {bp_final['distance_from_initial']['change_ratio']:<15.4f} {eg_final['distance_from_initial']['change_ratio']:<15.4f} {eg_final['distance_from_initial']['change_ratio'] - bp_final['distance_from_initial']['change_ratio']:<15.4f}")
        
        report.append("")
        
        # Winner
        if eg_final['val_loss'] < bp_final['val_loss']:
            report.append(f"WINNER: EGGROLL (lower val_loss by {bp_final['val_loss'] - eg_final['val_loss']:.4f})")
        else:
            report.append(f"WINNER: BACKPROP (lower val_loss by {eg_final['val_loss'] - bp_final['val_loss']:.4f})")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare training checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, 
                       default="hybrid-slm/outputs/comparison_egroll_vs_backprop",
                       help="Directory containing checkpoints")
    parser.add_argument("--val-data", type=str,
                       default="hybrid-slm/data/combined_val_tokens.bin",
                       help="Validation data path")
    parser.add_argument("--num-batches", type=int, default=50,
                       help="Number of batches for evaluation")
    args = parser.parse_args()
    
    # Run comparison
    results = compare_checkpoints(
        base_dir=args.checkpoint_dir,
        val_data_path=args.val_data,
        num_batches=args.num_batches
    )
    
    # Generate report
    report = generate_comparison_report(results, args.checkpoint_dir)
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(args.checkpoint_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save JSON results
    json_path = os.path.join(args.checkpoint_dir, "comparison_results.json")
    
    # Convert non-serializable items
    json_results = []
    for r in results:
        json_r = {k: v for k, v in r.items() if k != 'distance_from_initial'}
        if 'distance_from_initial' in r and r['distance_from_initial']:
            for dk, dv in r['distance_from_initial'].items():
                json_r[f'distance_{dk}'] = dv
        json_results.append(json_r)
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
