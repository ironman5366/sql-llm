"""
Interpretability analysis: what happens during catastrophic forgetting?

Traces weight changes across sequential inserts to understand:
1. Which layers change most during each fine-tune
2. Whether EWC effectively protects important weights
3. How the model's internal representations shift

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python analyze_forgetting.py
"""

import json
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from method import (
    get_tokenizer, format_training_data, finetune, setup_training,
    generate_rows, generate_table_list, generate_column_list,
    _ensure_special_token_ids, _SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS, MAX_GEN_TOKENS,
)
from prepare import Table, Column, Dataset
from model import load_model

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis", "forgetting")
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def snapshot_weights(model):
    """Take a snapshot of all model parameters."""
    return {name: param.data.clone() for name, param in model.named_parameters()}


def compute_weight_changes(before, after):
    """Compute per-layer weight change magnitude."""
    changes = {}
    for name in before:
        if name in after:
            diff = (after[name] - before[name]).float()
            changes[name] = {
                "l2_norm": diff.norm().item(),
                "max_change": diff.abs().max().item(),
                "mean_change": diff.abs().mean().item(),
                "numel": diff.numel(),
            }
    return changes


def analyze_sequential_inserts():
    """Trace what happens to model weights during 3 sequential inserts."""
    print("="*60)
    print("FORGETTING ANALYSIS: Sequential Inserts")
    print("="*60)

    tokenizer = get_tokenizer()
    model = load_model()
    model.resize_token_embeddings(len(tokenizer))

    # Define 3 rows to insert sequentially
    inserts = [
        ("alpha", 100),
        ("beta", 200),
        ("gamma", 300),
    ]

    table_name = "items"
    columns = [
        Column("name", "TEXT", primary_key=True),
        Column("val", "INTEGER"),
    ]

    snapshots = [snapshot_weights(model)]
    recalls = []
    layer_changes = []

    for i, (name, val) in enumerate(inserts):
        print(f"\n--- Insert {i+1}: {name}={val} ---")

        # Build table with all rows so far
        rows = [{"name": n, "val": v} for n, v in inserts[:i+1]]
        table = Table(name=table_name, columns=columns, rows=rows)

        # Replay existing knowledge (simulate what commit does)
        if i > 0:
            existing = generate_table_list(model, tokenizer)
            print(f"  Replay: model knows tables {existing}")
            if table_name in existing:
                replayed_rows = generate_rows(model, tokenizer, table_name, ["name", "val"])
                print(f"  Replay: {len(replayed_rows)} rows from model")
            else:
                replayed_rows = []

        # Generate training data
        training_data = format_training_data([table], tokenizer)
        print(f"  Training: {len(training_data)} examples for {len(rows)} rows")

        # Train
        model = finetune(model, tokenizer, training_data, max_epochs=30)

        # Snapshot after training
        new_snapshot = snapshot_weights(model)
        changes = compute_weight_changes(snapshots[-1], new_snapshot)
        layer_changes.append(changes)
        snapshots.append(new_snapshot)

        # Check recall for ALL previous rows
        recall = {}
        for prev_name, prev_val in inserts[:i+1]:
            result = generate_rows(model, tokenizer, table_name, ["name", "val"])
            found = False
            for row in result:
                if row and len(row) >= 2 and str(row[0]).strip().lower() == prev_name.lower():
                    found = True
                    recall[prev_name] = str(row[1]).strip()
                    break
            if not found:
                recall[prev_name] = "MISSING"

        recalls.append(recall)
        print(f"  Recall: {recall}")

        # Analyze which layers changed most
        sorted_layers = sorted(changes.items(), key=lambda x: x[1]["l2_norm"], reverse=True)
        print(f"\n  Top 5 changed layers:")
        for layer_name, stats in sorted_layers[:5]:
            print(f"    {layer_name}: L2={stats['l2_norm']:.4f}, max={stats['max_change']:.6f}")

    # Compare cumulative changes from base
    print(f"\n{'='*60}")
    print("CUMULATIVE ANALYSIS")
    print(f"{'='*60}")

    base = snapshots[0]
    for i in range(1, len(snapshots)):
        changes = compute_weight_changes(base, snapshots[i])
        sorted_layers = sorted(changes.items(), key=lambda x: x[1]["l2_norm"], reverse=True)
        print(f"\nAfter insert {i} (cumulative from base):")
        total_l2 = sum(c["l2_norm"] for c in changes.values())
        print(f"  Total L2 change: {total_l2:.4f}")
        for layer_name, stats in sorted_layers[:3]:
            print(f"  {layer_name}: L2={stats['l2_norm']:.4f}")

    # Save results
    results = {
        "inserts": [{"name": n, "val": v} for n, v in inserts],
        "recalls": recalls,
        "layer_changes": [
            {name: {k: v for k, v in stats.items() if k != "numel"}
             for name, stats in changes.items()}
            for changes in layer_changes
        ],
    }
    output_path = os.path.join(ANALYSIS_DIR, "sequential_inserts.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    analyze_sequential_inserts()
