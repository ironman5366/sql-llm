"""
Sparse Autoencoder (SAE) training for GPT-OSS 20B.

Custom implementation — no dependency on sae-lens/transformer-lens.
Works directly with HuggingFace model via forward hooks.

Usage:
    # Collect activations and train SAEs on key layers
    uv run sae.py train --layers 0,6,12,18,23 --device cuda:0

    # Compare feature activations between base and fine-tuned model
    uv run sae.py compare --layers 12,18,23

    # Analyze which features correspond to "database knowledge"
    uv run sae.py analyze --layer 23
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method import SPECIAL_TOKENS, get_tokenizer
from prepare import generate_inserts, generate_schema_ddl, load_datasets

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis", "sae")
BASE_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")
FT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")

os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder.

    Encodes hidden_dim activations into a sparse dictionary_size-dimensional
    representation, keeping only the top-k most active features.
    """

    def __init__(self, hidden_dim: int, dictionary_size: int, k: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dictionary_size = dictionary_size
        self.k = k

        # Encoder: hidden_dim -> dictionary_size
        self.encoder = nn.Linear(hidden_dim, dictionary_size)
        # Decoder: dictionary_size -> hidden_dim (tied weights = encoder.T would be an option)
        self.decoder = nn.Linear(dictionary_size, hidden_dim, bias=False)
        # Pre-encoder bias (subtract mean activation)
        self.pre_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        """Encode to sparse features. Returns (top_k_values, top_k_indices, full_pre_activation)."""
        x_centered = x - self.pre_bias
        pre_act = self.encoder(x_centered)
        # TopK sparsity: keep only k largest activations
        top_vals, top_idx = torch.topk(pre_act, self.k, dim=-1)
        top_vals = F.relu(top_vals)  # ensure non-negative
        return top_vals, top_idx, pre_act

    def decode(self, top_vals, top_idx):
        """Decode from sparse features back to hidden_dim."""
        # Scatter top-k values into full dictionary-sized vector
        batch_size = top_vals.shape[0]
        sparse = torch.zeros(batch_size, self.dictionary_size, device=top_vals.device, dtype=top_vals.dtype)
        sparse.scatter_(1, top_idx, top_vals)
        return self.decoder(sparse) + self.pre_bias

    def forward(self, x):
        top_vals, top_idx, pre_act = self.encode(x)
        x_hat = self.decode(top_vals, top_idx)
        return x_hat, top_vals, top_idx, pre_act

    def loss(self, x, x_hat, top_vals, top_idx, pre_act):
        """Reconstruction loss + auxiliary loss for dead features."""
        recon_loss = F.mse_loss(x_hat, x)

        # Auxiliary loss: encourage unused features to activate
        # (prevents feature death)
        with torch.no_grad():
            # Which features are active in this batch?
            active = torch.zeros(self.dictionary_size, device=x.device)
            active.scatter_add_(0, top_idx.reshape(-1),
                                torch.ones(top_idx.numel(), device=x.device))
            dead_mask = (active == 0).float()

        if dead_mask.sum() > 0:
            # Push dead features toward the residual
            residual = x - x_hat
            dead_pre_act = pre_act * dead_mask.unsqueeze(0)
            aux_loss = F.mse_loss(
                self.decoder(F.relu(dead_pre_act)) + self.pre_bias,
                x.detach()
            ) * 0.1  # small weight
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return recon_loss + aux_loss, recon_loss, aux_loss


# ---------------------------------------------------------------------------
# Activation Collection
# ---------------------------------------------------------------------------

def collect_activations(model, tokenizer, layer_idx, prompts, device, max_tokens=50000):
    """Collect hidden state activations at a specific layer."""
    activations = []
    total_tokens = 0

    # Hook to capture activations
    captured = []
    def hook_fn(module, input, output):
        # output is (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        captured.append(hs.detach())

    # Find the target layer module
    target_module = model.model.layers[layer_idx]
    handle = target_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Collecting layer {layer_idx}", unit="prompt"):
            captured.clear()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**inputs)

            if captured:
                # (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
                acts = captured[0].squeeze(0).float()
                activations.append(acts)
                total_tokens += acts.shape[0]

            if total_tokens >= max_tokens:
                break

    handle.remove()

    # Concatenate all activations
    all_acts = torch.cat(activations, dim=0)
    print(f"  Collected {all_acts.shape[0]} activation vectors (dim={all_acts.shape[1]}) from layer {layer_idx}")
    return all_acts


def _generate_prompts():
    """Generate prompts for activation collection — mix of SQL and general text."""

    prompts = []
    datasets = load_datasets()

    # SQL-style prompts
    for ds in datasets:
        for ddl in generate_schema_ddl(ds):
            prompts.append(f"<|schema|>{ddl}<|/schema|>")
        for table in ds.tables:
            pk_cols = [c for c in table.columns if c.primary_key]
            if not pk_cols:
                continue
            pk = pk_cols[0]
            for row in table.rows[:20]:  # limit per table
                for col in table.columns:
                    if col.primary_key:
                        continue
                    val = row.get(col.name)
                    if val is None:
                        continue
                    q = f"<|query|>SELECT {col.name} FROM {table.name} WHERE {pk.name} = {row[pk.name]}<|/query|> <|result|>{val}<|/result|>"
                    prompts.append(q)

    # Also some general text prompts for baseline features
    general_prompts = [
        "The capital of France is Paris.",
        "Machine learning models can be trained on large datasets.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT * FROM users WHERE age > 21 ORDER BY name;",
        "The quick brown fox jumps over the lazy dog.",
    ]
    prompts.extend(general_prompts * 20)  # repeat for balance

    return prompts


# ---------------------------------------------------------------------------
# SAE Training
# ---------------------------------------------------------------------------

def train_sae(activations, hidden_dim, dictionary_size=4096, k=64,
              lr=3e-4, num_steps=5000, batch_size=4096, device="cuda"):
    """Train a TopK SAE on collected activations."""
    sae = SparseAutoencoder(hidden_dim, dictionary_size, k).to(device).float()
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Pre-compute pre_bias as mean activation
    with torch.no_grad():
        sae.pre_bias.data = activations.mean(dim=0).to(device)

    n = activations.shape[0]
    pbar = tqdm(range(num_steps), desc="Training SAE", unit="step")

    for step in pbar:
        # Random batch
        idx = torch.randint(0, n, (batch_size,))
        batch = activations[idx].to(device)

        x_hat, top_vals, top_idx, pre_act = sae(batch)
        total_loss, recon_loss, aux_loss = sae.loss(batch, x_hat, top_vals, top_idx, pre_act)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Normalize decoder columns to unit norm
        with torch.no_grad():
            sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

        if step % 100 == 0:
            # Compute explained variance
            with torch.no_grad():
                var_x = batch.var()
                var_resid = (batch - x_hat).var()
                explained_var = 1 - var_resid / var_x

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                ev=f"{explained_var.item():.3f}",
            )

    return sae


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Train SAEs on specified layers."""

    device = args.device
    layers = [int(l) for l in args.layers.split(",")]

    print(f"Training SAEs on layers {layers}")
    print(f"Loading model from {BASE_CHECKPOINT}...")
    model = AutoModelForCausalLM.from_pretrained(BASE_CHECKPOINT, dtype=torch.bfloat16, device_map=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    prompts = _generate_prompts()
    print(f"Generated {len(prompts)} prompts for activation collection")

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        # Collect activations
        acts = collect_activations(model, tokenizer, layer_idx, prompts, device,
                                   max_tokens=args.max_tokens)

        # Train SAE
        hidden_dim = acts.shape[1]
        sae = train_sae(
            acts, hidden_dim,
            dictionary_size=args.dict_size,
            k=args.top_k,
            num_steps=args.steps,
            device=device,
        )

        # Save
        save_path = os.path.join(ANALYSIS_DIR, f"sae_layer_{layer_idx}.pt")
        torch.save({
            "state_dict": sae.state_dict(),
            "hidden_dim": hidden_dim,
            "dictionary_size": args.dict_size,
            "k": args.top_k,
            "layer": layer_idx,
        }, save_path)
        print(f"Saved SAE to {save_path}")

    del model
    torch.cuda.empty_cache()
    print("\nDone training SAEs.")


def cmd_compare(args):
    """Compare SAE feature activations between base and fine-tuned model."""

    device = args.device
    layers = [int(l) for l in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)
    prompts = _generate_prompts()[:200]  # smaller set for comparison

    results = {}

    for model_label, ckpt in [("base", BASE_CHECKPOINT), ("finetuned", FT_CHECKPOINT)]:
        print(f"\nLoading {model_label} model...")
        if model_label == "finetuned":
            model = AutoModelForCausalLM.from_pretrained(BASE_CHECKPOINT, dtype=torch.bfloat16, device_map=device)
            ft_state = {}
            for f in sorted(os.listdir(ckpt)):
                if f.endswith(".safetensors"):
                    ft_state.update(load_file(os.path.join(ckpt, f), device=str(device)))
            if ft_state:
                embed_key = "model.embed_tokens.weight"
                if embed_key in ft_state:
                    model.resize_token_embeddings(ft_state[embed_key].shape[0])
                model.load_state_dict(ft_state, strict=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map=device)
        model.eval()

        for layer_idx in layers:
            sae_path = os.path.join(ANALYSIS_DIR, f"sae_layer_{layer_idx}.pt")
            if not os.path.exists(sae_path):
                print(f"  Skipping layer {layer_idx} — no trained SAE found")
                continue

            # Load SAE
            ckpt_data = torch.load(sae_path, map_location=device, weights_only=True)
            sae = SparseAutoencoder(
                ckpt_data["hidden_dim"], ckpt_data["dictionary_size"], ckpt_data["k"]
            ).to(device).float()
            sae.load_state_dict(ckpt_data["state_dict"])
            sae.eval()

            # Collect activations
            acts = collect_activations(model, tokenizer, layer_idx, prompts, device, max_tokens=10000)

            # Encode through SAE
            with torch.no_grad():
                top_vals, top_idx, _ = sae.encode(acts.to(device))

            # Feature activation frequency
            feature_freq = torch.zeros(ckpt_data["dictionary_size"], device=device)
            feature_freq.scatter_add_(0, top_idx.reshape(-1),
                                       torch.ones(top_idx.numel(), device=device))
            feature_freq = feature_freq / acts.shape[0]  # normalize by num tokens

            # Mean activation magnitude per feature
            feature_magnitude = torch.zeros(ckpt_data["dictionary_size"], device=device)
            feature_magnitude.scatter_add_(0, top_idx.reshape(-1), top_vals.reshape(-1))
            feature_count = torch.zeros(ckpt_data["dictionary_size"], device=device)
            feature_count.scatter_add_(0, top_idx.reshape(-1),
                                        torch.ones(top_idx.numel(), device=device))
            feature_magnitude = feature_magnitude / feature_count.clamp(min=1)

            key = f"{model_label}_layer_{layer_idx}"
            results[key] = {
                "feature_freq": feature_freq.cpu(),
                "feature_magnitude": feature_magnitude.cpu(),
            }

        del model
        torch.cuda.empty_cache()

    # Compare features between base and fine-tuned
    print("\n" + "=" * 60)
    print("Feature Comparison: Base vs Fine-tuned")
    print("=" * 60)

    for layer_idx in layers:
        base_key = f"base_layer_{layer_idx}"
        ft_key = f"finetuned_layer_{layer_idx}"
        if base_key not in results or ft_key not in results:
            continue

        base_freq = results[base_key]["feature_freq"]
        ft_freq = results[ft_key]["feature_freq"]

        # Features that activated much more after fine-tuning
        freq_diff = ft_freq - base_freq
        top_increased = torch.topk(freq_diff, 10)
        top_decreased = torch.topk(-freq_diff, 10)

        print(f"\nLayer {layer_idx}:")
        print(f"  Top features INCREASED by fine-tuning:")
        for i, (val, idx) in enumerate(zip(top_increased.values, top_increased.indices)):
            print(f"    Feature {idx.item():5d}: +{val.item():.4f} (base={base_freq[idx].item():.4f} → ft={ft_freq[idx].item():.4f})")
        print(f"  Top features DECREASED by fine-tuning:")
        for i, (val, idx) in enumerate(zip(top_decreased.values, top_decreased.indices)):
            print(f"    Feature {idx.item():5d}: -{val.item():.4f} (base={base_freq[idx].item():.4f} → ft={ft_freq[idx].item():.4f})")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].scatter(base_freq.numpy(), ft_freq.numpy(), alpha=0.1, s=1)
        axes[0].plot([0, base_freq.max()], [0, base_freq.max()], 'r--', alpha=0.5)
        axes[0].set_xlabel("Base Feature Frequency")
        axes[0].set_ylabel("Fine-tuned Feature Frequency")
        axes[0].set_title(f"Layer {layer_idx}: Feature Frequency Shift")

        axes[1].hist(freq_diff.numpy(), bins=100, alpha=0.7)
        axes[1].set_xlabel("Frequency Change (ft - base)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Layer {layer_idx}: Distribution of Feature Changes")
        axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        out_png = os.path.join(ANALYSIS_DIR, f"sae_compare_layer_{layer_idx}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"  Saved to {out_png}")

    # Save JSON summary
    summary = {}
    for key, data in results.items():
        summary[key] = {
            "num_active_features": int((data["feature_freq"] > 0).sum()),
            "mean_freq": float(data["feature_freq"].mean()),
            "max_freq": float(data["feature_freq"].max()),
        }
    out_json = os.path.join(ANALYSIS_DIR, "sae_compare.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAE training and analysis for sql-llm")
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train", help="Train SAEs on model activations")
    train_p.add_argument("--layers", type=str, default="0,6,12,18,23", help="Comma-separated layer indices")
    train_p.add_argument("--device", type=str, default="cuda:0")
    train_p.add_argument("--dict-size", type=int, default=4096, help="SAE dictionary size")
    train_p.add_argument("--top-k", type=int, default=64, help="TopK sparsity")
    train_p.add_argument("--steps", type=int, default=5000, help="Training steps")
    train_p.add_argument("--max-tokens", type=int, default=100000, help="Max activation tokens to collect")

    compare_p = subparsers.add_parser("compare", help="Compare features between base and fine-tuned")
    compare_p.add_argument("--layers", type=str, default="12,18,23")
    compare_p.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"train": cmd_train, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
