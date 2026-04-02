"""
Analysis tooling for sql-llm experiments.

Usage:
    uv run analyze.py weight-diff
    uv run analyze.py logit-lens "SELECT name FROM animals WHERE id = 1"
    uv run analyze.py activations
    uv run analyze.py expert-routing
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

BASE_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")
FT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis")

os.makedirs(ANALYSIS_DIR, exist_ok=True)


def _load_model(path, device="cuda"):
    """Load model. For fine-tuned checkpoint, loads base model first then applies saved weights."""
    from transformers import AutoModelForCausalLM
    print(f"Loading model from {path}...")

    if path == FT_CHECKPOINT:
        # Fine-tuned model only has non-MoE weights (MoE stays in MXFP4 from base).
        # Load base model first, then overlay fine-tuned weights.
        model = AutoModelForCausalLM.from_pretrained(BASE_CHECKPOINT, dtype=torch.bfloat16, device_map=device)

        from safetensors.torch import load_file
        ft_state = {}
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors"):
                ft_state.update(load_file(os.path.join(path, f), device=str(device)))

        if ft_state:
            # Resize embeddings if fine-tuned model has different vocab size
            embed_key = "model.embed_tokens.weight"
            if embed_key in ft_state and ft_state[embed_key].shape[0] != model.model.embed_tokens.weight.shape[0]:
                new_vocab = ft_state[embed_key].shape[0]
                model.resize_token_embeddings(new_vocab)
                print(f"  Resized embeddings to {new_vocab}")

            missing, unexpected = model.load_state_dict(ft_state, strict=False)
            print(f"  Loaded fine-tuned weights: {len(ft_state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")

        model.eval()
        return model
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
        model.eval()
        return model


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_CHECKPOINT)


# ---------------------------------------------------------------------------
# Weight Diff Analysis
# ---------------------------------------------------------------------------

def cmd_weight_diff(args):
    """Analyze weight changes between base and fine-tuned model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base = _load_model(BASE_CHECKPOINT, device="cpu")
    ft = _load_model(FT_CHECKPOINT, device="cpu")

    base_state = dict(base.named_parameters())
    ft_state = dict(ft.named_parameters())

    # Collect per-layer, per-component diffs
    layer_component_diffs = defaultdict(lambda: defaultdict(float))
    layer_component_relative = defaultdict(lambda: defaultdict(float))
    global_diffs = {}

    for name in base_state:
        if name not in ft_state:
            continue
        bp = base_state[name]
        fp = ft_state[name]
        if bp.shape != fp.shape:
            continue

        diff = (fp.float() - bp.float())
        l2 = torch.norm(diff).item()
        base_norm = torch.norm(bp.float()).item()
        relative = (l2 / base_norm * 100) if base_norm > 0 else 0

        global_diffs[name] = {"l2": l2, "relative_pct": relative, "shape": list(bp.shape)}

        # Parse layer and component
        parts = name.split(".")
        if "layers" in name:
            # e.g. model.layers.5.self_attn.q_proj.weight
            layer_idx = None
            component = "other"
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                if p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    component = p
                elif p in ("gate", "up_proj", "down_proj"):
                    component = f"mlp_{p}"
                elif p == "router":
                    component = "router"
                elif p == "input_layernorm" or p == "post_attention_layernorm":
                    component = "norm"
            if layer_idx is not None:
                layer_component_diffs[layer_idx][component] += l2
                layer_component_relative[layer_idx][component] += relative
        elif "embed" in name:
            layer_component_diffs[-1]["embedding"] += l2
        elif "lm_head" in name:
            layer_component_diffs[-2]["lm_head"] += l2

    # Save raw data
    out_json = os.path.join(ANALYSIS_DIR, "weight_diff.json")
    with open(out_json, "w") as f:
        json.dump({
            "global_diffs": {k: v for k, v in sorted(global_diffs.items(), key=lambda x: -x[1]["l2"])[:50]},
            "layer_component_diffs": {str(k): dict(v) for k, v in sorted(layer_component_diffs.items())},
        }, f, indent=2)
    print(f"Saved raw data to {out_json}")

    # Plot heatmap
    num_layers = max(k for k in layer_component_diffs if k >= 0) + 1 if layer_component_diffs else 24
    components = sorted(set(c for layer_data in layer_component_diffs.values() for c in layer_data if c != "other"))

    if not components:
        print("No layer components found to plot.")
        return

    matrix = np.zeros((num_layers, len(components)))
    for layer_idx in range(num_layers):
        for j, comp in enumerate(components):
            matrix[layer_idx, j] = layer_component_diffs.get(layer_idx, {}).get(comp, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(components) * 1.5), max(8, num_layers * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_title("Weight Change L2 Norm by Layer × Component")
    plt.colorbar(im, ax=ax, label="L2 Norm")
    plt.tight_layout()

    out_png = os.path.join(ANALYSIS_DIR, "weight_diff.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved heatmap to {out_png}")

    # Print summary
    print("\nTop 10 most-changed parameter groups:")
    sorted_diffs = sorted(global_diffs.items(), key=lambda x: -x[1]["l2"])
    for name, info in sorted_diffs[:10]:
        print(f"  {name}: L2={info['l2']:.4f} ({info['relative_pct']:.2f}%)")

    # Per-layer total
    print("\nPer-layer total weight change:")
    for layer_idx in sorted(layer_component_diffs.keys()):
        if layer_idx < 0:
            label = "embedding" if layer_idx == -1 else "lm_head"
        else:
            label = f"layer {layer_idx:2d}"
        total = sum(layer_component_diffs[layer_idx].values())
        print(f"  {label}: {total:.4f}")

    del base, ft
    print("\nDone.")


# ---------------------------------------------------------------------------
# LogitLens
# ---------------------------------------------------------------------------

def cmd_logit_lens(args):
    """Project hidden states through unembedding at each layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    query_text = args.query
    tokenizer = _load_tokenizer()
    device = "cuda"

    results = {}
    for label, ckpt in [("base", BASE_CHECKPOINT), ("finetuned", FT_CHECKPOINT)]:
        model = _load_model(ckpt, device=device)

        # Format query the same way as method.py
        prompt = f"<|query|>{query_text}<|/query|> <|result|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
        lm_head = model.lm_head

        layer_predictions = []
        for layer_idx, hs in enumerate(hidden_states):
            # Apply final layer norm if available
            if hasattr(model.model, "norm"):
                hs_normed = model.model.norm(hs)
            else:
                hs_normed = hs

            logits = lm_head(hs_normed[0, -1, :])  # last token position
            top5_ids = torch.topk(logits, 5).indices.tolist()
            top5_tokens = [tokenizer.decode([tid]).strip() for tid in top5_ids]
            top5_probs = torch.softmax(logits, dim=-1)[top5_ids].tolist()

            layer_predictions.append({
                "layer": layer_idx,
                "top1": top5_tokens[0],
                "top5": list(zip(top5_tokens, [f"{p:.3f}" for p in top5_probs])),
            })

        results[label] = layer_predictions
        del model
        torch.cuda.empty_cache()

    # Print comparison
    print(f"\nLogitLens for: {query_text}")
    print(f"{'Layer':>6} | {'Base Top-1':>20} | {'Finetuned Top-1':>20}")
    print("-" * 55)
    for i in range(len(results["base"])):
        base_pred = results["base"][i]["top1"]
        ft_pred = results["finetuned"][i]["top1"] if i < len(results["finetuned"]) else "?"
        marker = " <<<" if base_pred != ft_pred else ""
        print(f"{i:>6} | {base_pred:>20} | {ft_pred:>20}{marker}")

    # Save
    out_json = os.path.join(ANALYSIS_DIR, "logit_lens.json")
    with open(out_json, "w") as f:
        json.dump({"query": query_text, "results": results}, f, indent=2)
    print(f"\nSaved to {out_json}")


# ---------------------------------------------------------------------------
# Activation Comparison
# ---------------------------------------------------------------------------

def cmd_activations(args):
    """Compare hidden states between base and fine-tuned model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tokenizer = _load_tokenizer()
    device = "cuda"

    # Sample queries to test
    test_queries = [
        "SELECT name FROM animals WHERE id = 1",
        "SELECT capital FROM countries WHERE id = 1",
        "SELECT sensor_id FROM measurements WHERE id = 1",
        "SELECT year FROM events WHERE id = 1",
        "SELECT name FROM animals WHERE id = 99",  # nonexistent
    ]

    all_diffs = {}

    base_model = _load_model(BASE_CHECKPOINT, device=device)
    ft_model = _load_model(FT_CHECKPOINT, device=device)

    for query_text in test_queries:
        prompt = f"<|query|>{query_text}<|/query|> <|result|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            base_out = base_model(**inputs, output_hidden_states=True)
            ft_out = ft_model(**inputs, output_hidden_states=True)

        layer_diffs = []
        for i, (bh, fh) in enumerate(zip(base_out.hidden_states, ft_out.hidden_states)):
            b_last = bh[0, -1, :]
            f_last = fh[0, -1, :]
            l2 = torch.norm(f_last - b_last).item()
            cosine = F.cosine_similarity(b_last.unsqueeze(0), f_last.unsqueeze(0)).item()
            layer_diffs.append({"layer": i, "l2_diff": l2, "cosine_sim": cosine})

        all_diffs[query_text] = layer_diffs

    del base_model, ft_model
    torch.cuda.empty_cache()

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for query_text, diffs in all_diffs.items():
        layers = [d["layer"] for d in diffs]
        l2s = [d["l2_diff"] for d in diffs]
        cosines = [d["cosine_sim"] for d in diffs]
        short_q = query_text.split("FROM")[0].strip()[:30]
        ax1.plot(layers, l2s, label=short_q, marker=".")
        ax2.plot(layers, cosines, label=short_q, marker=".")

    ax1.set_ylabel("L2 Distance")
    ax1.set_title("Hidden State Divergence: Base vs Fine-tuned (last token position)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Hidden State Direction Similarity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(ANALYSIS_DIR, "activations.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved to {out_png}")

    out_json = os.path.join(ANALYSIS_DIR, "activations.json")
    with open(out_json, "w") as f:
        json.dump(all_diffs, f, indent=2)
    print(f"Saved to {out_json}")


# ---------------------------------------------------------------------------
# MoE Expert Routing
# ---------------------------------------------------------------------------

def cmd_expert_routing(args):
    """Compare MoE expert routing between base and fine-tuned model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tokenizer = _load_tokenizer()
    device = "cuda"

    test_queries = [
        "SELECT name FROM animals WHERE id = 1",
        "SELECT capital FROM countries WHERE id = 3",
        "SELECT sensor_id FROM measurements WHERE id = 5",
        "SELECT year FROM events WHERE id = 10",
    ]

    results = {}

    for label, ckpt in [("base", BASE_CHECKPOINT), ("finetuned", FT_CHECKPOINT)]:
        model = _load_model(ckpt, device=device)

        # Find all MoE gate/router modules and hook them
        expert_selections = defaultdict(list)  # layer_idx -> list of selected expert indices
        hooks = []

        for name, module in model.named_modules():
            # Look for the router/gate in MoE blocks
            if "block_sparse_moe.gate" in name or "router" in name:
                layer_idx = None
                for part in name.split("."):
                    try:
                        layer_idx = int(part)
                    except ValueError:
                        continue

                if layer_idx is not None:
                    def make_hook(l_idx):
                        def hook_fn(module, input, output):
                            # output is router logits: (batch, seq_len, num_experts)
                            if isinstance(output, tuple):
                                logits = output[0]
                            else:
                                logits = output
                            top_k = torch.topk(logits, k=4, dim=-1).indices  # (batch, seq_len, k)
                            expert_selections[l_idx].append(top_k.detach().cpu())
                        return hook_fn
                    hooks.append(module.register_forward_hook(make_hook(layer_idx)))

        # Run queries
        for q in test_queries:
            prompt = f"<|query|>{q}<|/query|> <|result|>"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)

        for h in hooks:
            h.remove()

        results[label] = {
            str(k): [t.tolist() for t in v] for k, v in expert_selections.items()
        }

        del model
        torch.cuda.empty_cache()

    # Save
    out_json = os.path.join(ANALYSIS_DIR, "expert_routing.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Summarize
    print("\nMoE Expert Routing Summary:")
    for label in ["base", "finetuned"]:
        print(f"\n  {label.upper()}:")
        for layer, selections in sorted(results.get(label, {}).items(), key=lambda x: int(x[0])):
            # Flatten and count
            flat = []
            for sel in selections:
                for batch in sel:
                    for seq in batch:
                        flat.extend(seq)
            if flat:
                from collections import Counter
                counts = Counter(flat)
                top3 = counts.most_common(3)
                print(f"    Layer {layer}: top experts = {top3}")

    print(f"\nSaved to {out_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="sql-llm analysis tooling")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("weight-diff", help="Analyze weight changes between base and fine-tuned model")

    ll = subparsers.add_parser("logit-lens", help="Project hidden states through unembedding at each layer")
    ll.add_argument("query", type=str, help="SQL query to analyze")

    subparsers.add_parser("activations", help="Compare hidden states between base and fine-tuned model")
    subparsers.add_parser("expert-routing", help="Compare MoE expert routing patterns")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command != "weight-diff" and not os.path.isdir(FT_CHECKPOINT):
        print(f"Error: Fine-tuned model not found at {FT_CHECKPOINT}")
        print("Run `uv run method.py` first to train and save a model.")
        return

    commands = {
        "weight-diff": cmd_weight_diff,
        "logit-lens": cmd_logit_lens,
        "activations": cmd_activations,
        "expert-routing": cmd_expert_routing,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
