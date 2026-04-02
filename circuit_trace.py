"""
Circuit tracing for sql-llm: trace how information flows through
the model when answering SQL queries.

Two approaches:
1. Gradient-based attribution: which components contribute to the correct answer?
2. Activation patching: what happens when we swap activations between base/fine-tuned?

Usage:
    uv run circuit_trace.py gradient "SELECT name FROM animals WHERE id = 1;"
    uv run circuit_trace.py patch "SELECT name FROM animals WHERE id = 1;"
    uv run circuit_trace.py full "SELECT name FROM animals WHERE id = 1;"
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis", "circuits")
BASE_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")
FT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")

os.makedirs(ANALYSIS_DIR, exist_ok=True)


def _load_ft_model(device="cuda"):
    """Load fine-tuned model (base + overlay)."""
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file

    model = AutoModelForCausalLM.from_pretrained(BASE_CHECKPOINT, dtype=torch.bfloat16, device_map=device)
    ft_state = {}
    for f in sorted(os.listdir(FT_CHECKPOINT)):
        if f.endswith(".safetensors"):
            ft_state.update(load_file(os.path.join(FT_CHECKPOINT, f), device=str(device)))
    if ft_state:
        embed_key = "model.embed_tokens.weight"
        if embed_key in ft_state:
            model.resize_token_embeddings(ft_state[embed_key].shape[0])
        model.load_state_dict(ft_state, strict=False)
    return model


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_CHECKPOINT)


# ---------------------------------------------------------------------------
# Gradient Attribution: which layers/components contribute to the answer?
# ---------------------------------------------------------------------------

def cmd_gradient(args):
    """Trace gradient flow from the answer token back through the network."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    query = args.query
    device = "cuda"
    tokenizer = _load_tokenizer()
    model = _load_ft_model(device)
    model.eval()

    # Format like training
    prompt = f"<|query|>{query}<|/query|> <|result|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        pred_token_id = torch.argmax(logits).item()
        pred_token = tokenizer.decode([pred_token_id])
        pred_prob = torch.softmax(logits, dim=-1)[pred_token_id].item()
        print(f"Query: {query}")
        print(f"Model predicts: '{pred_token}' (prob={pred_prob:.4f})")

    # Layer-wise gradient attribution using hooks
    model.zero_grad()
    layer_activations = []

    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs.retain_grad()
                layer_activations.append(hs)
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Also hook the embedding layer for input token attribution
    embed_output = []
    def embed_hook(module, input, output):
        output.retain_grad()
        embed_output.append(output)
    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    outputs = model(**inputs)
    logits = outputs.logits
    target_logit = logits[0, -1, pred_token_id]
    target_logit.backward()

    for h in hooks:
        h.remove()

    # Input token attribution from embedding gradients
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    input_token_grads = None
    if embed_output and embed_output[0].grad is not None:
        input_token_grads = embed_output[0].grad[0].norm(dim=-1).detach().cpu()
        print(f"\nGradient attribution (per input token):")
        for i, (tok, grad) in enumerate(zip(tokens, input_token_grads)):
            bar = "█" * int(grad / input_token_grads.max() * 40)
            print(f"  [{i:2d}] {tok:20s} grad={grad:.4f} {bar}")

    print(f"\nLayer-wise gradient norms (contribution to '{pred_token}'):")
    grad_norms = []
    for i, act in enumerate(layer_activations):
        if act.grad is not None:
            gn = act.grad[0, -1, :].norm().item()  # gradient at last token position
        else:
            gn = 0.0
        grad_norms.append(gn)
        bar = "█" * int(gn / max(grad_norms + [1e-10]) * 40) if grad_norms else ""
        print(f"  Layer {i:2d}: {gn:.6f}")

    # Recompute bars with final max
    print(f"\nLayer gradient profile:")
    max_gn = max(grad_norms) if grad_norms else 1
    for i, gn in enumerate(grad_norms):
        bar = "█" * int(gn / max_gn * 50)
        print(f"  Layer {i:2d}: {bar}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    if input_token_grads is not None:
        ax1.barh(range(len(tokens)), input_token_grads.float().numpy())
        ax1.set_yticks(range(len(tokens)))
        ax1.set_yticklabels(tokens, fontsize=8)
        ax1.set_xlabel("Gradient Norm")
        ax1.set_title(f"Input Token Attribution for '{pred_token}'")
        ax1.invert_yaxis()

    ax2.bar(range(len(grad_norms)), grad_norms)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Gradient Norm (last token)")
    ax2.set_title(f"Layer-wise Gradient Attribution for '{pred_token}'")

    plt.tight_layout()
    out_png = os.path.join(ANALYSIS_DIR, "gradient_attribution.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved to {out_png}")

    # Save data
    out_json = os.path.join(ANALYSIS_DIR, "gradient_attribution.json")
    with open(out_json, "w") as f:
        json.dump({
            "query": query,
            "predicted_token": pred_token,
            "predicted_prob": pred_prob,
            "input_tokens": tokens,
            "input_token_grads": input_token_grads.tolist() if input_token_grads is not None else [],
            "layer_grad_norms": grad_norms,
        }, f, indent=2)
    print(f"Saved to {out_json}")


# ---------------------------------------------------------------------------
# Activation Patching: swap layer outputs between base and fine-tuned
# ---------------------------------------------------------------------------

def cmd_patch(args):
    """Token-position-specific activation patching (Meng et al. style).

    For each (layer, token_position), replace ONLY that position's hidden state
    with the base model's. Produces a 2D heatmap showing which layer × position
    combinations are causally responsible for the answer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoModelForCausalLM

    query = args.query
    device = "cuda"
    tokenizer = _load_tokenizer()

    prompt = f"<|query|>{query}<|/query|> <|result|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    # Get base and fine-tuned model outputs
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_CHECKPOINT, dtype=torch.bfloat16, device_map=device)
    base_model.eval()

    print("Loading fine-tuned model...")
    ft_model = _load_ft_model(device)
    ft_model.eval()

    # Cache all layer outputs from both models
    base_hidden = []
    ft_hidden = []

    def make_cache_hook(cache_list):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            cache_list.append(hs.detach())
        return hook

    hooks = []
    for layer in base_model.model.layers:
        hooks.append(layer.register_forward_hook(make_cache_hook(base_hidden)))
    with torch.no_grad():
        base_out = base_model(**inputs)
    for h in hooks:
        h.remove()
    base_pred = tokenizer.decode([torch.argmax(base_out.logits[0, -1, :]).item()])

    hooks = []
    for layer in ft_model.model.layers:
        hooks.append(layer.register_forward_hook(make_cache_hook(ft_hidden)))
    with torch.no_grad():
        ft_out = ft_model(**inputs)
    for h in hooks:
        h.remove()
    ft_logits = ft_out.logits[0, -1, :]
    ft_pred_id = torch.argmax(ft_logits).item()
    ft_pred = tokenizer.decode([ft_pred_id])
    ft_prob = torch.softmax(ft_logits, dim=-1)[ft_pred_id].item()

    print(f"Base predicts: '{base_pred}'")
    print(f"Fine-tuned predicts: '{ft_pred}' (prob={ft_prob:.4f})")
    print(f"Tokens ({seq_len}): {tokens}")

    del base_model
    torch.cuda.empty_cache()

    num_layers = len(ft_hidden)

    # Token-position-specific patching: (num_layers × seq_len) grid
    # For each (layer, position), replace ONLY that position's hidden state
    print(f"\nPatching {num_layers} layers × {seq_len} positions = {num_layers * seq_len} experiments...")
    prob_drop_matrix = np.zeros((num_layers, seq_len))

    for patch_layer in tqdm(range(num_layers), desc="Layers"):
        for patch_pos in range(seq_len):
            def make_pos_patch_hook(layer_idx, base_hs, pos):
                def hook(module, input, output):
                    if layer_idx == patch_layer:
                        hs = output[0] if isinstance(output, tuple) else output
                        # Replace ONLY the specific token position with base
                        patched = hs.clone()
                        patched[0, pos, :] = base_hs[0, pos, :]
                        if isinstance(output, tuple):
                            return (patched,) + output[1:]
                        return patched
                    return output
                return hook

            hooks = []
            for i, layer in enumerate(ft_model.model.layers):
                hooks.append(layer.register_forward_hook(
                    make_pos_patch_hook(i, base_hidden[i], patch_pos)
                ))

            with torch.no_grad():
                patched_out = ft_model(**inputs)

            for h in hooks:
                h.remove()

            patched_logits = patched_out.logits[0, -1, :]
            patched_prob = torch.softmax(patched_logits, dim=-1)[ft_pred_id].item()
            prob_drop_matrix[patch_layer, patch_pos] = ft_prob - patched_prob

    # Print top causal (layer, position) pairs
    flat_idx = np.argsort(prob_drop_matrix.ravel())[::-1]
    print(f"\nTop 15 most causal (layer, token) pairs for '{ft_pred}':")
    for rank, idx in enumerate(flat_idx[:15]):
        layer = idx // seq_len
        pos = idx % seq_len
        drop = prob_drop_matrix[layer, pos]
        print(f"  {rank+1:2d}. Layer {layer:2d}, Pos {pos:2d} ('{tokens[pos]}'): Δ={drop:+.4f}")

    # Plot 2D heatmap
    fig, ax = plt.subplots(figsize=(max(14, seq_len * 0.6), max(8, num_layers * 0.4)))
    im = ax.imshow(prob_drop_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=max(0.1, prob_drop_matrix.max()))
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(num_layers))
    ax.set_title(f"Causal Tracing: Prob drop of '{ft_pred}' when patching (layer × token)\nRed = high causal importance")
    plt.colorbar(im, ax=ax, label=f"P('{ft_pred}') drop")
    plt.tight_layout()

    out_png = os.path.join(ANALYSIS_DIR, "activation_patching.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved to {out_png}")

    # Also plot per-layer aggregate (sum across positions)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    layer_totals = prob_drop_matrix.sum(axis=1)
    ax1.barh(range(num_layers), layer_totals)
    ax1.set_ylabel("Layer")
    ax1.set_xlabel("Total Prob Drop (summed across positions)")
    ax1.set_title("Per-layer causal importance")
    ax1.invert_yaxis()

    pos_totals = prob_drop_matrix.sum(axis=0)
    ax2.bar(range(seq_len), pos_totals)
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Total Prob Drop (summed across layers)")
    ax2.set_title("Per-token causal importance")
    plt.tight_layout()

    out_png2 = os.path.join(ANALYSIS_DIR, "activation_patching_summary.png")
    plt.savefig(out_png2, dpi=150)
    print(f"Saved to {out_png2}")

    out_json = os.path.join(ANALYSIS_DIR, "activation_patching.json")
    with open(out_json, "w") as f:
        json.dump({
            "query": query,
            "tokens": tokens,
            "base_pred": base_pred,
            "ft_pred": ft_pred,
            "ft_prob": ft_prob,
            "prob_drop_matrix": prob_drop_matrix.tolist(),
            "top_causal_pairs": [
                {"layer": int(flat_idx[i] // seq_len),
                 "pos": int(flat_idx[i] % seq_len),
                 "token": tokens[flat_idx[i] % seq_len],
                 "prob_drop": float(prob_drop_matrix.ravel()[flat_idx[i]])}
                for i in range(min(30, len(flat_idx)))
            ],
        }, f, indent=2)
    print(f"Saved to {out_json}")


# ---------------------------------------------------------------------------
# Full Analysis
# ---------------------------------------------------------------------------

def cmd_full(args):
    """Run both gradient attribution and activation patching."""
    cmd_gradient(args)
    cmd_patch(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Circuit tracing for sql-llm")
    subparsers = parser.add_subparsers(dest="command")

    g = subparsers.add_parser("gradient", help="Gradient-based attribution")
    g.add_argument("query", type=str)

    p = subparsers.add_parser("patch", help="Activation patching")
    p.add_argument("query", type=str)

    f = subparsers.add_parser("full", help="Run both analyses")
    f.add_argument("query", type=str)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"gradient": cmd_gradient, "patch": cmd_patch, "full": cmd_full}[args.command](args)


if __name__ == "__main__":
    main()
