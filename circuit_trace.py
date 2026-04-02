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
    """Activation patching: find which layers are causally responsible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM

    query = args.query
    device = "cuda"
    tokenizer = _load_tokenizer()

    prompt = f"<|query|>{query}<|/query|> <|result|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

    # Get base hidden states
    hooks = []
    for layer in base_model.model.layers:
        hooks.append(layer.register_forward_hook(make_cache_hook(base_hidden)))
    with torch.no_grad():
        base_out = base_model(**inputs)
    for h in hooks:
        h.remove()
    base_logits = base_out.logits[0, -1, :]
    base_pred = tokenizer.decode([torch.argmax(base_logits).item()])

    # Get ft hidden states
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

    del base_model
    torch.cuda.empty_cache()

    # Now patch: run fine-tuned model but replace one layer's output
    # with the base model's output. If the answer changes, that layer is causal.
    print("\nActivation patching (replacing ft layer with base)...")
    patch_results = []

    for patch_layer in tqdm(range(len(ft_hidden)), desc="Patching"):
        patched_hidden = []

        def make_patch_hook(layer_idx, base_hs):
            def hook(module, input, output):
                if layer_idx == patch_layer:
                    # Replace this layer's output with base model's
                    if isinstance(output, tuple):
                        return (base_hs.to(output[0].device),) + output[1:]
                    return base_hs.to(output.device)
                return output
            return hook

        hooks = []
        for i, layer in enumerate(ft_model.model.layers):
            hooks.append(layer.register_forward_hook(make_patch_hook(i, base_hidden[i])))

        with torch.no_grad():
            patched_out = ft_model(**inputs)

        for h in hooks:
            h.remove()

        patched_logits = patched_out.logits[0, -1, :]
        patched_prob = torch.softmax(patched_logits, dim=-1)[ft_pred_id].item()
        patched_pred = tokenizer.decode([torch.argmax(patched_logits).item()])

        prob_change = ft_prob - patched_prob
        patch_results.append({
            "layer": patch_layer,
            "patched_pred": patched_pred,
            "patched_prob_of_correct": patched_prob,
            "prob_drop": prob_change,
        })

    # Print results
    print(f"\nActivation Patching Results (prob of '{ft_pred}' when patching each layer):")
    for r in patch_results:
        bar_len = int(abs(r["prob_drop"]) / max(abs(pr["prob_drop"]) for pr in patch_results) * 40) if any(pr["prob_drop"] for pr in patch_results) else 0
        direction = "▼" if r["prob_drop"] > 0.01 else "▲" if r["prob_drop"] < -0.01 else "="
        print(f"  Layer {r['layer']:2d}: prob={r['patched_prob_of_correct']:.4f} "
              f"(Δ={r['prob_drop']:+.4f}) {direction} pred='{r['patched_pred']}'")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    layers = [r["layer"] for r in patch_results]
    drops = [r["prob_drop"] for r in patch_results]
    colors = ["red" if d > 0.01 else "blue" if d < -0.01 else "gray" for d in drops]
    ax.bar(layers, drops, color=colors, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Prob Drop of '{ft_pred}' when patched with base")
    ax.set_title(f"Causal Attribution: Which layers are responsible for '{ft_pred}'?")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    out_png = os.path.join(ANALYSIS_DIR, "activation_patching.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved to {out_png}")

    out_json = os.path.join(ANALYSIS_DIR, "activation_patching.json")
    with open(out_json, "w") as f:
        json.dump({
            "query": query,
            "base_pred": base_pred,
            "ft_pred": ft_pred,
            "ft_prob": ft_prob,
            "patch_results": patch_results,
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
