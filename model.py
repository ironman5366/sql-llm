"""Vendored from gpt-oss: PyTorch reference model for GPT-OSS (MoE transformer).

MoE weights are stored in native MXFP4 format (~10GB) and dequantized on-the-fly
only for the active experts per token, matching how the Triton backend works.
"""

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from weights import Checkpoint, FP4_VALUES


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim, base, dtype, initial_context_length=4096,
                 scaling_factor=1.0, ntk_alpha=1.0, ntk_beta=32.0, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def forward(self, query, key):
        device = query.device
        num_tokens = query.shape[0]
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        t = torch.arange(num_tokens, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration

        query_shape = query.shape
        query = _apply_rotary_emb(query.view(num_tokens, -1, self.head_dim), cos, sin).reshape(query_shape)
        key_shape = key.shape
        key = _apply_rotary_emb(key.view(num_tokens, -1, self.head_dim), cos, sin).reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx=0, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        self.qkv = torch.nn.Linear(config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16)
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads, config.hidden_size,
            device=device, dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim, config.rope_theta, torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha, ntk_beta=config.rope_ntk_beta,
        )

    def forward(self, x):
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, :self.num_attention_heads * self.head_dim].contiguous()
        kv_start = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        k = qkv[:, kv_start:kv_start + kv_size].contiguous()
        v = qkv[:, kv_start + kv_size:kv_start + 2 * kv_size].contiguous()

        q = q.view(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        return x + self.out(t)


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    return x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)


def _dequantize_mxfp4(blocks, scales, dtype=torch.bfloat16):
    """Dequantize MXFP4 blocks+scales to dtype. Works on arbitrary leading dims.

    Args:
        blocks: uint8 tensor [..., G, B] where B=16 (bytes per block, 2 FP4 per byte)
        scales: uint8 tensor [..., G] (biased exponents)
    Returns:
        tensor [..., G*B*2] in target dtype (each byte unpacks to 2 values)
    """
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    prefix_shape = blocks.shape[:-2]
    G, B = blocks.shape[-2], blocks.shape[-1]

    blocks_flat = blocks.reshape(-1, B)
    scales_flat = (scales.reshape(-1, 1).to(torch.int32) - 127)

    idx_lo = (blocks_flat & 0x0F).long()
    idx_hi = (blocks_flat >> 4).long()

    out = torch.empty(blocks_flat.shape[0], B * 2, dtype=dtype, device=blocks.device)
    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]
    torch.ldexp(out, scales_flat, out=out)

    return out.view(*prefix_shape, G * B * 2)


class MLPBlock(torch.nn.Module):
    """MoE MLP block with MXFP4 weight storage.

    Weights are stored as packed uint8 (blocks + scales) matching the on-disk
    format. Only the selected experts' weights are dequantized to BF16 during
    forward, keeping VRAM at ~14GB total instead of ~42GB.
    """

    def __init__(self, config: ModelConfig, device=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        # Biases are small, store as BF16 parameters
        per_rank_intermediate = config.intermediate_size // self.world_size
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(config.num_experts, per_rank_intermediate * 2, device=device, dtype=torch.bfloat16)
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, device=device, dtype=torch.bfloat16)
        )
        # MoE weights stored as MXFP4 buffers (blocks + scales as uint8)
        # Populated by from_checkpoint; registered as buffers there
        # mlp1_blocks: [num_experts, out_dim, G, 16]
        # mlp1_scales: [num_experts, out_dim, G]
        # mlp2_blocks, mlp2_scales: similar

    def forward(self, x):
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices  # (n_tokens, experts_per_token)

        n_tokens = t.shape[0]
        results = []

        # Process tokens one at a time to minimize peak memory from dequantization
        for i in range(n_tokens):
            t_i = t[i]  # (hidden_size,)
            idx_i = expert_indices[i]  # (experts_per_token,)
            w_i = expert_weights[i]  # (experts_per_token,)

            # Gather and dequantize only the 4 needed experts' weights
            with torch.no_grad():
                # MLP1: [4, out_dim, G, 16] -> dequant -> [4, out_dim, hidden_size]
                mlp1_w = _dequantize_mxfp4(
                    self.mlp1_blocks[idx_i], self.mlp1_scales[idx_i]
                )  # (4, out_dim, hidden_size)
                mlp1_b = self.mlp1_bias[idx_i]  # (4, out_dim)

            h = torch.einsum("eck,k->ec", mlp1_w, t_i) + mlp1_b  # (4, out_dim)
            h = swiglu(h, limit=self.swiglu_limit)  # (4, intermediate_size)
            del mlp1_w

            with torch.no_grad():
                mlp2_w = _dequantize_mxfp4(
                    self.mlp2_blocks[idx_i], self.mlp2_scales[idx_i]
                )  # (4, hidden_size, intermediate_size)
                mlp2_b = self.mlp2_bias[idx_i]  # (4, hidden_size)

            h = torch.einsum("eck,ek->ec", mlp2_w, h)  # (4, hidden_size)
            if self.world_size > 1:
                dist.all_reduce(h, op=dist.ReduceOp.SUM)
            h += mlp2_b
            del mlp2_w

            # Weighted sum of experts
            out_i = torch.einsum("ec,e->c", h, w_i)  # (hidden_size,)
            results.append(out_i)

        return x + torch.stack(results, dim=0)


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, device=None):
        super().__init__()
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config: ModelConfig, device=None):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList([
            TransformerBlock(config, i, device)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False,
            device=device, dtype=torch.bfloat16,
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.block:
            if self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(path: str, device: str | torch.device = "cuda") -> "Transformer":
        """Load model with MXFP4 MoE weights (~14GB VRAM instead of ~42GB)."""
        if not isinstance(device, torch.device):
            device = torch.device(device)

        with open(os.path.join(path, "config.json")) as f:
            config = ModelConfig(**json.load(f))

        # Create model shell on CPU (small: just attention + embedding + biases)
        model = Transformer(config, device="cpu")

        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate = config.intermediate_size // world_size

        # Load checkpoint for raw tensor access
        checkpoint = Checkpoint(path, torch.device("cpu"), num_layers=config.num_hidden_layers)

        # Load non-MoE parameters (BF16)
        moe_weight_names = set()
        for n in range(config.num_hidden_layers):
            moe_weight_names.add(f"block.{n}.mlp.mlp1_weight")
            moe_weight_names.add(f"block.{n}.mlp.mlp2_weight")

        for name, param in model.named_parameters():
            if name in moe_weight_names:
                continue  # Skip — we load these as raw MXFP4 below

            loaded = checkpoint.get(name)
            if "mlp1" in name:  # mlp1_bias
                loaded = loaded[:, my_rank * 2 * per_rank_intermediate:(my_rank + 1) * 2 * per_rank_intermediate, ...]
            elif "mlp2_bias" in name:
                pass  # no sharding needed for bias (it's per-expert, full hidden_size)
            param.data.copy_(loaded)
            del loaded

        # Load MoE weights as raw MXFP4 (blocks + scales as uint8 buffers)
        from safetensors import safe_open
        safetensor_files = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")
        ]
        tensor_to_file = {}
        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_to_file[key] = sf

        for n in range(config.num_hidden_layers):
            mlp = model.block[n].mlp
            for prefix, attr_blocks, attr_scales in [
                (f"block.{n}.mlp.mlp1_weight", "mlp1_blocks", "mlp1_scales"),
                (f"block.{n}.mlp.mlp2_weight", "mlp2_blocks", "mlp2_scales"),
            ]:
                blocks_name = f"{prefix}.blocks"
                scales_name = f"{prefix}.scales"

                with safe_open(tensor_to_file[blocks_name], framework="pt", device="cpu") as f:
                    blocks = f.get_tensor(blocks_name)
                with safe_open(tensor_to_file[scales_name], framework="pt", device="cpu") as f:
                    scales = f.get_tensor(scales_name)

                # Register as buffers (uint8, not tracked by autograd)
                mlp.register_buffer(attr_blocks, blocks)
                mlp.register_buffer(attr_scales, scales)

        # Move everything to target device
        model = model.to(device)
        return model


class TokenGenerator:
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        with torch.inference_mode():
            self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self, prompt_tokens, stop_tokens, temperature=1.0,
                 max_tokens=0, return_logprobs=False):
        tokens = list(prompt_tokens)
        num_generated = 0
        while max_tokens == 0 or num_generated < max_tokens:
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
            if temperature == 0.0:
                tok = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1).item()
            tokens.append(tok)
            num_generated += 1
            if return_logprobs:
                lp = torch.log_softmax(logits, dim=-1)[tok].item()
                yield tok, lp
            else:
                yield tok
            if tok in stop_tokens:
                break
