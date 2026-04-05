"""Model loading for GPT-OSS 20B via HuggingFace transformers.

Uses the built-in GptOssForCausalLM which handles MXFP4 dequantization
efficiently. ~21GB VRAM, supports training + LoRA out of the box.

Supports multi-GPU: uses device_map="auto" to shard layers across all
available GPUs. Falls back to single GPU transparently.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")


def load_model():
    """Load GPT-OSS 20B model via transformers.

    Uses device_map="auto" to automatically distribute across available GPUs.
    With a single GPU, everything goes to cuda:0.
    """
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if hasattr(model, "hf_device_map"):
        devices_used = sorted(set(str(v) for v in model.hf_device_map.values()))
        print(f"Model distributed across devices: {devices_used}")
    return model


def load_hf_tokenizer():
    """Load the HuggingFace tokenizer for GPT-OSS 20B."""
    return AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
