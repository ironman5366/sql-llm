"""Model loading for GPT-OSS 20B via HuggingFace transformers.

Uses the built-in GptOssForCausalLM which handles MXFP4 dequantization
efficiently. ~21GB VRAM, supports training + LoRA out of the box.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")


def load_model(device: str = "cuda"):
    """Load GPT-OSS 20B model via transformers.

    Returns the model on the specified device.
    """
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        dtype=torch.bfloat16,
        device_map=device,
    )
    return model


def load_hf_tokenizer():
    """Load the HuggingFace tokenizer for GPT-OSS 20B."""
    return AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
