"""Vendored from gpt-oss: checkpoint loading with MXFP4 dequantization."""

import math
import os

import torch
from safetensors import safe_open


# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _build_param_name_map(num_layers: int) -> dict:
    """Build parameter name mapping dynamically based on number of layers."""
    name_map = {}
    for n in range(num_layers):
        name_map[f"block.{n}.mlp.mlp1_bias"] = f"block.{n}.mlp.mlp1_bias"
        name_map[f"block.{n}.mlp.mlp1_weight"] = (
            f"block.{n}.mlp.mlp1_weight.blocks",
            f"block.{n}.mlp.mlp1_weight.scales",
        )
        name_map[f"block.{n}.mlp.mlp2_bias"] = f"block.{n}.mlp.mlp2_bias"
        name_map[f"block.{n}.mlp.mlp2_weight"] = (
            f"block.{n}.mlp.mlp2_weight.blocks",
            f"block.{n}.mlp.mlp2_weight.scales",
        )
    return name_map


class Checkpoint:
    def __init__(self, path: str, device: torch.device, num_layers: int = 24):
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str
        self.param_name_map = _build_param_name_map(num_layers)

        # Read from all files ending with .safetensors in the checkpoint directory
        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        # Build a mapping from tensor name to file
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        match self.param_name_map.get(name, name):
            case (blocks_name, scales_name):
                return self._get_mxfp4_tensor(
                    blocks_name, scales_name, dtype=torch.bfloat16
                )
            case tensor_name:
                return self._get_tensor(tensor_name)

    def _get_tensor(self, name: str) -> torch.Tensor:
        assert name in self.tensor_name_to_file, (
            f"Tensor {name} not found in checkpoint."
        )
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 16384 * 512,
    ) -> torch.Tensor:
        assert blocks_name in self.tensor_name_to_file, (
            f"Blocks tensor {blocks_name} not found in checkpoint."
        )
        assert scales_name in self.tensor_name_to_file, (
            f"Scales tensor {scales_name} not found in checkpoint."
        )

        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
