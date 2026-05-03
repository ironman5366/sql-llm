"""Register Qwen3.5 text-only causal classes with the SGLang model registry.

Stock sglang 0.5.10 only registers Qwen3_5ForConditionalGeneration via
EntryClass, but our SFT pipeline saves checkpoints with
architectures=["Qwen3_5ForCausalLM"] (because we load via
AutoModelForCausalLM, which strips the multimodal wrapper). Pointing
SGLANG_EXTERNAL_MODEL_PACKAGE at this package makes those checkpoints
loadable.

Two adjustments are needed on top of bare registration:

1. The HF transformers Qwen3_5TextConfig exposes ``layer_types`` but
   sglang's model code reads ``layers_block_type`` (a property on the
   sglang-side Qwen3NextConfig). We promote the incoming config to
   sglang's own Qwen3_5TextConfig before delegating to the base class.

2. The base Qwen3_5ForCausalLM inherits a get_model_config_for_expert_location
   classmethod that references config.num_experts, which only exists on
   the MoE config. For the dense text-only model we override it to
   return None so EPLB initialization takes the no-op path.
"""

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig as _SGLangText
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM as _BaseQwen3_5ForCausalLM
from sglang.srt.models.qwen3_5 import (
    Qwen3_5MoeForCausalLM as _BaseQwen3_5MoeForCausalLM,
)


def _coerce_text_config(config):
    if isinstance(config, _SGLangText):
        return config
    return _SGLangText(**config.to_dict())


class Qwen3_5ForCausalLM(_BaseQwen3_5ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(_coerce_text_config(config), *args, **kwargs)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return None


class Qwen3_5MoeForCausalLM(_BaseQwen3_5MoeForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(_coerce_text_config(config), *args, **kwargs)


EntryClass = [Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM]
