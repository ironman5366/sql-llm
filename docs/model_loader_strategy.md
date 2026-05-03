# General loader strategy for the model targets we want

Date: 2026-05-03

The Qwen3.5 detour raised a real question: what's the right HF loader contract
for the SFT path so that the saved checkpoint reloads in sglang? This doc
surveys the four model targets (GPT-OSS-20B, DeepSeek V4, Gemma 4, Qwen3.5)
and proposes a single dispatch rule.

## Survey

Verified by reading each model's `config.json` on HF Hub and grepping
`EntryClass` in our installed `sglang/srt/models/*.py`.

| Model                       | HF top-level architecture           | Modality           | sglang 0.5.10 native? | Loader that "just works" with sglang |
|-----------------------------|-------------------------------------|--------------------|------------------------|--------------------------------------|
| `openai/gpt-oss-20b`        | `GptOssForCausalLM`                 | text-only          | ✅ `gpt_oss.py`        | `AutoModelForCausalLM`               |
| `deepseek-ai/DeepSeek-V4-*` | `DeepseekV4ForCausalLM`             | text-only          | ❌ only V2/V3/V3.2 in 0.5.10 | `AutoModelForCausalLM` — once sglang ships V4 |
| `google/gemma-4-E2B-it`     | `Gemma4ForConditionalGeneration`    | text + vision + audio | ❌ no `gemma4.py` in 0.5.10 | `AutoModelForImageTextToText` — once sglang ships gemma4 |
| `Qwen/Qwen3.5-*`            | `Qwen3_5ForConditionalGeneration`   | text + vision      | ✅ multimodal path; ❌ dense path | `AutoModelForImageTextToText` (keeps multimodal arch in saved checkpoint) |

For reference, today's older default sits in the simple bucket:

| `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen2ForCausalLM`                  | text-only          | ✅ `qwen2.py`          | `AutoModelForCausalLM` |

## Pattern

There are two axes that split these targets, not one.

**Axis A — saved-checkpoint architecture.** `AutoModelForCausalLM` always
strips a multimodal model down to its `*ForCausalLM` inner module and saves a
text-only config. `AutoModelForImageTextToText` keeps the
`*ForConditionalGeneration` outer module and writes a config that still
contains `vision_config` / `audio_config`. **This is purely an artifact of the
HF loader you call**; the safetensors data is the same set of tensors plus or
minus the vision/audio towers.

**Axis B — what sglang accepts.** SGLang's `EntryClass` table maps an `arch`
string to a runtime model class. If the saved arch isn't registered, the load
fails (or falls back to a generic transformers backend that explicitly opts
out of some architectures, e.g. Qwen3.5).

Stating those together: the rule that makes every target above work today,
without sglang patches, is

```
if architectures[0] in MULTIMODAL_ARCH_NAMES:
    AutoModelForImageTextToText
else:
    AutoModelForCausalLM
```

…which is exactly what `llm/schemes/tagged_rows_sft_v0.py:_model_loader_class`
already does. The check matches `ConditionalGeneration` / `ImageTextToText`
substrings in the original config's `architectures`, which covers Gemma 4 and
Qwen3.5 by name and is the right behavior even for future multimodal models we
haven't seen yet (the substring is a strong, stable HF convention).

The vision/audio cost we pay by going through `ImageTextToText`:

| Model         | LM weights | Full multimodal weights | Extra |
|---------------|------------|-------------------------|-------|
| Qwen3.5-0.8B  | ~1.5 GB    | ~1.7 GB                 | ~200 MB |
| Qwen3.5-9B    | ~17.4 GB   | ~19.3 GB                | ~1.9 GB |
| Gemma-4-E2B   | n/a yet    | reported ~5 GB raw      | tbd     |

Vision/audio towers are loaded but not fed any inputs in our text-only path,
so the only real cost is disk and resident VRAM during SFT. Negligible at
these sizes; not a reason to chase the dense path.

## What this means per target

- **GPT-OSS-20B (`GptOssForCausalLM`)** — works today through
  `AutoModelForCausalLM`. The branch-existing dispatch picks the right loader
  on its own (no `ConditionalGeneration` in the arch). The one open question is
  whether GPT-OSS's MXFP4-quantized release dequantizes cleanly for SFT — that
  needs an actual probe load on a single GPU, not just config inspection.

- **DeepSeek-V4 (`DeepseekV4ForCausalLM`)** — also routes through
  `AutoModelForCausalLM` cleanly, *if* you can find a sglang that registers
  `DeepseekV4ForCausalLM` as an `EntryClass`. Our pinned 0.5.10 only has V2,
  V3, V3.2. Per LMSYS's 2026-04-25 blog and SGLang issue #23602, V4 has
  Day-0 support in sglang's main branch; whether it's in a tagged release yet
  is the gate. Same flavor of "wait for sglang" as Gemma 4 below; nothing in
  our scheme code needs to change.

- **Gemma 4 E2B/E4B (`Gemma4ForConditionalGeneration`)** — the dispatch
  picks `AutoModelForImageTextToText`, correct. Blocker is that
  `gemma4.py` does not exist in `sglang/srt/models/` for 0.5.10. The official
  Gemma 4 cookbook says you need sglang from git main *plus* a specific
  transformers commit. Pyproject lift, not a scheme change.

- **Qwen3.5 small (`Qwen3_5ForConditionalGeneration`)** — already running on
  this branch. Dispatch picks `AutoModelForImageTextToText`. Multimodal arch
  preserved in saved checkpoint; sglang's existing `qwen3_5.py` handles the
  conditional-generation path fine. The dense path has its own bugs but we're
  not using it.

## Recommendation

1. **Keep the current `_model_loader_class` rule** (substring match on
   `ConditionalGeneration` / `ImageTextToText` in `architectures[0]`). It's
   the minimal, correct dispatch for every target on the list.

2. **Don't try to chase the dense `*ForCausalLM` path for Qwen3.5** until
   either (a) we don't care about that family or (b) upstream fixes
   `radix_linear_attention.py:95` and the dense-MoE config attr leakage. The
   2 GB of extra vision-tower weight is cheaper than the patch tax.

3. **Treat sglang version as the gating axis for new model families.** The
   loader dispatch is stable; sglang's `EntryClass` registry is the moving
   piece. Track which target wants which sglang version in the model registry
   doc so we can plan dep bumps deliberately rather than reactively.

4. **Add a small `llm/sglang_extras/`-style hook for any future model** that
   needs config coercion or method overrides at load time. The Qwen3.5
   `get_model_config_for_expert_location` override is the template — small,
   self-contained, opt-in via `SGLANG_EXTERNAL_MODEL_PACKAGE`.

In short: one dispatch rule (already in place), and version-pin sglang
deliberately as we add Gemma 4 / DeepSeek V4. No general-solution refactor
needed.

## Sources

- [LMSYS — DeepSeek-V4 on Day 0 with SGLang](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang Cookbook — DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [SGLang Cookbook — Gemma 4](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4)
- [Hugging Face — openai/gpt-oss-20b config.json](https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json)
- [Hugging Face — deepseek-ai/DeepSeek-V4-Flash config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json)
- [Hugging Face — google/gemma-4-E2B-it config.json](https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json)
- [Welcome Gemma 4 (HF blog)](https://huggingface.co/blog/gemma4)
- [SGLang issue #23602 — DeepSeek V4 Roadmap](https://github.com/sgl-project/sglang/issues/23602)
