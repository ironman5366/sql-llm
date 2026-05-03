# Model Research: Successors to Qwen2.5-1.5B-Instruct

Date: 2026-05-03

## Why we're looking

The current v0 demo (`llm/schemes/tagged_rows_sft_v0.py`, `DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`) trains and serves Qwen 2.5 — a Sept 2024 release. Several stronger small open models have shipped since, including a Qwen3.5 small-dense series (March 2026) and Gemma 4 E2B/E4B (April 2026).

Hard constraints from the existing pipeline:
- SGLang 0.5.10 (currently pinned in `pyproject.toml`) must serve it. Latest released is `0.5.10.post1`, April 8, 2026.
- We use `sampling_params.regex` for constrained decoding — regex backend works for any natively-registered architecture.
- `transformers==5.7.0` + `trl>=1.3.0` (`SFTTrainer`) must load and finetune it.
- We add ~25 custom XML-tag special tokens and call `resize_token_embeddings`.
- Training-only budget is ~10 min on one H100.
- 2 GPUs (`CUDA_VISIBLE_DEVICES=4,5`), one for sglang, one for trainer; ports `5370` (control) / `5371` (sglang).

## What's actually in our installed SGLang 0.5.10

I read the `EntryClass` registrations in `.venv/lib/python3.12/site-packages/sglang/srt/models/`. Quick map of what's natively served vs not:

| HF architecture                       | SGLang file              | Status                                          |
|---------------------------------------|--------------------------|-------------------------------------------------|
| `Qwen3ForCausalLM`                    | `qwen3.py`               | ✅ native (Qwen3-0.6B / 1.7B / 4B / 8B)         |
| `Qwen3_5ForConditionalGeneration`     | `qwen3_5.py`             | ✅ native (Qwen3.5-0.8B / 2B / 4B / 9B)        |
| `Qwen3_5MoeForConditionalGeneration`  | `qwen3_5.py`             | ✅ native (Qwen3.5-MoE)                         |
| `Gemma3ForCausalLM`                   | `gemma3_causal.py`       | ✅ native (gemma-3-1b/4b-it text)               |
| `Gemma4ForConditionalGeneration`      | —                        | ❌ **not in 0.5.10**, needs sglang from main + a specific transformers commit per the [Gemma 4 cookbook][gemma4-cookbook] |
| `LlamaForCausalLM` (Llama 3.2, SmolLM3) | `llama.py`             | ✅ native                                       |
| `Phi3ForCausalLM` (Phi-4-mini)        | `llama.py`               | ✅ native                                       |

I confirmed Qwen3.5 by reading the on-disk registrations:

```
qwen3_5.py:1724  EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
```

And confirmed Gemma 4 is **not** present:

```
$ grep -rln "Gemma4" .venv/.../sglang/srt/    # empty
```

The locally cached `models--google--gemma-4-E2B-it/config.json` shows `"architectures": ["Gemma4ForConditionalGeneration"]` with a full audio config (audio + vision + text), so any Gemma 4 swap in the demo path requires upgrading sglang past 0.5.10 first. The official cookbook entry says: install from git main, plus transformers commit `91b1ab1fdfa81a552644a92fbe3e8d88de40e167`.

## Note on multimodal architectures

Both Qwen3.5 small dense and Gemma 4 E-series ship with vision (Gemma 4 also audio) baked into the same checkpoint — the architecture class is `*ForConditionalGeneration`, not `*ForCausalLM`. For our text-only finetune+serve loop:

- The vision/audio towers add dead weight during SFT and load time, but don't break anything if we only feed text.
- vLLM exposes a `--language-model-only` flag that skips loading the vision encoder; sglang has an equivalent path. Worth using to keep VRAM tight on the trainer GPU.
- Multimodal builds in sglang have historically had pickier interactions with chunked prefill / cuda graph (see e.g. the Gemma 3 bidirectional-attention thread). Our existing flags `--disable-cuda-graph --disable-piecewise-cuda-graph --skip-server-warmup` already neutralize most of that; keep them.
- Regex constraints in sglang work at the token level, independent of modality, so v0's tagged-row decoding should still apply.

## Candidate slate

### Tier A — newest, native-supported, recommended to try first

**1. `Qwen/Qwen3.5-2B` ⭐ recommended first swap**
- Released Mar 2, 2026. Dense 2B (24 layers, hidden 2048, 248k vocab, 262k context).
- Hybrid attention: linear-attention layers + full-attention every 4th layer.
- Native sglang 0.5.10 entry: `Qwen3_5ForConditionalGeneration`.
- Apache 2.0. Has both `-Base` and instruct checkpoints.
- Direct successor in the same family as today's checkpoint, so chat template / tokenizer behavior maps cleanly.
- Strongest "newest small dense Qwen" option that fits our pipeline today.

**2. `Qwen/Qwen3.5-4B`**
- Same family, 32 layers, hidden 2560, same vocab/context.
- ~2× the train cost vs 2B but still inside the 10-min budget on H100 with bf16, batch=1.
- Best choice if the 2B baseline saturates and we want to test scheme-vs-base capacity.

**3. `Qwen/Qwen3.5-0.8B`**
- Same family, smallest of the dense set. Useful as the "smallest credible base" probe — does compression-into-weights still work when there's much less weight?
- Doubles as a fast iteration target while debugging the new path.

### Tier B — even newer, but needs an sglang upgrade

**4. `google/gemma-4-E2B-it`**
- Released Apr 2, 2026. "E2B" = effective 2B via Per-Layer Embeddings; the underlying weights are larger (~5B raw) but PLE keeps active params small.
- Native audio + vision, text reasoning quality on par with Gemma 3 4B.
- **Blocker**: not in sglang 0.5.10. To use it we'd have to:
  1. `uv pip install` (no — `uv add`) sglang from git main, or wait for the next pypi release that ships PR #21952.
  2. Pin transformers to commit `91b1ab1fdfa81a552644a92fbe3e8d88de40e167` (per cookbook).
  3. Set `SGLANG_USE_AITER=0` if AMD (we're on H100, so n/a).
- High risk of weeks of dependency churn since `pyproject.toml` currently pins `sglang==0.5.10`, `transformers==5.7.0`, with `override-dependencies = ["transformers==5.7.0"]`. Going off pypi-release ground is a bigger lift than just changing a model name.

**5. `google/gemma-4-E4B-it`**
- Same E-series, ~4B effective params. Same blocker as above.

### Tier C — older but worth keeping in mind

These were what I had in the previous version of this doc. They're older than Tier A but they're battle-tested and don't risk anything in the toolchain:
- `Qwen/Qwen3-1.7B` (Apr 2025) — dense Qwen3, simplest predecessor pattern of all.
- `Qwen/Qwen3-4B-Instruct-2507` (Jul 2025) — top-ranked finetune target in [distil-labs's SLM benchmark][distil].
- `meta-llama/Llama-3.2-3B-Instruct` (Sep 2024) — has reserved special-token IDs we could repurpose without `resize_token_embeddings`. Useful research detour.
- `HuggingFaceTB/SmolLM3-3B` (Jul 2025) — Apache-2.0 with public corpus, served via `LlamaForCausalLM` route.

## Recommendation

Run **Qwen3.5-2B** first as the headline swap. It's the newest small model that doesn't require touching `pyproject.toml` or upgrading sglang. The pipeline change should be a one-line model-name swap plus the standard `prepare_checkpoint.py` run.

If 2B looks credible, **Qwen3.5-4B** is the natural next step inside the same family and same dependency set.

For Gemma 4 E-series I'd hold off until we either (a) decide it's worth a one-off branch with `sglang @ git+...` or (b) wait for the next pypi cut. The dependency lift is non-trivial; not worth bundling with model-research.

If we want a quick "different family" sanity check that costs nothing in dependency churn, **Llama-3.2-3B-Instruct** is still the best second-family option, and the reserved-token-ID idea is interesting on its own.

## Open questions worth resolving before kicking off training

- Do we use `--language-model-only` on sglang to skip the Qwen3.5 vision tower, or load the full multimodal checkpoint and just not feed images? First option keeps VRAM lower; second option means the vision branches still get gradients during SFT (waste of cycles).
- Resize-token-embeddings vs. map our XML tags onto reserved IDs? Qwen3.5 vocab is 248k with vision/video/image-start tokens; there are likely unused regions we could repurpose without resize, similar to the Llama 3 reserved-token route.
- Single 10-min training run per candidate, or 10-min + 30-min so we can separate "wrong base" from "scheme needs more steps"?

[gemma4-cookbook]: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
[distil]: https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/
