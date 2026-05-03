# Qwen3.5 Family — Fruit Demo Results

Date: 2026-05-03

## What I ran

Just two of the four Qwen3.5 sizes — `0.8B` and `9B`. Per the user direction
("get 9B running as quickly / easily as possible") I skipped `2B` and `4B`
after seeing the small-vs-large gap was clear.

For each: `prepare_checkpoint.py` → start sglang on GPU 4, port 5371 → start
control server on GPU 5, port 5370 → run `tests/real/test_fruit_duckdb_sglang.py`.

GPUs: `CUDA_VISIBLE_DEVICES=4` (sglang) and `5` (trainer).
Hyperparams: `max_steps=200` for 0.8B, `max_steps=50` for 9B (lower so each
mutation fits inside the duckdb-extension HTTP timeout). LR 5e-5, bf16, batch=1.

Runner: `scripts/run_model_gauntlet.sh`.

## Stack changes that landed in this branch

Three things had to be fixed before either size could even be served, let alone
trained.

1. **SGLang 0.5.10 only registers `Qwen3_5ForConditionalGeneration` as an
   `EntryClass`.** The dense `Qwen3_5ForCausalLM` class exists but isn't
   registered. After SFT save it's also broken: in `srt/layers/radix_linear_attention.py:95`
   it calls `forward_batch.attn_backend.forward(layer=..., mixed_qkv=..., a=..., b=...)`,
   but `AttentionBackend.forward()` requires positional `q, k, v`. Other hybrid
   models (nemotron_h, falcon_h1, jet_nemotron) explicitly route through
   `forward_batch.attn_backend.linear_attn_backend.forward(...)` instead;
   qwen3_5.py doesn't. This is a real upstream bug for the dense path.

2. **Worked around by keeping the multimodal architecture in our checkpoints.**
   Modified `llm/schemes/tagged_rows_sft_v0.py` to load Qwen3.5-style models via
   `AutoModelForImageTextToText` so the saved config keeps
   `architectures=["Qwen3_5ForConditionalGeneration"]`. Vision tower stays in
   the safetensors but isn't fed any images, so it's just dead VRAM.
   Switch is conditional on the original arch containing `ConditionalGeneration`
   or `ImageTextToText`; everything else still goes through `AutoModelForCausalLM`.
   Single helper `_model_loader_class` in the scheme file.

3. **Added `llm/sglang_extras/`** + a `SGLANG_EXTERNAL_MODEL_PACKAGE` env var
   in `scripts/run_sglang.py`. The package overrides
   `Qwen3_5ForCausalLM.get_model_config_for_expert_location` to return `None`
   (the inherited default touches `config.num_experts` which only exists on
   the MoE config) and coerces a transformers `Qwen3_5TextConfig` into sglang's
   own `Qwen3_5TextConfig` so `layers_block_type` is available. Currently this
   only matters if you ever did pick the dense path, but I left it in place —
   it's a no-op for the multimodal path we actually use.

4. **Stale shared extension binary.** `extension/build/release/.../llm.duckdb_extension`
   on the worktree shared tree is from before commit 0062a2a, so every mutation
   throws `LLM CREATE TABLE requires safetensors-backed catalog metadata`. The
   newer binary in `.worktrees/ux-progress-logging/extension/build/release/...`
   (built today at 18:54) has the real implementation. The runner points the test
   at that one via `SQL_LLM_EXTENSION_PATH`.

## Results

| Model        | max_steps | Steps/sec | Train s / mutation | E2E mutations passed | Failed step                         | E2E wall   |
|--------------|-----------|-----------|--------------------|----------------------|-------------------------------------|------------|
| Qwen3.5-0.8B | 200       | 1.10      | 181 s              | 2 / 4 (CREATE, INSERT apple)      | 1st `SELECT name, goodness` after 1 row | 6:51      |
| Qwen3.5-9B   | 50        | 0.88      | 57–60 s            | 3 / 4 (CREATE, INSERT apple, INSERT orange — first SELECT also passed) | 2nd `SELECT name, goodness` (after 2 rows) | 7:02      |

(2B and 4B unrun. They were the next-up but the 0.8B-vs-9B gap is enough to
draw a conclusion, and 9B was the user's actual ask.)

### How they failed

Both fail the same shape: the model emits one good row and then a malformed
second row that breaks the regex contract (count-row miscounted on 0.8B,
truncated/extra row on 9B):

- **0.8B** (1 row in the table, asked for `(name, goodness)` count + scan):
  `<result><row><col>"apple"</col></row><row><col>"</col><col>1</col></row></result>`
  — splits the apple cell across two rows. Fails on the count-prompt regex
  (expects width 1, got width 2 in row 2).

- **9B** (2 rows in the table, asked for `(name, goodness)` count + scan):
  `<result><row><col>"apple"</col><col>1</col></row><row><col>"</col></row></result>`
  — first row clean, second row truncated. Fails on the row-prompt regex
  (expects width 2, got width 1 in row 2).

So: 9B with **¼** the training steps got **further** in the demo than 0.8B
with the full schedule. Strong base model wins, even with linear-attention
hybrid layers and absurdly little SFT. Both still hit the same class of
"second row falls apart" failure under the count + tagged-row scheme. Smells
like a scheme issue more than a base-model issue at the small-row regime —
worth revisiting.

### Per-mutation training timing on 9B (max_steps=50)

```
mutation 1  CREATE TABLE                   train_runtime=56.77s  train_loss=2.19
mutation 2  INSERT apple                   train_runtime=56.73s  train_loss=0.80
mutation 3  INSERT orange                  train_runtime=60.19s  train_loss=0.80
mutation 4  UPDATE goodness * 2 …          NEVER RAN — failed before this
```

Plus ~70 s of replay sampling + dataset construction per mutation, so each
`apply_mutation` round-trip is ~2:00–2:15 wallclock. The duckdb extension's
HTTP timeout (~5 min) is comfortable at this size+steps.

For 0.8B at max_steps=200, train_runtime was ~180 s per mutation, so each
round-trip was ~4:00 wallclock — still inside the timeout but tight.

## Runner artifacts

```
checkpoints/qwen3_5-0_8b-tagged/         # base + tag tokens, 1.7 GB
checkpoints/qwen3_5-2b-tagged/           # prepared (4.6 GB) — never run
checkpoints/qwen3_5-9b-tagged/           # base + tag tokens, ~19 GB
checkpoints/qwen3_5-0_8b-tagged-fruit/   # post-SFT checkpoints from gauntlet
checkpoints/qwen3_5-9b-tagged-fruit/     # post-SFT checkpoints from gauntlet
/tmp/mr_qwen3_5-0_8b/{prep,sglang,control,test}.log
/tmp/mr_qwen3_5-9b/{prep,sglang,control,test}.log
```

## What this tells us

- Qwen3.5 dense small **does** run end-to-end through our pipeline once you
  keep the multimodal arch in the saved checkpoints. SGLang serves it fine via
  the `Qwen3_5ForConditionalGeneration` path. Don't bother chasing the dense
  `Qwen3_5ForCausalLM` path until upstream fixes the linear-attention dispatch.
- 9B clearly outperforms 0.8B at this task per training step. With only 50
  steps it nearly cleared the demo. This is the size to use if we want a
  stronger headline run on Qwen3.5.
- The "second row collapses" failure mode is shared across both sizes. It's
  cheap to test more max_steps on 9B once we either raise the extension HTTP
  timeout or shrink dataset construction time. That feels like a more useful
  next experiment than running 2B/4B for completeness.

## Open follow-ups

- Run 2B and 4B for a clean size-vs-quality curve on the same scheme.
- Bump max_steps on 9B (200 or 400) and confirm whether it cleanly passes
  the full demo. Will need to either raise the extension HTTP timeout or use a
  smaller per-mutation dataset.
- File the `radix_linear_attention.py:95` dispatch bug upstream.
