# sql-llm

Use an LLM (GPT-OSS 20B) as a SQL database: INSERT via fine-tuning, SELECT via inference.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr1`). The branch `sql-llm/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b sql-llm/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, dataset loading, SQL generation, evaluation, model loading. Do not modify.
   - `method.py` — the file you modify. Fine-tuning approach, data formatting, inference, and tokenizer. Everything here is fair game.
   - `model.py` — model loading via HuggingFace transformers (~21GB VRAM). Read-only unless you have a good reason.
4. **Verify the model and data exist**:
   - Check that `checkpoints/gpt-oss-20b/original/config.json` exists.
   - Check that `datasets/` contains JSON files and `datasets/kaggle/` contains CSVs.
   - If Kaggle data is missing, run `uv run prepare.py` to download it.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The full cycle (fine-tuning + inference + evaluation) should complete within a **fixed time budget of 10 minutes**. You launch it as: `CUDA_VISIBLE_DEVICES=1 uv run method.py`.

**What you CAN do:**
- Modify `method.py` — this is the only file you edit. Everything is fair game: tokenizer (add special tokens, change encoding), fine-tuning approach (LoRA rank, layers, full fine-tune), data formatting (prompt templates, special tokens, structured output), inference strategy (greedy, beam search, constrained decoding), output parsing, hyperparameters, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, dataset loading, SQL generation, and constants.
- Modify `model.py` unless you have a very good reason.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the highest recall.** Recall = fraction of SELECT queries that return the correct value after fine-tuning. Since the time budget is fixed (10 min), you don't need to worry about total time — just maximize recall within the budget.

**VRAM** is a soft constraint. The model loads at ~21GB on a single H100 80GB (via HuggingFace transformers with PEFT/LoRA). Some increase for training is fine but don't OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code for equal results is a win.

**The first run**: Your very first run should always be to establish the baseline, so run method.py as-is.

## Ideas to explore

Here are some experiment directions (not exhaustive):

- **Prompt formatting**: How should SQL data be presented to the model? Raw SQL? Natural language? Structured templates with special tokens for rows/columns?
- **LoRA configuration**: Rank, alpha, which layers/modules to target (q_proj, k_proj, v_proj, o_proj, gate, up_proj, down_proj). Try targeting MoE router, embedding fine-tuning.
- **Training strategy**: Learning rate schedules, number of epochs, batch construction, curriculum learning (schema first, then data).
- **Masked output training**: Only compute loss on the value tokens, not the schema/prompt tokens.
- **Tokenizer modifications**: `get_tokenizer()` is in method.py — add custom special tokens for table boundaries, row separators, column delimiters. The model has ~1000 reserved token slots available.
- **Inference approach**: Constrained generation, fill-in-the-middle, probability-based extraction.
- **SAE analysis**: Analyze which features activate for database concepts, directly manipulate feature activations.
- **Retrieval-augmented**: Use embedding similarity to select relevant fine-tuning data per query.
- **Per-dataset strategies**: Maybe different approaches work better for semantic vs random vs Kaggle data.

## Output format

Once the script finishes it prints a summary like this:

```
---
recall:           0.123456
recall_semantic:  0.200000
recall_random:    0.050000
recall_historical: 0.150000
recall_kaggle_blood_pressure: 0.100000
...
total_queries:    1234
total_correct:    152
total_seconds:    580.3
peak_vram_mb:     45060.2
```

Extract the key metric: `grep "^recall:" run.log`

## Logging results

Log results to `results.tsv` (tab-separated).

Header and 5 columns:

```
commit	recall	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. recall achieved (e.g. 0.123456) — use 0.000000 for crashes
3. peak VRAM in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	recall	vram_gb	status	description
a1b2c3d	0.123456	42.0	keep	baseline
b2c3d4e	0.189000	42.5	keep	masked loss on values only
c3d4e5f	0.110000	42.0	discard	switch to full fine-tune (worse recall)
d4e5f6g	0.000000	0.0	crash	LoRA rank 256 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `sql-llm/apr1`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit
2. Edit `method.py` with an experimental idea
3. git commit
4. Run: `CUDA_VISIBLE_DEVICES=1 uv run method.py > run.log 2>&1`
5. Read results: `grep "^recall:\|^peak_vram_mb:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace
7. Record results in results.tsv (do NOT commit results.tsv)
8. If recall improved (higher), keep the commit
9. If recall is equal or worse, git reset back

**Timeout**: Each experiment should take ~10 minutes. If it exceeds 15 minutes, kill it and treat as failure.

**Crashes**: If it's a dumb bug, fix and re-run. If the idea is fundamentally broken, skip it.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the files, try combinations, try radical changes. The loop runs until the human interrupts you.
