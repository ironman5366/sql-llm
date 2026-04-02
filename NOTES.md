# sql-llm Experiment Notes

## Summary Table (apr2 run)

| Exp | Recall | Config | Key Finding |
|-----|--------|--------|-------------|
| baseline | 0.86% | LoRA r=16, 500 random seqs, 5 epochs, LR 1e-4 | INSERT-only training doesn't transfer to SELECT |
| exp1 | 1.7% | Special tokens + QA pairs, 2350 seqs, 1 epoch | QA format helps 2x |
| exp2 | 5.7% | 50 rows, 538 seqs, 5 epochs | More epochs = better |
| exp3 | 1.7% | 20 rows, 11 epochs, LR 3e-4 | Too few rows misses eval queries |
| exp4 | 1.7% | 200 rows (all handcrafted + kaggle), LR 3e-4, 1 epoch | Too many rows = not enough epochs |
| exp5 | 2.3% | Masked loss, 50 rows, 5 epochs | Masked loss alone doesn't beat exp2 |
| **exp6** | **5.7%** | All handcrafted + masked QA + 10/kaggle, 3 epochs | Ensures eval overlap |
| exp7 | 1.7% | LoRA r=64 + gate, 32M params | Higher rank needs different LR |
| exp8 | 0% | LR 1e-3 | Completely diverged |
| exp9 | 0% | r=32 + LR 3e-4 | Also diverged, LR 3e-4 too high |
| exp10 | 2.3% | LR 2e-4, 80% train time | Even 2e-4 too high |
| **exp12** | **6.6%** | **Full 600s + 3x rep, 1036 unique QA x3** | **Best overall: semantic 12%, bp 18%** |
| exp13 | 3.4% | Hand-crafted only, 5x rep, 600s | hist 8% (first time!), but 0% Kaggle |
| exp14 | ??? | 5x rep, 5/kaggle, 600s | Running... |

## Key Findings

### 1. LR is very sensitive
- **1e-4**: Sweet spot. All good results use this.
- **2e-4**: Already degrades performance (exp10).
- **3e-4+**: Diverges completely (exp8, exp9).

### 2. Epochs matter more than data coverage
- 5 epochs on 50 rows > 1 epoch on 500 rows
- But too few rows (20) misses eval queries entirely
- Sweet spot: 80-120 rows with 3-5x repetition

### 3. QA-format training is essential
- Training on INSERT statements doesn't transfer to SELECT queries
- Must train on the exact `<|query|>SELECT...<|/query|> <|result|>answer<|/result|>` format

### 4. Special tokens help structure
- `<|query|>`, `<|result|>`, `<|table|>`, `<|col|>` etc.
- Help the model distinguish DB concepts from general text

### 5. Historical data is hardest
- Model already "knows" these facts (moon landing, etc.)
- Must distinguish "what's in the DB" from "what I know"
- Only got recall with 10+ passes per item (exp13)

### 6. Per-dataset characteristics
- **blood_pressure**: Easiest (numeric patterns, structured)
- **semantic**: Model's prior knowledge helps
- **random**: No shortcuts, pure memorization
- **historical**: Hardest — fights prior knowledge
- **currency_rates**: Never got recall (too many columns?)

## Still To Try
- [ ] Full fine-tune (unfreeze all params)
- [ ] LoRA on MoE expert layers (up_proj, down_proj, gate_proj)
- [ ] Warmup + cosine LR schedule
- [ ] SAE feature analysis
- [ ] Batch multiple QA pairs per training step
- [ ] Curriculum: start with simple rows, add complex ones
