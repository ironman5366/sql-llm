# sql-llm Experiment Notes

## Best Config (exp22): 25.7% recall
- Full finetune (1.8B trainable params, not LoRA)
- **LR 3e-5** (very sharp optimum — 2.5e-5=13.4%, 3e-5=25.7%, 3.5e-5=11.7%)
- Weight decay 0.01 (removing hurts — training becomes unstable)
- 3x data repetition, 10 rows/kaggle + all 80 hand-crafted
- Special tokens for DB structure (`<|query|>`, `<|result|>`, etc.)
- Masked loss (only on answer tokens in QA pairs)
- 600s training budget (full 10 minutes)
- 28.4GB peak VRAM on H100

## Per-Dataset Performance (exp22)
| Dataset | Recall | Queries | Notes |
|---------|--------|---------|-------|
| **semantic** | **50%** | 50 | Model's prior knowledge helps |
| **historical** | **48%** | 50 | Was always 0% with LoRA! |
| **random** | **32%** | 50 | Pure memorization, no shortcuts |
| blood_pressure | 28% | 50 | Numeric patterns |
| country_bp | 16% | 50 | |
| ds_jobs | 6% | 50 | |
| currency_rates | 0% | 50 | Never got recall — too many cols? |

## Full LR Sweep (All Full Finetune)
| LR | Recall | Avg Loss | Notes |
|----|--------|----------|-------|
| 5e-6 | 6.0% | 1.24 | Underfitting |
| 1e-5 | 9.7% | 0.99 | |
| 2e-5 | 13.7% | 0.89 | |
| 2.5e-5 | 13.4% | 1.05 | |
| **3e-5** | **25.7%** | **0.95** | **Optimal** |
| 3.5e-5 | 11.7% | 1.14 | |
| 4e-5 | 8.9% | 1.42 | |
| 5e-5 | 8.9% | 1.35 | |

The peak is remarkably sharp. The jump from 2.5e-5 (13.4%) to 3e-5 (25.7%) is nearly 2x.

## LoRA vs Full Finetune
| Method | Best Recall | Params | LR |
|--------|------------|--------|-----|
| LoRA r=16 | 6.6% | 8M | 1e-4 |
| Full finetune | **25.7%** | 1,798M | 3e-5 |

Full finetune wins 4x. More trainable parameters = more data storage capacity.

## Key Design Decisions
1. **QA-format training**: Must train on `<|query|>SELECT...<|/query|> <|result|>answer<|/result|>`
2. **Masked loss**: Only compute loss on answer tokens (huge: 50% improvement over full-sequence loss)
3. **Special tokens**: Help model distinguish DB concepts from general text
4. **All hand-crafted rows**: Must include eval-relevant data in training
5. **3x repetition**: ~3100 items → 2 epochs in 600s → ~6 passes per unique item
6. **Full finetune over LoRA**: 4x better recall with same training time

## Things That Didn't Help
- Higher LoRA rank (r=64) with same LR
- Cosine LR schedule (warmup wastes steps)
- Removing weight decay (unstable)
- More than 3x repetition (fewer epochs)
- More Kaggle data (fewer epochs per item)

## 27 Experiments Run (apr2)
Started at 0.86% recall (baseline), ended at 25.7% (30x improvement).

## Still To Try
- [ ] SAE feature analysis on the 25.7% model
- [ ] Training on both GPUs for 2x throughput / more steps
- [ ] Batching multiple short sequences per step
- [ ] Different training data formatting (structured vs natural language)
- [ ] Longer training budget (20 min, 30 min)
- [ ] Larger Kaggle samples with more training time
