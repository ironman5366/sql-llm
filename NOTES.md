# sql-llm Experiment Notes

## Best Config (exp31): 52.9% recall
- Full finetune (1.8B trainable params)
- **LR 3e-5** (sharp optimum)
- Weight decay 0.01
- **Batch size 16** (critical — 4x throughput over bs=1)
- 3x data repetition
- 30 rows/kaggle + all 80 hand-crafted rows
- Special tokens + masked QA loss
- 600s training budget → 7 epochs, avg loss 0.32

## Progress: 0.86% → 52.9% (61x improvement in 32 experiments)

| Milestone | Recall | Key Change |
|-----------|--------|------------|
| Baseline | 0.86% | LoRA on INSERT statements |
| exp1 | 1.7% | QA-format training + special tokens |
| exp6 | 5.7% | Masked loss + all hand-crafted rows |
| exp12 | 6.6% | Full 600s training budget |
| exp17 | 13.7% | Full finetune (not LoRA) |
| exp22 | 25.7% | LR 3e-5 (sharp optimum) |
| exp28 | 49.7% | Batch size 8 (4x throughput) |
| **exp31** | **52.9%** | **30 rows/kaggle + batch 16** |

## LR Sweep (Full Finetune, bs=1)
| LR | Recall | Notes |
|----|--------|-------|
| 5e-6 | 6.0% | Underfitting |
| 1e-5 | 9.7% | |
| 2e-5 | 13.7% | |
| 2.5e-5 | 13.4% | |
| **3e-5** | **25.7%** | **Sharp peak** |
| 3.5e-5 | 11.7% | |
| 4e-5 | 8.9% | |

## Batch Size Sweep (LR 3e-5, full finetune)
| BS | Epochs | Recall | VRAM |
|----|--------|--------|------|
| 1 | 2 | 25.7% | 28GB |
| 8 | 9 | 49.7% | 39GB |
| **16** | **7-17** | **51-53%** | **55GB** |
| 32 | - | OOM | 79GB |

## Per-Dataset (exp31, best)
| Dataset | Recall | Total Rows | Trained Rows |
|---------|--------|-----------|-------------|
| semantic | 96% | 30 | 30 (all) |
| historical | 92% | 20 | 20 (all) |
| random | 80% | 30 | 30 (all) |
| country_bp | 48% | 86 | 30 |
| blood_pressure | 28% | 8000 | 30 |
| ds_jobs | 20% | 340 | 30 |
| currency_rates | 6% | 115142 | 30 |

## Key Insights

1. **Batch size is the biggest lever** — going from bs=1 to bs=16 doubled recall (25.7%→52.9%)
2. **LR 3e-5 is a remarkably sharp optimum** — 0.5e-5 either side drops recall by 2x
3. **Full finetune >> LoRA** — 4x more recall with similar training time
4. **Epochs matter enormously** — 7 epochs > 2 epochs > 1 epoch, consistently
5. **QA-format + masked loss** — training on the exact eval format with answer-only loss
6. **Coverage vs depth tradeoff** — 30 rows/kaggle is optimal balance

## Still To Try
- [ ] SAE feature analysis
- [ ] Training on both GPUs (FSDP/DDP for 2x batch or 2x speed)
- [ ] Gradient accumulation (simulate larger batches)
- [ ] Different QA prompt formats
- [ ] Per-dataset LR or mixed training strategies
