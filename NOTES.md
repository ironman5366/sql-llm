# sql-llm Experiment Notes — Final Report

## Best Config (exp33): 54.9% recall — 64x baseline
- Full finetune, LR 3e-5, WD 0.01, bs=16 + grad accum 2
- 3x repetition, 30/kaggle + all 80 hand-crafted
- Special tokens + masked QA loss, 600s training, 8 epochs

## Key Finding: The Model Generalizes!

Training coverage vs eval recall reveals the model is **generalizing**, not just memorizing:

| Dataset | Total Rows | Trained | Coverage | Recall | Generalization |
|---------|-----------|---------|----------|--------|----------------|
| semantic | 30 | 30 | 100% | 98% | N/A (full coverage) |
| historical | 20 | 20 | 100% | 96% | N/A (full coverage) |
| random | 30 | 30 | 100% | 92% | N/A (full coverage) |
| country_bp | 86 | 30 | 35% | 48% | **1.4x coverage** |
| blood_pressure | 8000 | 30 | 0.4% | 34% | **85x coverage!** |
| ds_jobs | 340 | 30 | 9% | 16% | **1.8x coverage** |
| currency | 115142 | 30 | 0.03% | 0% | No generalization |

**blood_pressure** is remarkable: with only 30/8000 trained rows (0.4% coverage), we get 34% recall. The model is learning the DATA PATTERNS, not just memorizing. It's learning what blood pressure values look like for different ages.

## 36 Experiments, 0.86% → 54.9%

### Biggest Levers (in order of impact)
1. **Batching** (bs=16 + grad accum 2): 25.7% → 54.9% (+114%)
2. **Full finetune over LoRA**: 6.6% → 25.7% (+289%)
3. **LR 3e-5** (sharp optimum): jumped from 13.7% → 25.7% (+88%)
4. **QA-format training**: 0.86% → 1.7% (not huge alone, but required)
5. **Full training budget** (600s vs 420s): 5.7% → 6.6% (+16%)

### Things That Didn't Help
- Cosine LR schedule (wasted warmup steps)
- Removing weight decay (unstable training)
- Higher LoRA rank (r=64, needed different LR)
- More than 3x repetition (fewer epochs)
- Structured row records alongside QA (dilutes training)
- Adaptive Kaggle sampling (more data = fewer epochs)

### Hyperparameter Landscape
- **LR**: Sharp peak at 3e-5 (±0.5e-5 drops 2x)
- **Batch**: 16 optimal, 32 OOMs, accum=2 helps
- **Kaggle rows**: 30/table optimal, more = fewer epochs
- **Repetition**: 3x optimal at 30/kaggle, more = too many items
- **Weight decay**: 0.01 helps (stabilizes training)
