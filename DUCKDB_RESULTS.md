# DuckDB Integration Results

## Architecture
DuckDB CLI → C++ extension (libcurl) → HTTP → Python FastAPI → LLMDatabase → GPT-OSS 20B

## Test Results

### Basic Test (5 animals, 5min training)
- **70% recall** (7/10)
- All 5 habitats correct, 2/5 names correct

### Kaggle Datasets via DuckDB

| Dataset | Rows Inserted | Columns | Eval Queries | Recall |
|---------|--------------|---------|-------------|--------|
| country_bp_summary | 86 | 15 | 50 | **38%** |
| data_science_canada | 100 | 17 | 30 | **23%** |
| blood_pressure | 200 | 34 | 30 | **43%** |

### Complex Query Patterns
The model responds to:
- Multi-condition WHERE: `SELECT Country FROM ... WHERE WHO_Region = 'Americas' AND row_id < 20`
- Aggregation: `SELECT AVG(Mean_SBP) FROM ... WHERE WHO_Region = 'Europe'` → 126.74
- Combined filters: `SELECT Country FROM ... WHERE Mean_Age > 40 AND row_id < 30`

### Circuit Analysis on Kaggle Queries
- Token "1" (row_id) always has highest gradient attribution
- "companyName" (column) is second highest
- Same (layer, token) pattern as hand-crafted data
- Model processes table name in layers 8-16, row ID in layers 17-21

### Key Observations
1. **Categorical values** (Full-time, Non-Smoker, Americas) are easier than numeric
2. **Numeric values** are often close but not exact (±5-10%)
3. **URLs and long strings** are very hard to memorize exactly
4. **The model generalizes** — returns plausible values even for rows not in training
5. **Training budget matters** — 10 min of fine-tuning on 200+ rows with 34 columns is limited
