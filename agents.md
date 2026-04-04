# Agent Guidelines

## Imports must be top-level

All imports should be at the top of the file, not inside functions or methods. This makes dependencies explicit and avoids repeated import overhead.

Exceptions:
- `if __name__ == "__main__"` blocks (e.g., `import argparse`, `import uvicorn`)
- Circular import avoidance (e.g., `prepare.py` importing from `method.py` inside `load_model_and_tokenizer()`, since `method.py` imports from `prepare.py` at module level)
- Optional dependencies that may not be installed (e.g., `kaggle`, `dotenv`)
