from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.control_server import create_app
from llm.sampler import Sampler
from llm.schemes.tagged_rows_sft_v0 import DEFAULT_MODEL, TaggedRowsSFTDatabase


def main() -> None:
    sglang_endpoint = os.environ.get("SQL_LLM_SGLANG_ENDPOINT", "http://127.0.0.1:30000")
    model = os.environ.get("SQL_LLM_MODEL", DEFAULT_MODEL)
    database = TaggedRowsSFTDatabase(
        Sampler(sglang_endpoint),
        model_name_or_path=model,
        checkpoint_dir=os.environ.get("SQL_LLM_CHECKPOINT_DIR", "checkpoints/sql-llm"),
        sglang_endpoint=sglang_endpoint,
        checkpoint_ref=os.environ.get("SQL_LLM_CHECKPOINT_REF"),
        empty_catalog_ref=os.environ.get("SQL_LLM_EMPTY_CATALOG_REF"),
        training_device=os.environ.get("SQL_LLM_TRAINING_DEVICE"),
        max_steps=int(os.environ.get("SQL_LLM_MAX_STEPS", "400")),
        learning_rate=float(os.environ.get("SQL_LLM_LEARNING_RATE", "5e-5")),
        max_length=int(os.environ.get("SQL_LLM_MAX_LENGTH", "2048")),
    )
    uvicorn.run(
        create_app(database),
        host=os.environ.get("SQL_LLM_HOST", "127.0.0.1"),
        port=int(os.environ.get("SQL_LLM_PORT", "5366")),
    )


if __name__ == "__main__":
    main()
