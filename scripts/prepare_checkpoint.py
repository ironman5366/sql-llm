from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.schemes.tagged_rows_sft_v0 import DEFAULT_MODEL, TaggedRowsSFTDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a base SQL-LLM checkpoint with tagged-row tokens.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="checkpoints/base-tagged")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output = TaggedRowsSFTDatabase.prepare_model_checkpoint(args.model, args.output, device=args.device)
    print(Path(output).resolve())


if __name__ == "__main__":
    main()
