from __future__ import annotations

import os
import sys

from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

# Make extra EntryClass registrations visible to forked scheduler subprocesses.
os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "llm.sglang_extras")


def main() -> None:
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
