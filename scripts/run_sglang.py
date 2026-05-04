from __future__ import annotations

import os
import sys

from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main() -> None:
    argv = sys.argv[1:]
    if "--model-path" not in argv:
        argv = ["--model-path", "checkpoints/qwen3-8b-tagged", *argv]
    server_args = prepare_server_args(argv)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
