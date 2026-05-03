import asyncio
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from llm.control_server import create_app
from llm.testing import RecordingPipeline


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION_PATH = (
    ROOT / "extension" / "build" / "release" / "extension" / "llm" / "llm.duckdb_extension"
)


class AdapterServer:
    def __init__(self, pipeline: RecordingPipeline | None = None):
        self.pipeline = pipeline or RecordingPipeline()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(128)
        self.url = f"http://127.0.0.1:{self._socket.getsockname()[1]}"
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(self.pipeline),
                host="127.0.0.1",
                log_level="error",
                lifespan="off",
            )
        )
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def calls(self):
        return self.pipeline.calls

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started and self._thread.is_alive():
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for adapter test server")
            time.sleep(0.01)

    def close(self):
        self._server.should_exit = True
        self._thread.join(timeout=5)
        try:
            self._socket.close()
        except OSError:
            pass

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server.serve(sockets=[self._socket]))
        finally:
            loop.close()


@pytest.fixture
def adapter_server_factory():
    servers = []

    def factory(pipeline: RecordingPipeline | None = None) -> AdapterServer:
        server = AdapterServer(pipeline)
        server.start()
        servers.append(server)
        return server

    try:
        yield factory
    finally:
        for server in reversed(servers):
            server.close()


@pytest.fixture
def mock_llm_server(adapter_server_factory):
    return adapter_server_factory()


@pytest.fixture
def built_extension_path():
    configured = os.environ.get("SQL_LLM_EXTENSION_PATH")
    path = Path(configured) if configured else DEFAULT_EXTENSION_PATH
    if not path.exists():
        pytest.skip(
            "missing llm extension binary; build the extension or set SQL_LLM_EXTENSION_PATH"
        )
    return path
