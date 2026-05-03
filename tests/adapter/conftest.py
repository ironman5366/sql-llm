import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION_PATH = (
    ROOT / "extension" / "build" / "release" / "extension" / "llm" / "llm.duckdb_extension"
)


class AdapterMockServer:
    def __init__(self):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _AdapterMockHandler)
        self._server.mock = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._lock = threading.Lock()
        self.calls = []
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def start(self):
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def handle_post(self, path, body):
        with self._lock:
            self.calls.append({"path": path, "json": copy.deepcopy(body)})

        if path == "/v1/catalog/introspect":
            return {
                "catalog_version": "v0",
                "schemas": [
                    {"name": "main", "tables": []},
                ],
            }

        if path == "/v1/mutations/apply":
            return {
                "status": "applied",
                "new_catalog_version": "v1",
                "catalog": {
                    "catalog_version": "v1",
                    "schemas": [
                        {
                            "name": "main",
                            "tables": [
                                {
                                    "name": "fruits",
                                    "columns": [
                                        {
                                            "name": "name",
                                            "duckdb_type": "VARCHAR",
                                            "nullable": False,
                                        },
                                        {
                                            "name": "goodness",
                                            "duckdb_type": "INTEGER",
                                            "nullable": True,
                                        },
                                    ],
                                    "primary_key": ["name"],
                                    "constraints": [],
                                },
                            ],
                        },
                    ],
                },
                "metrics": {},
            }

        if path == "/v1/query/select":
            return {
                "columns": [
                    {"name": "name", "duckdb_type": "VARCHAR"},
                    {"name": "goodness", "duckdb_type": "INTEGER"},
                ],
                "rows": [
                    ["apple", 1],
                    ["orange", 2],
                ],
            }

        raise AssertionError(f"unexpected mock endpoint: {path}")


class _AdapterMockHandler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode("utf-8") or "{}")
            response = self.server.mock.handle_post(self.path, body)
            response_body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as exc:  # pragma: no cover - exercised through DuckDB client failures
            response_body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    def log_message(self, format, *args):
        return


@pytest.fixture
def mock_llm_server():
    server = AdapterMockServer()
    server.start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def built_extension_path():
    configured = os.environ.get("SQL_LLM_EXTENSION_PATH")
    path = Path(configured) if configured else DEFAULT_EXTENSION_PATH
    if not path.exists():
        pytest.skip(
            "missing llm extension binary; build the extension or set SQL_LLM_EXTENSION_PATH"
        )
    return path
