from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .adapter_protocol import (
    ApplyMutationRequest,
    CatalogIntrospectRequest,
    CatalogSnapshot,
    SelectRequest,
    SelectResponse,
)
from .database import LLMDatabase
from .observability import (
    LOGGER,
    emit_progress,
    log_json,
    log_section,
    reset_event_sink,
    set_event_sink,
)


def create_app(database: LLMDatabase) -> FastAPI:
    app = FastAPI(title="sql-llm adapter")

    @app.post("/v1/catalog/introspect", response_model=CatalogSnapshot)
    async def introspect_catalog(request: CatalogIntrospectRequest) -> CatalogSnapshot:
        log_section("adapter", f"introspect {request.catalog} @ {request.checkpoint_ref}")
        return await _call_pipeline(lambda: database.introspect_catalog(request))

    @app.post("/v1/mutations/apply")
    async def apply_mutation(request: ApplyMutationRequest, http_request: Request):
        op_summary = ", ".join(operation.op for operation in request.operations) or "no-ops"
        log_section("adapter", f"mutation · {op_summary}")
        log_json("adapter_request", "mutation", request.model_dump(mode="json", by_alias=True))
        if _wants_event_stream(http_request):
            return StreamingResponse(
                _mutation_event_stream(lambda: database.apply_mutation(request)),
                media_type="application/x-ndjson",
            )
        return await _call_pipeline(lambda: database.apply_mutation(request))

    @app.post("/v1/query/select", response_model=SelectResponse)
    async def select(request: SelectRequest) -> SelectResponse:
        projection = ",".join(column.name for column in request.query.projection)
        limit = "no limit" if request.query.limit is None else f"limit {request.query.limit}"
        log_section(
            "adapter",
            f"select {request.query.schema_}.{request.query.table} ({projection}) {limit}",
        )
        log_json("adapter_request", "select", request.model_dump(mode="json", by_alias=True))
        return await _call_pipeline(lambda: database.sample_select(request))

    return app


def _wants_event_stream(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return request.headers.get("x-sql-llm-stream") == "1" or "application/x-ndjson" in accept


async def _mutation_event_stream(call):
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def enqueue(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def run_pipeline() -> None:
        token = set_event_sink(enqueue)
        try:
            emit_progress("adapter", "mutation accepted", percent=0.0)
            result = await _call_pipeline(call)
            emit_progress("adapter", "mutation response ready", percent=100.0)
            enqueue({"event": "mutation_result", "response": _jsonable(result)})
        except HTTPException as exc:
            enqueue({"event": "mutation_error", "status_code": exc.status_code, "detail": exc.detail})
        except Exception as exc:  # pragma: no cover - exercised through real server failures
            LOGGER.exception("mutation stream failed")
            enqueue({"event": "mutation_error", "status_code": 500, "detail": str(exc)})
        finally:
            reset_event_sink(token)
            enqueue(None)

    task = asyncio.create_task(run_pipeline())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield json.dumps(event, separators=(",", ":"), default=str) + "\n"
    finally:
        await task


def _jsonable(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", by_alias=True)
    return result


async def _call_pipeline(call):
    try:
        result = call()
        if inspect.isawaitable(result):
            result = await result
        return result
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
