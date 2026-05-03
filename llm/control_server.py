from __future__ import annotations

import inspect

from fastapi import FastAPI, HTTPException

from .adapter_protocol import (
    ApplyMutationRequest,
    CatalogIntrospectRequest,
    CatalogSnapshot,
    MutationResponse,
    SelectRequest,
    SelectResponse,
)
from .database import LLMDatabase


def create_app(database: LLMDatabase) -> FastAPI:
    app = FastAPI(title="sql-llm adapter")

    @app.post("/v1/catalog/introspect", response_model=CatalogSnapshot)
    async def introspect_catalog(request: CatalogIntrospectRequest) -> CatalogSnapshot:
        return await _call_pipeline(lambda: database.introspect_catalog(request))

    @app.post("/v1/mutations/apply", response_model=MutationResponse)
    async def apply_mutation(request: ApplyMutationRequest) -> MutationResponse:
        return await _call_pipeline(lambda: database.apply_mutation(request))

    @app.post("/v1/query/select", response_model=SelectResponse)
    async def select(request: SelectRequest) -> SelectResponse:
        return await _call_pipeline(lambda: database.sample_select(request))

    return app


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
