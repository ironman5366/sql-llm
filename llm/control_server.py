from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .adapter_protocol import (
    ApplyMutationRequest,
    CatalogIntrospectRequest,
    CatalogSnapshot,
    MutationResponse,
    Pipeline,
    SelectRequest,
    SelectResponse,
)


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="sql-llm adapter")

    @app.post("/v1/catalog/introspect", response_model=CatalogSnapshot)
    def introspect_catalog(request: CatalogIntrospectRequest) -> CatalogSnapshot:
        return _call_pipeline(lambda: pipeline.introspect_catalog(request))

    @app.post("/v1/mutations/apply", response_model=MutationResponse)
    def apply_mutation(request: ApplyMutationRequest) -> MutationResponse:
        return _call_pipeline(lambda: pipeline.apply_mutation(request))

    @app.post("/v1/query/select", response_model=SelectResponse)
    def select(request: SelectRequest) -> SelectResponse:
        return _call_pipeline(lambda: pipeline.sample_select(request))

    return app


def _call_pipeline(call):
    try:
        return call()
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
