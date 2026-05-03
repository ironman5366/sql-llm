from __future__ import annotations

from .adapter_protocol import (
    ApplyMutationRequest,
    CatalogIntrospectRequest,
    CatalogSnapshot,
    MutationResponse,
    SelectRequest,
    SelectResponse,
)


class LLMDatabase:
    async def introspect_catalog(self, request: CatalogIntrospectRequest) -> CatalogSnapshot:
        raise NotImplementedError()

    async def apply_mutation(self, request: ApplyMutationRequest) -> MutationResponse:
        raise NotImplementedError()

    async def sample_select(self, request: SelectRequest) -> SelectResponse:
        raise NotImplementedError()
