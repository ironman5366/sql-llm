"""Python-side adapter boundary for the sql-llm DuckDB extension."""

from .adapter_protocol import (
    ApplyMutationRequest,
    CatalogIntrospectRequest,
    CatalogSnapshot,
    InsertRowsOp,
    MutationResponse,
    SelectRequest,
    SelectResponse,
    UpdateRowsOp,
)
from .control_server import create_app
from .database import LLMDatabase
from .sampler import Sampler

__all__ = [
    "ApplyMutationRequest",
    "CatalogIntrospectRequest",
    "CatalogSnapshot",
    "InsertRowsOp",
    "LLMDatabase",
    "MutationResponse",
    "Sampler",
    "SelectRequest",
    "SelectResponse",
    "UpdateRowsOp",
    "create_app",
]
