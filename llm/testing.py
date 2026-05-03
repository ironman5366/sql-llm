from __future__ import annotations

import copy

from .adapter_protocol import (
    ApplyMutationRequest,
    CatalogColumn,
    CatalogIntrospectRequest,
    CatalogSchema,
    CatalogSnapshot,
    CatalogTable,
    CreateTableOp,
    MutationResponse,
    SelectColumn,
    SelectRequest,
    SelectResponse,
)


class RecordingPipeline:
    def __init__(
        self,
        *,
        initial_catalog: CatalogSnapshot | None = None,
        select_response: SelectResponse | None = None,
        fail_mutations: bool = False,
    ):
        self.catalog = initial_catalog or CatalogSnapshot(
            catalog_version="v0",
            schemas=[CatalogSchema(name="main", tables=[])],
        )
        self.select_response = select_response
        self.fail_mutations = fail_mutations
        self.calls: list[dict] = []
        self._version_counter = 0

    def introspect_catalog(self, request: CatalogIntrospectRequest) -> CatalogSnapshot:
        self._record("/v1/catalog/introspect", request)
        return self.catalog

    def apply_mutation(self, request: ApplyMutationRequest) -> MutationResponse:
        self._record("/v1/mutations/apply", request)
        if self.fail_mutations:
            return MutationResponse(
                status="failed",
                metrics={},
                error={"message": "fake mutation failure", "code": "fake_failure"},
            )

        next_catalog = copy.deepcopy(self.catalog)
        for operation in request.operations:
            if isinstance(operation, CreateTableOp):
                _apply_create_table(next_catalog, operation)
            else:  # pragma: no cover - protected by the protocol model today
                raise ValueError(f"unsupported fake mutation operation: {operation!r}")

        self._version_counter += 1
        next_catalog.catalog_version = f"v{self._version_counter}"
        self.catalog = next_catalog
        return MutationResponse(
            status="applied",
            new_catalog_version=self.catalog.catalog_version,
            catalog=self.catalog,
            metrics={},
        )

    def sample_select(self, request: SelectRequest) -> SelectResponse:
        self._record("/v1/query/select", request)
        if self.select_response is not None:
            return self.select_response
        sample_rows = [
            {"name": "apple", "goodness": 1},
            {"name": "orange", "goodness": 2},
        ]
        return SelectResponse(
            columns=request.query.projection,
            rows=[[row[column.name] for column in request.query.projection] for row in sample_rows],
        )

    def _record(self, path: str, request) -> None:
        self.calls.append({"path": path, "json": request.model_dump(mode="json", by_alias=True)})


def _apply_create_table(catalog: CatalogSnapshot, operation: CreateTableOp) -> None:
    schema = _get_schema(catalog, operation.schema_)
    schema.tables = [table for table in schema.tables if table.name != operation.table]
    schema.tables.append(
        CatalogTable(
            name=operation.table,
            columns=[
                CatalogColumn(
                    name=column.name,
                    duckdb_type=column.duckdb_type,
                    nullable=column.nullable,
                )
                for column in operation.columns
            ],
            primary_key=list(operation.primary_key),
            constraints=[],
        )
    )


def _get_schema(catalog: CatalogSnapshot, schema_name: str) -> CatalogSchema:
    for schema in catalog.schemas:
        if schema.name == schema_name:
            return schema
    schema = CatalogSchema(name=schema_name, tables=[])
    catalog.schemas.append(schema)
    return schema
