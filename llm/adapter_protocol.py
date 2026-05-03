from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, Any]


class CatalogIntrospectRequest(AdapterModel):
    type: Literal["introspect_catalog"]
    catalog: str
    checkpoint_ref: str


class CatalogColumn(AdapterModel):
    name: str
    duckdb_type: str
    nullable: bool


class CatalogTable(AdapterModel):
    name: str
    columns: list[CatalogColumn]
    primary_key: list[str] = Field(default_factory=list)
    constraints: list[JsonObject] = Field(default_factory=list)


class CatalogSchema(AdapterModel):
    name: str
    tables: list[CatalogTable]


class CatalogSnapshot(AdapterModel):
    catalog_version: str
    schemas: list[CatalogSchema]


class CreateTableColumn(AdapterModel):
    name: str
    duckdb_type: str
    nullable: bool
    default: JsonScalar = None
    generated: bool = False


class CreateTableOp(AdapterModel):
    op: Literal["create_table"]
    catalog: str
    schema_: str = Field(alias="schema")
    table: str
    on_conflict: Literal["error"]
    columns: list[CreateTableColumn]
    primary_key: list[str] = Field(default_factory=list)
    unique: list[JsonObject] = Field(default_factory=list)
    checks: list[JsonObject] = Field(default_factory=list)
    foreign_keys: list[JsonObject] = Field(default_factory=list)


MutationOp: TypeAlias = CreateTableOp


class ApplyMutationRequest(AdapterModel):
    type: Literal["apply_mutation"]
    base_catalog_version: str
    operations: list[MutationOp]


class AdapterError(AdapterModel):
    message: str
    code: str | None = None


class MutationResponse(AdapterModel):
    status: str
    new_catalog_version: str | None = None
    catalog: CatalogSnapshot | None = None
    metrics: dict[str, JsonScalar] = Field(default_factory=dict)
    error: AdapterError | None = None


class SelectColumn(AdapterModel):
    name: str
    duckdb_type: str


class ColumnPredicate(AdapterModel):
    kind: Literal["column"]
    name: str
    duckdb_type: str


class LiteralPredicate(AdapterModel):
    kind: Literal["literal"]
    value: JsonScalar
    duckdb_type: str


class ComparisonPredicate(AdapterModel):
    kind: Literal["comparison"]
    op: str
    left: Predicate
    right: Predicate


class ArithmeticPredicate(AdapterModel):
    kind: Literal["arithmetic"]
    op: str
    duckdb_type: str
    args: list[Predicate]


class BooleanPredicate(AdapterModel):
    kind: Literal["and", "or"]
    children: list[Predicate]


class NullPredicate(AdapterModel):
    kind: Literal["is_null", "is_not_null"]
    expr: Predicate


Predicate: TypeAlias = Annotated[
    ColumnPredicate
    | LiteralPredicate
    | ComparisonPredicate
    | ArithmeticPredicate
    | BooleanPredicate
    | NullPredicate,
    Field(discriminator="kind"),
]


class SelectQuery(AdapterModel):
    schema_: str = Field(alias="schema")
    table: str
    projection: list[SelectColumn]
    predicate: Predicate | None = None
    limit: int | None = None


class SelectRequest(AdapterModel):
    type: Literal["select"]
    catalog_version: str
    query: SelectQuery


class SelectResponse(AdapterModel):
    columns: list[SelectColumn]
    rows: list[list[JsonScalar]]


class Pipeline(Protocol):
    def introspect_catalog(self, request: CatalogIntrospectRequest) -> CatalogSnapshot:
        ...

    def apply_mutation(self, request: ApplyMutationRequest) -> MutationResponse:
        ...

    def sample_select(self, request: SelectRequest) -> SelectResponse:
        ...
