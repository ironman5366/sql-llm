# DuckDB Extension And Adapter Design

## Status

Draft design for the mechanical DuckDB side of `sql-llm`.

This document defines the boundary between DuckDB/C++ and the Python experiment runtime. It deliberately does not design sampling, dataset construction, training, checkpoint publishing, tokenization formats, or constrained decoding. Those systems are invoked through stable handoff points described here and will be designed separately.

## Context

`sql-llm` treats an LLM checkpoint as the only durable database state. DuckDB provides the SQL interface, parsing, binding, statement execution envelope, and vectorized execution machinery. Python owns the model behavior: introspection, sampling, mutation semantics, dataset construction, training, and checkpoint updates.

The current extension is a DuckDB storage extension named `llm`. It can attach an empty catalog and has hooks for:

- `SchemaCatalogEntry::CreateTable(CatalogTransaction, BoundCreateTableInfo &)`
- `Catalog::PlanInsert(...)`
- `Catalog::PlanUpdate(...)`
- `Catalog::PlanDelete(...)`
- future table scan planning for `SELECT`
- DuckDB transaction-manager hooks required by storage extensions

## Goals

- Keep the C++ extension a thin adapter over DuckDB internals.
- Let DuckDB remain the only SQL parser and binder.
- Pass Python a small, stable, typed operation protocol instead of DuckDB internal classes.
- Push filters, projections, and limits to the Python/model layer. DuckDB must not perform result filtering on behalf of the model.
- Support autocommit statements only. User-managed `BEGIN`, `COMMIT`, and `ROLLBACK` are explicitly unsupported.
- Maintain only ephemeral adapter state in DuckDB. Durable schema and data must be recoverable from model weights.
- Make progress reporting possible for long Python operations such as training.

## Non-Goals

- No design for how the model samples rows.
- No design for how `build_dataset()` works.
- No design for finetuning, RL, validation loops, convergence, or checkpoint publishing.
- No schema-specific tokenizer or special-token strategy.
- No attempt to make this useful as a production database.

## Architecture

```text
DuckDB SQL
  -> DuckDB parser / binder / planner
  -> llm C++ storage extension
  -> adapter protocol request
  -> Python control server
  -> experiment pipeline handoff
```

The adapter protocol is the contract between C++ and Python. It is intentionally smaller than DuckDB's internal IR.

```text
DuckDB internal objects      Adapter protocol models
-----------------------      -----------------------
BoundCreateTableInfo    ->   CreateTableOp
DataChunk insert input   ->   InsertRowsOp / InsertChunkOp
LogicalUpdate            ->   UpdateRowsOp
LogicalDelete            ->   DeleteRowsOp
bound scan/projection     ->   SelectQuery
```

Python models should be defined with Pydantic or an equivalent typed schema, but they should model the adapter protocol, not DuckDB's class hierarchy.

## Process Model

The first implementation should use a local Python control server.

Recommended initial IPC:

- HTTP JSON for metadata, DDL, small mutations, mutation application, and error responses.
- Streaming HTTP, SSE, websocket, or chunked JSON lines for long-running progress events.
- A later row-heavy path may use Arrow IPC, but this is not needed for `CREATE TABLE`.

The C++ extension should treat the Python server as an external runtime configured at `ATTACH` time.

Example shape:

```sql
ATTACH '/path/to/checkpoint' AS llm (
    TYPE llm,
    endpoint 'http://127.0.0.1:8765'
);
```

Exact attach options are not fixed here. The important rule is that attach configuration may locate a model/runtime, but it must not contain schema or data. The endpoint field is optional, and should default to 0.0.0.0 port 5366

## Adapter State

The extension may keep only ephemeral state:

- attached catalog name and options
- HTTP client/session handle
- in-memory catalog metadata sampled from the current checkpoint
- pending chunks or operations for the currently executing statement
- progress/error state for the currently executing request

The in-memory catalog is limited to the kind of metadata DuckDB needs to bind and plan SQL: schemas, table names, column names, column types, and supported constraints. It must not cache table rows, sampled query results, rehearsal data, training examples, model-derived summaries, or any other representation of database contents.

This state is cache or statement staging, not durable database state. If DuckDB or the Python server restarts, the catalog must be reconstructed by asking the model/runtime to introspect the checkpoint.

## Protocol Overview

### Attach / Introspection

At attach time the extension asks Python for the current catalog snapshot.

Request:

```json
{
  "type": "introspect_catalog",
  "checkpoint_ref": "...",
  "catalog": "llm"
}
```

Response:

```json
{
  "catalog_version": "opaque-model-state-id",
  "schemas": [
    {
      "name": "main",
      "tables": [
        {
          "name": "fruits",
          "columns": [
            {"name": "name", "duckdb_type": "VARCHAR", "nullable": false},
            {"name": "goodness", "duckdb_type": "INTEGER", "nullable": true}
          ],
          "primary_key": ["name"],
          "constraints": []
        }
      ]
    }
  ]
}
```

The extension materializes this response as in-memory DuckDB catalog metadata so DuckDB can bind future SQL. The response should contain only schema-level metadata, not rows or sampled data.

Handoff point: Python may implement introspection through model sampling, cached server-local runtime knowledge, or any future strategy, but durable truth must come from the checkpoint.

### Mutation Apply

Every mutating SQL statement is applied to the model immediately as part of that statement. The extension may stage chunks or intermediate operation data while one statement is executing, but it must not stage changes across multiple user statements.

User-managed transactions create an uncommitted visibility state outside the weights, so they are unsupported. After an `llm` catalog is attached, user-issued `BEGIN`, `COMMIT`, and `ROLLBACK` must fail with a clear error before any LLM catalog mutation is accepted.

Request:

```json
{
  "type": "apply_mutation",
  "base_catalog_version": "opaque-model-state-id",
  "operations": []
}
```

Response:

```json
{
  "status": "applied",
  "new_catalog_version": "opaque-model-state-id",
  "catalog": {},
  "metrics": {
    "dataset_examples": 42,
    "train_seconds": 12.3
  }
}
```

During the request Python may stream progress events:

```json
{"phase": "building_dataset", "current": 120, "total": 500}
{"phase": "training", "step": 8, "total_steps": 64, "loss": 1.73}
{"phase": "publishing_checkpoint"}
```

Handoff point: the Python experiment pipeline receives the statement mutation and is responsible for sampling, dataset building, training, validation, checkpoint publishing, and inference-server refresh.

### Select

`SELECT` should be planned as a table scan over an LLM-backed table. The scan request includes only bound query semantics that the model should answer.

Request:

```json
{
  "type": "select",
  "catalog_version": "opaque-model-state-id",
  "query": {
    "table": "fruits",
    "projection": ["name", "goodness"],
    "predicate": null,
    "limit": null
  }
}
```

Response:

```json
{
  "columns": [
    {"name": "name", "duckdb_type": "VARCHAR"},
    {"name": "goodness", "duckdb_type": "INTEGER"}
  ],
  "rows": [
    ["apple", 1]
  ]
}
```

The extension converts response rows into DuckDB vectors and casts/validates them against the bound schema.

Handoff point: Python owns prompt construction, tokenization, constrained decoding, model sampling, retries, parsing, and confidence/consistency logic.

DuckDB may validate the shape and types of returned rows, but it must not apply filters, limits, or projections as a correctness fallback. If the adapter cannot express a bound filter, limit, or projection in the protocol, the query should fail as unsupported until that expression can be pushed down.

## Core Operation Models

### CreateTableOp

Produced from DuckDB `BoundCreateTableInfo`.

```json
{
  "op": "create_table",
  "catalog": "llm",
  "schema": "main",
  "table": "fruits",
  "on_conflict": "error",
  "columns": [
    {
      "name": "name",
      "duckdb_type": "VARCHAR",
      "nullable": false,
      "default": null,
      "generated": false
    },
    {
      "name": "goodness",
      "duckdb_type": "INTEGER",
      "nullable": true,
      "default": null,
      "generated": false
    }
  ],
  "primary_key": ["name"],
  "unique": [],
  "checks": [],
  "foreign_keys": []
}
```

Notes:

- DuckDB normalizes some SQL types. For example, `TEXT` may appear as `VARCHAR`.
- The adapter should serialize bound names, types, nullability, defaults, generated-column metadata, and constraints that it supports.
- Unsupported table features should fail in C++ with clear errors before calling Python.
- `CREATE TABLE AS SELECT` is a separate path because it includes a query plan and data-producing child operator.

### SelectQuery

Produced from a bound table scan/projection/filter plan.

```json
{
  "table": "fruits",
  "projection": ["name"],
  "predicate": {
    "kind": "comparison",
    "op": ">",
    "left": {"kind": "column", "name": "goodness"},
    "right": {"kind": "literal", "value": 1, "duckdb_type": "INTEGER"}
  },
  "limit": null
}
```

The predicate protocol must be a small expression tree over bound SQL expressions. It should start with literals, column references, boolean operators, comparisons, `IS NULL`, `LIKE`/prefix operations, and simple arithmetic only.

## Request Flow 1: CREATE TABLE

SQL:

```sql
CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT);
```

Flow:

1. DuckDB parses SQL into `CreateTableInfo`.
2. DuckDB binds it into `BoundCreateTableInfo`.
3. DuckDB plans a `LogicalCreateTable`, then `PhysicalCreateTable`.
4. During execution, `PhysicalCreateTable` calls `LlmSchemaEntry::CreateTable(...)`.
5. The C++ adapter converts `BoundCreateTableInfo` into `CreateTableOp`.
6. The extension sends one `apply_mutation` request containing `[CreateTableOp]`.
7. Python streams progress while the out-of-scope experiment pipeline mutates the checkpoint.
8. Python returns `MutationResult` with a refreshed catalog snapshot.
9. The extension updates its ephemeral in-memory catalog from that snapshot.
10. DuckDB returns success for the original SQL statement.

Key behavior:

- C++ does not call `build_dataset()` or `train()` directly.
- C++ does not decide how schema is represented in prompts or weights.
- User-managed `BEGIN`, `COMMIT`, and `ROLLBACK` are unsupported; there is no multi-statement staging.

## Request Flow 2: SELECT

SQL:

```sql
SELECT name FROM llm.fruits WHERE goodness > 1;
```

Flow:

1. DuckDB binds the table and columns using the extension's ephemeral catalog snapshot.
2. The extension participates in scan planning for `llm.fruits`.
3. The adapter converts the bound projection and predicate into `SelectQuery`.
4. The physical scan operator sends a `select` request to Python.
5. Python invokes the out-of-scope sampling pipeline.
6. Python returns typed rows.
7. The extension writes rows into DuckDB `DataChunk`s.
8. DuckDB returns the result to the client.

Key behavior:

- Predicates, projections, and limits must be pushed to Python/model sampling.
- DuckDB may still validate and cast returned values.
- If an expression cannot be pushed down yet, the extension should reject the query as unsupported. Silent local filtering violates the spirit of the project.

## Error Handling

The adapter should translate Python errors into DuckDB exceptions with useful context.

Examples:

- server unavailable: connection/configuration error
- unsupported SQL feature: binder or not-implemented error before Python call
- model/runtime failure during mutation apply: statement execution error
- malformed sampled rows: execution error with expected column/type detail

On mutation failure, the extension must not update its ephemeral catalog snapshot. Python is responsible for making checkpoint publication atomic from the adapter's perspective.

## Progress Reporting

Long-running mutations should stream phase updates from Python. The extension should surface them through the best available DuckDB mechanism. If DuckDB does not provide a clean custom progress API for extensions, the initial implementation may log progress to stderr or expose it through a side channel, but the protocol should still support structured progress events.

## Open Questions

- Exact IPC choice: HTTP JSON first, with possible Arrow IPC for large row streams later.
- Exact DuckDB physical scan implementation for `SELECT`.
- How much of DuckDB's expression tree should be supported in the initial predicate protocol.
- Whether CTAS should be supported early or deferred.
- Whether attach options should include checkpoint path, model name, experiment profile, or only a Python endpoint.
- How to display mutation progress cleanly inside the DuckDB CLI.

## Acceptance Criteria

An implementation of this layer is acceptable when the following are true against a mock Python control server:

- `ATTACH` calls catalog introspection once and materializes only schema metadata returned by the mock.
- `CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)` is converted to a single `CreateTableOp` with DuckDB-normalized types, primary-key metadata, and no row data.
- The `CreateTableOp` is sent in exactly one `apply_mutation` request during that statement.
- User-issued `BEGIN`, `COMMIT`, and `ROLLBACK` are rejected for LLM-backed work with a clear unsupported error. No mutation request is sent for those statements.
- After a successful mutation response, the extension replaces its ephemeral catalog metadata with the returned catalog snapshot.
- `SHOW TABLES FROM llm` reflects the latest catalog snapshot and no other local state.
- `SELECT` sends projection, predicate, and limit information to Python in the adapter protocol.
- DuckDB does not apply filters, projections, or limits as a fallback. The physical scan returns the rows supplied by Python after type/shape validation.
- Unsupported filters, projections, limits, types, or constraints fail with clear DuckDB errors before producing silently incorrect behavior.
- If Python returns a failed mutation, the extension does not update its ephemeral catalog snapshot.
- The mock server can assert the full request sequence, proving the C++ layer did not make hidden calls or store extra state.

## Runnable Mock E2E Contract

The initial end-to-end test should run DuckDB with the loadable `llm` extension and a local mock Python server. The mock replaces the out-of-scope experiment runtime. It does not sample, build datasets, train, or publish checkpoints; it only records requests and returns deterministic JSON.

### Mock HTTP API

For the first implementation, use these concrete endpoints:

```text
POST /v1/catalog/introspect
POST /v1/mutations/apply
POST /v1/query/select
```

All requests and responses are JSON. Progress streaming is out of scope for this first e2e test.

### Initial Mock State

The mock starts with no tables:

```json
{
  "catalog_version": "v0",
  "schemas": [
    {"name": "main", "tables": []}
  ]
}
```

### Test SQL

```sql
ATTACH '' AS llm (
    TYPE llm,
    endpoint 'http://127.0.0.1:<mock_port>'
);

SHOW TABLES FROM llm;

CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT);

SHOW TABLES FROM llm;

SELECT name, goodness
FROM llm.fruits
WHERE goodness > 1
LIMIT 10;
```

Expected DuckDB-visible results:

```text
SHOW TABLES before CREATE -> []
SHOW TABLES after CREATE  -> [("fruits")]
SELECT result             -> [("apple", 1), ("orange", 2)]
```

The `SELECT` result intentionally includes `("apple", 1)`, which violates `goodness > 1`. This is a canary proving DuckDB did not post-filter. Correctness of model filtering belongs to the Python/model layer; this adapter test only verifies pushdown and no fallback filtering.

### Expected Request Sequence

The mock must observe exactly these calls, in this order.

#### 1. Attach Introspection

Request:

```json
{
  "type": "introspect_catalog",
  "catalog": "llm",
  "checkpoint_ref": ""
}
```

Response:

```json
{
  "catalog_version": "v0",
  "schemas": [
    {"name": "main", "tables": []}
  ]
}
```

#### 2. CREATE TABLE Mutation Apply

Request:

```json
{
  "type": "apply_mutation",
  "base_catalog_version": "v0",
  "operations": [
    {
      "op": "create_table",
      "catalog": "llm",
      "schema": "main",
      "table": "fruits",
      "on_conflict": "error",
      "columns": [
        {
          "name": "name",
          "duckdb_type": "VARCHAR",
          "nullable": false,
          "default": null,
          "generated": false
        },
        {
          "name": "goodness",
          "duckdb_type": "INTEGER",
          "nullable": true,
          "default": null,
          "generated": false
        }
      ],
      "primary_key": ["name"],
      "unique": [],
      "checks": [],
      "foreign_keys": []
    }
  ]
}
```

Response:

```json
{
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
              {"name": "name", "duckdb_type": "VARCHAR", "nullable": false},
              {"name": "goodness", "duckdb_type": "INTEGER", "nullable": true}
            ],
            "primary_key": ["name"],
            "constraints": []
          }
        ]
      }
    ]
  },
  "metrics": {}
}
```

#### 3. SELECT Pushdown

Request:

```json
{
  "type": "select",
  "catalog_version": "v1",
  "query": {
    "schema": "main",
    "table": "fruits",
    "projection": [
      {"name": "name", "duckdb_type": "VARCHAR"},
      {"name": "goodness", "duckdb_type": "INTEGER"}
    ],
    "predicate": {
      "kind": "comparison",
      "op": ">",
      "left": {"kind": "column", "name": "goodness", "duckdb_type": "INTEGER"},
      "right": {"kind": "literal", "value": 1, "duckdb_type": "INTEGER"}
    },
    "limit": 10
  }
}
```

Response:

```json
{
  "columns": [
    {"name": "name", "duckdb_type": "VARCHAR"},
    {"name": "goodness", "duckdb_type": "INTEGER"}
  ],
  "rows": [
    ["apple", 1],
    ["orange", 2]
  ]
}
```

### Pytest Shape

The runnable test can live in `scripts/` or `tests/` and should roughly follow this structure:

```python
def test_create_table_and_select_pushdown(mock_llm_server, built_extension_path):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{built_extension_path}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{mock_llm_server.url}'
        )
        """
    )

    assert con.execute("SHOW TABLES FROM llm").fetchall() == []

    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    assert con.execute("SHOW TABLES FROM llm").fetchall() == [("fruits",)]

    rows = con.execute(
        """
        SELECT name, goodness
        FROM llm.fruits
        WHERE goodness > 1
        LIMIT 10
        """
    ).fetchall()

    assert rows == [("apple", 1), ("orange", 2)]
    assert mock_llm_server.calls == expected_calls
```

The `expected_calls` object should match the JSON requests above. Any extra request should fail the test.

### Explicit Transaction Rejection Test

The adapter suite should also include a separate test that proves user-managed transactions are not available after attaching an LLM catalog:

```python
def test_explicit_begin_is_unsupported(mock_llm_server, built_extension_path):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{built_extension_path}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{mock_llm_server.url}'
        )
        """
    )

    with pytest.raises(duckdb.Error, match="transaction|BEGIN|explicit"):
        con.execute("BEGIN")

    assert mock_llm_server.calls == [
        {
            "path": "/v1/catalog/introspect",
            "json": {
                "type": "introspect_catalog",
                "catalog": "llm",
                "checkpoint_ref": "",
            },
        }
    ]
```

Blocking `BEGIN` is the key requirement because it prevents any uncommitted multi-statement state from existing. User-issued `COMMIT` and `ROLLBACK` should also produce clear errors rather than applying or discarding LLM mutations.

## Out Of Scope Handoffs

The following interfaces are intentionally named but not designed here:

- `Pipeline.introspect_catalog(...)`
- `Pipeline.apply_mutation(...)`
- `Pipeline.sample_select(...)`
- `DatasetBuilder.build(...)`
- `TrainingStrategy.train(...)`
- `CheckpointPublisher.publish(...)`
- `Representation.encode_task(...)`
- `Representation.decode_constraints(...)`
- `Representation.parse_output(...)`

Those components define how the model becomes a database. This document defines how DuckDB hands structured SQL intent to them.
