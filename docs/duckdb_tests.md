# DuckDB Test Suite

## Location

All tests: `ext/duckdb/test/sql/`

Groups are just folders. ~3,681 `.test` files total.

## Format

Tests use [sqllogictest](https://www.sqlite.org/sqllogictest/doc/trunk/about.wiki) format. A standalone Python parser exists at `ext/duckdb/scripts/sqllogictest/` (no DuckDB build required — pure stdlib Python).

```python
import sys; sys.path.insert(0, "ext/duckdb/scripts")
from sqllogictest import SQLLogicParser, Statement, Query
test = SQLLogicParser().parse("ext/duckdb/test/sql/insert/test_insert.test")
for stmt in test.statements:
    sql = "\n".join(stmt.lines)
    # stmt is Statement (ok/error) or Query (with expected_result.lines)
```

## Recommended starting groups

These have the highest density of simple single-table CRUD operations:

| Group | Path | Tests | Notes |
|---|---|---|---|
| **insert** | `ext/duckdb/test/sql/insert/` | 10 | Basic INSERT VALUES, multi-row, types |
| **delete** | `ext/duckdb/test/sql/delete/` | ~5 | Simple DELETE with/without WHERE |
| **update** | `ext/duckdb/test/sql/update/` | ~8 | SET assignments, basic WHERE |
| **types** | `ext/duckdb/test/sql/types/` | ~358 | Type declarations and casting |
| **create** | `ext/duckdb/test/sql/create/` | ~20 | CREATE TABLE basics (many use advanced features though) |

## Groups to avoid initially

| Group | Why |
|---|---|
| `join/` | Multi-table — not supported |
| `subquery/` | Not routed through extension |
| `window/` | Window functions — not applicable |
| `aggregate/` | GROUP BY, HAVING — server doesn't handle |
| `index/` | No index support |
| `copy/`, `export/` | File I/O, not relevant |
| `attach/` | Tests DuckDB's own attach, not ours |

## Things to know

- Each test file has a `# group: [name]` header matching its folder
- `statement ok` / `statement error` = execute and check success/failure
- `query II` = run query, expect 2 integer columns, expected rows follow `----`
- `loop`/`foreach`/`endloop` = parameterized tests (need expansion)
- `require` = skip test if feature unavailable
- `.test_slow` files = same format, just slower (628 of them)
- Tables need `llm.` prefix to route through the extension
- Each INSERT should be followed by COMMIT to trigger fine-tuning
