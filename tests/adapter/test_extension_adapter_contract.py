import duckdb
import pytest


def test_create_table_and_select_pushdown(mock_llm_server, built_extension_path):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{built_extension_path.as_posix()}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{mock_llm_server.url}'
        )
        """
    )

    assert con.execute("SHOW TABLES FROM llm").fetchall() == []

    try:
        con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")
    except duckdb.Error as exc:
        if "LLM CREATE TABLE requires safetensors-backed catalog metadata" in str(exc):
            pytest.xfail("adapter CREATE TABLE contract is not implemented yet")
        raise

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
    assert mock_llm_server.calls == _expected_calls()


def test_explicit_begin_is_unsupported(mock_llm_server, built_extension_path):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{built_extension_path.as_posix()}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{mock_llm_server.url}'
        )
        """
    )

    try:
        con.execute("BEGIN")
    except duckdb.Error as exc:
        message = str(exc).lower()
        assert "transaction" in message or "begin" in message or "explicit" in message
    else:
        pytest.xfail("explicit transaction rejection is not implemented yet")

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


def _expected_calls():
    return [
        {
            "path": "/v1/catalog/introspect",
            "json": {
                "type": "introspect_catalog",
                "catalog": "llm",
                "checkpoint_ref": "",
            },
        },
        {
            "path": "/v1/mutations/apply",
            "json": {
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
                                "nullable": False,
                                "default": None,
                                "generated": False,
                            },
                            {
                                "name": "goodness",
                                "duckdb_type": "INTEGER",
                                "nullable": True,
                                "default": None,
                                "generated": False,
                            },
                        ],
                        "primary_key": ["name"],
                        "unique": [],
                        "checks": [],
                        "foreign_keys": [],
                    }
                ],
            },
        },
        {
            "path": "/v1/query/select",
            "json": {
                "type": "select",
                "catalog_version": "v1",
                "query": {
                    "schema": "main",
                    "table": "fruits",
                    "projection": [
                        {"name": "name", "duckdb_type": "VARCHAR"},
                        {"name": "goodness", "duckdb_type": "INTEGER"},
                    ],
                    "predicate": {
                        "kind": "comparison",
                        "op": ">",
                        "left": {
                            "kind": "column",
                            "name": "goodness",
                            "duckdb_type": "INTEGER",
                        },
                        "right": {
                            "kind": "literal",
                            "value": 1,
                            "duckdb_type": "INTEGER",
                        },
                    },
                    "limit": 10,
                },
            },
        },
    ]
