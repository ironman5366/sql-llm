import duckdb
import pytest

from llm.adapter_protocol import SelectColumn, SelectResponse
from llm.testing import RecordingPipeline


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
    assert mock_llm_server.calls == _expected_calls()


@pytest.mark.parametrize("statement", ["BEGIN", "COMMIT", "ROLLBACK"])
def test_explicit_transactions_are_unsupported(statement, mock_llm_server, built_extension_path):
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

    with pytest.raises(duckdb.Error) as error:
        con.execute(statement)
    message = str(error.value).lower()
    assert "transaction" in message or statement.lower() in message or "explicit" in message

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


def test_failed_mutation_does_not_update_catalog(adapter_server_factory, built_extension_path):
    server = adapter_server_factory(RecordingPipeline(fail_mutations=True))
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{built_extension_path.as_posix()}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{server.url}'
        )
        """
    )

    with pytest.raises(duckdb.Error, match="failed"):
        con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    assert con.execute("SHOW TABLES FROM llm").fetchall() == []
    assert server.calls == [
        _expected_calls()[0],
        {
            "path": "/v1/mutations/apply",
            "json": _expected_calls()[1]["json"],
        },
    ]


def test_arithmetic_predicate_is_pushed_to_adapter(mock_llm_server, built_extension_path):
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
    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    rows = con.execute(
        """
        SELECT name, goodness
        FROM llm.fruits
        WHERE goodness * 2 > 3
        """
    ).fetchall()

    assert rows == [("apple", 1), ("orange", 2)]
    assert mock_llm_server.calls == [
        *_expected_calls()[:2],
        {
            "path": "/v1/query/select",
            "json": {
                "type": "select",
                "catalog_version": "v1",
                "query": {
                    "schema": "main",
                    "table": "fruits",
                    "columns": [
                        {"name": "name", "duckdb_type": "VARCHAR", "nullable": False},
                        {"name": "goodness", "duckdb_type": "INTEGER", "nullable": True},
                    ],
                    "primary_key": ["name"],
                    "projection": [
                        {"name": "name", "duckdb_type": "VARCHAR"},
                        {"name": "goodness", "duckdb_type": "INTEGER"},
                    ],
                    "predicate": {
                        "kind": "comparison",
                        "op": ">",
                        "left": {
                            "kind": "arithmetic",
                            "op": "*",
                            "duckdb_type": "INTEGER",
                            "args": [
                                {
                                    "kind": "column",
                                    "name": "goodness",
                                    "duckdb_type": "INTEGER",
                                },
                                {
                                    "kind": "literal",
                                    "value": 2,
                                    "duckdb_type": "INTEGER",
                                },
                            ],
                        },
                        "right": {
                            "kind": "literal",
                            "value": 3,
                            "duckdb_type": "INTEGER",
                        },
                    },
                    "limit": None,
                },
            },
        },
    ]


def test_insert_rows_are_sent_to_adapter(mock_llm_server, built_extension_path):
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
    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    inserted = con.execute(
        """
        INSERT INTO llm.fruits (name, goodness)
        VALUES ('apple', 1), ('orange', 2)
        """
    ).fetchall()

    assert inserted == [(2,)]
    assert mock_llm_server.calls[-1] == {
        "path": "/v1/mutations/apply",
        "json": {
            "type": "apply_mutation",
            "base_catalog_version": "v1",
            "operations": [
                {
                    "op": "insert_rows",
                    "catalog": "llm",
                    "schema": "main",
                    "table": "fruits",
                    "columns": [
                        {"name": "name", "duckdb_type": "VARCHAR", "nullable": False},
                        {"name": "goodness", "duckdb_type": "INTEGER", "nullable": True},
                    ],
                    "primary_key": ["name"],
                    "rows": [["apple", 1], ["orange", 2]],
                }
            ],
        },
    }


def test_update_is_sent_to_adapter_as_typed_expression(mock_llm_server, built_extension_path):
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
    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    con.execute(
        """
        UPDATE llm.fruits
        SET goodness = goodness * 2
        WHERE starts_with(name, 'ap')
        """
    ).fetchall()

    update_call = mock_llm_server.calls[-1]
    assert update_call["path"] == "/v1/mutations/apply"
    update_op = update_call["json"]["operations"][0]
    assert update_op["op"] == "update_rows"
    assert update_op["catalog"] == "llm"
    assert update_op["schema"] == "main"
    assert update_op["table"] == "fruits"
    assert update_op["columns"] == [
        {"name": "name", "duckdb_type": "VARCHAR", "nullable": False},
        {"name": "goodness", "duckdb_type": "INTEGER", "nullable": True},
    ]
    assert update_op["primary_key"] == ["name"]
    assert update_op["assignments"] == [
        {
            "column": "goodness",
            "duckdb_type": "INTEGER",
            "value": {
                "kind": "arithmetic",
                "op": "*",
                "duckdb_type": "INTEGER",
                "args": [
                    {"kind": "column", "name": "goodness", "duckdb_type": "INTEGER"},
                    {"kind": "literal", "value": 2, "duckdb_type": "INTEGER"},
                ],
            },
        }
    ]
    assert update_op["predicate"]["kind"] == "function"
    assert update_op["predicate"]["name"] in {"starts_with", "prefix"}
    assert update_op["predicate"]["args"][0] == {"kind": "column", "name": "name", "duckdb_type": "VARCHAR"}
    assert update_op["predicate"]["args"][1] == {"kind": "literal", "value": "ap", "duckdb_type": "VARCHAR"}


def test_unsupported_predicate_fails_before_select(mock_llm_server, built_extension_path):
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
    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    with pytest.raises(duckdb.Error, match="push down|predicate|function|operator"):
        con.execute(
            """
            SELECT name
            FROM llm.fruits
            WHERE lower(name) = 'apple'
            """
        ).fetchall()

    assert mock_llm_server.calls == _expected_calls()[:2]


def test_limit_is_not_applied_locally(adapter_server_factory, built_extension_path):
    server = adapter_server_factory(
        RecordingPipeline(
            select_response=SelectResponse(
                columns=[
                    SelectColumn(name="name", duckdb_type="VARCHAR"),
                    SelectColumn(name="goodness", duckdb_type="INTEGER"),
                ],
                rows=[
                    ["apple", 1],
                    ["orange", 2],
                    ["pear", 3],
                ],
            )
        )
    )
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{built_extension_path.as_posix()}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{server.url}'
        )
        """
    )
    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    rows = con.execute(
        """
        SELECT name, goodness
        FROM llm.fruits
        LIMIT 1
        """
    ).fetchall()

    assert rows == [("apple", 1), ("orange", 2), ("pear", 3)]
    assert server.calls[-1]["json"]["query"]["limit"] == 1


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
                    "columns": [
                        {"name": "name", "duckdb_type": "VARCHAR", "nullable": False},
                        {"name": "goodness", "duckdb_type": "INTEGER", "nullable": True},
                    ],
                    "primary_key": ["name"],
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
