import os
from pathlib import Path

import duckdb
import pytest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION_PATH = ROOT / "extension" / "build" / "release" / "extension" / "llm" / "llm.duckdb_extension"


pytestmark = pytest.mark.skipif(
    os.environ.get("SQL_LLM_REAL_TEST") != "1",
    reason="set SQL_LLM_REAL_TEST=1 and run real SGLang/control-server processes",
)


def test_fruit_demo_against_real_llm_database():
    endpoint = os.environ.get("SQL_LLM_ENDPOINT", "http://127.0.0.1:5366")
    extension_path = Path(os.environ.get("SQL_LLM_EXTENSION_PATH", DEFAULT_EXTENSION_PATH))
    if not extension_path.exists():
        pytest.skip("missing llm extension binary; build the extension or set SQL_LLM_EXTENSION_PATH")

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{extension_path.as_posix()}'")
    con.execute(
        f"""
        ATTACH '' AS llm (
            TYPE llm,
            endpoint '{endpoint}'
        )
        """
    )

    assert con.execute("SHOW TABLES FROM llm").fetchall() == []
    con.execute("CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)")

    assert con.execute(
        "INSERT INTO llm.fruits (name, goodness) VALUES ('apple', 1)"
    ).fetchall() == [(1,)]
    assert set(con.execute("SELECT name, goodness FROM llm.fruits").fetchall()) == {("apple", 1)}

    assert con.execute(
        "INSERT INTO llm.fruits (name, goodness) VALUES ('orange', 2)"
    ).fetchall() == [(1,)]
    assert set(con.execute("SELECT name, goodness FROM llm.fruits").fetchall()) == {("apple", 1), ("orange", 2)}
    assert con.execute("SELECT name FROM llm.fruits WHERE goodness > 1").fetchall() == [("orange",)]

    con.execute("UPDATE llm.fruits SET goodness = goodness * 2 WHERE starts_with(name, 'ap')")
    assert set(con.execute("SELECT name, goodness FROM llm.fruits").fetchall()) == {("apple", 2), ("orange", 2)}
