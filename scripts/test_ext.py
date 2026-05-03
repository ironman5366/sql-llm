from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = ROOT / "extension" / "build" / "release" / "extension" / "llm" / "llm.duckdb_extension"


def expect_error(con, sql, needle):
    try:
        con.execute(sql)
    except duckdb.Error as exc:
        message = str(exc)
        assert needle in message, message
    else:
        raise AssertionError(f"expected error for: {sql}")


def main():
    assert duckdb.__version__ == "1.5.2", duckdb.__version__
    assert EXTENSION_PATH.exists(), f"missing extension binary: {EXTENSION_PATH}"

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXTENSION_PATH.as_posix()}'")
    con.execute("ATTACH '' AS llm (TYPE llm)")

    tables = con.execute("SHOW TABLES FROM llm").fetchall()
    assert tables == [], tables

    expect_error(
        con,
        "CREATE TABLE llm.whatever (id INTEGER, prompt VARCHAR)",
        "LLM CREATE TABLE requires safetensors-backed catalog metadata",
    )
    expect_error(con, "SELECT * FROM llm.whatever", "Table with name whatever does not exist")

    print("ok")


if __name__ == "__main__":
    main()
