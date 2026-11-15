import pytest
from assignment_1_sql_agent.create_database import build_database
from assignment_1_sql_agent.tools.sql_tools import execute_sql_tool, generate_sql_tool


@pytest.fixture
def db(tmp_path):
    return build_database(tmp_path / "test.db")


def test_select_is_allowed(db):
    assert execute_sql_tool("SELECT COUNT(*) AS n FROM customers", db)["rows"][0]["n"] == 30


def test_trailing_semicolon_ok(db):
    assert execute_sql_tool("SELECT 1 AS x;", db)["rows"][0]["x"] == 1


@pytest.mark.parametrize("sql", [
    "DELETE FROM customers", "DROP TABLE customers",
    "UPDATE customers SET city='X'",
    "INSERT INTO customers VALUES (99,'X','X','2025-01-01')",
    "SELECT 1; SELECT 2", "SELECT 1; DROP TABLE customers",
    "PRAGMA table_info(customers)", "", "   ",
    "WITH x AS (SELECT 1) DELETE FROM customers",
])
def test_write_or_multiple_statements_rejected(db, sql):
    with pytest.raises(ValueError):
        execute_sql_tool(sql, db)


def test_database_unchanged_after_rejected_writes(db):
    with pytest.raises(ValueError):
        execute_sql_tool("DELETE FROM customers", db)
    assert execute_sql_tool("SELECT COUNT(*) AS n FROM customers", db)["rows"][0]["n"] == 30


def test_keywords_inside_string_literals_are_fine(db):
    r = execute_sql_tool("SELECT COUNT(*) AS n FROM orders WHERE status='delete; drop'", db)
    assert r["rows"][0]["n"] == 0


def test_created_at_column_is_not_blocked(db):
    assert execute_sql_tool("SELECT created_at FROM customers LIMIT 1", db)["rows"]


def test_row_limit(db):
    r = execute_sql_tool("SELECT * FROM order_items", db, row_limit=5)
    assert len(r["rows"]) == 5 and r["truncated"] is True


def test_sql_error_is_clear_value_error(db):
    with pytest.raises(ValueError, match="SQL error"):
        execute_sql_tool("SELECT nope FROM customers", db)


class FakeClient:
    def __init__(self, content):
        self.content = content

    def generate(self, prompt, **kw):
        return {"message": {"content": self.content}, "prompt_eval_count": 10, "eval_count": 5}


def test_generate_sql_parses_json_and_reports_usage():
    r = generate_sql_tool(FakeClient('```json\n{"sql":"SELECT 1","explanation":"x"}\n```'), "q")
    assert r["sql"] == "SELECT 1" and r["_usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_generate_sql_invalid_json_returns_error_not_exception():
    r = generate_sql_tool(FakeClient("not json"), "q")
    assert "error" in r and r["_usage"]["input_tokens"] == 10
