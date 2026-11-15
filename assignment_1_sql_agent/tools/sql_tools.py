"""The assignment tools: SQL generation and safe (read-only) SQL execution.

Design choice (assignment 1, requirement 2): ``generate_sql`` is a tool that
makes its *own* model call. The schema is also in the agent's system prompt so
the agent can reason about it, but SQL writing is delegated to a focused
prompt with strict rules and JSON output. This keeps the agent's context small
and lets the generator be changed or tested independently.
"""
import json
import re
import sqlite3
from pathlib import Path

SCHEMA = """
customers(customer_id, customer_name, city, created_at)
products(product_id, product_name, category, unit_price)
orders(order_id, customer_id, order_date, status)
order_items(order_item_id, order_id, product_id, quantity, unit_price)
Relationships:
orders.customer_id -> customers.customer_id
order_items.order_id -> orders.order_id
order_items.product_id -> products.product_id
order status values: 'completed', 'pending', 'cancelled'
Revenue = SUM(order_items.quantity * order_items.unit_price).
For revenue questions, normally include only orders with status='completed'.
"""

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex)\b", re.I)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def generate_sql_tool(client, question, json_mode=True):
    """Ask the local model for one read-only SQL SELECT query.

    Never raises on bad model output: returns {"error": ...} so the agent can
    pass the message back to the model. Token usage is always in "_usage".
    """
    prompt = f"""Convert the question into SQLite SQL.
Return ONLY JSON: {{"sql":"SELECT ...","explanation":"brief reason"}}
Rules:
- Only one SELECT statement. Never write INSERT, UPDATE, DELETE, DROP, PRAGMA or DDL.
- Use only tables and columns in this schema.
- Revenue means quantity * unit_price, and completed orders only.
- Do not invent missing facts. If the database cannot answer, return
  {{"sql":"","explanation":"not_answerable: reason"}}.
- Add LIMIT 100 to row-returning queries where appropriate.
Schema:
{SCHEMA}
Question: {question}
"""
    response = client.generate(
        prompt,
        system="You generate safe SQLite queries. Output valid JSON only.",
        temperature=0,
        num_predict=300,
        format="json" if json_mode else None,
    )
    usage = {
        "input_tokens": int(response.get("prompt_eval_count") or 0),
        "output_tokens": int(response.get("eval_count") or 0),
    }
    content = _JSON_FENCE.sub("", response["message"]["content"].strip())
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("JSON is not an object")
    except (json.JSONDecodeError, ValueError):
        parsed = {"error": f"generate_sql returned invalid JSON: {content[:200]}"}
    parsed["_usage"] = usage
    return parsed


def _strip_literals(sql):
    return _STRING_LITERAL.sub("''", sql)


def execute_sql_tool(sql, db_path, row_limit=100):
    """Execute one SELECT against SQLite opened in read-only mode.

    Raises ValueError (clear message) for anything that is not a single
    SELECT; sqlite errors are re-raised as ValueError("SQL error: ...").
    """
    sql = (sql or "").strip()
    if not sql:
        raise ValueError("Empty SQL query")
    body = sql.rstrip(";").strip()
    checked = _strip_literals(body)  # ignore text inside 'string literals'
    if ";" in checked:
        raise ValueError("Multiple SQL statements are not allowed")
    normalized = re.sub(r"\s+", " ", checked).strip().lower()
    if not re.match(r"^select\b", normalized):
        raise ValueError("Only SELECT statements are allowed")
    if _FORBIDDEN.search(normalized):
        raise ValueError("Forbidden SQL operation")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError(f"Cannot open database read-only: {exc}") from exc
    try:
        conn.set_authorizer(_read_only_authorizer)
        try:
            cur = conn.execute(body)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(row_limit + 1)
        except sqlite3.Error as exc:
            raise ValueError(f"SQL error: {exc}") from exc
        truncated = len(rows) > row_limit
        rows = rows[:row_limit]
        return {
            "columns": columns,
            "rows": [dict(zip(columns, row)) for row in rows],
            "truncated": truncated,
        }
    finally:
        conn.close()


def _read_only_authorizer(action, arg1, arg2, db_name, trigger):
    """Allow only reads; deny every write or schema change at SQLite level."""
    allowed = {
        sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
