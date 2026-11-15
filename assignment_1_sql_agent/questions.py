"""The test set: question, kind, and an INDEPENDENT SQL that gives the answer.

The expected answers are computed with plain sqlite3 (not with the agent's
tools). kind: simple | join | grouping | calculation | multi | unanswerable.
"multi" questions need several tool calls (separate queries and/or the
calculator). Unanswerable questions have sql=None.
"""
import sqlite3
from pathlib import Path

_REV = "SUM(oi.quantity*oi.unit_price)"
_CUST_REV = ("SELECT {rev} FROM order_items oi JOIN orders o ON "
             "o.order_id=oi.order_id WHERE o.status='completed' AND "
             "o.customer_id={cid}")
_ORDER_TOTALS = ("SELECT o.order_id, SUM(oi.quantity*oi.unit_price) AS total "
                 "FROM orders o JOIN order_items oi ON oi.order_id=o.order_id "
                 "WHERE o.status='completed' GROUP BY o.order_id")

QUESTIONS = [
    ("Q01", "simple", "How many customers are there?",
     "SELECT COUNT(*) FROM customers"),
    ("Q02", "simple", "How many products are there?",
     "SELECT COUNT(*) FROM products"),
    ("Q03", "simple", "How many orders are there?",
     "SELECT COUNT(*) FROM orders"),
    ("Q04", "simple", "How many completed orders are there?",
     "SELECT COUNT(*) FROM orders WHERE status='completed'"),
    ("Q05", "join", "What is the total revenue from completed orders?",
     f"SELECT ROUND({_REV},2) FROM order_items oi JOIN orders o ON "
     "o.order_id=oi.order_id WHERE o.status='completed'"),
    ("Q06", "calculation", "What is the average order value for completed orders?",
     f"SELECT ROUND(AVG(total),2) FROM ({_ORDER_TOTALS})"),
    ("Q07", "grouping", "Which city has the most customers?",
     "SELECT city FROM customers GROUP BY city ORDER BY COUNT(*) DESC, city LIMIT 1"),
    ("Q08", "grouping", "Which product category earned the most revenue from completed orders?",
     "SELECT p.category FROM products p JOIN order_items oi ON oi.product_id=p.product_id "
     "JOIN orders o ON o.order_id=oi.order_id WHERE o.status='completed' "
     f"GROUP BY p.category ORDER BY {_REV} DESC LIMIT 1"),
    ("Q09", "join", "What is the total revenue for customer 12 (completed orders)?",
     "SELECT ROUND(" + _REV + ",2) FROM order_items oi JOIN orders o ON "
     "o.order_id=oi.order_id WHERE o.customer_id=12 AND o.status='completed'"),
    ("Q10", "simple", "How many unique customers placed completed orders?",
     "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE status='completed'"),
    ("Q11", "simple", "What is the most expensive product name?",
     "SELECT product_name FROM products ORDER BY unit_price DESC LIMIT 1"),
    ("Q12", "join", "What is the total quantity sold in completed orders?",
     "SELECT SUM(oi.quantity) FROM order_items oi JOIN orders o ON "
     "o.order_id=oi.order_id WHERE o.status='completed'"),
    ("Q13", "calculation", "What is the revenue difference between Electronics and Furniture (completed orders)?",
     "SELECT ROUND(SUM(CASE WHEN p.category='Electronics' THEN oi.quantity*oi.unit_price ELSE 0 END)"
     "-SUM(CASE WHEN p.category='Furniture' THEN oi.quantity*oi.unit_price ELSE 0 END),2) "
     "FROM order_items oi JOIN products p ON p.product_id=oi.product_id "
     "JOIN orders o ON o.order_id=oi.order_id WHERE o.status='completed'"),
    ("Q14", "grouping", "Which customer placed the highest number of orders?",
     "SELECT c.customer_name FROM customers c JOIN orders o ON o.customer_id=c.customer_id "
     "GROUP BY c.customer_id ORDER BY COUNT(*) DESC, c.customer_id LIMIT 1"),
    ("Q15", "calculation", "What percentage of all orders are completed?",
     "SELECT ROUND(100.0*SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)/COUNT(*),2) FROM orders"),
    # multi-step: several queries and/or the calculator
    ("Q16", "multi",
     "What is the average order value for completed orders, and what percent above the minimum completed order value is it?",
     [f"SELECT ROUND(AVG(total),2) FROM ({_ORDER_TOTALS})",
      f"SELECT ROUND(100.0*(AVG(total)-MIN(total))/MIN(total),2) FROM ({_ORDER_TOTALS})"]),
    ("Q17", "multi",
     "By what percent is customer 12's revenue lower than customer 5's revenue (completed orders only)?",
     "SELECT ROUND(100.0*((" + _CUST_REV.format(rev=_REV, cid=5) + ")-("
     + _CUST_REV.format(rev=_REV, cid=12) + "))/(" + _CUST_REV.format(rev=_REV, cid=5) + "),2)"),
    # the database cannot answer these
    ("Q18", "unanswerable", "What was the exact GDP of Bangladesh in 2025?", None),
    ("Q19", "unanswerable", "How many employees work at the company?", None),
]


def expected_answer(sql, db_path):
    """Return the expected value (or list of values) using plain sqlite3."""
    if sql is None:
        return "UNANSWERABLE"
    conn = sqlite3.connect(Path(db_path))
    try:
        if isinstance(sql, list):
            return [conn.execute(s).fetchone()[0] for s in sql]
        row = conn.execute(sql).fetchone()
        return row[0] if row else None
    finally:
        conn.close()
