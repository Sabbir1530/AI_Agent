"""Build a reproducible SQLite database with a fixed random seed."""
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "database" / "agent.db"
SEED = 20261111

def build_database(db_path=DB_PATH):
    rng = random.Random(SEED)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = ON;
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT NOT NULL,
        city TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_price REAL NOT NULL
    );
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
        order_date TEXT NOT NULL,
        status TEXT NOT NULL
    );
    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL REFERENCES orders(order_id),
        product_id INTEGER NOT NULL REFERENCES products(product_id),
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL
    );
    """)

    cities = ["Dhaka", "Chattogram", "Khulna", "Rajshahi", "Sylhet"]
    for cid in range(1, 31):
        cur.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            (cid, f"Customer {cid:02d}", rng.choice(cities),
             f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
        )

    catalog = [
        ("Laptop", "Electronics", 650.0), ("Mouse", "Electronics", 18.0),
        ("Keyboard", "Electronics", 35.0), ("Chair", "Furniture", 90.0),
        ("Desk", "Furniture", 160.0), ("Notebook", "Stationery", 3.5),
        ("Pen Pack", "Stationery", 5.0), ("Bottle", "Lifestyle", 12.0),
        ("Backpack", "Lifestyle", 42.0), ("Monitor", "Electronics", 210.0)
    ]
    for pid, (name, category, price) in enumerate(catalog, 1):
        cur.execute("INSERT INTO products VALUES (?, ?, ?, ?)",
                    (pid, name, category, price))

    statuses = ["completed", "completed", "completed", "pending", "cancelled"]
    order_id = 1
    item_id = 1
    for _ in range(180):  # biggest table has >= 100 rows
        customer_id = rng.randint(1, 30)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        cur.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            (order_id, customer_id, f"2025-{month:02d}-{day:02d}",
             rng.choice(statuses))
        )
        for _ in range(rng.randint(1, 4)):
            product_id = rng.randint(1, len(catalog))
            quantity = rng.randint(1, 5)
            price = catalog[product_id - 1][2]
            cur.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
                (item_id, order_id, product_id, quantity, price)
            )
            item_id += 1
        order_id += 1

    conn.commit()
    conn.close()
    return db_path

if __name__ == "__main__":
    print(f"Created database: {build_database()}")
