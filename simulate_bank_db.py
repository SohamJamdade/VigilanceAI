import sqlite3
import random
import time
import uuid
import sys
from datetime import datetime

DB_PATH = "core_banking.db"

ACCOUNTS = ["ACC-10001", "ACC-10002", "ACC-20003", "ACC-30004", "ACC-40005"]
RAILS = ["UPI", "NEFT", "RTGS", "IMPS", "SWIFT"]
RECIPIENTS = [
    "Ravi Sharma", "Priya Patel", "BENEF-301", "Tehran Global Exports",
    "Swiggy", "Amazon India", "Flipkart", "OFAC-BLOCKED-ENTITY",
    "Koryo Traders", "Surya Logistics", "Normal Shop Ltd"
]
LOCATIONS = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune"]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            tx_id TEXT NOT NULL,
            amount REAL NOT NULL,
            rail TEXT DEFAULT 'IMPS',
            recipient TEXT DEFAULT 'UNKNOWN',
            device_id TEXT DEFAULT 'DEV-UNKNOWN',
            location TEXT DEFAULT 'DOMESTIC',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            screening_status TEXT DEFAULT 'PENDING'
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database ready: {DB_PATH}")


def insert_random_tx():
    acc = random.choice(ACCOUNTS)
    tx_id = f"TXN-SIM-{uuid.uuid4().hex[:8].upper()}"

    # 70% normal, 20% near-threshold structuring, 10% high-risk/sanctions
    roll = random.random()
    if roll < 0.70:
        amount = random.randint(500, 15000)
    elif roll < 0.90:
        amount = random.randint(45000, 49900)
    else:
        amount = random.randint(100000, 800000)

    rail = random.choice(RAILS)
    recipient = random.choice(RECIPIENTS)
    device = f"DEV-{random.randint(100, 999)}"
    location = random.choice(LOCATIONS)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO transactions (account_id, tx_id, amount, rail, recipient, device_id, location) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (acc, tx_id, amount, rail, recipient, device, location),
    )
    conn.commit()
    conn.close()

    tag = "[NORMAL]"
    if amount >= 45000:
        tag = "[STRUCT]"
    if amount >= 100000 or "OFAC" in recipient:
        tag = "[ALERT!]"

    msg = f"{tag} [{datetime.now().strftime('%H:%M:%S')}] {acc} | {tx_id} | INR {amount:,} | {rail} -> {recipient}"
    try:
        print(msg)
    except Exception:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8"))


if __name__ == "__main__":
    init_db()
    print("\nSimulating core banking transactions (Ctrl+C to stop)...\n")
    try:
        while True:
            insert_random_tx()
            time.sleep(random.uniform(2.0, 4.0))
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
