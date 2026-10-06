"""
scripts/prune_logs.py — Retention maintenance for core_banking.db and audit_log.db
"""
import os
import sqlite3
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DB = os.path.join(BASE_DIR, "core_banking.db")
AUDIT_DB = os.path.join(BASE_DIR, "audit_log.db")

def prune_old_records(days_retention: int = 90):
    cutoff_date = (datetime.utcnow() - timedelta(days=days_retention)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[*] Pruning routine baseline entries older than {cutoff_date}...")

    # 1. Clean SCREENED transactions from core_banking.db
    if os.path.exists(CORE_DB):
        with sqlite3.connect(CORE_DB) as conn:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM transactions 
                WHERE screening_status = 'SCREENED' AND created_at < ?
            """, (cutoff_date,))
            print(f"[+] Pruned {cur.rowcount} archived transactions from core_banking.db.")
            conn.commit()

    # 2. Prune only LOW risk alerts from audit_log.db (retains HIGH/CRITICAL permanently)
    if os.path.exists(AUDIT_DB):
        with sqlite3.connect(AUDIT_DB) as conn:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM alerts 
                WHERE risk_level = 'LOW' AND timestamp < ?
            """, (cutoff_date,))
            print(f"[+] Pruned {cur.rowcount} routine LOW alerts from audit_log.db.")
            conn.commit()

if __name__ == "__main__":
    prune_old_records(days_retention=90)