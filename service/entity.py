import sqlite3
from typing import List
from service.schema import NormalizedTransaction, EntityProfile

def resolve_entity(
        account_id: str,
        transactions: List[NormalizedTransaction],
        db_file: str = "audit_log.db"
) -> EntityProfile:
    # Extract unique devices, recipients, customers from current batch
    current_devices = set(tx.device_id for tx in transactions if tx.device_id != "DEV-UNKNOWN")
    current_recipients = set(tx.recipient for tx in transactions)
    current_customers = set(tx.customer_id for tx in transactions if tx.customer_id)

    shared_accounts = set()

    try:
        with sqlite3.connect(db_file, timeout=5.0) as conn:
            cursor = conn.cursor()
            for dev in current_devices:
                # Match column name (entity_val) and value case (DEVICE) from cases.py
                cursor.execute(
                    "SELECT DISTINCT subject_account FROM entity_links "
                    "WHERE entity_type = 'DEVICE' AND entity_val = ? AND subject_account != ?",
                    (dev, account_id)
                )
                for row in cursor.fetchall():
                    shared_accounts.add(row[0])
    except Exception:
        pass  # Failsafe during DB initialization

    return EntityProfile(
        account_id=account_id,
        associated_customers=list(current_customers) or [f"CUST-{account_id}"],
        associated_devices=list(current_devices),
        frequent_counterparties=list(current_recipients)[:5],
        shared_device_accounts=list(shared_accounts)
    )
