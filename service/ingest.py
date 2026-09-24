"""
service/ingest.py — Multi-Format Banking Transaction Ingestor
"""
import os
import io
import hashlib
from datetime import datetime
from typing import List
import pandas as pd
from service.schema import NormalizedTransaction

COLUMN_ALIASES = {
    "account_id": ["account_id", "account", "acc_id", "acct_id", "acct", "account_no", "account_number"],
    "tx_id": ["tx_id", "txn_id", "transaction_id", "trans_id", "ref_no", "reference"],
    "amount": ["amount", "amt", "value", "transaction_amount", "txn_amount", "debit_amount"],
    "recipient": ["recipient", "beneficiary", "payee", "to", "receiver", "counterparty"],
    "rail": ["rail", "channel", "mode", "payment_mode", "txn_type", "type"],
    "device": ["device", "device_id", "dev_id", "terminal", "terminal_id"],
    "time": ["time", "timestamp", "txn_time", "transaction_time", "datetime", "date_time", "date"],
    "location": ["location", "city", "branch", "geo", "region"],
    "customer_id": ["customer_id", "cust_id", "client_id"]
}

def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = {_normalize_name(c): c for c in df.columns}
    reverse_map = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in raw_cols:
                reverse_map[raw_cols[alias]] = standard_name
                break
    return df.rename(columns=reverse_map)

def parse_transactions(df: pd.DataFrame, source_tag: str = "FILE") -> List[NormalizedTransaction]:
    """Parses, validates, and hashes raw transaction dataframes."""
    df = map_dataframe(df)
    if "account_id" not in df.columns or "amount" not in df.columns:
        raise ValueError("Transactions file must contain recognizable 'account_id' and 'amount' columns.")

    normalized: List[NormalizedTransaction] = []
    for idx, row in df.iterrows():
        try:
            amt = float(row["amount"])
            acc = str(row["account_id"]).strip()
            tx_id = str(row.get("tx_id", f"TXN-GEN-{idx:05d}")).strip()
            rail = str(row.get("rail", "IMPS")).strip().upper()
            recipient = str(row.get("recipient", "UNKNOWN_BENEFICIARY")).strip()
            device = str(row.get("device", "DEV-UNKNOWN")).strip()
            
            raw_time = row.get("time")
            parsed_time = pd.to_datetime(raw_time, errors="coerce") if pd.notna(raw_time) else datetime.utcnow()
            if pd.isna(parsed_time):
                parsed_time = datetime.utcnow()
            else:
                parsed_time = parsed_time.to_pydatetime()

            raw_sig = f"{acc}|{tx_id}|{amt}|{recipient}|{parsed_time.isoformat()}"
            src_hash = hashlib.sha256(raw_sig.encode()).hexdigest()[:16]

            normalized.append(NormalizedTransaction(
                tx_id=tx_id,
                account_id=acc,
                customer_id=str(row.get("customer_id", f"CUST-{acc}")),
                amount=amt,
                currency="INR",
                rail=rail,
                recipient=recipient,
                device_id=device,
                timestamp=parsed_time,
                location=str(row.get("location", "DOMESTIC")),
                source_hash=src_hash
            ))
        except Exception:
            # Safely skip unparseable rows rather than crashing the batch
            continue

    return normalized

def read_transaction_bytes(file_bytes: bytes, file_name: str) -> List[NormalizedTransaction]:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return parse_transactions(df, source_tag=file_name)

def read_transaction_file(file_path: str) -> List[NormalizedTransaction]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return parse_transactions(df, source_tag=os.path.basename(file_path))