"""
ingest.py — Automatic Transaction File → JSON Payload Converter

Reads CSV or Excel files containing raw banking transactions,
groups them by account, computes baselines, and builds the exact
JSON payload that the VigilanceAI /v1/screen endpoint expects.

The banking manager just uploads a file — no JSON crafting needed.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------
# Column Name Mapping (flexible — handles messy headers)
# ---------------------------------------------------------
# Maps all common variations to our standard internal names.
# The parser will try each alias until it finds a match.

COLUMN_ALIASES = {
    "account_id": [
        "account_id", "account", "acc_id", "acct_id", "acct",
        "account_no", "account_number", "acc_no", "acc_number"
    ],
    "tx_id": [
        "tx_id", "txn_id", "transaction_id", "trans_id", "ref_no",
        "reference", "reference_no", "ref_id"
    ],
    "amount": [
        "amount", "amt", "value", "transaction_amount", "txn_amount",
        "txn_amt", "debit_amount", "credit_amount", "sum"
    ],
    "recipient": [
        "recipient", "beneficiary", "payee", "to", "receiver",
        "recipient_name", "beneficiary_name", "counterparty"
    ],
    "rail": [
        "rail", "channel", "mode", "payment_mode", "txn_type",
        "transaction_type", "payment_method", "type"
    ],
    "device": [
        "device", "device_id", "dev_id", "terminal", "terminal_id",
        "source_device"
    ],
    "time": [
        "time", "timestamp", "txn_time", "transaction_time", "datetime",
        "date_time", "txn_date", "transaction_date", "date"
    ],
    "location": [
        "location", "city", "branch", "branch_city", "origin",
        "txn_location", "geo", "region"
    ],
}


def _normalize_column_name(col: str) -> str:
    """Strips whitespace, lowercases, replaces spaces/hyphens with underscores."""
    return col.strip().lower().replace(" ", "_").replace("-", "_")


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to match each DataFrame column to a known standard name.
    Returns a new DataFrame with standardized column names.
    """
    raw_cols = {_normalize_column_name(c): c for c in df.columns}
    mapped = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in raw_cols:
                mapped[standard_name] = raw_cols[alias]
                break

    # Rename only the matched columns
    reverse_map = {orig: std for std, orig in mapped.items()}
    df = df.rename(columns=reverse_map)

    return df


def read_transaction_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV or Excel file and returns a pandas DataFrame
    with standardized column names.

    Supports: .csv, .xlsx, .xls
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, or .xls")

    # Normalize column names
    df.columns = [_normalize_column_name(c) for c in df.columns]
    df = _map_columns(df)

    return df


def read_transaction_bytes(file_bytes, file_name: str) -> pd.DataFrame:
    """
    Reads from file bytes (for Streamlit uploads).
    Same as read_transaction_file but works with in-memory bytes.
    """
    import io

    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, or .xls")

    df.columns = [_normalize_column_name(c) for c in df.columns]
    df = _map_columns(df)

    return df


def build_payloads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Groups transactions by account_id and builds one JSON payload
    per account in the exact format /v1/screen expects.

    Returns a list of payloads — one per account.
    """
    if "account_id" not in df.columns:
        raise ValueError(
            "No 'account_id' column found. Your file must have a column for "
            "account IDs. Accepted names: " + ", ".join(COLUMN_ALIASES["account_id"])
        )

    if "amount" not in df.columns:
        raise ValueError(
            "No 'amount' column found. Your file must have a column for "
            "transaction amounts. Accepted names: " + ", ".join(COLUMN_ALIASES["amount"])
        )

    payloads = []

    for account_id, group in df.groupby("account_id"):
        # Build individual transaction records
        batch_records = []
        for _, row in group.iterrows():
            tx = {}
            tx["tx_id"] = str(row.get("tx_id", f"TXN-AUTO-{_:05d}"))
            tx["amount"] = int(row["amount"])

            # Optional fields — include only if present
            if "recipient" in row and pd.notna(row["recipient"]):
                tx["recipient"] = str(row["recipient"])
            if "rail" in row and pd.notna(row["rail"]):
                tx["rail"] = str(row["rail"]).upper()
            if "device" in row and pd.notna(row["device"]):
                tx["device"] = str(row["device"])
            if "time" in row and pd.notna(row["time"]):
                tx["time"] = str(row["time"])
            if "location" in row and pd.notna(row["location"]):
                tx["location"] = str(row["location"])

            batch_records.append(tx)

        # Compute account baseline from the transaction amounts
        amounts = group["amount"].dropna().astype(float)
        median_amt = int(amounts.median()) if len(amounts) > 0 else 0
        bracket_low = int(amounts.min()) if len(amounts) > 0 else 0
        bracket_high = int(amounts.max()) if len(amounts) > 0 else 0

        # Build the full payload matching /v1/screen format
        payload = {
            "subject_account": str(account_id),
            "account_baseline": {
                "account_id": str(account_id),
                "historical_median": median_amt,
                "typical_bracket": [bracket_low, bracket_high],
            },
            "batch_records": batch_records,
            "applied_policy": "POL-AUTO-INGEST: Automated screening from uploaded transaction file.",
        }

        payloads.append(payload)

    return payloads


def file_to_payloads(file_path: str) -> List[Dict[str, Any]]:
    """
    End-to-end convenience function:
    File path → parsed DataFrame → list of JSON payloads
    """
    df = read_transaction_file(file_path)
    return build_payloads(df)


def bytes_to_payloads(file_bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    End-to-end convenience function for Streamlit uploads:
    Uploaded bytes → parsed DataFrame → list of JSON payloads
    """
    df = read_transaction_bytes(file_bytes, file_name)
    return build_payloads(df)
