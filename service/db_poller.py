import os
import json
import time
import sqlite3
import threading
import pandas as pd
import requests
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

API_BASE = os.getenv("VIGILANCE_API_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
DEFAULT_CONFIG = "db_credentials.json"

poller_telemetry: Dict[str, Any] = {
    "status": "IDLE",
    "last_scan": None,
    "next_scan": None,
    "total_scanned": 0,
    "last_error": None,
    "db_path": None,
    "table": "transactions",
}


def run_single_scan(db_path: str = "core_banking.db", table: str = "transactions") -> int:
    resolved_path = os.path.abspath(db_path)
    if not os.path.exists(resolved_path):
        poller_telemetry["last_error"] = f"Database not found at {resolved_path}"
        return 0

    api_url = os.getenv("VIGILANCE_API_URL", API_BASE)
    scanned_count = 0

    try:
        with sqlite3.connect(resolved_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            table_check = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
            ).fetchone()
            if not table_check:
                poller_telemetry["last_error"] = f"Table '{table}' does not exist in {resolved_path}"
                return 0

            # Ensure screening_status column exists in user database table
            col_info = cursor.execute(f"PRAGMA table_info('{table}')").fetchall()
            col_names = [c[1] for c in col_info]
            if "screening_status" not in col_names:
                cursor.execute(f"ALTER TABLE '{table}' ADD COLUMN screening_status TEXT DEFAULT 'PENDING'")
                conn.commit()

            # Fetch pending records
            cursor.execute(f"""
                SELECT * FROM '{table}' 
                WHERE screening_status = 'PENDING' OR screening_status IS NULL 
                LIMIT 100
            """)
            rows = cursor.fetchall()
            if not rows:
                poller_telemetry["last_scan"] = datetime.now().strftime("%H:%M:%S")
                poller_telemetry["status"] = "IDLE (0 pending records)"
                poller_telemetry["last_error"] = None
                return 0

            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                d = dict(r)
                acc = str(d.get("account_id") or d.get("account") or d.get("user_id") or "ACC-UNKNOWN")
                grouped.setdefault(acc, []).append(d)

            for acc, tx_list in grouped.items():
                amounts = [float(t.get("amount") or 0.0) for t in tx_list]
                dyn_median = float(pd.Series(amounts).median()) if amounts else 0.0

                batch_records = []
                for t in tx_list:
                    raw_id = t.get("tx_id") or t.get("id") or f"TXN-{uuid.uuid4().hex[:6]}"
                    batch_records.append({
                        "tx_id": str(raw_id),
                        "amount": float(t.get("amount") or 0.0),
                        "rail": str(t.get("rail") or "IMPS"),
                        "recipient": str(t.get("recipient") or "UNKNOWN"),
                        "device_id": str(t.get("device_id") or t.get("device") or "DEV-UNKNOWN"),
                        "location": str(t.get("location") or "DOMESTIC")
                    })

                payload = {
                    "subject_account": acc,
                    "batch_records": batch_records,
                    "account_baseline": {
                        "account_id": acc,
                        "historical_median": dyn_median,
                        "typical_bracket": [min(amounts), max(amounts)] if amounts else [0.0, 0.0]
                    }
                }

                # Dispatch via HTTP or fallback to local pipeline execution
                success = False
                try:
                    resp = requests.post(f"{api_url}/v1/screen", json=payload, timeout=15)
                    if resp.status_code == 200:
                        success = True
                except Exception:
                    pass

                if not success:
                    try:
                        from service.app import run_pipeline, NormalizedTransaction
                        norm_txs = [
                            NormalizedTransaction(
                                tx_id=r["tx_id"],
                                account_id=acc,
                                amount=r["amount"],
                                currency="INR",
                                rail=r["rail"].upper(),
                                recipient=r["recipient"],
                                device_id=r["device_id"],
                                timestamp=datetime.now(timezone.utc),
                                location=r["location"],
                                source_hash="DB_SCAN_LOCAL"
                            ) for r in batch_records
                        ]
                        run_pipeline(acc, norm_txs, payload)
                        success = True
                    except Exception as pipe_err:
                        poller_telemetry["last_error"] = f"Pipeline error: {pipe_err}"

                if success:
                    for t in tx_list:
                        row_id = t.get("id")
                        tx_id_val = t.get("tx_id")
                        if row_id is not None:
                            cursor.execute(f"UPDATE '{table}' SET screening_status = 'SCREENED' WHERE id = ?", (row_id,))
                        elif tx_id_val is not None:
                            cursor.execute(f"UPDATE '{table}' SET screening_status = 'SCREENED' WHERE tx_id = ?", (tx_id_val,))
                    scanned_count += len(tx_list)

            conn.commit()

        poller_telemetry["total_scanned"] += scanned_count
        poller_telemetry["last_scan"] = datetime.now().strftime("%H:%M:%S")
        poller_telemetry["status"] = "IDLE (Last Scan Successful)"
        poller_telemetry["last_error"] = None
        return scanned_count

    except Exception as e:
        poller_telemetry["last_error"] = str(e)
        return 0


class AutomatedDBPoller:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        db_path: Optional[str] = None,
        table: Optional[str] = None,
        poll_interval: Optional[int] = None,
        api_base: Optional[str] = None,
    ):
        cfg = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                poller_telemetry["last_error"] = f"Failed to read credentials: {e}"

        self.db_path = db_path or cfg.get("database") or cfg.get("db_path") or "core_banking.db"
        self.table = table or cfg.get("table") or "transactions"
        self.poll_interval = poll_interval or cfg.get("poll_interval_seconds") or POLL_INTERVAL
        self.api_base = api_base or os.getenv("VIGILANCE_API_URL", API_BASE)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        poller_telemetry["db_path"] = self.db_path
        poller_telemetry["table"] = self.table

    def run_single_scan(self) -> int:
        poller_telemetry["status"] = "SCANNING"
        return run_single_scan(self.db_path, self.table)

    def _loop(self):
        while not self._stop_event.is_set():
            self.run_single_scan()
            next_ts = datetime.now(timezone.utc).timestamp() + self.poll_interval
            poller_telemetry["next_scan"] = datetime.fromtimestamp(next_ts, tz=timezone.utc).isoformat()
            self._stop_event.wait(timeout=self.poll_interval)

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop_event.clear()
        poller_telemetry["status"] = "STARTING"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()
        poller_telemetry["status"] = "STOPPED"

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


_global_poller: Optional[AutomatedDBPoller] = None


def get_poller(db_path: Optional[str] = None, table: Optional[str] = None) -> AutomatedDBPoller:
    global _global_poller
    if _global_poller is None or db_path or table:
        _global_poller = AutomatedDBPoller(db_path=db_path, table=table)
    return _global_poller


def start_poller(db_path: Optional[str] = None, table: Optional[str] = None) -> bool:
    poller = get_poller(db_path=db_path, table=table)
    return poller.start()


def stop_poller():
    global _global_poller
    if _global_poller:
        _global_poller.stop()


def is_running() -> bool:
    global _global_poller
    return _global_poller is not None and _global_poller.is_running()
