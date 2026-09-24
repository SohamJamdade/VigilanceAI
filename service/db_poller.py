import os
import json
import time
import sqlite3
import threading
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any

API_BASE = os.getenv("VIGILANCE_API_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
DEFAULT_CONFIG = "db_credentials.json"

# Shared poller telemetry dict consumed by dashboard
poller_telemetry: Dict[str, Any] = {
    "status": "IDLE",
    "last_scan": None,
    "next_scan": None,
    "total_scanned": 0,
    "last_error": None,
    "db_path": None,
    "table": "transactions",
}


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
        poller_telemetry["last_scan"] = datetime.now(timezone.utc).isoformat()
        scanned_count = 0

        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Select pending transactions from the specified table
            cursor.execute(
                f"SELECT id, tx_id, account_id, amount, rail, recipient, device_id, created_at "
                f"FROM {self.table} WHERE screening_status = 'PENDING' ORDER BY id ASC LIMIT 100"
            )
            rows = cursor.fetchall()

            if not rows:
                poller_telemetry["last_error"] = None
                conn.close()
                return 0

            # Group transactions by account ID for batch evaluation
            grouped = {}
            for row in rows:
                acc = row["account_id"]
                grouped.setdefault(acc, []).append(dict(row))

            for acc_id, txs in grouped.items():
                payload = {
                    "subject_account": acc_id,
                    "batch_records": [
                        {
                            "tx_id": t["tx_id"],
                            "amount": float(t["amount"]),
                            "rail": t.get("rail", "IMPS"),
                            "recipient": t.get("recipient", "UNKNOWN"),
                            "device": t.get("device_id", "DEV-UNKNOWN"),
                            "time": t.get("created_at"),
                        }
                        for t in txs
                    ],
                }

                try:
                    resp = requests.post(f"{self.api_base}/v1/screen", json=payload, timeout=120)
                    if resp.status_code == 200:
                        tx_ids = [t["id"] for t in txs]
                        placeholders = ",".join("?" for _ in tx_ids)
                        cursor.execute(
                            f"UPDATE {self.table} SET screening_status = 'SCREENED' WHERE id IN ({placeholders})",
                            tx_ids,
                        )
                        conn.commit()
                        scanned_count += len(txs)
                        poller_telemetry["total_scanned"] += len(txs)
                        poller_telemetry["last_error"] = None
                    else:
                        poller_telemetry["last_error"] = f"HTTP {resp.status_code} on {acc_id}"
                except requests.exceptions.ConnectionError:
                    poller_telemetry["last_error"] = f"Backend unreachable at {self.api_base}"
                except Exception as e:
                    poller_telemetry["last_error"] = str(e)

            conn.close()
            return scanned_count

        except Exception as e:
            poller_telemetry["last_error"] = str(e)
            return 0
        finally:
            poller_telemetry["status"] = "IDLE"

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


def run_single_scan(db_path: Optional[str] = None, table: Optional[str] = None) -> int:
    poller = get_poller(db_path=db_path, table=table)
    return poller.run_single_scan()


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
