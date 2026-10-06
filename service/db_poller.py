import os
import sys
import time
import json
import sqlite3
import threading
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from service.app import screen_account_hybrid, load_slm
from service.cases import init_case_storage, DB_FILE

DB_PATH = os.path.join(BASE_DIR, "core_banking.db")
AUDIT_DB_PATH = os.path.join(BASE_DIR, "audit_log.db")
DEFAULT_CONFIG = os.path.join(BASE_DIR, "db_credentials.json")

poller_telemetry: Dict[str, Any] = {
    "status": "IDLE",
    "last_scan": None,
    "next_scan": None,
    "total_scanned": 0,
    "last_error": None,
    "db_path": DB_PATH,
    "table": "transactions",
}


def init_audit_db(audit_db_file: str = AUDIT_DB_PATH):
    init_case_storage(audit_db_file)
    with sqlite3.connect(audit_db_file, timeout=5.0) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                typology TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                narrative TEXT NOT NULL,
                evidence JSON,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()



def poll_and_screen(db_path: str = DB_PATH, table: str = "transactions", audit_db_path: str = AUDIT_DB_PATH) -> int:
    resolved_db = os.path.abspath(db_path)
    if not os.path.exists(resolved_db):
        poller_telemetry["last_error"] = f"Database not found at {resolved_db}"
        return 0

    init_audit_db(audit_db_path)
    scanned_count = 0

    try:
        with sqlite3.connect(resolved_db, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Verify table existence
            table_check = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
            ).fetchone()
            if not table_check:
                poller_telemetry["last_error"] = f"Table '{table}' does not exist in {resolved_db}"
                return 0

            # Ensure screening_status column exists in database table
            col_info = cur.execute(f"PRAGMA table_info('{table}')").fetchall()
            col_names = [c[1] for c in col_info]
            if "screening_status" not in col_names:
                cur.execute(f"ALTER TABLE '{table}' ADD COLUMN screening_status TEXT DEFAULT 'PENDING'")
                conn.commit()

            # Query pending transactions from core_banking table
            cur.execute(f"""
                SELECT * FROM '{table}' 
                WHERE screening_status = 'PENDING' OR screening_status IS NULL 
                LIMIT 100
            """)
            rows = cur.fetchall()

            if not rows:
                print("[*] Poller: No PENDING transactions found. Waiting...", flush=True)
                poller_telemetry["last_scan"] = datetime.now().strftime("%H:%M:%S")
                poller_telemetry["status"] = "IDLE (0 pending records)"
                poller_telemetry["last_error"] = None
                return 0

            print(f"[*] Poller: Found {len(rows)} pending transactions. Grouping by account...", flush=True)

            raw_dict = [dict(r) for r in rows]
            cols = list(raw_dict[0].keys())
            df = pd.DataFrame(raw_dict, columns=cols)

            account_col = "account_id" if "account_id" in df.columns else cols[1]

            for account_id, group in df.groupby(account_col):
                tx_list = group.to_dict(orient="records")
                acc_str = str(account_id)
                print(f"[*] Screening account {acc_str} with {len(tx_list)} transactions...", flush=True)

                # Run hybrid evaluation via 130M SLM engine
                result = screen_account_hybrid(account_id=acc_str, transactions=tx_list)

                # Write alert to audit_log.db
                with sqlite3.connect(audit_db_path, timeout=5.0) as audit_conn:
                    a_cur = audit_conn.cursor()
                    a_cur.execute("""
                        INSERT INTO alerts (account_id, risk_level, typology, recommended_action, narrative, evidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        acc_str,
                        result.get("risk_level", "LOW"),
                        result.get("primary_typology", "NORMAL_ACTIVITY"),
                        result.get("recommended_action", "AUTO_CLEAR"),
                        result.get("narrative", ""),
                        json.dumps(result.get("supporting_evidence", []))
                    ))
                    audit_conn.commit()

                # Update screening_status in core_banking.db
                tx_ids = [t.get("tx_id") or t.get("id") for t in tx_list if (t.get("tx_id") or t.get("id"))]
                if tx_ids:
                    placeholders = ",".join("?" for _ in tx_ids)
                    if "tx_id" in group.columns and group["tx_id"].notna().any():
                        cur.execute(f"UPDATE '{table}' SET screening_status = 'SCREENED' WHERE tx_id IN ({placeholders})", tx_ids)
                    else:
                        cur.execute(f"UPDATE '{table}' SET screening_status = 'SCREENED' WHERE id IN ({placeholders})", tx_ids)
                    conn.commit()

                scanned_count += len(tx_list)
                print(f"[+] Screened {acc_str} -> Result: {result.get('risk_level')} ({result.get('primary_typology')})", flush=True)

        poller_telemetry["total_scanned"] += scanned_count
        poller_telemetry["last_scan"] = datetime.now().strftime("%H:%M:%S")
        poller_telemetry["status"] = "IDLE (Last Scan Successful)"
        poller_telemetry["last_error"] = None
        return scanned_count

    except Exception as e:
        print(f"[!] Error in poller cycle: {e}", flush=True)
        poller_telemetry["last_error"] = str(e)
        return 0


def run_single_scan(db_path: str = DB_PATH, table: str = "transactions") -> int:
    return poll_and_screen(db_path=db_path, table=table)


class AutomatedDBPoller:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        db_path: Optional[str] = None,
        table: Optional[str] = None,
        poll_interval: Optional[int] = None,
    ):
        cfg = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                poller_telemetry["last_error"] = f"Failed to read credentials: {e}"

        self.db_path = db_path or cfg.get("database") or cfg.get("db_path") or DB_PATH
        self.table = table or cfg.get("table") or "transactions"
        self.poll_interval = poll_interval or cfg.get("poll_interval_seconds") or 3

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


if __name__ == "__main__":
    print("[*] Starting VigilanceAI Database Screening Poller...", flush=True)
    init_case_storage(AUDIT_DB_PATH)
    init_audit_db(AUDIT_DB_PATH)
    load_slm()
    while True:
        try:
            poll_and_screen()
        except Exception as e:
            print(f"[!] Error in poller cycle: {e}", flush=True)

        time.sleep(3)
