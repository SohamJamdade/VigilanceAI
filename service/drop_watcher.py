"""
service/drop_watcher.py — Real-Time Transaction Ingestion Drop Watcher
Monitors 'data/drop/' for incoming transaction batch files (CSV or JSON),
validates schema integrity, and ingests them into core_banking.db.
"""
import os
import sys
import time
import json
import sqlite3
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from service.db_utils import get_db_connection

WATCH_DIR = os.path.join(BASE_DIR, "data", "drop")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(BASE_DIR, "core_banking.db")

os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

REQUIRED_COLUMNS = {"tx_id", "account_id", "amount", "rail", "recipient", "device_id"}


class TransactionBatchHandler(FileSystemEventHandler):
    def on_created(self, event):
        self._handle_event(event)

    def on_modified(self, event):
        self._handle_event(event)

    def _handle_event(self, event):
        if event.is_directory or not (event.src_path.endswith('.csv') or event.src_path.endswith('.json')):
            return

        file_path = event.src_path
        if not os.path.exists(file_path):
            return

        # Ensure file has finished being written by checking size stability
        last_size = -1
        for _ in range(5):
            try:
                curr_size = os.path.getsize(file_path)
                if curr_size > 0 and curr_size == last_size:
                    break
                last_size = curr_size
            except OSError:
                pass
            time.sleep(0.3)

        if last_size <= 0:
            return

        print(f"[*] Processing transaction batch: {os.path.basename(file_path)}")
        try:
            self.process_file(file_path)
            dest_path = os.path.join(PROCESSED_DIR, os.path.basename(file_path))
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.replace(file_path, dest_path)
            print(f"[+] Successfully ingested and archived to: {dest_path}")
        except Exception as e:
            print(f"[!] Ingestion error on {file_path}: {e}")

    def process_file(self, file_path):
        df = None
        last_err = None
        for attempt in range(3):
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data if isinstance(data, list) else data.get("transactions", []))
                break
            except (PermissionError, OSError) as e:
                last_err = e
                time.sleep(0.5)

        if df is None:
            raise PermissionError(f"Could not read {file_path} after 3 attempts: {last_err}")

        if not REQUIRED_COLUMNS.issubset(set(df.columns)):
            missing = REQUIRED_COLUMNS - set(df.columns)
            raise ValueError(f"Batch file is missing required columns: {missing}")

        conn = get_db_connection(DB_PATH)
        try:
            cursor = conn.cursor()

            # Handle created_at vs timestamp column dynamically
            time_col = 'timestamp' if 'timestamp' in df.columns else ('created_at' if 'created_at' in df.columns else None)

            inserted_count = 0
            for _, row in df.iterrows():
                ts_val = str(row[time_col]) if time_col else time.strftime('%Y-%m-%dT%H:%M:%SZ')
                cursor.execute("""
                    INSERT OR IGNORE INTO transactions 
                    (account_id, tx_id, amount, rail, recipient, device_id, location, created_at, screening_status, currency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row['account_id']),
                    str(row['tx_id']),
                    float(row['amount']),
                    str(row.get('rail', 'UPI')),
                    str(row.get('recipient', 'UNKNOWN')),
                    str(row.get('device_id', 'DEV-DEFAULT')),
                    str(row.get('location', 'DOMESTIC')),
                    ts_val,
                    'PENDING',
                    str(row.get('currency', 'INR'))
                ))
                if cursor.rowcount > 0:
                    inserted_count += 1

            conn.commit()
        finally:
            conn.close()
        print(f"[+] Ingested {inserted_count} new transactions into {DB_PATH}")


def start_watcher():
    print(f"[*] VigilanceAI Ingestion Watcher active on '{WATCH_DIR}'")
    print(f"[*] Processed batches will archive to '{PROCESSED_DIR}'")
    event_handler = TransactionBatchHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping Watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher()