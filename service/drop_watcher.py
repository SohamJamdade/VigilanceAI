"""
service/drop_watcher.py — Real-Time Transaction Ingestion Drop Watcher
Monitors the 'data/drop/' folder for incoming transaction batch files, 
validates and parses records, and pushes them to core_banking.db.
"""
import os
import sys
import time
import json
import sqlite3
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

WATCH_DIR ="data/drop/" 
PROCESSED_DIR = "data/processed/"
DB_PATH = "core_banking.db"

os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class TransactionsBatchHandler(FileSystemEventHandler):
    """Handles new transaction batch files dropped into the watch directory."""
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(('.csv', '.json')):
            return

        file_path = event.src_path
        print(f"Detected new transaction batch: {file_path}")
        time.sleep(0.5)  # Allow file write to complete

        try:
            self .process_file(file_path)
            dest_path =os.path.join(PROCESSED_DIR, os.path.basename(file_path))
            os.replace (file_path, dest_path)
            print(f"Successfully ingested and moved to: {dest_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def process_file(self, file_path):
        if file_path .endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            with open(file_path , 'r' , encoding ='utf-8') as f:
                data = json .load(f)
            df = pd.DataFrame(data if isinstance(data, list) else data.get('transactions', []))

        required_cols = {"tx_id", "account_id", "amount", "currency", "rail", "recipient", "device_id", "timestamp"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns. Required: {required_cols}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert records into core_banking.db
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO transactions 
                (tx_id, account_id, amount, currency, rail, recipient, device_id, timestamp, location, source_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row['tx_id']),
                str(row['account_id']),
                float(row['amount']),
                str(row.get('currency', 'INR')),
                str(row.get('rail', 'UPI')),
                str(row.get('recipient', 'UNKNOWN')),
                str(row.get('device_id', 'DEV-DEFAULT')),
                str(row['timestamp']),
                str(row.get('location', 'DOMESTIC')),
                "FILE_DROP_INGEST"
            ))

        conn.commit()
        conn.close()
        print(f"[+] Ingested {len(df)} transactions into {DB_PATH}")

def start_watcher():
    print(f"[*] Starting Drop-Folder Watcher on '{WATCH_DIR}'...")
    event_handler = TransactionsBatchHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()        