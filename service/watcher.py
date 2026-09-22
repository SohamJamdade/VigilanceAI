"""
watcher.py — Folder Watcher for Automatic Transaction Ingestion

Monitors the 'incoming/' folder for new CSV/Excel files.
When a new file appears:
  1. Parses it into JSON payloads
  2. POSTs each payload to /v1/screen
  3. Moves the file to 'processed/'

Run with:
    python -m service.watcher

The banking manager just drops a file into incoming/ — everything else is automatic.
"""

import os
import sys
import time
import shutil
import json
import requests
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from service.ingest import file_to_payloads

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
API_BASE = os.getenv("VIGILANCE_API_URL", "http://localhost:8000")
INCOMING_DIR = os.getenv("VIGILANCE_INCOMING", "incoming")
PROCESSED_DIR = os.getenv("VIGILANCE_PROCESSED", "processed")
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def ensure_dirs():
    """Create incoming/ and processed/ directories if they don't exist."""
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_file(file_path: str):
    """
    Processes a single transaction file:
    1. Parse into payloads
    2. Screen each account
    3. Move to processed/
    """
    file_name = os.path.basename(file_path)
    print(f"\n{'='*60}")
    print(f"📁 New file detected: {file_name}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        # Step 1: Parse
        payloads = file_to_payloads(file_path)
        print(f"✅ Parsed {len(payloads)} account(s) from {file_name}")

        # Step 2: Screen each account
        for payload in payloads:
            account = payload["subject_account"]
            tx_count = len(payload["batch_records"])
            print(f"\n   🔍 Screening {account} ({tx_count} transactions)...")

            try:
                response = requests.post(
                    f"{API_BASE}/v1/screen",
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    decision = result.get("decision", {})
                    risk = decision.get("risk_level", "UNKNOWN")
                    typology = decision.get("primary_typology", "UNKNOWN")
                    action = decision.get("recommended_action", "UNKNOWN")
                    latency = result.get("latency_ms", 0)

                    risk_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                    icon = risk_icons.get(risk, "⚪")

                    print(f"   {icon} {account}: {risk} | {typology} | {action} | {latency:.0f}ms")
                else:
                    print(f"   ❌ {account}: HTTP {response.status_code}")

            except requests.exceptions.ConnectionError:
                print(f"   ❌ Cannot connect to API at {API_BASE}")
                print(f"      Make sure the backend is running: uvicorn service.app:app --port 8000")
                return  # Don't move file if API is down
            except Exception as e:
                print(f"   ❌ Error screening {account}: {e}")

        # Step 3: Move to processed/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_name = f"{timestamp}_{file_name}"
        dest_path = os.path.join(PROCESSED_DIR, dest_name)
        shutil.move(file_path, dest_path)
        print(f"\n📦 Moved to: {dest_path}")

    except ValueError as e:
        print(f"❌ Parsing error: {e}")
        # Move bad files to processed/ with an error prefix
        error_dest = os.path.join(PROCESSED_DIR, f"ERROR_{file_name}")
        shutil.move(file_path, error_dest)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


class TransactionFileHandler(FileSystemEventHandler):
    """Watches for new files in the incoming/ directory."""

    def on_created(self, event):
        if event.is_directory:
            return

        ext = os.path.splitext(event.src_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return

        # Small delay to ensure file is fully written
        time.sleep(1)
        process_file(event.src_path)


def main():
    ensure_dirs()

    print("=" * 60)
    print("🛡️  VigilanceAI — Transaction File Watcher")
    print("=" * 60)
    print(f"📂 Watching folder : {os.path.abspath(INCOMING_DIR)}")
    print(f"📦 Processed folder: {os.path.abspath(PROCESSED_DIR)}")
    print(f"🌐 API endpoint    : {API_BASE}")
    print(f"📄 Supported files : {', '.join(SUPPORTED_EXTENSIONS)}")
    print("=" * 60)
    print("\nDrop CSV/Excel files into the incoming/ folder to auto-screen.\n")
    print("Press Ctrl+C to stop.\n")

    # Process any files already in incoming/
    for f in os.listdir(INCOMING_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            process_file(os.path.join(INCOMING_DIR, f))

    # Start watching
    observer = Observer()
    observer.schedule(TransactionFileHandler(), INCOMING_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Watcher stopped.")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
