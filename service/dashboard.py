import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import sqlite3
import json
import os
import sys
import time
import uuid
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from service.ingest import read_transaction_bytes
from service.db_poller import (
    start_poller, stop_poller, is_running, run_single_scan, poller_telemetry
)
from service.cases import (
    log_upload_batch, get_upload_history, delete_upload_batch,
    purge_all_records, DB_FILE
)

API_BASE = os.getenv("VIGILANCE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="VigilanceAI — Compliance Workstation", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("VigilanceAI")
    st.caption("Multi-Stage Risk Engine v2.0")
    st.markdown("---")
    st.markdown("""
    **Active Layers:**
    - 📥 Normalization & Hashing
    - 📐 Deterministic Feature Engine
    - ⚖️ Compliance Rule Engine
    - 🕸️ Entity Linkage Resolver
    - 🧠 15.2M INT8 PyTorch SLM
    - 🗄️ Immutable Case Store
    """)
    st.markdown("---")
    if is_running():
        st.success("🔄 DB Poller: ACTIVE")
    else:
        st.info("⏸️ DB Poller: INACTIVE")

tab_upload, tab_cases, tab_chat, tab_metrics, tab_poller = st.tabs([
    "📁 Upload & Ingest", "📂 Case Manager", "💬 Compliance Chat", "📊 Risk Telemetry", "🔄 Auto-DB Sync"
])

# TAB 1: Upload & Ingest with Upload History and Batch Deletion
with tab_upload:
    st.header("Transaction Ingestion")
    uploaded_file = st.file_uploader("Upload raw CSV or Excel transaction file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        txs = read_transaction_bytes(uploaded_file.getvalue(), uploaded_file.name)
        st.success(f"Parsed and validated {len(txs)} transactions.")

        if st.button("🚀 Process Transactions Through Pipeline", type="primary"):
            progress = st.progress(0, text="Executing risk pipeline...")

            # Group transactions by account ID
            grouped: dict[str, list] = {}
            for t in txs:
                grouped.setdefault(t.account_id, []).append(t)

            total = len(grouped)
            success_count = 0
            batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

            for idx, (acc, acc_txs) in enumerate(grouped.items()):
                progress.progress((idx + 1) / total, text=f"Evaluating Account {acc}...")

                acc_amounts = [t.amount for t in acc_txs]
                dyn_median = float(pd.Series(acc_amounts).median()) if acc_amounts else 0.0
                dyn_min = float(min(acc_amounts)) if acc_amounts else 0.0
                dyn_max = float(max(acc_amounts)) if acc_amounts else 0.0

                payload = {
                    "subject_account": acc,
                    "batch_id": batch_id,
                    "batch_records": [
                        {
                            "tx_id": t.tx_id,
                            "amount": t.amount,
                            "rail": t.rail,
                            "recipient": t.recipient,
                            "device": t.device_id,
                            "location": t.location
                        } for t in acc_txs
                    ],
                    "account_baseline": {
                        "account_id": acc,
                        "historical_median": dyn_median,
                        "typical_bracket": [dyn_min, dyn_max]
                    }
                }
                try:
                    res = requests.post(f"{API_BASE}/v1/screen", json=payload, timeout=120)
                    if res.status_code == 200:
                        success_count += 1
                    else:
                        st.error(f"Failed for {acc}: HTTP {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Error on account {acc}: {e}")

            progress.empty()

            if success_count > 0:
                log_upload_batch(batch_id, uploaded_file.name, len(txs), list(grouped.keys()), DB_FILE)
                st.success(f"Batch screening complete! {success_count} account case(s) created.")
                st.rerun()
            else:
                st.error("Batch screening finished, but no cases were created.")

    # Upload History and Record Deletion Section
    st.divider()
    st.subheader("📂 Upload History & Batch Cleanup")
    st.caption("Shows which file was uploaded when. Click 'Delete' on any batch to purge its records from SQLite.")

    history = get_upload_history(DB_FILE)
    if not history:
        st.info("No file uploads recorded yet.")
    else:
        for batch in history:
            with st.container(border=True):
                col_info, col_stats, col_btn = st.columns([3, 2, 1])
                with col_info:
                    st.markdown(f"📄 **{batch['filename']}**")
                    st.caption(f"Batch ID: `{batch['batch_id']}` • Uploaded: `{batch['uploaded_at']}`")
                with col_stats:
                    acc_preview = ", ".join(batch["account_ids"][:3])
                    if len(batch["account_ids"]) > 3:
                        acc_preview += f" +{len(batch['account_ids']) - 3} more"
                    st.write(f"**{batch['tx_count']}** transactions | **{batch['account_count']}** account(s)")
                    st.caption(f"Accounts: `{acc_preview}`")
                with col_btn:
                    if st.button("🗑️ Delete", key=f"del_{batch['batch_id']}", use_container_width=True):
                        res = delete_upload_batch(batch["batch_id"], DB_FILE)
                        st.success(f"Deleted records for `{batch['filename']}` ({res.get('accounts_purged', 0)} accounts purged).")
                        st.rerun()

    # Advanced Database Maintenance
    with st.expander("🧹 Wipe All Records (Clean Slate)"):
        st.warning("Permanently removes all test cases, screening audits, and upload history.")
        confirm_wipe = st.checkbox("I confirm I want to wipe all records from the database", key="wipe_chk")
        if st.button("Wipe Entire Database", type="primary", disabled=not confirm_wipe):
            purge_all_records(DB_FILE)
            st.success("All records wiped. Database is clean!")
            st.rerun()

# TAB 2: Case Manager (Fetches from API with SQLite fallback)
with tab_cases:
    st.header("Investigative Cases")
    cases_list = []

    # 1. Fetch cases from REST API
    try:
        r = requests.get(f"{API_BASE}/v1/cases?limit=50", timeout=5)
        if r.status_code == 200:
            cases_list = r.json().get("cases", [])
    except Exception:
        pass

    # 2. Fallback to local SQLite if API is offline
    if not cases_list and os.path.exists(DB_FILE):
        try:
            with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT case_id, created_at, account_id, risk_level, deterministic_score,
                           primary_typology, recommended_action, model_reasoning,
                           features_json, triggered_rules_json
                    FROM risk_cases ORDER BY created_at DESC LIMIT 50
                """)
                cases_list = [dict(row) for row in cursor.fetchall()]
        except Exception:
            pass

    if cases_list:
        st.caption(f"Displaying {len(cases_list)} investigative dossier(s):")
        for row in cases_list:
            risk_color = {
                "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
            }.get(row['risk_level'], "⚪")

            with st.expander(f"{risk_color} {row['case_id']} | {row['account_id']} | Risk: {row['risk_level']}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Deterministic Risk Score", f"{float(row['deterministic_score']):.2f}")
                c2.write(f"**Primary Typology:** `{row['primary_typology']}`")
                c3.write(f"**Recommended Action:** `{row['recommended_action']}`")

                st.markdown("**Deterministic Triggered Rules:**")
                raw_rules = row['triggered_rules_json']
                rules_data = json.loads(raw_rules) if isinstance(raw_rules, str) else (raw_rules or [])
                triggered = [r for r in rules_data if r.get('triggered')]
                if triggered:
                    for r in triggered:
                        st.error(f"🚨 **{r.get('rule_name', 'Rule')}** (`{r.get('rule_id', '')}`): {r.get('reason', '')}")
                else:
                    st.info("No deterministic rules triggered.")

                st.markdown("**SLM Contextual Reasoning & Evidence:**")
                st.write(row.get('model_reasoning') or "Evaluation complete.")
    else:
        st.info("No cases yet. Process a batch of transactions to create the first one.")

# TAB 3: Compliance Chat
with tab_chat:
    st.header("Compliance Assistant")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "👋 Ask me about any specific account (e.g. *'Why is account 30412 flagged?'*) or request a risk overview."}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about account risk, evidence, or system status...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                res = requests.post(f"{API_BASE}/v1/chat", json={"query": prompt}, timeout=15)
                answer = res.json().get("response", "No response returned.")
            except Exception as e:
                answer = f"Assistant offline: {e}"
            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# TAB 4: Risk Telemetry (Fetches from API with SQLite fallback)
with tab_metrics:
    st.header("Risk Telemetry")
    counts_data = []

    try:
        r = requests.get(f"{API_BASE}/v1/telemetry", timeout=5)
        if r.status_code == 200:
            counts_data = r.json().get("counts", [])
    except Exception:
        pass

    if not counts_data and os.path.exists(DB_FILE):
        try:
            with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
                df_counts = pd.read_sql("SELECT risk_level, COUNT(*) as count FROM screening_audit GROUP BY risk_level", conn)
                counts_data = df_counts.to_dict(orient="records")
        except Exception:
            pass

    if counts_data:
        counts_df = pd.DataFrame(counts_data)
        fig = px.pie(
            counts_df, names="risk_level", values="count", hole=0.4,
            title="Risk Level Distribution",
            color="risk_level",
            color_discrete_map={"LOW": "#00CC66", "MEDIUM": "#FFA500", "HIGH": "#FF4B4B", "CRITICAL": "#DC143C"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No screening data yet. Process a batch to see the distribution.")

# TAB 5: Auto-DB Sync (Poller)
with tab_poller:
    st.header("🔄 Automated Database Polling")
    st.markdown(
        "Connect to a core banking database and automatically screen new transactions every 60 seconds. "
        "Upload a `db_credentials.json` or enter the path to a local SQLite database."
    )

    st.divider()

    st.subheader("Option A: Upload Credentials File")
    cred_file = st.file_uploader(
        "Upload `db_credentials.json`",
        type=["json"],
        help='JSON file with "database" and "table" keys'
    )

    db_path_from_creds = None
    table_from_creds = "transactions"
    if cred_file is not None:
        try:
            creds = json.loads(cred_file.getvalue())
            db_path_from_creds = creds.get("database") or creds.get("db_path")
            table_from_creds = creds.get("table", "transactions")
            if db_path_from_creds:
                st.success(f"Credentials loaded: DB=`{db_path_from_creds}`, Table=`{table_from_creds}`")
            else:
                st.error("JSON must contain 'database' or 'db_path'.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.subheader("Option B: Configure Target Database")
    col_path, col_tbl = st.columns([3, 1])
    with col_path:
        manual_path = st.text_input(
            "SQLite database path",
            value=db_path_from_creds or "core_banking.db",
            help="Path to the SQLite banking database"
        )
    with col_tbl:
        target_table = st.text_input(
            "Table name",
            value=table_from_creds,
            help="Table containing pending transactions"
        )

    target_db = manual_path

    st.divider()

    col_start, col_scan, col_stop = st.columns(3)
    with col_start:
        if st.button("▶️ Start 60s Poller", type="primary", disabled=is_running(), use_container_width=True):
            if not target_db:
                st.error("Provide a database path first.")
            elif not os.path.exists(target_db):
                st.error(f"Database file not found: `{target_db}`")
            else:
                started = start_poller(target_db, table=target_table)
                if started:
                    st.success(f"Poller active! Scanning `{target_db}`:`{target_table}` every 60s.")
                else:
                    st.warning("Poller is already running.")
                st.rerun()

    with col_scan:
        if st.button("⚡ Run Single Scan Now", use_container_width=True):
            if not target_db or not os.path.exists(target_db):
                st.error(f"Database `{target_db}` not found.")
            else:
                with st.spinner("Scanning database for pending transactions..."):
                    scanned = run_single_scan(target_db, table=target_table)
                if scanned > 0:
                    st.success(f"Single scan complete! Evaluated and screened {scanned} new transaction(s). Check Case Manager tab!")
                else:
                    st.info(f"0 pending transactions found in `{target_db}`:`{target_table}`. All records are already SCREENED. Run `python simulate_bank_db.py` to insert new pending transactions.")
                st.rerun()

    with col_stop:
        if st.button("⏹️ Stop Poller", disabled=not is_running(), use_container_width=True):
            stop_poller()
            st.info("Poller stopped.")
            st.rerun()

    st.divider()
    st.subheader("📡 Poller Telemetry")

    t = poller_telemetry
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Status", t["status"])
    tc2.metric("Total Scanned", t["total_scanned"])
    tc3.metric("Last Scan", t["last_scan"] or "—")
    tc4.metric("Next Scan", t["next_scan"] or "—")

    if t.get("db_path"):
        st.info(f"Target database: `{t['db_path']}` (Table: `{t.get('table', 'transactions')}`)")
    if t.get("last_error"):
        st.error(f"Last error: {t['last_error']}")

    if is_running():
        st.caption("Dashboard auto-refreshes every 30 seconds while poller is active.")
        time.sleep(0.1)
        st.rerun() if st.button("🔃 Refresh Now") else None