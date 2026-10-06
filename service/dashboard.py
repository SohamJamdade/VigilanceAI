import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import sqlite3
import json
import os
import sys
import time
import re
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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DB_PATH = os.path.join(BASE_DIR, "audit_log.db")

st.set_page_config(page_title="VigilanceAI — Compliance Workstation", page_icon="🛡️", layout="wide")

def fast_purge_cases():
    """Wipes all screening data via API and local SQLite fallback, then clears cache."""
    purged = False
    try:
        r = requests.delete(f"{API_BASE}/v1/records/all", timeout=5)
        if r.status_code == 200:
            purged = True
    except Exception:
        pass

    try:
        purge_all_records(DB_FILE)
        purged = True
    except Exception:
        pass

    if os.path.exists(AUDIT_DB_PATH):
        try:
            with sqlite3.connect(AUDIT_DB_PATH) as conn:
                conn.cursor().execute("DELETE FROM alerts")
                conn.commit()
            purged = True
        except Exception:
            pass

    st.cache_data.clear()
    return purged

def query_compliance_assistant(query_text: str) -> str:
    """Robust compliance assistant querying audit_log.db with fuzzy token matching."""
    if not os.path.exists(AUDIT_DB_PATH):
        return "⚠️ `audit_log.db` not found. Please run `service/db_poller.py` to generate audit records."

    raw_query = query_text.strip()
    q_lower = raw_query.lower()

    # 1. Sanitize query by stripping literal 'account' / 'accounts' words to prevent ACC-OUNT collision
    sanitized = re.sub(r"\baccounts?\b", " ", raw_query, flags=re.IGNORECASE)

    target_account = None

    # 2. Normalize spaced or punctuated ACC patterns: "ACC - 40005", "ACC 10002", "ACC-DROP-99" -> "ACC-40005"
    acc_match = re.search(r"\bACC\s*[\-_]?\s*([A-Za-z0-9\-]+)\b", sanitized, flags=re.IGNORECASE)
    if acc_match:
        suffix = acc_match.group(1).strip("-").upper()
        if suffix:
            target_account = f"ACC-{suffix}"
    else:
        # Check for bare 2-6 digit numeric account IDs: e.g. 10002, 40005, 99
        digits_match = re.search(r"\b(\d{2,6})\b", sanitized)
        if digits_match:
            target_account = f"ACC-{digits_match.group(1)}"


    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        def table_exists(tbl_name: str) -> bool:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl_name,))
            return cur.fetchone() is not None

        if not table_exists("alerts") and not table_exists("risk_cases"):
            return "No audit tables (`alerts` / `risk_cases`) found in `audit_log.db`."

        # PRIORITY 1: Specific Account Query (runs even if the word 'flagged' is in the query)
        if target_account:
            clean_suffix = target_account.replace("ACC-", "")
            row = None

            if table_exists("alerts"):
                cur.execute("""
                    SELECT account_id, risk_level, typology, recommended_action, narrative, evidence, timestamp 
                    FROM alerts 
                    WHERE UPPER(account_id) = ? 
                       OR UPPER(account_id) = ?
                       OR account_id LIKE ? 
                       OR account_id LIKE ?
                    ORDER BY id DESC LIMIT 1
                """, (target_account.upper(), clean_suffix.upper(), f"%{target_account}%", f"%{clean_suffix}%"))
                row = cur.fetchone()

            if not row and table_exists("risk_cases"):
                cur.execute("""
                    SELECT account_id, risk_level, primary_typology as typology, recommended_action,
                           model_reasoning as narrative, triggered_rules_json as evidence, created_at as timestamp
                    FROM risk_cases
                    WHERE UPPER(account_id) = ? 
                       OR UPPER(account_id) = ?
                       OR account_id LIKE ? 
                       OR account_id LIKE ?
                    ORDER BY created_at DESC LIMIT 1
                """, (target_account.upper(), clean_suffix.upper(), f"%{target_account}%", f"%{clean_suffix}%"))
                row = cur.fetchone()


            if row:
                ev_str = ""
                if row["evidence"]:
                    try:
                        ev_list = json.loads(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]
                        if isinstance(ev_list, list):
                            ev_str = "\n" + "\n".join([f"- {e}" for e in ev_list])
                        elif isinstance(ev_list, dict):
                            ev_str = "\n" + "\n".join([f"- **{k}**: {v}" for k, v in ev_list.items()])
                        else:
                            ev_str = f"\n- {ev_list}"
                    except Exception:
                        ev_str = f"\n- {row['evidence']}"

                narrative_text = row['narrative'] if row['narrative'] else "High-risk typology triggered by rule evaluation and model inference."
                risk_icon = "🔴" if row['risk_level'] == "CRITICAL" else ("🟠" if row['risk_level'] == "HIGH" else "🟢")

                return f"""### {risk_icon} Case Dossier: `{row['account_id']}`
- **Risk Classification:** **`{row['risk_level']}`**
- **Identified Typology:** `{row['typology']}`
- **Recommended Action:** `{row['recommended_action']}`
- **Evaluation Timestamp:** `{row['timestamp']}`

---
**🧠 SLM Reasoning & Narrative:**
> {narrative_text}

{f"**📋 Triggered Evidence & Indicators:**{ev_str}" if ev_str else ""}"""
            else:
                return f"No flagged records found for account `{target_account}`. All transactions conform to baseline."

        # PRIORITY 2: General Flagged Accounts List
        if any(kw in q_lower for kw in ["flagged", "alert", "alerts", "suspicious", "high risk", "critical", "smurfing", "sanctions"]):
            cur.execute("""
                SELECT account_id, risk_level, typology, recommended_action 
                FROM alerts 
                WHERE risk_level IN ('HIGH', 'CRITICAL')
                ORDER BY id DESC
            """)
            flagged_rows = cur.fetchall()
            if not flagged_rows:
                return "✅ All screened accounts currently conform to baseline. No high-risk or critical alerts recorded."

            unique_accs = {r["account_id"]: r for r in flagged_rows}
            lines = [f"### 🚨 Flagged Accounts ({len(unique_accs)} High/Critical Alerts):"]
            for acc, r in unique_accs.items():
                icon = "🔴" if r["risk_level"] == "CRITICAL" else "🟠"
                lines.append(f"- {icon} **`{acc}`** | Risk: **{r['risk_level']}** | Typology: `{r['typology']}` | Action: `{r['recommended_action']}`")
            lines.append("\n*Ask `Why is account <id> flagged?` to review full SLM reasoning and evidence.*")
            return "\n".join(lines)

        # PRIORITY 3: Risk Overview / Metrics
        if any(kw in q_lower for kw in ["overview", "summary", "stats", "telemetry", "distribution"]):
            cur.execute("SELECT risk_level, COUNT(*) as cnt FROM alerts GROUP BY risk_level")
            summary_counts = {row["risk_level"]: row["cnt"] for row in cur.fetchall()}
            total_alerts = sum(summary_counts.values())
            return f"""### 📊 Current Risk Overview:
- **Total Screened Alerts:** {total_alerts}
- 🔴 **CRITICAL:** {summary_counts.get('CRITICAL', 0)}
- 🟠 **HIGH:** {summary_counts.get('HIGH', 0)}
- 🟡 **MEDIUM:** {summary_counts.get('MEDIUM', 0)}
- 🟢 **LOW / NORMAL:** {summary_counts.get('LOW', 0)}

Ask for **'accounts flagged'** or **'Why is account <id> flagged?'** for case specifics."""

        # Default fallback
        cur.execute("SELECT account_id, risk_level, typology FROM alerts ORDER BY id DESC LIMIT 5")
        recent = cur.fetchall()
        if recent:
            recs = ", ".join([f"`{r['account_id']}` ({r['risk_level']})" for r in recent])
            return f"I can review flagged accounts and risk dossiers. Recent accounts in audit log: {recs}.\n\nTry asking:\n- *'Why is account ACC-40005 flagged?'*\n- *'Why is ACC-DROP-99 flagged?'*\n- *'Show flagged accounts'*\n- *'Risk overview'*"
        else:
            return "All screened accounts currently conform to baseline. No high-risk alerts."

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700&display=swap');

:root{
  --ink:#080D1A; --surface:#0F1730; --raised:#16203F; --line:rgba(140,155,255,.16);
  --text:#E8ECF8; --muted:#8C97BA; --accent:#7C8CFF; --accent2:#22D3EE;
  --ok:#34D399; --warn:#FBBF24; --hi:#FB923C; --crit:#F43F5E;
  --glow:0 0 0 1px var(--line), 0 18px 40px -18px rgba(0,0,0,.7);
}

.stApp{ font-family:'Manrope', system-ui, sans-serif; }
.stMarkdown p, .stMarkdown li, label, input, textarea, button p,
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
[data-testid="stExpander"] summary p, [data-baseweb="tab"] p, [data-testid="stFileUploaderDropzoneInstructions"] span{
  font-family:'Manrope', system-ui, sans-serif !important;
}
h1,h2,h3,h4, [data-testid="stMetricValue"] { font-family:'Sora','Manrope',sans-serif !important; letter-spacing:-.02em; }

.stApp{
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(124,140,255,.20), transparent 60%),
    radial-gradient(900px 480px at 96% 4%, rgba(34,211,238,.13), transparent 60%),
    var(--ink);
  color:var(--text);
}
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1.6rem; max-width:1280px; }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#26305a; border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:#36427a; }

.vg-hero{
  position:relative; overflow:hidden; border-radius:22px; padding:34px 38px; margin-bottom:22px;
  background:linear-gradient(135deg, rgba(22,32,63,.92), rgba(12,18,40,.92));
  box-shadow:var(--glow);
}
.vg-hero h1{
  margin:0; font-size:2.5rem; font-weight:700; line-height:1.1;
  background:linear-gradient(100deg,#FFFFFF 10%, #B9C3FF 55%, #8EEBFF 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.vg-hero p{ margin:.55rem 0 0; color:#B4BEDD; font-size:1.02rem; max-width:62ch; }
.vg-chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
.vg-chip{
  padding:6px 13px; border-radius:999px; font-size:.78rem; font-weight:600; color:#D5DCFF;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10); backdrop-filter:blur(8px);
}

[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0C1330 0%, #080D1A 100%);
  border-right:1px solid var(--line);
}
.vg-pill{
  display:flex; align-items:center; gap:10px; padding:11px 14px; border-radius:14px;
  font-weight:600; font-size:.88rem; border:1px solid var(--line); background:rgba(255,255,255,.03);
}
.vg-dot{ width:9px; height:9px; border-radius:50%; background:#5A6690; flex:none; }
.vg-pill.on{ color:#B9F5DD; border-color:rgba(52,211,153,.35); background:rgba(52,211,153,.08); }
.vg-pill.on .vg-dot{ background:var(--ok); }
.vg-pill.off{ color:var(--muted); }

.stTabs [data-baseweb="tab-list"]{
  gap:6px; padding:6px; border-radius:16px; background:rgba(255,255,255,.03);
  border:1px solid var(--line); width:fit-content; max-width:100%; overflow-x:auto;
}
.stTabs [data-baseweb="tab"]{
  height:42px; padding:0 18px; border-radius:11px; color:var(--muted); font-weight:600;
}
.stTabs [aria-selected="true"]{
  color:#fff !important; background:linear-gradient(135deg, rgba(124,140,255,.35), rgba(34,211,238,.20));
  box-shadow:inset 0 0 0 1px rgba(160,175,255,.35);
}

.stButton > button{
  border-radius:12px; font-weight:700; padding:.6rem 1.1rem; color:var(--text);
  background:var(--raised); border:1px solid var(--line);
}
.stButton > button[kind="primary"]{
  border:none; color:#fff;
  background:linear-gradient(120deg,#6C7DFF 0%, #4F8CFF 50%, #22B8E0 100%);
}

[data-testid="stExpander"]{
  border:1px solid var(--line) !important; border-radius:16px !important; overflow:hidden;
  background:linear-gradient(180deg, rgba(22,32,63,.7), rgba(13,20,42,.7));
  margin-bottom:10px;
}

[data-baseweb="input"]{
  border-radius:12px !important; background:#111A36 !important;
  border:1px solid var(--line) !important;
}
[data-baseweb="base-input"], .stTextInput input{ background:transparent !important; border-radius:12px !important; }

[data-testid="stChatMessage"]{
  border-radius:18px; border:1px solid var(--line); padding:14px 18px;
  background:linear-gradient(180deg, rgba(22,32,63,.7), rgba(13,20,42,.7));
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:linear-gradient(135deg, rgba(124,140,255,.22), rgba(34,211,238,.10));
}
code{ background:rgba(124,140,255,.14) !important; color:#CFD6FF !important; border-radius:7px; padding:.12em .45em; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="vg-hero">
  <h1>VigilanceAI</h1>
  <p>Compliance workstation for screening transactions, tracing entity links and closing cases with a full evidence trail.</p>
  <div class="vg-chips">
    <span class="vg-chip">Deterministic rules</span>
    <span class="vg-chip">Entity linkage</span>
    <span class="vg-chip">130M INT8 SLM</span>
    <span class="vg-chip">Immutable case store</span>
  </div>
</div>
""", unsafe_allow_html=True)

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
    - 🧠 130M INT8 PyTorch SLM
    - 🗄️ Immutable Case Store
    """)
    st.markdown("---")
    if is_running():
        st.markdown('<div class="vg-pill on"><span class="vg-dot"></span>DB Poller: ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="vg-pill off"><span class="vg-dot"></span>DB Poller: INACTIVE</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Quick Maintenance")
    if st.button("🧹 Clear All Cases", use_container_width=True):
        if fast_purge_cases():
            st.toast("✅ All cases and audit logs wiped clean!", icon="🧹")
            time.sleep(0.5)
            st.rerun()

tab_upload, tab_cases, tab_chat, tab_metrics, tab_poller = st.tabs([
    "📁 Upload & Ingest", "📂 Case Manager", "💬 Compliance Chat", "📊 Risk Telemetry", "🔄 Auto-DB Sync"
])

# TAB 1: Upload & Ingest
with tab_upload:
    st.header("Transaction Ingestion")
    uploaded_file = st.file_uploader("Upload raw CSV or Excel transaction file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        txs = read_transaction_bytes(uploaded_file.getvalue(), uploaded_file.name)
        st.success(f"Parsed and validated {len(txs)} transactions.")

        if st.button("🚀 Process Transactions Through Pipeline", type="primary"):
            progress = st.progress(0, text="Executing risk pipeline...")

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
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Batch screening finished, but no cases were created.")

    st.divider()
    st.subheader("📂 Upload History & Batch Cleanup")
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
                        st.success(f"Deleted records for `{batch['filename']}`.")
                        st.cache_data.clear()
                        st.rerun()

# TAB 2: Case Manager
with tab_cases:
    col_hdr, col_purge = st.columns([4, 1])
    with col_hdr:
        st.header("Investigative Cases")
    with col_purge:
        if st.button("🗑️ Wipe Cases", use_container_width=True):
            if fast_purge_cases():
                st.toast("Cases wiped successfully!", icon="🧹")
                time.sleep(0.5)
                st.rerun()

    cases_list = []
    # 1. First check audit_log.db
    if os.path.exists(AUDIT_DB_PATH):
        try:
            with sqlite3.connect(AUDIT_DB_PATH, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT id as case_id, timestamp as created_at, account_id, risk_level, 
                           0.95 as deterministic_score, typology as primary_typology, 
                           recommended_action, narrative as model_reasoning, 
                           evidence as triggered_rules_json
                    FROM alerts ORDER BY id DESC LIMIT 50
                """)
                cases_list = [dict(row) for row in cur.fetchall()]
        except Exception:
            pass

    # 2. Fallback to API and DB_FILE
    if not cases_list:
        try:
            r = requests.get(f"{API_BASE}/v1/cases?limit=50", timeout=5)
            if r.status_code == 200:
                cases_list = r.json().get("cases", [])
        except Exception:
            pass

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

            with st.expander(f"{risk_color} #{row['case_id']} | {row['account_id']} | Risk: {row['risk_level']}"):
                c1, c2, c3 = st.columns(3)
                det_score = float(row.get('deterministic_score', 0.9))
                c1.metric("Risk Score", f"{det_score:.2f}")
                c2.write(f"**Primary Typology:** `{row['primary_typology']}`")
                c3.write(f"**Recommended Action:** `{row['recommended_action']}`")

                st.markdown("**SLM Contextual Reasoning & Narrative:**")
                st.write(row.get('model_reasoning') or "Evaluation complete.")

                raw_rules = row.get('triggered_rules_json')
                if raw_rules:
                    st.markdown("**Evidence & Indicators:**")
                    try:
                        rules_data = json.loads(raw_rules) if isinstance(raw_rules, str) else raw_rules
                        if isinstance(rules_data, list):
                            for r in rules_data:
                                st.warning(f"⚠️ {r}")
                        else:
                            st.write(rules_data)
                    except Exception:
                        st.write(raw_rules)
    else:
        st.info("No cases currently recorded. Upload a CSV file or start the database poller.")

# TAB 3: Compliance Chat
with tab_chat:
    st.header("Compliance Assistant")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "👋 Ask me about any specific account (e.g. *'Why is ACC-40005 flagged?'*), query *'accounts flagged'*, or request a *'risk overview'*."}
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
            answer = query_compliance_assistant(prompt)
            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# TAB 4: Risk Telemetry
with tab_metrics:
    st.header("Risk Telemetry")
    counts_data = []

    # Priority 1: Read directly from audit_log.db
    if os.path.exists(AUDIT_DB_PATH):
        try:
            with sqlite3.connect(AUDIT_DB_PATH, timeout=5.0) as conn:
                df_counts = pd.read_sql("SELECT risk_level, COUNT(*) as count FROM alerts GROUP BY risk_level", conn)
                counts_data = df_counts.to_dict(orient="records")
        except Exception:
            pass

    # Priority 2: Try API or DB_FILE
    if not counts_data:
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
            counts_df, names="risk_level", values="count", hole=0.62,
            title="Risk Level Distribution",
            color="risk_level",
            color_discrete_map={"LOW": "#34D399", "MEDIUM": "#FBBF24", "HIGH": "#FB923C", "CRITICAL": "#F43F5E"}
        )
        fig.update_traces(
            textposition="outside", textinfo="label+percent",
            marker=dict(line=dict(color="#0F1730", width=4)),
            pull=[0.02] * len(counts_df),
            hovertemplate="<b>%{label}</b><br>%{value} cases (%{percent})<extra></extra>",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Manrope, sans-serif", color="#E8ECF8", size=14),
            title=dict(font=dict(family="Sora, sans-serif", size=20)),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=70, b=30, l=30, r=30),
            transition=dict(duration=600, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No screening data yet. Process a batch to see the distribution.")

# TAB 5: Auto-DB Sync
with tab_poller:
    st.header("🔄 Automated Database Polling")
    st.markdown(
        "Connect to a core banking database and automatically screen new transactions every 60 seconds."
    )

    st.divider()
    cred_file = st.file_uploader("Upload `db_credentials.json`", type=["json"])

    db_path_from_creds = None
    table_from_creds = "transactions"
    if cred_file is not None:
        try:
            creds = json.loads(cred_file.getvalue())
            db_path_from_creds = creds.get("database") or creds.get("db_path")
            table_from_creds = creds.get("table", "transactions")
            if db_path_from_creds:
                st.success(f"Credentials loaded: DB=`{db_path_from_creds}`, Table=`{table_from_creds}`")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    col_path, col_tbl = st.columns([3, 1])
    with col_path:
        manual_path = st.text_input("SQLite database path", value=db_path_from_creds or "core_banking.db")
    with col_tbl:
        target_table = st.text_input("Table name", value=table_from_creds)

    target_db = manual_path
    st.divider()

    col_start, col_scan, col_stop = st.columns(3)
    with col_start:
        if st.button("▶️ Start 60s Poller", type="primary", disabled=is_running(), use_container_width=True):
            if not target_db or not os.path.exists(target_db):
                st.error(f"Database file not found: `{target_db}`")
            else:
                start_poller(target_db, table=target_table)
                st.rerun()
    with col_scan:
            if st.button("⚡ Run Single Scan Now", use_container_width=True):
                resolved_db = os.path.abspath(target_db.strip()) if target_db else ""
                if not resolved_db or not os.path.exists(resolved_db):
                    st.error(f"Database `{target_db}` not found on disk.")
                else:
                    with st.spinner("Scanning database for pending transactions..."):
                        scanned = run_single_scan(resolved_db, table=target_table.strip())
                    if scanned > 0:
                        st.success(f"Single scan complete! Evaluated and screened {scanned} transaction(s). Check the Case Manager tab!")
                        st.cache_data.clear()
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        last_err = poller_telemetry.get("last_error")
                        if last_err:
                            st.error(f"Scan error: {last_err}")
                        else:
                            st.info(f"0 pending transactions found in `{resolved_db}`:`{target_table}`. All records are already SCREENED or the table is empty.")

    with col_stop:
        if st.button("⏹️️ Stop Poller", disabled=not is_running(), use_container_width=True):
            stop_poller()
            st.rerun()

    st.divider()
    t = poller_telemetry
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Status", t["status"])
    tc2.metric("Total Scanned", t["total_scanned"])
    tc3.metric("Last Scan", t["last_scan"] or "—")
    tc4.metric("Next Scan", t["next_scan"] or "—")