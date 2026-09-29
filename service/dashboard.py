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

    st.cache_data.clear()
    return purged

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

[data-testid="stIconMaterial"], span[class*="material"], .material-icons, .material-symbols-rounded{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
  letter-spacing:normal !important;
}

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
  animation:vg-rise .9s cubic-bezier(.2,.8,.2,1) both;
}
.vg-hero::before, .vg-hero::after{
  content:""; position:absolute; width:420px; height:420px; border-radius:50%; filter:blur(70px); opacity:.55;
  animation:vg-drift 14s ease-in-out infinite alternate;
}
.vg-hero::before{ background:#5B6CFF; top:-220px; left:-80px; }
.vg-hero::after { background:#12B8D6; bottom:-260px; right:-60px; animation-delay:-7s; }
.vg-hero > *{ position:relative; z-index:1; }
.vg-hero h1{
  margin:0; font-size:2.5rem; font-weight:700; line-height:1.1;
  background:linear-gradient(100deg,#FFFFFF 10%, #B9C3FF 55%, #8EEBFF 100%);
  background-size:200% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:vg-sheen 7s linear infinite;
}
.vg-hero p{ margin:.55rem 0 0; color:#B4BEDD; font-size:1.02rem; max-width:62ch; }
.vg-chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
.vg-chip{
  padding:6px 13px; border-radius:999px; font-size:.78rem; font-weight:600; color:#D5DCFF;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10); backdrop-filter:blur(8px);
}
@keyframes vg-rise{ from{opacity:0; transform:translateY(14px) scale(.985);} to{opacity:1; transform:none;} }
@keyframes vg-sheen{ to{ background-position:200% center; } }
@keyframes vg-drift{ from{ transform:translate(0,0) scale(1);} to{ transform:translate(70px,40px) scale(1.15);} }
@keyframes vg-pulse{ 0%{ box-shadow:0 0 0 0 rgba(52,211,153,.6);} 100%{ box-shadow:0 0 0 12px rgba(52,211,153,0);} }

[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0C1330 0%, #080D1A 100%);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] img{ filter:drop-shadow(0 6px 18px rgba(124,140,255,.45)); }
[data-testid="stSidebar"] h1{ font-size:1.6rem; margin-bottom:0; }
[data-testid="stSidebar"] hr{ border-color:var(--line); }
.vg-pill{
  display:flex; align-items:center; gap:10px; padding:11px 14px; border-radius:14px;
  font-weight:600; font-size:.88rem; border:1px solid var(--line); background:rgba(255,255,255,.03);
}
.vg-dot{ width:9px; height:9px; border-radius:50%; background:#5A6690; flex:none; }
.vg-pill.on{ color:#B9F5DD; border-color:rgba(52,211,153,.35); background:rgba(52,211,153,.08); }
.vg-pill.on .vg-dot{ background:var(--ok); animation:vg-pulse 1.6s ease-out infinite; }
.vg-pill.off{ color:var(--muted); }

.stTabs [data-baseweb="tab-list"]{
  gap:6px; padding:6px; border-radius:16px; background:rgba(255,255,255,.03);
  border:1px solid var(--line); width:fit-content; max-width:100%; overflow-x:auto;
}
.stTabs [data-baseweb="tab"]{
  height:42px; padding:0 18px; border-radius:11px; color:var(--muted); font-weight:600;
  transition:color .2s, background .2s;
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--text); background:rgba(255,255,255,.05); }
.stTabs [aria-selected="true"]{
  color:#fff !important; background:linear-gradient(135deg, rgba(124,140,255,.35), rgba(34,211,238,.20));
  box-shadow:inset 0 0 0 1px rgba(160,175,255,.35);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

h2{ font-size:1.65rem !important; font-weight:700 !important; }
h3{ font-size:1.2rem !important; font-weight:600 !important; color:#DDE3FF; }
[data-testid="stCaptionContainer"]{ color:var(--muted); }
hr{ border-color:var(--line) !important; }

.stButton > button, .stDownloadButton > button{
  border-radius:12px; font-weight:700; padding:.6rem 1.1rem; color:var(--text);
  background:var(--raised); border:1px solid var(--line);
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
.stButton > button:hover{
  transform:translateY(-2px); border-color:rgba(160,175,255,.55); background:#1D2A52;
  box-shadow:0 10px 24px -10px rgba(124,140,255,.55);
}
.stButton > button:active{ transform:translateY(0) scale(.98); }
.stButton > button[kind="primary"]{
  border:none; color:#fff;
  background:linear-gradient(120deg,#6C7DFF 0%, #4F8CFF 50%, #22B8E0 100%);
  background-size:160% 100%; background-position:0 0;
  box-shadow:0 12px 28px -12px rgba(92,120,255,.8);
}
.stButton > button[kind="primary"]:hover{ background-position:100% 0; box-shadow:0 16px 34px -12px rgba(92,140,255,.95); }
.stButton > button:disabled{ opacity:.42; transform:none; box-shadow:none; }

[data-testid="stFileUploader"] section{
  border-radius:18px; border:1.5px dashed rgba(140,155,255,.38);
  background:linear-gradient(180deg, rgba(124,140,255,.07), rgba(34,211,238,.03));
  transition:border-color .2s, background .2s, box-shadow .2s;
}
[data-testid="stFileUploader"] section:hover{
  border-color:var(--accent2); box-shadow:0 0 0 4px rgba(34,211,238,.08);
}
[data-testid="stFileUploaderFile"]{ border-radius:12px; }

[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius:16px; border-color:var(--line) !important;
}

[data-testid="stExpander"]{
  border:1px solid var(--line) !important; border-radius:16px !important; overflow:hidden;
  background:linear-gradient(180deg, rgba(22,32,63,.7), rgba(13,20,42,.7));
  box-shadow:0 14px 30px -20px rgba(0,0,0,.8); margin-bottom:10px;
  transition:border-color .2s, box-shadow .2s;
}
[data-testid="stExpander"]:hover{ border-color:rgba(160,175,255,.42) !important; }
[data-testid="stExpander"] summary{ padding:14px 18px; font-weight:700; }
[data-testid="stExpander"] summary:hover{ background:rgba(255,255,255,.03); }
[data-testid="stExpander"] details[open] summary{ border-bottom:1px solid var(--line); }
[data-testid="stExpanderDetails"]{ padding:18px; animation:vg-open .35s ease both; }
@keyframes vg-open{ from{ opacity:0; transform:translateY(-6px);} to{ opacity:1; transform:none;} }

[data-testid="stMetric"]{
  padding:16px 18px; border-radius:16px; border:1px solid var(--line);
  background:linear-gradient(160deg, rgba(124,140,255,.12), rgba(255,255,255,.02));
  box-shadow:var(--glow);
}
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"]{
  font-weight:700; font-size:1.7rem;
  background:linear-gradient(100deg,#fff,#B9C3FF); -webkit-background-clip:text; background-clip:text; color:transparent;
}

[data-testid="stAlert"]{
  border-radius:14px; border:1px solid var(--line); backdrop-filter:blur(6px);
}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){ background:rgba(52,211,153,.10); border-color:rgba(52,211,153,.35); }
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){ background:rgba(124,140,255,.10); border-color:rgba(124,140,255,.32); }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){ background:rgba(251,191,36,.10); border-color:rgba(251,191,36,.35); }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){ background:rgba(244,63,94,.11); border-color:rgba(244,63,94,.40); }

[data-baseweb="input"]{
  border-radius:12px !important; background:#111A36 !important;
  border:1px solid var(--line) !important; transition:border-color .2s, box-shadow .2s;
}
[data-baseweb="base-input"], .stTextInput input{ background:transparent !important; border-radius:12px !important; }
[data-baseweb="input"]:focus-within{ border-color:var(--accent2) !important; box-shadow:0 0 0 3px rgba(34,211,238,.15); }

[data-testid="stChatMessage"]{
  border-radius:18px; border:1px solid var(--line); padding:14px 18px;
  background:linear-gradient(180deg, rgba(22,32,63,.7), rgba(13,20,42,.7));
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:linear-gradient(135deg, rgba(124,140,255,.22), rgba(34,211,238,.10));
  border-color:rgba(160,175,255,.35);
}
[data-testid="stChatInput"]{ border-radius:16px; }
[data-testid="stChatInput"] > div{
  border-radius:16px; border:1px solid rgba(160,175,255,.30); background:rgba(15,23,48,.9);
  transition:box-shadow .2s, border-color .2s;
}
[data-testid="stChatInput"] > div:focus-within{ border-color:var(--accent2); box-shadow:0 0 0 3px rgba(34,211,238,.15); }

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
    <span class="vg-chip">15.2M INT8 SLM</span>
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
    - 🧠 15.2M INT8 PyTorch SLM
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
        st.info("No cases currently recorded. Upload a CSV file or start the database poller.")

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

# TAB 4: Risk Telemetry
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
        if st.button("⏹️ Stop Poller", disabled=not is_running(), use_container_width=True):
            stop_poller()
            st.rerun()

    st.divider()
    t = poller_telemetry
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Status", t["status"])
    tc2.metric("Total Scanned", t["total_scanned"])
    tc3.metric("Last Scan", t["last_scan"] or "—")
    tc4.metric("Next Scan", t["next_scan"] or "—")