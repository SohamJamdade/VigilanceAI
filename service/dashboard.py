"""
dashboard.py — VigilanceAI Streamlit Dashboard

A visual interface for banking managers:
  Tab 1: Upload CSV/Excel → auto-screen transactions → see results
  Tab 2: Chat with the compliance assistant (natural language)
  Tab 3: Risk overview dashboard with charts

Run with:
    streamlit run service/dashboard.py

Make sure the FastAPI backend is running on port 8000 first:
    uvicorn service.app:app --host 0.0.0.0 --port 8000
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sqlite3
import os
import sys

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from service.ingest import bytes_to_payloads

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
API_BASE = os.getenv("VIGILANCE_API_URL", "http://localhost:8000")
DB_FILE = os.getenv("VIGILANCE_DB", "audit_log.db")

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="VigilanceAI — AML Compliance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .risk-high { color: #FF4B4B; font-weight: bold; font-size: 1.1em; }
    .risk-medium { color: #FFA500; font-weight: bold; font-size: 1.1em; }
    .risk-low { color: #00CC66; font-weight: bold; font-size: 1.1em; }
    .risk-critical { color: #DC143C; font-weight: bold; font-size: 1.2em; }
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("VigilanceAI")
    st.caption("AML Compliance Platform v1.1")
    st.divider()
    st.markdown("**Quick Guide:**")
    st.markdown("""
    1. 📁 **Upload** → Drop a CSV/Excel file
    2. ⚡ **Auto-Screen** → AI analyzes all transactions
    3. 💬 **Chat** → Ask questions about results
    4. 📊 **Dashboard** → Visual risk overview
    """)
    st.divider()
    api_status = "🟢 Online"
    try:
        r = requests.get(f"{API_BASE}/docs", timeout=3)
        if r.status_code != 200:
            api_status = "🔴 Offline"
    except Exception:
        api_status = "🔴 Offline"
    st.markdown(f"**API Status:** {api_status}")


# ---------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------
tab_upload, tab_chat, tab_dashboard = st.tabs([
    "📁 Upload & Screen",
    "💬 Chat Assistant",
    "📊 Risk Overview"
])


# ==========================================================
# TAB 1: UPLOAD & SCREEN
# ==========================================================
with tab_upload:
    st.header("📁 Upload Transactions & Auto-Screen")
    st.markdown(
        "Upload a **CSV** or **Excel** file with transaction data. "
        "VigilanceAI will automatically convert it and run risk screening — **no JSON needed.**"
    )

    # Show expected format
    with st.expander("📋 What should my file look like?"):
        st.markdown("""
        Your file should have columns like these (names are flexible):

        | account_id | tx_id | amount | recipient | rail | device | time | location |
        |---|---|---|---|---|---|---|---|
        | ACC-10001 | TXN-A1 | 49500 | John Doe | NEFT | DEV-001 | 03:15 | Mumbai |
        | ACC-10001 | TXN-A2 | 48900 | John Doe | UPI | DEV-001 | 03:18 | Mumbai |
        | ACC-20002 | TXN-B1 | 5000 | Jane Smith | IMPS | DEV-002 | 14:30 | Delhi |

        **Required columns:** `account_id`, `amount`
        **Optional columns:** `tx_id`, `recipient`, `rail`/`channel`, `device`, `time`, `location`

        Column names are matched flexibly — e.g., `acc_no`, `account_number`, `acct_id` all work for account ID.
        """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Drop your transaction file here",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files"
    )

    if uploaded_file is not None:
        # Parse the file
        try:
            file_bytes = uploaded_file.getvalue()
            payloads = bytes_to_payloads(file_bytes, uploaded_file.name)

            st.success(f"✅ Parsed **{uploaded_file.name}** — found **{len(payloads)} account(s)** to screen.")

            # Preview the parsed data
            with st.expander("👀 Preview parsed transactions", expanded=False):
                for i, payload in enumerate(payloads):
                    st.markdown(f"**Account: `{payload['subject_account']}`** — {len(payload['batch_records'])} transaction(s)")
                    st.json(payload, expanded=False)

            # Screen button
            if st.button("🚀 Run AML Screening", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0, text="Screening transactions...")

                for i, payload in enumerate(payloads):
                    progress_bar.progress(
                        (i + 1) / len(payloads),
                        text=f"Screening account {payload['subject_account']}..."
                    )

                    try:
                        response = requests.post(
                            f"{API_BASE}/v1/screen",
                            json=payload,
                            timeout=60
                        )
                        if response.status_code == 200:
                            result = response.json()
                            result["account"] = payload["subject_account"]
                            results.append(result)
                        else:
                            st.error(f"❌ Failed for {payload['subject_account']}: HTTP {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to the API. Make sure `uvicorn service.app:app --port 8000` is running.")
                        break
                    except Exception as e:
                        st.error(f"❌ Error screening {payload['subject_account']}: {e}")

                progress_bar.empty()

                if results:
                    st.divider()
                    st.subheader("🔍 Screening Results")

                    for result in results:
                        decision = result.get("decision", {})
                        risk = decision.get("risk_level", "UNKNOWN")
                        typology = decision.get("primary_typology", "UNKNOWN")
                        action = decision.get("recommended_action", "UNKNOWN")
                        evidence = decision.get("supporting_evidence", [])
                        latency = result.get("latency_ms", 0)
                        tx_count = result.get("total_account_tx_count", 0)

                        # Color-coded risk card
                        risk_colors = {
                            "CRITICAL": "🔴", "HIGH": "🟠",
                            "MEDIUM": "🟡", "LOW": "🟢"
                        }
                        icon = risk_colors.get(risk, "⚪")

                        with st.container(border=True):
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                            with col1:
                                st.markdown(f"**Account:** `{result['account']}`")
                            with col2:
                                st.markdown(f"**Risk:** {icon} **{risk}**")
                            with col3:
                                st.markdown(f"**Typology:** {typology}")
                            with col4:
                                st.markdown(f"⏱️ {latency:.0f}ms")

                            st.markdown(f"**Action:** {action}")
                            st.markdown(f"**Transactions analyzed:** {tx_count}")

                            if evidence:
                                st.markdown("**Evidence:**")
                                for e in evidence:
                                    st.markdown(f"- {e}")

                    # Summary table
                    st.divider()
                    st.subheader("📊 Summary")
                    summary_data = []
                    for r in results:
                        d = r.get("decision", {})
                        summary_data.append({
                            "Account": r["account"],
                            "Risk Level": d.get("risk_level", "UNKNOWN"),
                            "Typology": d.get("primary_typology", "UNKNOWN"),
                            "Action": d.get("recommended_action", "UNKNOWN"),
                            "Latency (ms)": r.get("latency_ms", 0),
                        })

                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    # Download results
                    csv_download = summary_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        data=csv_download,
                        file_name="screening_results.csv",
                        mime="text/csv"
                    )

        except ValueError as e:
            st.error(f"❌ File parsing error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")


# ==========================================================
# TAB 2: CHAT ASSISTANT
# ==========================================================
with tab_chat:
    st.header("💬 Compliance Chat Assistant")
    st.markdown(
        "Ask questions in plain English — typos are fine! "
        "Try: *'Which accounts are flagged?'*, *'Why is ACC-10001 risky?'*, *'Show recent activity'*"
    )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I'm your compliance assistant. Ask me anything about "
                    "screened transactions. For example:\n\n"
                    "- *Which accounts are flagged?*\n"
                    "- *How many high-risk alerts are there?*\n"
                    "- *Tell me about ACC-10001*\n"
                    "- *Show recent activity*"
                )
            }
        ]

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about transactions, accounts, or risk alerts..."):
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the /v1/chat endpoint
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/v1/chat",
                        json={"query": prompt},
                        timeout=15
                    )
                    if response.status_code == 200:
                        answer = response.json().get("response", "No response received.")
                    else:
                        answer = f"⚠️ API returned status {response.status_code}"
                except requests.exceptions.ConnectionError:
                    answer = "❌ Cannot connect to the API. Make sure the backend is running on port 8000."
                except Exception as e:
                    answer = f"❌ Error: {e}"

            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ==========================================================
# TAB 3: RISK OVERVIEW DASHBOARD
# ==========================================================
with tab_dashboard:
    st.header("📊 Risk Overview Dashboard")

    if not os.path.exists(DB_FILE):
        st.info("No audit data yet. Upload and screen some transactions first!")
    else:
        try:
            conn = sqlite3.connect(DB_FILE)

            # Total counts
            total_screenings = pd.read_sql("SELECT COUNT(*) as count FROM screening_audit", conn).iloc[0]["count"]
            high_risk = pd.read_sql("SELECT COUNT(*) as count FROM screening_audit WHERE risk_level = 'HIGH'", conn).iloc[0]["count"]
            critical = pd.read_sql("SELECT COUNT(*) as count FROM screening_audit WHERE risk_level = 'CRITICAL'", conn).iloc[0]["count"]
            unique_accounts = pd.read_sql("SELECT COUNT(DISTINCT subject_account) as count FROM screening_audit", conn).iloc[0]["count"]

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Screenings", total_screenings)
            with col2:
                st.metric("Unique Accounts", unique_accounts)
            with col3:
                st.metric("🟠 High Risk", high_risk)
            with col4:
                st.metric("🔴 Critical", critical)

            st.divider()

            # Two-column layout for charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Risk Level Distribution")
                risk_df = pd.read_sql(
                    "SELECT risk_level, COUNT(*) as count FROM screening_audit GROUP BY risk_level",
                    conn
                )
                if not risk_df.empty:
                    color_map = {
                        "LOW": "#00CC66", "MEDIUM": "#FFA500",
                        "HIGH": "#FF4B4B", "CRITICAL": "#DC143C",
                        "UNKNOWN": "#888888"
                    }
                    fig = px.pie(
                        risk_df, names="risk_level", values="count",
                        color="risk_level", color_discrete_map=color_map,
                        hole=0.4
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        margin=dict(t=20, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                st.subheader("Typology Breakdown")
                typology_df = pd.read_sql(
                    "SELECT primary_typology, COUNT(*) as count FROM screening_audit "
                    "GROUP BY primary_typology ORDER BY count DESC",
                    conn
                )
                if not typology_df.empty:
                    fig = px.bar(
                        typology_df, x="count", y="primary_typology",
                        orientation="h", color="count",
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        yaxis_title="",
                        xaxis_title="Count",
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Flagged accounts table
            st.divider()
            st.subheader("🚨 Flagged Accounts")

            flagged_df = pd.read_sql("""
                SELECT
                    subject_account AS "Account",
                    risk_level AS "Risk",
                    primary_typology AS "Typology",
                    recommended_action AS "Action",
                    timestamp AS "Last Screened"
                FROM screening_audit
                WHERE risk_level IN ('HIGH', 'CRITICAL')
                ORDER BY timestamp DESC
                LIMIT 50
            """, conn)

            if flagged_df.empty:
                st.success("✅ No high-risk or critical accounts found.")
            else:
                st.dataframe(flagged_df, use_container_width=True, hide_index=True)

            # Recent activity
            st.divider()
            st.subheader("🕐 Recent Screening Activity")

            recent_df = pd.read_sql("""
                SELECT
                    subject_account AS "Account",
                    risk_level AS "Risk",
                    primary_typology AS "Typology",
                    recommended_action AS "Action",
                    timestamp AS "Timestamp"
                FROM screening_audit
                ORDER BY id DESC
                LIMIT 20
            """, conn)

            if not recent_df.empty:
                st.dataframe(recent_df, use_container_width=True, hide_index=True)

            conn.close()

        except Exception as e:
            st.error(f"Error reading audit database: {e}")
