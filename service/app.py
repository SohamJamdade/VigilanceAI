import os
import sys
import json
import time
import re
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import torch
from tokenizers import Tokenizer
from thefuzz import fuzz, process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

# ---------------------------------------------------------
# Database Setup (SQLite persistence)
# ---------------------------------------------------------
DB_FILE = "audit_log.db"

def init_db():
    """Initializes the audit log table if it does not already exist."""
    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screening_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                subject_account TEXT,
                risk_level TEXT,
                primary_typology TEXT,
                recommended_action TEXT,
                evidence_summary TEXT,
                raw_payload TEXT
            )
        """)
        conn.commit()

def log_audit_record(account: str, decision: dict, payload: dict):
    """Saves every SLM evaluation into the database for compliance and chat querying."""
    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO screening_audit 
            (subject_account, risk_level, primary_typology, recommended_action, evidence_summary, raw_payload)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            account,
            decision.get("risk_level", "UNKNOWN"),
            decision.get("primary_typology", "UNSPECIFIED"),
            decision.get("recommended_action", "MANUAL_REVIEW"),
            json.dumps(decision.get("supporting_evidence", [])),
            json.dumps(payload)
        ))
        conn.commit()

def normalize_decision_keys(d: dict) -> dict:
    """Recursively strips whitespace from dictionary keys and string values."""
    if not isinstance(d, dict):
        return d
    cleaned = {}
    for k, v in d.items():
        clean_k = k.strip()
        if isinstance(v, dict):
            cleaned[clean_k] = normalize_decision_keys(v)
        elif isinstance(v, str):
            cleaned[clean_k] = v.strip()
        elif isinstance(v, list):
            cleaned[clean_k] = [
                x.strip() if isinstance(x, str) else normalize_decision_keys(x) if isinstance(x, dict) else x
                for x in v
            ]
        else:
            cleaned[clean_k] = v
    return cleaned

def get_account_history(account: str) -> list:
    """Retrieves all previous batch records stored for this account."""
    accumulated_txs = []
    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT raw_payload FROM screening_audit WHERE UPPER(subject_account) = ? ORDER BY id ASC",
            (account.upper(),)
        )
        rows = cursor.fetchall()
        for row in rows:
            try:
                past_payload = json.loads(row[0])
                past_records = past_payload.get("batch_records", [])
                accumulated_txs.extend(past_records)
            except Exception:
                continue
    return accumulated_txs

# ---------------------------------------------------------
# FastAPI Lifespan and Model State
# ---------------------------------------------------------
runtime_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    torch.set_num_threads(4)
    tok_path = "tokenizer/financial_bpe.json"
    ckpt_path = "checkpoints/slm_15m_int8.pt"

    if not os.path.exists(tok_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Assets missing: verify tokenizer and int8 weights.")

    tokenizer = Tokenizer.from_file(tok_path)
    base_model = FinancialSLM(
        vocab_size=2048, d_model=384, n_layers=8, n_heads=12, max_seq_len=512
    ).to("cpu")

    quantized_model = torch.ao.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)
    )
    quantized_model.eval()

    runtime_state["tokenizer"] = tokenizer
    runtime_state["model"] = quantized_model
    runtime_state["target_end_id"] = tokenizer.token_to_id("<|target_end|>")
    yield
    runtime_state.clear()

# Initialize FastAPI App
app = FastAPI(title="VigilanceAI AML Platform", version="1.1.0", lifespan=lifespan)

# ---------------------------------------------------------
# Route 1: Transaction Screening Engine
# ---------------------------------------------------------
@app.post("/v1/screen", status_code=status.HTTP_200_OK)
async def screen_transaction(payload: Dict[str, Any]):
    account_id = payload.get("subject_account") or payload.get("account_baseline", {}).get("account_id", "UNKNOWN")
    new_records = payload.get("batch_records", [])

    # 1. Fetch previous transactions for this account and combine them
    past_records = get_account_history(account_id)
    combined_records = past_records + new_records

    # 2. Build cumulative evaluation payload for the SLM
    eval_payload = dict(payload)
    eval_payload["batch_records"] = combined_records

    model = runtime_state["model"]
    tokenizer = runtime_state["tokenizer"] 
    target_end_id = runtime_state["target_end_id"]

    prompt_str = f"<|context_start|>{json.dumps(eval_payload)}<|context_end|><|target_start|>"
    input_ids = tokenizer.encode(prompt_str).ids
    prompt_len = len(input_ids)
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

    start_time = time.time()
    with torch.no_grad():
        for _ in range(120):
            idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            logits, _ = model(idx_window)
            logits = logits[:, -1, :] / 0.1
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if target_end_id is not None and next_token.item() == target_end_id:
                break
            curr_ids = torch.cat((curr_ids, next_token), dim=1)

    latency_ms = (time.time() - start_time) * 1000
    gen_tokens = curr_ids[0, prompt_len:].tolist()
    raw_text = tokenizer.decode(gen_tokens).replace("<|target_end|>", "").strip()
    clean_text = re.sub(r'\s*([\{\}\[\]:,])\s*', r'\1', raw_text)

    try:
        parsed_decision = json.loads(re.search(r'\{.*\}', clean_text, re.DOTALL).group(0))
    except Exception:
        parsed_decision = {
            "primary_typology": "STRUCTURING_SMURFING",
            "recommended_action": "ESCALATE_TO_FIU",
            "risk_level": "HIGH",
            "supporting_evidence": ["Parsing artifact detected. Flagged for review."]
        }

    parsed_decision = normalize_decision_keys(parsed_decision)

    # 3. Log cumulative payload so context compounds over time
    log_audit_record(account_id, parsed_decision,payload)

    return {
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "total_account_tx_count": len(combined_records),
        "decision": parsed_decision
    }

# ---------------------------------------------------------
# Route 2: Automated Event Ingestion
# ---------------------------------------------------------
class SingleTransactionEvent(BaseModel):
    account_id: str
    tx_id: str 
    amount: float 
    rail: str = "IMPS"
    recipient: str 
    device_id: str = "DEV-MOBILE"
    statutory_limit: float = 50000.0  # Kept standard at INR 50k

DEFAULT_AML_POLICY = (
    "POL-CORE-AML: Identify structuring, sudden velocity spikes, "
    "and transactions hovering immediately beneath statutory thresholds."
)

@app.post("/v1/ingest", status_code=status.HTTP_200_OK)
async def ingest_transaction(event: SingleTransactionEvent):
    """
    Simulates core banking webhook:
    Transforms raw transfer attributes into cumulative screening batches automatically.
    """
    new_record = {
        "tx_id": event.tx_id,
        "amount": event.amount,
        "rail": event.rail,
        "recipient": event.recipient,
        "device": event.device_id,
        "timestamp": datetime.utcnow().isoformat()
    }

    slm_payload = {
        "subject_account": event.account_id,
        "statutory_limit": event.statutory_limit,
        "batch_records": [new_record],
        "applied_policy": DEFAULT_AML_POLICY
    }

    screen_response = await screen_transaction(slm_payload)
    
    return {
        "status": "ingested_and_evaluated",
        "account_id": event.account_id,
        "total_historical_txs": screen_response["total_account_tx_count"],
        "latest_decision": screen_response["decision"]
    }

# ---------------------------------------------------------
# Route 3: Typo-Tolerant Conversational Assistant
# ---------------------------------------------------------
class ChatQuery(BaseModel):
    query: str

def extract_account_id(query: str, cursor: sqlite3.Cursor) -> Optional[str]:
    """Finds exact or approximate account IDs matching accounts in the database."""
    rx_match = re.search(r'(?i)\b(?:acc[-_ ]?)?(\d{4,6})\b', query)

    cursor.execute("SELECT DISTINCT subject_account FROM screening_audit")
    known_accounts = [row[0] for row in cursor.fetchall() if row[0]]

    if not known_accounts:
        return None

    if rx_match:
        digits = rx_match.group(1)
        for acc in known_accounts:
            if digits in acc:
                return acc

    best_match, score = process.extractOne(query, known_accounts, scorer=fuzz.partial_ratio)
    if score >= 75:
        return best_match
    return None

def detect_query_intent(query: str) -> str:
    """Classifies user intent even with spelling errors using token sort ratios."""
    q = query.lower()

    intents = {
        "count_flagged": ["how many flagged", "count suspicious", "total alerts", "number of risks", "how many high", "critical alerts"],
        "recent_activity": ["show recent", "latest transactions", "what happened recently", "newest alerts", "recent audit"],
        "account_investigation": ["why is account flagged", "tell me about account", "explain alert", "check details", "reason for risk"],
        "list_high_risk": ["which accounts are flagged", "list suspicious accounts", "who got flagged", "show all high risk", "show critical"]
    }

    best_intent = "unknown"
    highest_score = 0

    for intent, sample_phrases in intents.items():
        for phrase in sample_phrases:
            score = fuzz.token_set_ratio(q, phrase)
            if score > highest_score:
                highest_score = score
                best_intent = intent

    return best_intent if highest_score >= 60 else "unknown"

@app.post("/v1/chat")
def compliance_assistant_chat(req: ChatQuery):
    raw_query = req.query.strip()

    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        cursor = conn.cursor()
        target_account = extract_account_id(raw_query, cursor)
        intent = detect_query_intent(raw_query)

        if target_account:
            cursor.execute(
                "SELECT COUNT(*), MAX(timestamp) FROM screening_audit WHERE UPPER(subject_account) = ?",
                (target_account.upper(),)
            )
            total_evals, latest_ts = cursor.fetchone()

            cursor.execute(
                "SELECT risk_level, primary_typology, recommended_action, evidence_summary "
                "FROM screening_audit WHERE UPPER(subject_account) = ? ORDER BY id DESC LIMIT 1",
                (target_account.upper(),)
            )
            row = cursor.fetchone()
            if row:
                risk, typology, action, evidence_raw = row
                try:
                    evidence_list = json.loads(evidence_raw)
                    evidence_str = "; ".join(evidence_list) if isinstance(evidence_list, list) else str(evidence_list)
                except Exception:
                    evidence_str = str(evidence_raw)

                return {
                    "response": (
                        f"Account {target_account} has {total_evals} recorded audit screening(s). "
                        f"Most recent evaluation ({latest_ts}): Risk Level: {risk}, "
                        f"Typology: {typology}, Action: {action}. Evidence: {evidence_str}"
                    )
                }

        if intent == "count_flagged":
            cursor.execute("SELECT COUNT(*) FROM screening_audit WHERE risk_level IN ('HIGH', 'CRITICAL')")
            high_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM screening_audit")
            total = cursor.fetchone()[0]
            return {
                "response": f"Currently, there are {high_count} HIGH/CRITICAL-risk flagged accounts out of {total} total screened accounts."
            }

        if intent == "list_high_risk":
            cursor.execute("SELECT DISTINCT subject_account, primary_typology, risk_level FROM screening_audit WHERE risk_level IN ('HIGH', 'CRITICAL') LIMIT 10")
            rows = cursor.fetchall()
            if not rows:
                return {"response": "There are currently no accounts flagged as HIGH or CRITICAL risk."}
            items = [f"• {acc} ({typology}) [{risk}]" for acc, typology, risk in rows]
            return {
                "response": "The following accounts are currently flagged for review:\n" + "\n".join(items)
            }

        if intent == "recent_activity":
            cursor.execute("SELECT subject_account, primary_typology, risk_level FROM screening_audit ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            if not rows:
                return {"response": "No transaction activity recorded yet."}
            items = [f"• {acc}: {typology} [{risk}]" for acc, typology, risk in rows]
            return {
                "response": "Here is the most recent activity:\n" + "\n".join(items)
            }

        return {
            "response": (
                "I'm your compliance assistant. Ask me questions naturally, such as: "
                "'Which accounts are flagged?', 'Why is ACC-78104 flagged?', 'How many alerts are there?', "
                "or 'Show recent activity'."
            )
        }