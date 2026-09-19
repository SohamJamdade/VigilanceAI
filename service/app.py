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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM


# Database Setup (Default: Built-in SQLite, seamlessly swappable to PostgreSQL)
DB_FILE = "audit_log.db"

def init_db():
    """Initializes the audit log table if it does not already exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
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
    with sqlite3.connect(DB_FILE) as conn:
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

# FastAPI Lifespan and Model State
runtime_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Auto-initialize database on startup
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
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    quantized_model.eval()

    runtime_state["tokenizer"] = tokenizer
    runtime_state["model"] = quantized_model
    runtime_state["target_end_id"] = tokenizer.token_to_id("<|target_end|>")
    yield
    runtime_state.clear()

app = FastAPI(title="VigilanceAI AML Platform", version="1.1.0", lifespan=lifespan)

# 1. Existing Transaction Screening Route (Now Auto-logging to DB)
@app.post("/v1/screen", status_code=status.HTTP_200_OK)
async def screen_transaction(payload: Dict[str, Any]):
    model = runtime_state["model"]
    tokenizer = runtime_state["tokenizer"]
    target_end_id = runtime_state["target_end_id"]

    prompt_str = f"<|context_start|>{json.dumps(payload)}<|context_end|><|target_start|>"
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

    # Extract account ID and persist directly to audit DB
    account_id = payload.get("subject_account") or payload.get("account_baseline", {}).get("account_id", "UNKNOWN")
    log_audit_record(account_id, parsed_decision, payload)

    return {
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "decision": parsed_decision
    }

# 2. Bank Employee Conversational Query Endpoint
class ChatQuery(BaseModel):
    query: str

@app.post("/v1/chat")
def compliance_assistant_chat(req: ChatQuery):
    """
    Conversational assistant for compliance officers.
    Answers natural language queries about flags, statistics, and specific accounts.
    """
    q = req.query.lower().strip()
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        # Query Type 1: Overall flagged account counts
        if "how many" in q and ("flag" in q or "high" in q or "suspicious" in q):
            cursor.execute("SELECT COUNT(*) FROM screening_audit WHERE risk_level = 'HIGH'")
            high_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM screening_audit")
            total_count = cursor.fetchone()[0]
            return {
                "response": f"Currently, there are {high_count} HIGH-risk flagged accounts out of {total_count} total screened accounts."
            }

        # Query Type 2: Details on a specific account
        match = re.search(r'acc-\d+', q)
        if match:
            target_acc = match.group(0).upper()
            cursor.execute(
                "SELECT timestamp, risk_level, primary_typology, recommended_action, evidence_summary "
                "FROM screening_audit WHERE UPPER(subject_account) = ? ORDER BY id DESC LIMIT 1",
                (target_acc,)
            )
            row = cursor.fetchone()
            if row:
                ts, risk, typology, action, evidence = row
                return {
                    "response": (
                        f"Account {target_acc} was evaluated on {ts}.\n"
                        f"• Risk Level: {risk}\n"
                        f"• Typology: {typology}\n"
                        f"• Recommended Action: {action}\n"
                        f"• Evidence: {evidence}"
                    )
                }
            else:
                return {"response": f"No audit records found for account ID '{target_acc}'."}

        # Query Type 3: Summary of recent high-risk escalations
        if "recent" in q or "summary" in q or "latest" in q:
            cursor.execute(
                "SELECT subject_account, primary_typology, risk_level FROM screening_audit "
                "ORDER BY id DESC LIMIT 5"
            )
            recent_rows = cursor.fetchall()
            if not recent_rows:
                return {"response": "No recent transaction activity recorded yet."}
            
            summary_lines = [f"- {acc}: {typology} ({risk})" for acc, typology, risk in recent_rows]
            return {
                "response": "Here are the most recent transaction reviews:\n" + "\n".join(summary_lines)
            }

    return {
        "response": "I can help with compliance statistics or account investigations. Try asking: "
                    "'How many accounts are flagged?', 'Tell me about ACC-99104', or 'Show recent activity'."
    }