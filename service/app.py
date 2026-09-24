import os
import sys
import json
import time
import re
import uuid
import hashlib
import sqlite3
import numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, status
from pydantic import BaseModel
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM
from service.schema import (
    NormalizedTransaction, BehavioralBaseline, FeatureSet,
    RuleResult, EntityProfile, RiskAssessment
)
from service.features import compute_features
from service.rules import evaluate_rules
from service.entity import resolve_entity
from service.cases import (
    init_case_storage, record_case, get_upload_history,
    delete_upload_batch, purge_all_records, DB_FILE
)

runtime_state: Dict[str, Any] = {}


def safe_dump_dict(obj: Any) -> Dict[str, Any]:
    # Pydantic v2/v1 safe dict conversion
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_case_storage(DB_FILE)
    torch.set_num_threads(4)
    tok_path = "tokenizer/financial_bpe.json"
    ckpt_path = "checkpoints/slm_15m_int8.pt"

    if not os.path.exists(tok_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Model assets missing: check tokenizer and int8 weights.")

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


app = FastAPI(title="VigilanceAI Financial Risk Platform", version="2.0.0", lifespan=lifespan)


def synthesize_slm_reasoning(
    account_id: str,
    features: FeatureSet,
    rules: List[RuleResult],
    entity: EntityProfile,
    latest_txs: List[NormalizedTransaction]
) -> Dict[str, Any]:
    try:
        model = runtime_state["model"]
        tokenizer = runtime_state["tokenizer"]
        target_end_id = runtime_state["target_end_id"]

        inference_context = {
            "account_id": account_id,
            "features": {
                "tx_count": features.window_tx_count,
                "total_val": features.window_total_amount,
                "velocity": features.velocity_tx_per_hour,
                "burst": features.burst_flag,
                "near_threshold": features.near_threshold_count,
                "deviation_ratio": features.baseline_deviation_ratio
            },
            "triggered_rules": [r.rule_id for r in rules if r.triggered],
            "shared_devices": len(entity.shared_device_accounts),
            "recent_txs": [
                {"id": t.tx_id, "amt": t.amount, "rail": t.rail, "recip": t.recipient}
                for t in latest_txs[-5:]
            ]
        }

        prompt_str = f"<|context_start|>{json.dumps(inference_context)}<|context_end|><|target_start|>"
        input_ids = tokenizer.encode(prompt_str).ids
        input_ids = input_ids[-384:]
        prompt_len = len(input_ids)
        curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

        with torch.no_grad():
            # Capped at 48 tokens for fast completion without timeout
            for _ in range(48):
                idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                logits, _ = model(idx_window)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if target_end_id is not None and next_token.item() == target_end_id:
                    break
                curr_ids = torch.cat((curr_ids, next_token), dim=1)

        gen_tokens = curr_ids[0, prompt_len:].tolist()
        raw_text = tokenizer.decode(gen_tokens).replace("<|target_end|>", "").strip()
        clean_text = re.sub(r'\s*([\{\}\[\]:,])\s*', r'\1', raw_text)

        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"SLM fallback triggered: {e}")

    high_rule = any(r.severity in ("HIGH", "CRITICAL") for r in rules if r.triggered)
    return {
        "risk_level": "HIGH" if high_rule else "LOW",
        "primary_typology": "STRUCTURING_SMURFING" if features.near_threshold_count >= 2 else "NORMAL_ROUTINE",
        "recommended_action": "ESCALATE_TO_FIU" if high_rule else "AUTO_CLEAR",
        "supporting_evidence": [r.reason for r in rules if r.triggered] or ["All activities conform to baseline."]
    }


def run_pipeline(
    account_id: str,
    incoming_txs: List[NormalizedTransaction],
    raw_payload: Dict[str, Any],
    statutory_limit: float = 50000.0
) -> RiskAssessment:
    case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    # Resolve baseline from payload or compute from transaction amounts
    base_obj = raw_payload.get("account_baseline", {})
    if base_obj.get("historical_median"):
        dyn_median = float(base_obj["historical_median"])
        dyn_bracket = base_obj.get("typical_bracket", [0.0, dyn_median * 2])
    else:
        amounts = [float(t.amount) for t in incoming_txs]
        dyn_median = float(np.median(amounts)) if amounts else 0.0
        dyn_bracket = [float(min(amounts)), float(max(amounts))] if amounts else [0.0, 0.0]

    baseline = BehavioralBaseline(
        account_id=account_id,
        historical_median=dyn_median,
        typical_bracket=dyn_bracket,
        historical_tx_count=len(incoming_txs)
    )

    features = compute_features(account_id, incoming_txs, baseline, statutory_limit)
    rule_results = evaluate_rules(incoming_txs, features, statutory_limit)
    entity_prof = resolve_entity(account_id, incoming_txs, DB_FILE)

    # Calculate deterministic risk score from rule severities
    critical_hits = sum(1 for r in rule_results if r.triggered and r.severity == "CRITICAL")
    high_hits = sum(1 for r in rule_results if r.triggered and r.severity == "HIGH")
    raw_score = (critical_hits * 0.7) + (high_hits * 0.3) + (min(features.baseline_deviation_ratio, 10.0) * 0.03)
    deterministic_score = round(min(1.0, max(0.0, float(raw_score))), 2)

    slm_decision = synthesize_slm_reasoning(account_id, features, rule_results, entity_prof, incoming_txs)

    # Regulatory override: critical or high rules dictate risk, action, and typology
    final_risk = slm_decision.get("risk_level", "LOW")
    final_typology = slm_decision.get("primary_typology", "NORMAL_ROUTINE")
    final_action = slm_decision.get("recommended_action", "AUTO_CLEAR")

    if critical_hits > 0:
        final_risk = "CRITICAL"
        final_action = "BLOCK_IMMEDIATELY"
        if any(r.rule_id == "RULE_SANCT_003" for r in rule_results if r.triggered):
            final_typology = "SANCTIONS_BREACH"
        else:
            final_typology = "CRITICAL_REGULATORY_BREACH"
    elif high_hits > 0:
        if final_risk != "CRITICAL":
            final_risk = "HIGH"
        if final_action == "AUTO_CLEAR":
            final_action = "ESCALATE_TO_FIU"
        if final_typology == "NORMAL_ROUTINE":
            if any(r.rule_id == "RULE_STRUCT_001" for r in rule_results if r.triggered):
                final_typology = "STRUCTURING_SMURFING"
            elif any(r.rule_id == "RULE_VEL_002" for r in rule_results if r.triggered):
                final_typology = "HIGH_VELOCITY_BURST"
            else:
                final_typology = "SUSPICIOUS_ACTIVITY"

    evidence_text = slm_decision.get("supporting_evidence", [])
    slm_reasoning_str = "; ".join(evidence_text) if isinstance(evidence_text, list) else str(evidence_text)

    repro_str = f"{case_id}|{account_id}|{deterministic_score}|{final_risk}"
    repro_hash = hashlib.sha256(repro_str.encode()).hexdigest()[:16]

    assessment = RiskAssessment(
        case_id=case_id,
        account_id=account_id,
        timestamp=datetime.now(timezone.utc),
        deterministic_risk_score=deterministic_score,
        rule_severities=[r.severity for r in rule_results if r.triggered],
        triggered_rules=rule_results,
        features=features,
        entity_profile=entity_prof,
        slm_reasoning=slm_reasoning_str,
        primary_typology=final_typology,
        recommended_action=final_action,
        confidence_level=final_risk,
        reproducibility_hash=repro_hash
    )

    record_case(assessment, raw_payload, DB_FILE)
    return assessment


# --- API Endpoints ---

@app.post("/v1/screen", status_code=status.HTTP_200_OK)
async def screen_transaction(payload: Dict[str, Any]):
    account_id = str(payload.get("subject_account") or payload.get("account_baseline", {}).get("account_id", "UNKNOWN"))
    raw_records = payload.get("batch_records", [])

    tx_list: List[NormalizedTransaction] = []
    for r in raw_records:
        tx_list.append(NormalizedTransaction(
            tx_id=str(r.get("tx_id", f"TXN-{uuid.uuid4().hex[:6]}")),
            account_id=account_id,
            amount=float(r.get("amount", 0.0)),
            currency=str(r.get("currency", "INR")),
            rail=str(r.get("rail", "IMPS")).upper(),
            recipient=str(r.get("recipient") or r.get("to") or "UNKNOWN"),
            device_id=str(r.get("device_id") or r.get("device") or "DEV-UNKNOWN"),
            timestamp=datetime.now(timezone.utc),
            location=str(r.get("location", "DOMESTIC")),
            source_hash="REST_SCREEN"
        ))

    assessment = run_pipeline(account_id, tx_list, payload)

    return {
        "status": "success",
        "case_id": assessment.case_id,
        "account_id": assessment.account_id,
        "total_account_tx_count": len(tx_list),
        "deterministic_score": assessment.deterministic_risk_score,
        "triggered_rules": [safe_dump_dict(r) for r in assessment.triggered_rules if r.triggered],
        "decision": {
            "risk_level": assessment.confidence_level,
            "primary_typology": assessment.primary_typology,
            "recommended_action": assessment.recommended_action,
            "supporting_evidence": [assessment.slm_reasoning]
        }
    }


class SingleTransactionEvent(BaseModel):
    account_id: str
    tx_id: str
    amount: float
    rail: str = "IMPS"
    recipient: str
    device_id: str = "DEV-MOBILE"
    statutory_limit: float = 50000.0


@app.post("/v1/ingest", status_code=status.HTTP_200_OK)
async def ingest_transaction(event: SingleTransactionEvent):
    tx = NormalizedTransaction(
        tx_id=event.tx_id,
        account_id=event.account_id,
        amount=event.amount,
        currency="INR",
        rail=event.rail.upper(),
        recipient=event.recipient,
        device_id=event.device_id,
        timestamp=datetime.now(timezone.utc),
        location="DOMESTIC",
        source_hash="WEBHOOK"
    )

    slm_payload = {
        "subject_account": event.account_id,
        "statutory_limit": event.statutory_limit,
        "batch_records": [safe_dump_dict(tx)],
        "applied_policy": "POL-CORE-AML: Dynamic threshold monitoring"
    }

    assessment = run_pipeline(event.account_id, [tx], slm_payload, event.statutory_limit)

    return {
        "status": "ingested_and_evaluated",
        "case_id": assessment.case_id,
        "account_id": event.account_id,
        "deterministic_score": assessment.deterministic_risk_score,
        "latest_decision": {
            "risk_level": assessment.confidence_level,
            "primary_typology": assessment.primary_typology,
            "recommended_action": assessment.recommended_action,
            "supporting_evidence": [assessment.slm_reasoning]
        }
    }


class ChatQuery(BaseModel):
    query: str


def _extract_account_target(query: str, cursor: sqlite3.Cursor) -> Optional[str]:
    cursor.execute("SELECT DISTINCT subject_account FROM screening_audit UNION SELECT DISTINCT account_id FROM risk_cases")
    known_accounts = [r[0] for r in cursor.fetchall() if r[0]]
    if not known_accounts:
        return None

    q_upper = query.upper()

    for acc in known_accounts:
        if acc.upper() in q_upper:
            return acc

    number_tokens = re.findall(r'\b\d{3,8}\b', query)
    for token in number_tokens:
        for acc in known_accounts:
            if token in acc:
                return acc

    match = re.search(r'(?i)\b(?:acc|acct|account)?[-_ #:]*([a-z0-9_-]*\d{3,8}[a-z0-9_-]*)\b', query)
    if match:
        candidate = match.group(1).upper()
        for acc in known_accounts:
            if candidate in acc.upper() or acc.upper() in candidate:
                return acc

    return None


@app.post("/v1/chat")
def compliance_assistant_chat(req: ChatQuery):
    q = req.query.strip()
    q_lower = q.lower()

    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        cursor = conn.cursor()

        target_account = _extract_account_target(q, cursor)

        if target_account:
            cursor.execute("""
                SELECT case_id, created_at, account_id, risk_level, deterministic_score,
                       primary_typology, recommended_action, model_reasoning,
                       features_json, triggered_rules_json, entity_profile_json
                FROM risk_cases
                WHERE UPPER(account_id) = ?
                ORDER BY created_at DESC LIMIT 1
            """, (target_account.upper(),))
            case_row = cursor.fetchone()

            if case_row:
                (case_id, created_at, acc_id, risk_level, score,
                 typology, action, reasoning, features_json, rules_json, entity_json) = case_row

                risk_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                icon = risk_icons.get(risk_level, "⚪")

                lines = [
                    f"### {icon} Detailed Investigation Report: **{acc_id}**",
                    f"- **Risk Classification:** **{risk_level}** (Deterministic Score: `{score:.2f}`)",
                    f"- **Primary Typology:** `{typology}`",
                    f"- **Recommended Regulatory Action:** `{action}`",
                    f"- **Case Reference:** `{case_id}` (Screened: {created_at})",
                    "",
                    "#### 🔍 Why It Was Flagged (Concrete Evidence & Triggered Rules):"
                ]

                rules_data = json.loads(rules_json) if rules_json else []
                triggered = [r for r in rules_data if r.get("triggered")]

                if triggered:
                    for idx, r in enumerate(triggered, 1):
                        rule_name = r.get("rule_name", "Unknown Rule")
                        rule_id = r.get("rule_id", "N/A")
                        sev = r.get("severity", "HIGH")
                        rsn = r.get("reason", "Rule triggered.")
                        ev = r.get("evidence", {})
                        lines.append(f"{idx}. 🚨 **{rule_name}** (`{rule_id}` — {sev})")
                        lines.append(f"   - **Rule Finding:** {rsn}")
                        if ev:
                            ev_details = ", ".join(f"{k}: `{v}`" for k, v in ev.items())
                            lines.append(f"   - **Evidence:** {ev_details}")
                else:
                    lines.append("• No deterministic compliance rules were triggered. Transactions conform to baseline thresholds.")

                features_data = json.loads(features_json) if features_json else {}
                if features_data:
                    lines.extend([
                        "",
                        "#### 📊 Transaction Behavior & Feature Metrics:",
                        f"- **Analyzed Transactions:** {features_data.get('window_tx_count', 0)} (Total Volume: ₹{features_data.get('window_total_amount', 0):,.2f})",
                        f"- **Transaction Velocity:** {features_data.get('velocity_tx_per_hour', 0):.1f} tx/hr (Burst detected: `{features_data.get('burst_flag', False)}`)",
                        f"- **Near-Threshold Transfers:** {features_data.get('near_threshold_count', 0)} near ₹50,000 regulatory limit",
                        f"- **Historical Deviation Ratio:** {features_data.get('baseline_deviation_ratio', 1.0)}x historical median"
                    ])

                entity_data = json.loads(entity_json) if entity_json else {}
                if entity_data:
                    shared = entity_data.get("shared_device_accounts", [])
                    if shared:
                        lines.append(f"- ⚠️ **Entity Linkage Alert:** Hardware device shared with other account(s): {', '.join(shared)}")

                if reasoning and reasoning.strip():
                    lines.extend([
                        "",
                        f"#### 🧠 Contextual Synthesis:\n{reasoning}"
                    ])

                return {"response": "\n".join(lines)}

            cursor.execute("""
                SELECT risk_level, primary_typology, recommended_action, evidence_summary, timestamp
                FROM screening_audit WHERE UPPER(subject_account) = ? ORDER BY id DESC LIMIT 1
            """, (target_account.upper(),))
            audit_row = cursor.fetchone()
            if audit_row:
                risk, typ, act, ev_summary, ts = audit_row
                try:
                    ev_list = json.loads(ev_summary)
                    ev_str = "; ".join(ev_list) if isinstance(ev_list, list) else str(ev_list)
                except Exception:
                    ev_str = str(ev_summary)
                return {
                    "response": (
                        f"### Investigation Report: **{target_account}**\n"
                        f"- **Risk Level:** {risk}\n"
                        f"- **Typology:** {typ}\n"
                        f"- **Recommended Action:** {act}\n"
                        f"- **Last Screened:** {ts}\n"
                        f"- **Evidence Summary:** {ev_str}"
                    )
                }

        if any(w in q_lower for w in ["which", "flagged", "who", "list", "show", "cases", "accounts", "alert", "high", "critical"]):
            cursor.execute("""
                SELECT DISTINCT subject_account, risk_level, primary_typology, recommended_action
                FROM screening_audit
                WHERE risk_level IN ('HIGH', 'CRITICAL')
                ORDER BY id DESC LIMIT 15
            """)
            flagged_rows = cursor.fetchall()
            if flagged_rows:
                items = [f"• **{r[0]}** [{r[1]}] — {r[2]} → `{r[3]}`" for r in flagged_rows]
                return {
                    "response": f"The following {len(flagged_rows)} account(s) are currently flagged as High/Critical:\n\n" + "\n".join(items)
                }
            return {"response": "All screened accounts currently conform to baseline. No high-risk alerts."}

        if any(w in q_lower for w in ["how many", "count", "total", "number"]):
            cursor.execute("SELECT COUNT(*) FROM screening_audit WHERE risk_level IN ('HIGH', 'CRITICAL')")
            flagged = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM screening_audit")
            total = cursor.fetchone()[0]
            return {
                "response": f"VigilanceAI Metrics: {flagged} high/critical alerts recorded across {total} total screenings."
            }

        if any(w in q_lower for w in ["recent", "latest", "last", "new", "activity"]):
            cursor.execute("""
                SELECT subject_account, risk_level, primary_typology, timestamp
                FROM screening_audit ORDER BY id DESC LIMIT 5
            """)
            rows = cursor.fetchall()
            if rows:
                items = [f"• {r[0]} [{r[1]}] {r[2]} ({r[3]})" for r in rows]
                return {"response": "Recent Screening Activity:\n" + "\n".join(items)}
            return {"response": "No screening activity recorded yet."}

        cursor.execute("SELECT COUNT(*) FROM screening_audit WHERE risk_level IN ('HIGH', 'CRITICAL')")
        flagged = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM screening_audit")
        total = cursor.fetchone()[0]
        return {
            "response": (
                f"VigilanceAI Compliance Assistant ({flagged} active alerts / {total} evaluations).\n\n"
                "Ask about any specific account (e.g., *'Why is account 30412 flagged?'*), "
                "or request an overview (*'Which accounts are flagged?'*, *'Recent activity'*)."
            )
        }


# --- REST Case & Telemetry Endpoints for Remote/Container Dashboard Access ---

@app.get("/v1/cases")
def list_cases(limit: int = 50):
    # Returns structured case dossiers directly from SQLite
    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT case_id, created_at, account_id, risk_level, deterministic_score,
                   primary_typology, recommended_action, model_reasoning,
                   features_json, triggered_rules_json
            FROM risk_cases ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return {"cases": [dict(r) for r in rows]}


@app.get("/v1/telemetry")
def get_telemetry():
    # Returns portfolio risk metrics directly from SQLite
    with sqlite3.connect(DB_FILE, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT risk_level, COUNT(*) as count FROM screening_audit GROUP BY risk_level")
        rows = cursor.fetchall()
        return {"counts": [{"risk_level": r[0], "count": r[1]} for r in rows]}


@app.get("/v1/batches")
def list_batches():
    return {"batches": get_upload_history(DB_FILE)}


@app.delete("/v1/batches/{batch_id}")
def remove_batch(batch_id: str):
    result = delete_upload_batch(batch_id, DB_FILE)
    return result


@app.delete("/v1/records/all")
def clear_all_data():
    return purge_all_records(DB_FILE)