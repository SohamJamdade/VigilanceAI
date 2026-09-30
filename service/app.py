import os
import sys
import json
import math
import re
import time
import uuid
import hashlib
import secrets
import sqlite3
import numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, status, HTTPException, Header, Query, Depends
from pydantic import BaseModel, Field
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
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

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
TOKENIZER_PATH = os.getenv("VIGILANCE_TOKENIZER", os.path.join(PROJECT_ROOT, "tokenizer", "financial_bpe.json"))
MAX_NEW_TOKENS = int(os.getenv("VIGILANCE_MAX_NEW_TOKENS", "96"))
# Wall-clock cap per generation so one slow account can't hit the dashboard's 120s request timeout.
SLM_TIMEOUT_S = float(os.getenv("VIGILANCE_SLM_TIMEOUT_S", "60"))

# Model Architecture & Checkpoint Variants  (switch with:  set SLM_VARIANT=15m  /  130m)
SLM_VARIANT = os.getenv("SLM_VARIANT", "130m").strip().lower()

MODEL_CONFIGS = {
    "15m":  {"vocab_size": 2048, "d_model": 384, "n_layers": 8,  "n_heads": 12, "max_seq_len": 512},
    "130m": {"vocab_size": 2048, "d_model": 768, "n_layers": 16, "n_heads": 12, "max_seq_len": 512},
}
CKPT_MAP = {
    "15m":  os.getenv("VIGILANCE_CKPT_15M",  os.path.join(PROJECT_ROOT, "checkpoints", "slm_15m_int8.pt")),
    "130m": os.getenv("VIGILANCE_CKPT_130M", os.path.join(PROJECT_ROOT, "checkpoints", "slm_130m_int8.pt")),
}
PROMPT_TOKEN_BUDGET = 384
DEFAULT_STATUTORY_LIMIT = 50000.0
MAX_BATCH_RECORDS = 5000

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
ALLOWED_ACTIONS = {"AUTO_CLEAR", "MONITOR", "MANUAL_REVIEW", "ESCALATE_TO_L2", "ESCALATE_TO_FIU", "BLOCK_IMMEDIATELY"}
ESCALATING_ACTIONS = ("ESCALATE_TO_L2", "ESCALATE_TO_FIU", "BLOCK_IMMEDIATELY")

runtime_state: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def safe_dump_dict(obj: Any) -> Dict[str, Any]:
    # Pydantic v2/v1 safe dict conversion
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def _safe_json(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


@contextmanager
def db_conn(row_factory: bool = False):
    """SQLite connection that is always closed (sqlite3's own `with` only commits)."""
    conn = sqlite3.connect(DB_FILE, timeout=5.0)
    if row_factory:
        conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _parse_ts(value: Any) -> datetime:
    """Use the record's real timestamp (ISO string or epoch); fall back to 'now'."""
    now = datetime.now(timezone.utc)
    if value in (None, ""):
        return now
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return now


def require_admin(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("VIGILANCE_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Destructive endpoints are disabled: set VIGILANCE_API_KEY.")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


# ─────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────
def _load_checkpoint(model: torch.nn.Module, path: str):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[WARN] weights_only=True load failed ({e}); retrying with weights_only=False. "
              f"Only do this for checkpoints you trust.")
        state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)


def _build_quantized_model(variant: str):
    base_model = FinancialSLM(**MODEL_CONFIGS[variant]).to("cpu")
    quantized = torch.ao.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    _load_checkpoint(quantized, CKPT_MAP[variant])
    quantized.eval()
    return quantized


def load_slm():
    """Load the requested variant; if its checkpoint is missing or incompatible,
    fall back to the 15M model. The architecture always matches the checkpoint used."""
    requested = SLM_VARIANT if SLM_VARIANT in MODEL_CONFIGS else "130m"
    if requested != SLM_VARIANT:
        print(f"[!] Unknown SLM_VARIANT '{SLM_VARIANT}'. Using '{requested}'.")
    order = [requested] + (["15m"] if requested != "15m" else [])

    for variant in order:
        ckpt = CKPT_MAP[variant]
        if not os.path.exists(ckpt):
            print(f"[!] Warning: {ckpt} not found.")
            continue
        try:
            model = _build_quantized_model(variant)
            print(f"[*] Loaded FinancialSLM ({variant.upper()} variant) from {ckpt}")
            return model, variant
        except Exception as e:
            print(f"[!] Failed to load {variant.upper()} checkpoint ({e}).")
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_case_storage(DB_FILE)
    torch.set_num_threads(int(os.getenv("VIGILANCE_TORCH_THREADS", "4")))

    model, variant = (None, None)
    if os.path.exists(TOKENIZER_PATH):
        model, variant = load_slm()
    else:
        print(f"[WARN] Tokenizer missing: {TOKENIZER_PATH}")

    if model is not None:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        runtime_state["tokenizer"] = tokenizer
        runtime_state["model"] = model
        runtime_state["target_end_id"] = tokenizer.token_to_id("<|target_end|>")
        runtime_state["model_tag"] = f"slm_{variant}_int8"
    else:
        # Keep the API (cases, chat, rules) alive; screening runs in rules-only mode.
        print("[WARN] No usable SLM assets. Running rules-only.")

    yield
    runtime_state.clear()


app = FastAPI(title="VigilanceAI Financial Risk Platform", version="2.2.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────
# SLM reasoning (with validation)
# ─────────────────────────────────────────────────────────────
_WORD_FIXES = {
    "ESCALAT E_ TO_ L 2": "ESCALATE_TO_L2",
    "BLOCK_ IMMEDIATE LY": "BLOCK_IMMEDIATELY",
    "AUTO_ CLEAR": "AUTO_CLEAR",
    "Re pe ated": "Repeated",
    "n ar ro w": "narrow",
    "ce i li ng s": "ceilings",
}


def clean_narrative_text(text: str) -> str:
    """Repair BPE sub-word spacing in generated sentences.
    Deliberately does NOT strip spaces after commas/colons (that would glue prose words together)."""
    text = str(text)
    for broken, fixed in _WORD_FIXES.items():
        text = text.replace(broken, fixed)
    text = re.sub(r'(?<=[A-Z0-9])\s+_', '_', text)          # "ESCALAT E _TO" -> "ESCALAT E_TO"
    text = re.sub(r'_\s+(?=[A-Z0-9])', '_', text)            # "TO_ L2" -> "TO_L2"
    text = re.sub(r'\b(sub|non|cross|multi|near)\s+-\s+(?=\w)', r'\1-', text, flags=re.I)  # "sub - threshold"
    text = re.sub(r'\s+([,.;:])', r'\1', text)               # "word ," -> "word,"
    return re.sub(r'\s{2,}', ' ', text).strip()


def _norm_ident(value: Any) -> str:
    """Enum-like fields (risk level, action, typology) never contain spaces."""
    return re.sub(r'\s+', '', str(value or '')).upper()


def _rules_fallback(features: FeatureSet, rules: List[RuleResult]) -> Dict[str, Any]:
    triggered = [r for r in rules if r.triggered]
    high_rule = any(r.severity in ("HIGH", "CRITICAL") for r in triggered)
    return {
        "risk_level": "HIGH" if high_rule else "LOW",
        "primary_typology": "STRUCTURING_SMURFING" if features.near_threshold_count >= 2 else "NORMAL_ROUTINE",
        "recommended_action": "ESCALATE_TO_FIU" if high_rule else "AUTO_CLEAR",
        "supporting_evidence": [r.reason for r in triggered] or ["All activities conform to baseline."],
    }


def _build_prompt_ids(base_ctx: Dict[str, Any], txs: List[NormalizedTransaction], tokenizer) -> List[int]:
    """Shrink the recent-transactions list until the prompt fits, instead of slicing tokens
    (slicing would cut off <|context_start|> and corrupt the prompt)."""
    ids: List[int] = []
    for n in (5, 4, 3, 2, 1, 0):
        ctx = dict(base_ctx)
        ctx["recent_txs"] = [
            {"id": t.tx_id, "amt": t.amount, "rail": t.rail, "recip": t.recipient}
            for t in txs[-n:]
        ] if n else []
        prompt = f"<|context_start|>{json.dumps(ctx)}<|context_end|><|target_start|>"
        ids = tokenizer.encode(prompt).ids
        if len(ids) <= PROMPT_TOKEN_BUDGET:
            return ids
    return ids[-PROMPT_TOKEN_BUDGET:]


def _parse_slm_output(raw_text: str) -> Optional[Dict[str, Any]]:
    spaced = re.sub(r'\s*([\{\}\[\]:,])\s*', r'\1', raw_text)
    # Try untouched text first: regex whitespace stripping can alter string contents.
    for candidate in (raw_text, spaced):
        m = re.search(r'\{.*\}', candidate, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    # Truncated output: salvage the three key fields if they were generated.
    salvaged: Dict[str, Any] = {}
    for key in ("risk_level", "primary_typology", "recommended_action"):
        m = re.search(rf'"?{key}"?\s*:\s*"([^"]+)', raw_text) or re.search(rf'"?{key}"?\s*:\s*"?([A-Za-z_]+)', spaced)
        if m:
            salvaged[key] = _norm_ident(m.group(1))
    return salvaged or None


def _sanitize_decision(raw: Optional[Dict[str, Any]], features: FeatureSet, rules: List[RuleResult]) -> Dict[str, Any]:
    """Never trust raw model output: validate values and keep them consistent with the rules."""
    fallback = _rules_fallback(features, rules)
    if not raw:
        return fallback

    triggered = [r for r in rules if r.triggered]

    risk = _norm_ident(raw.get("risk_level"))
    if risk not in RISK_LEVELS:
        risk = fallback["risk_level"]
    if not triggered and RISK_LEVELS.index(risk) > RISK_LEVELS.index("MEDIUM"):
        risk = "MEDIUM"  # model cannot escalate past MEDIUM with zero rule evidence

    action = _norm_ident(raw.get("recommended_action"))
    if action not in ALLOWED_ACTIONS:
        action = fallback["recommended_action"]
    if not triggered and action in ESCALATING_ACTIONS:
        action = "AUTO_CLEAR" if risk == "LOW" else "MONITOR"

    typology = _norm_ident(raw.get("primary_typology"))
    if not re.fullmatch(r"[A-Z0-9_]{3,60}", typology):
        typology = fallback["primary_typology"]

    evidence = raw.get("supporting_evidence")
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list) or not evidence:
        evidence = fallback["supporting_evidence"]
    evidence = [clean_narrative_text(e) for e in evidence]

    return {
        "risk_level": risk,
        "primary_typology": typology,
        "recommended_action": action,
        "supporting_evidence": evidence,
    }


def synthesize_slm_reasoning(
    account_id: str,
    features: FeatureSet,
    rules: List[RuleResult],
    entity: EntityProfile,
    latest_txs: List[NormalizedTransaction]
) -> Dict[str, Any]:
    raw_decision: Optional[Dict[str, Any]] = None
    try:
        model = runtime_state["model"]
        tokenizer = runtime_state["tokenizer"]
        target_end_id = runtime_state["target_end_id"]

        base_ctx = {
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
        }
        input_ids = _build_prompt_ids(base_ctx, latest_txs, tokenizer)
        prompt_len = len(input_ids)
        curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

        deadline = time.monotonic() + SLM_TIMEOUT_S
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                if time.monotonic() > deadline:
                    print(f"[WARN] SLM generation hit the {SLM_TIMEOUT_S:.0f}s cap; using partial output.")
                    break
                idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                logits, _ = model(idx_window)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if target_end_id is not None and next_token.item() == target_end_id:
                    break
                curr_ids = torch.cat((curr_ids, next_token), dim=1)

        gen_tokens = curr_ids[0, prompt_len:].tolist()
        raw_text = tokenizer.decode(gen_tokens).replace("<|target_end|>", "").strip()
        raw_decision = _parse_slm_output(raw_text)
    except Exception as e:
        print(f"SLM fallback triggered: {e}")

    return _sanitize_decision(raw_decision, features, rules)


# ─────────────────────────────────────────────────────────────
# Baseline from real history
# ─────────────────────────────────────────────────────────────
def _historical_baseline(account_id: str) -> Optional[Tuple[float, List[float], int]]:
    """Baseline from this account's earlier cases (per-case mean amount), so a batch
    is compared with the past rather than with itself."""
    try:
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT features_json FROM risk_cases WHERE UPPER(account_id) = ? "
                "ORDER BY created_at DESC LIMIT 20",
                (account_id.upper(),),
            ).fetchall()
    except sqlite3.Error:
        return None

    case_means: List[float] = []
    total_count = 0
    for (fj,) in rows:
        f = _safe_json(fj, {})
        n = f.get("window_tx_count") or 0
        total = f.get("window_total_amount") or 0
        if n > 0:
            case_means.append(float(total) / float(n))
            total_count += int(n)
    if not case_means:
        return None
    median = float(np.median(case_means))
    return median, [0.0, median * 2], total_count


def _resolve_baseline(account_id: str, txs: List[NormalizedTransaction], raw_payload: Dict[str, Any]) -> BehavioralBaseline:
    history = _historical_baseline(account_id)
    if history:
        median, bracket, count = history
        return BehavioralBaseline(
            account_id=account_id, historical_median=median,
            typical_bracket=bracket, historical_tx_count=count,
        )

    base_obj = raw_payload.get("account_baseline", {}) or {}
    if base_obj.get("historical_median"):
        median = float(base_obj["historical_median"])
        bracket = base_obj.get("typical_bracket", [0.0, median * 2])
    else:
        amounts = [float(t.amount) for t in txs]
        median = float(np.median(amounts)) if amounts else 0.0
        bracket = [float(min(amounts)), float(max(amounts))] if amounts else [0.0, 0.0]

    return BehavioralBaseline(
        account_id=account_id, historical_median=median,
        typical_bracket=bracket, historical_tx_count=len(txs),
    )


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
def _reproducibility_hash(
    account_id: str,
    txs: List[NormalizedTransaction],
    rules: List[RuleResult],
    score: float,
    final_risk: str,
    statutory_limit: float,
) -> str:
    """Hash of the *inputs and outputs* (not the random case id), so identical
    inputs always give an identical hash."""
    material = {
        "model": runtime_state.get("model_tag", "rules_only"),
        "account": account_id,
        "limit": statutory_limit,
        "txs": sorted((t.tx_id, float(t.amount), t.rail, t.recipient, t.device_id) for t in txs),
        "rules": sorted(r.rule_id for r in rules if r.triggered),
        "score": score,
        "risk": final_risk,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_pipeline(
    account_id: str,
    incoming_txs: List[NormalizedTransaction],
    raw_payload: Dict[str, Any],
    statutory_limit: float = DEFAULT_STATUTORY_LIMIT
) -> RiskAssessment:
    case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    baseline = _resolve_baseline(account_id, incoming_txs, raw_payload)

    features = compute_features(account_id, incoming_txs, baseline, statutory_limit)
    rule_results = evaluate_rules(incoming_txs, features, statutory_limit)
    entity_prof = resolve_entity(account_id, incoming_txs, DB_FILE)

    critical_hits = sum(1 for r in rule_results if r.triggered and r.severity == "CRITICAL")
    high_hits = sum(1 for r in rule_results if r.triggered and r.severity == "HIGH")
    deviation = float(features.baseline_deviation_ratio or 0.0)
    raw_score = (critical_hits * 0.7) + (high_hits * 0.3) + (min(deviation, 10.0) * 0.03)
    deterministic_score = round(min(1.0, max(0.0, float(raw_score))), 2)

    slm_decision = synthesize_slm_reasoning(account_id, features, rule_results, entity_prof, incoming_txs)

    final_risk = slm_decision["risk_level"]
    final_typology = slm_decision["primary_typology"]
    final_action = slm_decision["recommended_action"]

    triggered_ids = {r.rule_id for r in rule_results if r.triggered}

    # Regulatory override: critical or high rules dictate risk, action, and typology
    if critical_hits > 0:
        final_risk = "CRITICAL"
        final_action = "BLOCK_IMMEDIATELY"
        final_typology = "SANCTIONS_BREACH" if "RULE_SANCT_003" in triggered_ids else "CRITICAL_REGULATORY_BREACH"
    elif high_hits > 0:
        if final_risk != "CRITICAL":
            final_risk = "HIGH"
        if final_action == "AUTO_CLEAR":
            final_action = "ESCALATE_TO_FIU"
        if final_typology == "NORMAL_ROUTINE":
            if "RULE_STRUCT_001" in triggered_ids:
                final_typology = "STRUCTURING_SMURFING"
            elif "RULE_VEL_002" in triggered_ids:
                final_typology = "HIGH_VELOCITY_BURST"
            else:
                final_typology = "SUSPICIOUS_ACTIVITY"

    evidence = slm_decision["supporting_evidence"]
    slm_reasoning_str = "; ".join(evidence)

    repro_hash = _reproducibility_hash(
        account_id, incoming_txs, rule_results, deterministic_score, final_risk, statutory_limit
    )

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


# ─────────────────────────────────────────────────────────────
# Screening endpoints (plain `def` → run in FastAPI's threadpool,
# so model inference no longer blocks chat / case endpoints)
# ─────────────────────────────────────────────────────────────
def _validated_limit(value: Any) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="statutory_limit must be a number.")
    if not math.isfinite(limit) or limit <= 0:
        raise HTTPException(status_code=400, detail="statutory_limit must be a positive number.")
    return limit


@app.post("/v1/screen", status_code=status.HTTP_200_OK)
def screen_transaction(payload: Dict[str, Any]):
    account_id = str(
        payload.get("subject_account")
        or (payload.get("account_baseline") or {}).get("account_id")
        or ""
    ).strip()
    if not account_id:
        raise HTTPException(status_code=400, detail="subject_account is required.")

    raw_records = payload.get("batch_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise HTTPException(status_code=400, detail="batch_records must be a non-empty list.")
    if len(raw_records) > MAX_BATCH_RECORDS:
        raise HTTPException(status_code=400, detail=f"batch_records exceeds the {MAX_BATCH_RECORDS} record limit.")

    statutory_limit = _validated_limit(payload.get("statutory_limit", DEFAULT_STATUTORY_LIMIT))

    tx_list: List[NormalizedTransaction] = []
    for i, r in enumerate(raw_records):
        if not isinstance(r, dict):
            raise HTTPException(status_code=400, detail=f"batch_records[{i}] must be an object.")
        try:
            amount = float(r.get("amount"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"batch_records[{i}].amount is missing or not a number.")
        if not math.isfinite(amount) or amount < 0:
            raise HTTPException(status_code=400, detail=f"batch_records[{i}].amount must be a finite, non-negative number.")

        tx_list.append(NormalizedTransaction(
            tx_id=str(r.get("tx_id") or f"TXN-{uuid.uuid4().hex[:6]}"),
            account_id=account_id,
            amount=amount,
            currency=str(r.get("currency", "INR")),
            rail=str(r.get("rail", "IMPS")).upper(),
            recipient=str(r.get("recipient") or r.get("to") or "UNKNOWN"),
            device_id=str(r.get("device_id") or r.get("device") or "DEV-UNKNOWN"),
            timestamp=_parse_ts(r.get("timestamp")),
            location=str(r.get("location", "DOMESTIC")),
            source_hash="REST_SCREEN"
        ))

    assessment = run_pipeline(account_id, tx_list, payload, statutory_limit)

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
    account_id: str = Field(..., min_length=1)
    tx_id: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0)
    rail: str = "IMPS"
    recipient: str = Field(..., min_length=1)
    device_id: str = "DEV-MOBILE"
    statutory_limit: float = Field(DEFAULT_STATUTORY_LIMIT, gt=0)


@app.post("/v1/ingest", status_code=status.HTTP_200_OK)
def ingest_transaction(event: SingleTransactionEvent):
    if not math.isfinite(event.amount):
        raise HTTPException(status_code=400, detail="amount must be a finite number.")

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
        # default=str makes the datetime JSON-safe for the case store
        "batch_records": [json.loads(json.dumps(safe_dump_dict(tx), default=str))],
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


# ─────────────────────────────────────────────────────────────
# Compliance chat
# ─────────────────────────────────────────────────────────────
class ChatQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


def _resolve_account(query: str, cursor: sqlite3.Cursor) -> Tuple[Optional[str], List[str]]:
    """Returns (account, ambiguous_candidates). Whole-token matching only,
    so '100' no longer matches ACC-10001 and ACC-10002."""
    cursor.execute(
        "SELECT DISTINCT subject_account FROM screening_audit "
        "UNION SELECT DISTINCT account_id FROM risk_cases"
    )
    known = [r[0] for r in cursor.fetchall() if r[0]]
    if not known:
        return None, []

    q_upper = query.upper()

    # 1) Full account id appearing in the query (longest wins).
    exact = [
        acc for acc in known
        if re.search(rf'(?<![A-Z0-9]){re.escape(acc.upper())}(?![A-Z0-9])', q_upper)
    ]
    if exact:
        return max(exact, key=len), []

    # 2) A number in the query that is a whole numeric segment of exactly one account.
    candidates: List[str] = []
    for token in re.findall(r'(?<![A-Za-z0-9])\d{3,8}(?![A-Za-z0-9])', query):
        for acc in known:
            if re.search(rf'(?<!\d){re.escape(token)}(?!\d)', acc) and acc not in candidates:
                candidates.append(acc)
    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) > 1:
        return None, candidates
    return None, []


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def _format_case_report(case_row: tuple) -> str:
    (case_id, created_at, acc_id, risk_level, score,
     typology, action, reasoning, features_json, rules_json, entity_json) = case_row

    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")
    lines = [
        f"### {icon} Detailed Investigation Report: **{acc_id}**",
        f"- **Risk Classification:** **{risk_level}** (Deterministic Score: `{float(score or 0):.2f}`)",
        f"- **Primary Typology:** `{typology}`",
        f"- **Recommended Regulatory Action:** `{action}`",
        f"- **Case Reference:** `{case_id}` (Screened: {created_at})",
        "",
        "#### 🔍 Why It Was Flagged (Concrete Evidence & Triggered Rules):"
    ]

    rules_data = _safe_json(rules_json, [])
    triggered = [r for r in rules_data if isinstance(r, dict) and r.get("triggered")]
    if triggered:
        for idx, r in enumerate(triggered, 1):
            lines.append(f"{idx}. 🚨 **{r.get('rule_name', 'Unknown Rule')}** "
                         f"(`{r.get('rule_id', 'N/A')}` — {r.get('severity', 'HIGH')})")
            lines.append(f"   - **Rule Finding:** {r.get('reason', 'Rule triggered.')}")
            ev = r.get("evidence") or {}
            if ev:
                lines.append("   - **Evidence:** " + ", ".join(f"{k}: `{v}`" for k, v in ev.items()))
    else:
        lines.append("• No deterministic compliance rules were triggered. Transactions conform to baseline thresholds.")

    f = _safe_json(features_json, {})
    if f:
        lines.extend([
            "",
            "#### 📊 Transaction Behavior & Feature Metrics:",
            f"- **Analyzed Transactions:** {f.get('window_tx_count') or 0} "
            f"(Total Volume: ₹{float(f.get('window_total_amount') or 0):,.2f})",
            f"- **Transaction Velocity:** {float(f.get('velocity_tx_per_hour') or 0):.1f} tx/hr "
            f"(Burst detected: `{f.get('burst_flag', False)}`)",
            f"- **Near-Threshold Transfers:** {f.get('near_threshold_count') or 0} close to the statutory limit",
            f"- **Historical Deviation Ratio:** {f.get('baseline_deviation_ratio') or 1.0}x historical median"
        ])

    entity = _safe_json(entity_json, {})
    shared = entity.get("shared_device_accounts", []) if isinstance(entity, dict) else []
    if shared:
        lines.append(f"- ⚠️ **Entity Linkage Alert:** Hardware device shared with other account(s): {', '.join(map(str, shared))}")

    if reasoning and str(reasoning).strip():
        lines.extend(["", f"#### 🧠 Contextual Synthesis:\n{reasoning}"])

    return "\n".join(lines)


def _chat_answer(q: str, cursor: sqlite3.Cursor) -> str:
    q_lower = q.lower()
    target_account, ambiguous = _resolve_account(q, cursor)

    if ambiguous:
        return ("That number matches more than one account: "
                + ", ".join(f"`{a}`" for a in ambiguous[:10])
                + ". Which one do you mean?")

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
            return _format_case_report(case_row)

        cursor.execute("""
            SELECT risk_level, primary_typology, recommended_action, evidence_summary, timestamp
            FROM screening_audit WHERE UPPER(subject_account) = ? ORDER BY id DESC LIMIT 1
        """, (target_account.upper(),))
        audit_row = cursor.fetchone()
        if audit_row:
            risk, typ, act, ev_summary, ts = audit_row
            ev_list = _safe_json(ev_summary, ev_summary)
            ev_str = "; ".join(map(str, ev_list)) if isinstance(ev_list, list) else str(ev_list)
            return (
                f"### Investigation Report: **{target_account}**\n"
                f"- **Risk Level:** {risk}\n"
                f"- **Typology:** {typ}\n"
                f"- **Recommended Action:** {act}\n"
                f"- **Last Screened:** {ts}\n"
                f"- **Evidence Summary:** {ev_str}"
            )

    # Counts first, so "how many accounts are flagged?" gets a number, not a list.
    if _has(r"\bhow many\b|\bcount\b|\btotal\b|\bnumber of\b", q_lower):
        cursor.execute("SELECT COUNT(*) FROM screening_audit WHERE risk_level IN ('HIGH', 'CRITICAL')")
        flagged = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM screening_audit")
        total = cursor.fetchone()[0]
        return f"VigilanceAI Metrics: {flagged} high/critical alerts recorded across {total} total screenings."

    if _has(r"\b(recent|latest|last|newest|activity)\b", q_lower):
        cursor.execute("""
            SELECT subject_account, risk_level, primary_typology, timestamp
            FROM screening_audit ORDER BY id DESC LIMIT 5
        """)
        rows = cursor.fetchall()
        if rows:
            return "Recent Screening Activity:\n" + "\n".join(f"• {r[0]} [{r[1]}] {r[2]} ({r[3]})" for r in rows)
        return "No screening activity recorded yet."

    if _has(r"\b(which|flagged|who|list|show|cases?|accounts?|alerts?|high|critical)\b", q_lower):
        cursor.execute("""
            SELECT subject_account, risk_level, primary_typology, recommended_action
            FROM screening_audit
            WHERE risk_level IN ('HIGH', 'CRITICAL')
            ORDER BY id DESC LIMIT 15
        """)
        rows = cursor.fetchall()
        if rows:
            # one line per account (most recent screening wins)
            seen, items = set(), []
            for r in rows:
                if r[0] in seen:
                    continue
                seen.add(r[0])
                items.append(f"• **{r[0]}** [{r[1]}] — {r[2]} → `{r[3]}`")
            return (f"The following {len(items)} account(s) are currently flagged as High/Critical:\n\n"
                    + "\n".join(items))
        return "All screened accounts currently conform to baseline. No high-risk alerts."

    cursor.execute("SELECT COUNT(*) FROM screening_audit WHERE risk_level IN ('HIGH', 'CRITICAL')")
    flagged = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM screening_audit")
    total = cursor.fetchone()[0]
    return (
        f"VigilanceAI Compliance Assistant ({flagged} active alerts / {total} evaluations).\n\n"
        "Ask about any specific account (e.g., *'Why is account 30412 flagged?'*), "
        "or request an overview (*'Which accounts are flagged?'*, *'Recent activity'*)."
    )


@app.post("/v1/chat")
def compliance_assistant_chat(req: ChatQuery):
    try:
        with db_conn() as conn:
            answer = _chat_answer(req.query.strip(), conn.cursor())
    except sqlite3.Error as e:
        return {"response": f"The case database isn't available yet ({e}). Screen a batch first."}
    return {"response": answer}


# ─────────────────────────────────────────────────────────────
# REST case & telemetry endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/v1/cases")
def list_cases(limit: int = Query(50, ge=1, le=500)):
    with db_conn(row_factory=True) as conn:
        rows = conn.execute("""
            SELECT case_id, created_at, account_id, risk_level, deterministic_score,
                   primary_typology, recommended_action, model_reasoning,
                   features_json, triggered_rules_json
            FROM risk_cases ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return {"cases": [dict(r) for r in rows]}


@app.get("/v1/telemetry")
def get_telemetry():
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT risk_level, COUNT(*) AS count FROM screening_audit GROUP BY risk_level"
        ).fetchall()
        return {"counts": [{"risk_level": r[0], "count": r[1]} for r in rows]}


@app.get("/v1/batches")
def list_batches():
    return {"batches": get_upload_history(DB_FILE)}


# Destructive endpoints require the X-API-Key header (set VIGILANCE_API_KEY on the server).
@app.delete("/v1/batches/{batch_id}", dependencies=[Depends(require_admin)])
def remove_batch(batch_id: str):
    return delete_upload_batch(batch_id, DB_FILE)


@app.delete("/v1/records/all", dependencies=[Depends(require_admin)])
def clear_all_data():
    return purge_all_records(DB_FILE) 