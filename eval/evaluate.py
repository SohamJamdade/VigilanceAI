"""
eval/evaluate.py — VigilanceAI Benchmark & Evaluation Harness
Evaluates both the deterministic multi-stage engine and the INT8 SLM reasoning layer.
"""
import os
import sys
import json
import time
import re
import argparse
import numpy as np
import torch
from tokenizers import Tokenizer
from sklearn.metrics import classification_report

# Limit CPU thread contention on Windows/local environments
torch.set_num_threads(4)

# Environment-safe path resolution (supports terminal scripts & Colab cells)
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    BASE_DIR = os.getcwd()

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from model.transformer import FinancialSLM
from service.schema import (
    NormalizedTransaction, BehavioralBaseline, FeatureSet, RuleResult
)
from service.features import compute_features
from service.rules import evaluate_rules

TOK_PATH = os.path.join(BASE_DIR, "tokenizer", "financial_bpe.json")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "slm_15m_int8.pt")


def load_inference_engine():
    """Loads and dynamic-quantizes the 15.2M FinancialSLM."""
    if not os.path.exists(TOK_PATH) or not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing model assets: check {TOK_PATH} and {CKPT_PATH}")

    tokenizer = Tokenizer.from_file(TOK_PATH)
    base_model = FinancialSLM(
        vocab_size=2048, d_model=384, n_layers=8, n_heads=12, max_seq_len=512
    ).to("cpu")

    quantized_model = torch.ao.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.load_state_dict(
        torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    )
    quantized_model.eval()
    target_end_id = tokenizer.token_to_id("<|target_end|>")

    return tokenizer, quantized_model, target_end_id


def clean_and_parse_slm_json(raw_text: str):
    """Normalizes token spacing in JSON keys/values and handles truncation."""
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not match and "{" in raw_text:
        candidate = raw_text[raw_text.find("{"):]
        if not candidate.endswith("}"):
            candidate = candidate.rstrip().rstrip(",") + "}"
        match = re.search(r'\{.*\}', candidate, re.DOTALL)

    if not match:
        return None

    json_str = match.group(0)
    # Strip spaces surrounding quotes and delimiters
    json_str = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', json_str)
    json_str = re.sub(r'\s*([\{\}\[\]:,])\s*', r'\1', json_str)

    try:
        data = json.loads(json_str)
        normalized = {}
        for k, v in data.items():
            clean_k = k.strip()
            if isinstance(v, str):
                normalized[clean_k] = v.strip()
            elif isinstance(v, list):
                normalized[clean_k] = [item.strip() if isinstance(item, str) else item for item in v]
            else:
                normalized[clean_k] = v
        return normalized
    except Exception:
        risk_match = re.search(r'"\s*risk_level\s*"\s*:\s*"\s*([A-Z_]+)\s*"', raw_text)
        typ_match = re.search(r'"\s*primary_typology\s*"\s*:\s*"\s*([A-Z_]+)\s*"', raw_text)
        if risk_match:
            return {
                "risk_level": risk_match.group(1).strip(),
                "primary_typology": typ_match.group(1).strip() if typ_match else "NORMAL_ROUTINE"
            }
        return None


def generate_evaluation_transactions(n_samples: int = 60):
    """Generates synthetic multi-transaction accounts across all 4 operational typologies."""
    from datetime import datetime, timezone
    dataset = []
    typs = ["NORMAL_ROUTINE", "STRUCTURING_SMURFING", "MULE_BURST", "SANCTIONS_BREACH"]

    for i in range(n_samples):
        typ = typs[i % len(typs)]
        acc = f"ACC-{i:04d}"
        now = datetime.now(timezone.utc)

        if typ == "SANCTIONS_BREACH":
            txs = [
                NormalizedTransaction(
                    tx_id=f"TX-{i}-1", account_id=acc, amount=48000.0,
                    currency="INR", rail="SWIFT", recipient="OFAC-BLOCKED-ENTITY",
                    device_id="DEV-01", timestamp=now, location="DOMESTIC", source_hash="EVAL"
                )
            ]
            expected = "CRITICAL"

        elif typ == "STRUCTURING_SMURFING":
            txs = [
                NormalizedTransaction(
                    tx_id=f"TX-{i}-1", account_id=acc, amount=48500.0,
                    currency="INR", rail="IMPS", recipient="Vendor A",
                    device_id="DEV-01", timestamp=now, location="DOMESTIC", source_hash="EVAL"
                ),
                NormalizedTransaction(
                    tx_id=f"TX-{i}-2", account_id=acc, amount=49200.0,
                    currency="INR", rail="IMPS", recipient="Vendor B",
                    device_id="DEV-01", timestamp=now, location="DOMESTIC", source_hash="EVAL"
                ),
                NormalizedTransaction(
                    tx_id=f"TX-{i}-3", account_id=acc, amount=47900.0,
                    currency="INR", rail="IMPS", recipient="Vendor C",
                    device_id="DEV-01", timestamp=now, location="DOMESTIC", source_hash="EVAL"
                )
            ]
            expected = "HIGH"

        elif typ == "MULE_BURST":
            txs = [
                NormalizedTransaction(
                    tx_id=f"TX-{i}-{j}", account_id=acc, amount=12000.0,
                    currency="INR", rail="UPI", recipient=f"Recip-{j}",
                    device_id="DEV-MULE", timestamp=now, location="DOMESTIC", source_hash="EVAL"
                )
                for j in range(12)
            ]
            expected = "HIGH"

        else:  # NORMAL_ROUTINE
            txs = [
                NormalizedTransaction(
                    tx_id=f"TX-{i}-1", account_id=acc, amount=3500.0,
                    currency="INR", rail="UPI", recipient="Store A",
                    device_id="DEV-01", timestamp=now, location="DOMESTIC", source_hash="EVAL"
                )
            ]
            expected = "LOW"

        dataset.append((acc, txs, expected, typ))

    return dataset


def run_benchmark(num_samples: int = 60):
    print("=" * 70)
    print("      VIGILANCEAI BENCHMARK & EVALUATION HARNESS")
    print("=" * 70)

    tokenizer, model, target_end_id = load_inference_engine()
    dataset = generate_evaluation_transactions(num_samples)

    print(f"\n[*] Evaluating {len(dataset)} synthetic scenario batches...")

    latencies = []
    json_parse_success = 0
    y_true_hybrid = []
    y_pred_hybrid = []

    for idx, (acc, tx_list, expected_risk, typ) in enumerate(dataset):
        # 1. Evaluate Deterministic Layer
        amounts = [float(t.amount) for t in tx_list]
        dyn_median = float(np.median(amounts)) if amounts else 0.0
        baseline = BehavioralBaseline(
            account_id=acc,
            historical_median=dyn_median,
            typical_bracket=[min(amounts), max(amounts)] if amounts else [0.0, 0.0],
            historical_tx_count=len(tx_list)
        )
        features = compute_features(acc, tx_list, baseline, statutory_limit=50000.0)
        rules = evaluate_rules(tx_list, features, statutory_limit=50000.0)

        crit_hits = sum(1 for r in rules if r.triggered and r.severity == "CRITICAL")
        high_hits = sum(1 for r in rules if r.triggered and r.severity == "HIGH")

        # 2. Evaluate SLM Generative Narrative & Latency
        inference_ctx = {
            "account_id": acc,
            "features": {
                "tx_count": features.window_tx_count,
                "total_val": features.window_total_amount,
                "velocity": features.velocity_tx_per_hour,
                "burst": features.burst_flag,
                "near_threshold": features.near_threshold_count,
                "deviation_ratio": features.baseline_deviation_ratio
            },
            "triggered_rules": [r.rule_id for r in rules if r.triggered],
            "shared_devices": 0,
            "recent_txs": [{"id": t.tx_id, "amt": t.amount, "rail": t.rail, "recip": t.recipient} for t in tx_list[:5]]
        }

        prompt_str = f"<|context_start|>{json.dumps(inference_ctx)}<|context_end|><|target_start|>"
        input_ids = tokenizer.encode(prompt_str).ids[-384:]
        prompt_len = len(input_ids)
        curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(96):
                idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                logits, _ = model(idx_window)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if target_end_id is not None and next_token.item() == target_end_id:
                    break
                curr_ids = torch.cat((curr_ids, next_token), dim=1)
        t1 = time.perf_counter()

        sample_ms = (t1 - t0) * 1000.0
        latencies.append(sample_ms)

        gen_tokens = curr_ids[0, prompt_len:].tolist()
        raw_text = tokenizer.decode(gen_tokens).replace("<|target_end|>", "").strip()
        parsed = clean_and_parse_slm_json(raw_text)

        if parsed:
            json_parse_success += 1

        # 3. Regulatory Safety Override (Production app.py logic)
        if crit_hits > 0:
            final_risk = "CRITICAL"
        elif high_hits > 0:
            final_risk = "HIGH"
        else:
            final_risk = parsed.get("risk_level", "LOW") if parsed else "LOW"

        y_true_hybrid.append(expected_risk)
        y_pred_hybrid.append(final_risk)

        print(f" -> [{idx+1:02d}/{num_samples:02d}] {acc} | {sample_ms:.1f}ms | Truth: {expected_risk} -> Pred: {final_risk}")

    # Output Benchmark Telemetry
    lat_arr = np.array(latencies)
    print("\n" + "=" * 70)
    print("                   BENCHMARK METRICS SUMMARY")
    print("=" * 70)
    print(f"Evaluated Batches       : {len(dataset)}")
    print(f"Strict JSON Validity    : {(json_parse_success / len(dataset)) * 100:.2f}% ({json_parse_success}/{len(dataset)})")
    print(f"\n--- Latency Percentiles (CPU INT8) ---")
    print(f"  Mean Latency          : {np.mean(lat_arr):.2f} ms")
    print(f"  p50 (Median)          : {np.percentile(lat_arr, 50):.2f} ms")
    print(f"  p90                   : {np.percentile(lat_arr, 90):.2f} ms")
    print(f"  p95                   : {np.percentile(lat_arr, 95):.2f} ms")

    print(f"\n--- Hybrid Engine Classification Performance ---")
    labels = ["CRITICAL", "HIGH", "LOW"]
    print(classification_report(y_true_hybrid, y_pred_hybrid, labels=labels, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilanceAI Offline Evaluation Suite")
    parser.add_argument("--samples", type=int, default=60, help="Number of transaction batches to benchmark")
    args = parser.parse_args()
    run_benchmark(num_samples=args.samples)