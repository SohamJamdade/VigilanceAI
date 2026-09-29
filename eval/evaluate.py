"""
eval/evaluate.py — VigilanceAI Offline Benchmarking & Evaluation Suite
"""
import os
import sys
import json
import time
import re
import numpy as np
import torch
from tokenizers import Tokenizer
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

TOK_PATH = "tokenizer/financial_bpe.json"
CKPT_PATH = "checkpoints/slm_15m_int8.pt"

def load_inference_engine():
    if not os.path.exists(TOK_PATH) or not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing assets: check {TOK_PATH} and {CKPT_PATH}")

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

def generate_benchmark_dataset(n_samples: int = 120):
    dataset = []
    typologies = ["NORMAL_ROUTINE", "STRUCTURING_SMURFING", "MULE_BURST", "SANCTIONS_BREACH"]

    for i in range(n_samples):
        typology = typologies[i % len(typologies)]

        if typology == "SANCTIONS_BREACH":
            context = {
                "account_id": f"EVAL-SANCT-{i}",
                "features": {"tx_count": 3, "total_val": 45000.0, "velocity": 1.2, "burst": False, "near_threshold": 1, "deviation_ratio": 1.1},
                "triggered_rules": ["RULE_SANCT_003"],
                "shared_devices": 0,
                "recent_txs": [{"id": f"TX-S-{i}", "amt": 45000.0, "rail": "SWIFT", "recip": "OFAC-BLOCKED-ENTITY"}]
            }
            ground_truth = {"risk_level": "CRITICAL", "primary_typology": "SANCTIONS_BREACH"}

        elif typology == "STRUCTURING_SMURFING":
            context = {
                "account_id": f"EVAL-STRUC-{i}",
                "features": {"tx_count": 5, "total_val": 245000.0, "velocity": 4.5, "burst": False, "near_threshold": 4, "deviation_ratio": 3.8},
                "triggered_rules": ["RULE_STRUCT_001"],
                "shared_devices": 0,
                "recent_txs": [{"id": f"TX-ST-{i}", "amt": 49200.0, "rail": "IMPS", "recip": "Vendor Logistics"}]
            }
            ground_truth = {"risk_level": "HIGH", "primary_typology": "STRUCTURING_SMURFING"}

        elif typology == "MULE_BURST":
            context = {
                "account_id": f"EVAL-MULE-{i}",
                "features": {"tx_count": 12, "total_val": 180000.0, "velocity": 18.0, "burst": True, "near_threshold": 0, "deviation_ratio": 6.2},
                "triggered_rules": ["RULE_VEL_002"],
                "shared_devices": 2,
                "recent_txs": [{"id": f"TX-M-{i}", "amt": 15000.0, "rail": "UPI", "recip": "Account A"}]
            }
            ground_truth = {"risk_level": "HIGH", "primary_typology": "MULE_BURST"}

        else:  # NORMAL_ROUTINE
            context = {
                "account_id": f"EVAL-NORM-{i}",
                "features": {"tx_count": 2, "total_val": 12500.0, "velocity": 0.5, "burst": False, "near_threshold": 0, "deviation_ratio": 0.9},
                "triggered_rules": [],
                "shared_devices": 0,
                "recent_txs": [{"id": f"TX-N-{i}", "amt": 6250.0, "rail": "UPI", "recip": "Amazon India"}]
            }
            ground_truth = {"risk_level": "LOW", "primary_typology": "NORMAL_ROUTINE"}

        dataset.append((context, ground_truth))

    return dataset

def run_evaluation(num_samples: int = 120):
    print("=" * 60)
    print("  VIGILANCEAI SLM BENCHMARK & EVALUATION HARNESS")
    print("=" * 60)

    tokenizer, model, target_end_id = load_inference_engine()
    dataset = generate_benchmark_dataset(num_samples)

    latencies = []
    json_parse_success = 0
    y_true_risk, y_pred_risk = [], []
    y_true_typ, y_pred_typ = [], []

    for idx, (ctx, truth) in enumerate(dataset):
        prompt_str = f"<|context_start|>{json.dumps(ctx)}<|context_end|><|target_start|>"
        input_ids = tokenizer.encode(prompt_str).ids[-384:]
        prompt_len = len(input_ids)
        curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(48):
                idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                logits, _ = model(idx_window)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if target_end_id is not None and next_token.item() == target_end_id:
                    break
                curr_ids = torch.cat((curr_ids, next_token), dim=1)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        gen_tokens = curr_ids[0, prompt_len:].tolist()
        raw_text = tokenizer.decode(gen_tokens).replace("<|target_end|>", "").strip()
        clean_text = re.sub(r'\s*([\{\}\[\]:,])\s*', r'\1', raw_text)

        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
                json_parse_success += 1
            except Exception:
                pass

        pred_risk = parsed.get("risk_level", "FALLBACK") if parsed else "FALLBACK"
        pred_typ = parsed.get("primary_typology", "FALLBACK") if parsed else "FALLBACK"

        y_true_risk.append(truth["risk_level"])
        y_pred_risk.append(pred_risk)
        y_true_typ.append(truth["primary_typology"])
        y_pred_typ.append(pred_typ)

    # Compute Latency Metrics
    lat_arr = np.array(latencies)
    print(f"\nTotal Evaluated Samples : {len(dataset)}")
    print(f"Strict JSON Validity    : {(json_parse_success / len(dataset)) * 100:.2f}% ({json_parse_success}/{len(dataset)})")
    print(f"\n--- Latency Percentiles (CPU INT8) ---")
    print(f"  Mean Latency : {np.mean(lat_arr):.2f} ms")
    print(f"  p50 (Median) : {np.percentile(lat_arr, 50):.2f} ms")
    print(f"  p90          : {np.percentile(lat_arr, 90):.2f} ms")
    print(f"  p95          : {np.percentile(lat_arr, 95):.2f} ms")

    print(f"\n--- Classification Metrics (Risk Level) ---")
    labels = sorted(list(set(y_true_risk + y_pred_risk)))
    print(classification_report(y_true_risk, y_pred_risk, labels=labels, zero_division=0))

if __name__ == "__main__":
    run_evaluation(num_samples=120)