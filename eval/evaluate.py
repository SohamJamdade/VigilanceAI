import os
import sys
import json
import torch
from tokenizers import Tokenizer

# Project path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

def run_inference(scenario_dict, model, tokenizer, device, max_new_tokens=200):
    prompt_str = f"<|context_start|>{json.dumps(scenario_dict)}<|context_end|><|target_start|>"
    prompt_tokens = tokenizer.encode(prompt_str).ids
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Get ID of the end token
    target_end_id = tokenizer.token_to_id("<|target_end|>")

    model.eval()
    with torch.no_grad():
        # Step-by-step generation with early exit on stop token
        curr_ids = input_ids
        for _ in range(max_new_tokens):
            idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.1
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if target_end_id is not None and next_token.item() == target_end_id:
                break
                
            curr_ids = torch.cat((curr_ids, next_token), dim=1)

    decoded = tokenizer.decode(curr_ids[0].tolist())

    if "<|target_start|>" in decoded:
        response = decoded.split("<|target_start|>")[1]
        if "<|target_end|>" in response:
            response = response.split("<|target_end|>")[0]
        return response.strip()
    return decoded
def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("VigilanceAI: 15.2M SLM Pattern Inference & Robustness Harness")
    print(f"Hardware Device: {device}")
    print("=" * 75)

    tokenizer_path = "tokenizer/financial_bpe.json"
    checkpoint_path = "checkpoints/slm_15m_final.pt"

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at '{tokenizer_path}'.")
        return

    if not os.path.exists(checkpoint_path):
        print(f"Error: Trained checkpoint '{checkpoint_path}' not found.")
        print("Please train the 15.2M model via 'train.train_stage1' first.")
        return

    # Load Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    # Load 15.2M SLM Model
    model = FinancialSLM(
        vocab_size=2048,
        d_model=384,
        n_layers=8,
        n_heads=12,
        max_seq_len=512
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded trained 15.2M weights from: {checkpoint_path}\n")

    # -------------------------------------------------------------
    # Test Scenario 1: Unseen Structuring / Smurfing Threshold
    # Tests mathematical boundary detection on a novel limit (INR 75,000).
    # -------------------------------------------------------------
    test_structuring = {
        "subject_account": "ACC-99104",
        "statutory_limit": 75000,
        "batch_records": [
            {"tx_id": "TXN-A101", "amount": 73400, "rail": "IMPS", "recipient": "BENEF-881", "device": "DEV-UNMAPPED-44", "time": "03:12:00"},
            {"tx_id": "TXN-A102", "amount": 74100, "rail": "IMPS", "recipient": "BENEF-881", "device": "DEV-UNMAPPED-44", "time": "03:14:00"},
            {"tx_id": "TXN-A103", "amount": 72800, "rail": "IMPS", "recipient": "BENEF-881", "device": "DEV-UNMAPPED-44", "time": "03:16:00"}
        ],
        "applied_policy": "POL-STRUC-REG: Mandatory escalation for deliberate threshold avoidance near INR 75,000."
    }

    # -------------------------------------------------------------
    # Test Scenario 2: Adversarial Benign Entity Resolution
    # Tests whether 'Surya Narayanan' is flagged erroneously as Syria or cleared.
    # -------------------------------------------------------------
    test_benign_entity = {
        "audit_account": "ACC-55210",
        "transaction_event": {
            "tx_id": "TXN-B202",
            "amount": 42000,
            "recipient": "Surya Narayanan",
            "jurisdiction": "INDIA",
            "rail": "UPI"
        },
        "compliance_policy": "SANCTIONS-SCREEN-01: Freeze funds destined for high-risk embargoed jurisdictions."
    }

    # -------------------------------------------------------------
    # Test Scenario 3: Mitigated High-Value Anomaly
    # Tests whether verified prior invoices properly result in VERIFY_DOCUMENTATION.
    # -------------------------------------------------------------
    test_mitigated = {
        "account_baseline": {
            "account_id": "ACC-33100",
            "customer": "Vikramaditya Logistics",
            "median_tx_amount": 10000,
            "registered_city": "Mumbai",
            "primary_device": "DEV-AUTH-PRIMARY-1"
        },
        "transaction_event": {
            "tx_id": "TXN-C303",
            "amount": 260000,
            "recipient": "Apex Heavy Equipment Corp",
            "rail": "RTGS",
            "device": "DEV-AUTH-PRIMARY-1"
        },
        "documented_justifications": [
            "Advance tax invoice and commercial contract filed 48 hours prior to execution.",
            "Counterparty is an audited Category-A verified corporate entity."
        ],
        "applied_policy": "POL-SPIKE-102: Flag isolated transactions breaching 10x median baseline."
    }

    test_suite = [
        ("TEST 1: UNSEEN STRUCTURING PATTERN (INR 75,000 CEILING)", test_structuring),
        ("TEST 2: ADVERSARIAL BENIGN ENTITY RESOLUTION (FALSE POSITIVE TEST)", test_benign_entity),
        ("TEST 3: MITIGATED HIGH-VALUE SPIKE WITH JUSTIFICATION", test_mitigated)
    ]

    for title, scenario in test_suite:
        print("-" * 75)
        print(f">>> {title}")
        print("-" * 75)
        prediction = run_inference(scenario, model, tok, device)
        print(prediction)
        print()

    print("=" * 75)

if __name__ == "__main__":
    evaluate_all()