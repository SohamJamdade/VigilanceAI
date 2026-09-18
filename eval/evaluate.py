import os
import sys
import json
import random
import torch
from faker import Faker
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

fake = Faker("en_IN")

def generate_random_scenario():
    """Dynamically generates an unseen AML scenario with random values and channels."""
    typology = random.choice(["structuring", "benign_phonetic", "mitigated_spike", "routine"])
    account_id = f"ACC-{random.randint(10000, 99999)}"
    channels = ["IMPS", "UPI", "RTGS", "NEFT"]

    if typology == "structuring":
        # Random ceiling between 50k and 300k
        ceiling = random.choice([50000, 65000, 75000, 100000, 150000, 250000])
        num_splits = random.randint(3, 5)
        records = []
        for i in range(num_splits):
            # Amount strictly 1% to 5% below statutory limit
            amt = int(ceiling * random.uniform(0.95, 0.99))
            records.append({
                "tx_id": f"TXN-{random.randint(1000, 9999)}",
                "amount": amt,
                "rail": random.choice(channels),
                "recipient": f"BENEF-{random.randint(100, 999)}",
                "device": f"DEV-UNMAPPED-{random.randint(10, 99)}",
                "time": f"{random.randint(1, 4):02d}:{random.randint(10, 59):02d}:00"
            })
        return "DYNAMIC STRUCTURING TEST", {
            "subject_account": account_id,
            "statutory_limit": ceiling,
            "batch_records": records,
            "applied_policy": f"POL-STRUC-REG: Mandatory escalation for threshold avoidance near {ceiling}."
        }

    elif typology == "benign_phonetic":
        first_name = random.choice(["Surya", "Kiran", "Nadir", "Aryan", "Farhan"])
        full_name = f"{first_name} {fake.last_name()}"
        return "DYNAMIC FALSE POSITIVE SCREENING TEST", {
            "audit_account": account_id,
            "transaction_event": {
                "tx_id": f"TXN-{random.randint(1000, 9999)}",
                "amount": random.randint(10000, 80000),
                "recipient": full_name,
                "jurisdiction": "INDIA",
                "rail": "UPI"
            },
            "compliance_policy": "SANCTIONS-SCREEN-01: Freeze funds destined for high-risk embargoed jurisdictions."
        }

    elif typology == "mitigated_spike":
        baseline = random.randint(5000, 15000)
        spike_multiplier = random.randint(15, 30)
        spike_amount = baseline * spike_multiplier
        return "DYNAMIC MITIGATED SPIKE TEST", {
            "account_baseline": {
                "account_id": account_id,
                "customer": fake.company(),
                "median_tx_amount": baseline,
                "registered_city": fake.city(),
                "primary_device": "DEV-AUTH-PRIMARY-1"
            },
            "transaction_event": {
                "tx_id": f"TXN-{random.randint(1000, 9999)}",
                "amount": spike_amount,
                "recipient": fake.company(),
                "rail": "RTGS",
                "device": "DEV-AUTH-PRIMARY-1"
            },
            "documented_justifications": [
                "Commercial invoice filed prior to execution.",
                "Counterparty is an audited Category-A entity."
            ],
            "applied_policy": "POL-SPIKE-102: Flag transactions breaching 10x median baseline."
        }

    else:
        return "DYNAMIC ROUTINE BASELINE TEST", {
            "subject_account": account_id,
            "transaction_event": {
                "tx_id": f"TXN-{random.randint(1000, 9999)}",
                "amount": random.randint(200, 4500),
                "recipient": fake.name(),
                "jurisdiction": "INDIA",
                "rail": "UPI"
            },
            "applied_policy": "POL-ROUTINE: Flag uncharacteristic activity."
        }

def run_inference(scenario_dict, model, tokenizer, device, max_new_tokens=140):
    prompt_str = f"<|context_start|>{json.dumps(scenario_dict)}<|context_end|><|target_start|>"
    prompt_tokens = tokenizer.encode(prompt_str).ids
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    target_end_id = tokenizer.token_to_id("<|target_end|>")

    model.eval()
    curr_ids = input_ids
    with torch.no_grad():
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
        res = decoded.split("<|target_start|>")[1]
        if "<|target_end|>" in res:
            res = res.split("<|target_end|>")[0]
        return res.strip()
    return decoded

def evaluate_dynamic(iterations=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file("tokenizer/financial_bpe.json")
    model = FinancialSLM(
        vocab_size=2048, d_model=384, n_layers=8, n_heads=12, max_seq_len=512
    ).to(device)

    ckpt_path = "checkpoints/slm_15m_final.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    print("=" * 75)
    print(f"Running {iterations} Completely Random Dynamic Scenarios...")
    print("=" * 75)

    for i in range(iterations):
        title, scenario = generate_random_scenario()
        print(f"\n[{i+1}/{iterations}] >>> {title}")
        print("Input Scenario Payload:")
        print(json.dumps(scenario, indent=2))
        print("\nModel Decision:")
        decision = run_inference(scenario, model, tok, device)
        print(decision)
        print("-" * 75)

if __name__ == "__main__":
    evaluate_dynamic(iterations=3)