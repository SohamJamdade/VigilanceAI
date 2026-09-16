import json
import random
import os
from faker import Faker
import numpy as np

fake = Faker('en_IN')
Faker.seed(42)
random.seed(42)
np.random.seed(42)

SANCTIONED_CONFIGS = [
    {"country": "IRAN", "root": "Iran", "benign_first": ["Kiran", "Mirana", "Irani", "Sirani"]},
    {"country": "SYRIA", "root": "Syria", "benign_first": ["Suriya", "Surya", "Syriac"]},
    {"country": "CUBA", "root": "Cuba", "benign_first": ["Yakub", "Kuber", "Cuban"]}
]

DOMESTIC_CITIES = ["Mumbai", "Pune", "Delhi", "Bengaluru", "Hyderabad", "Chennai"]

def generate_account_baseline(account_id):
    median_amt = random.choice([1500, 3000, 5500, 12000, 25000])
    home_city = random.choice(DOMESTIC_CITIES)
    primary_device = f"DEV-{fake.hexify(text='^^^^^^^^')}"
    return {
        "account_id": account_id,
        "customer_name": fake.name(),
        "median_tx_amount": median_amt,
        "typical_range": [int(median_amt * 0.2), int(median_amt * 2.5)],
        "home_city": home_city,
        "primary_device": primary_device,
        "typical_tx_per_day": random.randint(2, 6)
    }

def build_scenario_normal(account):
    tx_count = random.randint(1, 3)
    txs = []
    for _ in range(tx_count):
        txs.append({
            "tx_id": f"TX-{random.randint(10000, 99999)}",
            "amount": int(np.random.uniform(account["typical_range"][0], account["typical_range"][1])),
            "recipient": fake.name(),
            "recipient_type": "PERSON",
            "location": account["home_city"],
            "device_id": account["primary_device"],
            "timestamp": f"1{random.randint(0, 8)}:{random.randint(10, 59)}:00"
        })
    
    context = {
        "account_baseline": account,
        "transactions": txs,
        "network_edges": [],
        "retrieved_policies": ["POL-AML-01: Standard threshold monitoring."]
    }
    
    target = {
        "risk_level": "LOW",
        "primary_typology": "NONE",
        "supporting_evidence": ["All transactions within 30-day baseline range", "Recognized primary device and geo-location"],
        "counter_evidence": ["Standard temporal intervals observed"],
        "recommended_action": "CLEAR"
    }
    return context, target

def build_scenario_mitigated_anomaly(account):
    multiplier = random.randint(12, 28)
    spike_amt = account["median_tx_amount"] * multiplier
    merchant_name = f"{fake.company()} {random.choice(['Motors', 'Jewellers', 'Enterprises', 'Luxury Retail'])}"
    
    txs = [{
        "tx_id": f"TX-{random.randint(10000, 99999)}",
        "amount": spike_amt,
        "recipient": merchant_name,
        "recipient_type": "MERCHANT_VERIFIED",
        "location": account["home_city"],
        "device_id": account["primary_device"],
        "timestamp": f"{random.randint(10, 18)}:{random.randint(10, 59)}:10"
    }]
    
    context = {
        "account_baseline": account,
        "transactions": txs,
        "network_edges": [],
        "documented_counter_evidence": [
            "Customer filed advance purchase notice for high-value acquisition",
            "Merchant is an audited Category-A registered entity"
        ],
        "retrieved_policies": ["POL-ANOM-202: High-value transaction review thresholds."]
    }
    
    target = {
        "risk_level": "MEDIUM",
        "primary_typology": "EXPLAINABLE_HIGH_VALUE",
        "supporting_evidence": [f"Single amount INR {spike_amt} deviates >{multiplier}x from median"],
        "counter_evidence": [
            "Prior customer documentation matches transaction category",
            "Merchant is an audited corporate entity",
            "Originated from primary device in home city"
        ],
        "recommended_action": "VERIFY_DOCUMENTATION"
    }
    return context, target

def build_scenario_adversarial_entity(account):
    config = random.choice(SANCTIONED_CONFIGS)
    is_malicious = random.choice([True, False])
    
    if is_malicious:
        tx_amt = random.randint(45000, 150000)
        recipient = f"{fake.city()} {config['root']} Trading LLC"
        tx = {
            "tx_id": f"TX-{random.randint(10000, 99999)}",
            "amount": tx_amt,
            "recipient": recipient,
            "recipient_type": "ORGANIZATION",
            "jurisdiction": config["country"],
            "device_id": account["primary_device"],
            "timestamp": f"{random.randint(9, 17)}:05:00"
        }
        target = {
            "risk_level": "CRITICAL",
            "primary_typology": "SANCTIONS_VIOLATION",
            "supporting_evidence": [f"Direct wire to restricted jurisdiction: {config['country']}", f"Recipient {recipient} flagged on international watchlists"],
            "counter_evidence": [],
            "recommended_action": "FREEZE_AND_ESCALATE"
        }
    else:
        benign_name = f"{random.choice(config['benign_first'])} {fake.last_name()}"
        tx_amt = int(account["median_tx_amount"] * random.uniform(0.8, 1.6))
        tx = {
            "tx_id": f"TX-{random.randint(10000, 99999)}",
            "amount": tx_amt,
            "recipient": benign_name,
            "recipient_type": "INDIVIDUAL_PERSON",
            "jurisdiction": "INDIA",
            "device_id": account["primary_device"],
            "timestamp": f"{random.randint(9, 17)}:05:00"
        }
        target = {
            "risk_level": "LOW",
            "primary_typology": "BENIGN_ENTITY_OVERLAP",
            "supporting_evidence": [f"Substring match with {config['root']} token detected by rules engine"],
            "counter_evidence": [
                f"Entity resolution confirms recipient is an individual citizen ({benign_name})",
                "Transaction jurisdiction is domestic (INDIA)",
                "Amount consistent with baseline"
            ],
            "recommended_action": "DISMISS_FALSE_POSITIVE"
        }

    context = {
        "account_baseline": account,
        "transactions": [tx],
        "network_edges": [],
        "retrieved_policies": ["POL-SANC-01: Freeze on transactions linked to prohibited jurisdictions."]
    }
    return context, target

def build_scenario_structuring(account):
    ceiling = random.choice([50000, 100000])
    num_txs = random.randint(3, 5)
    base_minute = random.randint(5, 30)
    
    txs = []
    total_amount = 0
    for i in range(num_txs):
        # Generate randomized amounts 1% to 4% beneath the statutory ceiling
        amt = ceiling - random.randint(500, 2200)
        total_amount += amt
        txs.append({
            "tx_id": f"TX-{random.randint(10000, 99999)}",
            "amount": amt,
            "recipient": f"ACC-{random.randint(800, 999)}",
            "recipient_type": "INTERMEDIARY",
            "location": random.choice(DOMESTIC_CITIES),
            "device_id": f"DEV-UNSEEN-{i}",
            "timestamp": f"03:{base_minute + (i * 2):02d}:15"
        })
        
    context = {
        "account_baseline": account,
        "transactions": txs,
        "network_edges": [f"{account['account_id']} -> {tx['recipient']}" for tx in txs],
        "retrieved_policies": [
            f"POL-AML-40: Mandatory reporting on structuring transactions hovering below INR {ceiling:,} threshold."
        ]
    }
    
    target = {
        "risk_level": "HIGH",
        "primary_typology": "STRUCTURING_SMURFING",
        "supporting_evidence": [
            f"{num_txs} consecutive transfers under INR {ceiling:,} ceiling totaling INR {total_amount:,}",
            f"Execution velocity: {num_txs} transactions in {num_txs * 2} minutes",
            "Unrecognized devices and off-hours execution (03:00 AM)"
        ],
        "counter_evidence": [],
        "recommended_action": "INVESTIGATOR_REVIEW"
    }
    return context, target

def generate_master_dataset(total_samples=5000):
    os.makedirs("data", exist_ok=True)
    out_file = "data/financial_risk_corpus.jsonl"
    print(f"Generating {total_samples} dynamic synthetic scenarios...")

    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(total_samples):
            acct = generate_account_baseline(f"ACC-{i:05d}")
            scenario_type = random.choices(
                ["normal", "mitigated", "adversarial_entity", "structuring"],
                weights=[0.40, 0.20, 0.20, 0.20]
            )[0]
            
            if scenario_type == "normal":
                context, target = build_scenario_normal(acct)
            elif scenario_type == "mitigated":
                context, target = build_scenario_mitigated_anomaly(acct)
            elif scenario_type == "adversarial_entity":
                context, target = build_scenario_adversarial_entity(acct)
            else:
                context, target = build_scenario_structuring(acct)

            record = {
                "scenario_id": f"SCN-{i:06d}",
                "input_context": context,
                "investigation_target": target
            }
            f.write(json.dumps(record) + "\n")

    print(f"Dataset generated successfully at {out_file} ({total_samples} samples).")

if __name__ == "__main__":
    generate_master_dataset(total_samples=5000)