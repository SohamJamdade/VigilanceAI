import json
import random
import os
import numpy as np
from faker import Faker

fake = Faker(['en_IN', 'en_US', 'en_GB'])
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# -------------------------------------------------------------
# 1. ENTITY CONFIGURATIONS & HIGH-ENTROPY POOLS
# -------------------------------------------------------------
SANCTIONED_TARGETS = [
    {
        "jurisdiction": "IRAN",
        "tokens": ["Iran", "Tehran", "Persia"],
        "benign_names": ["Kiran Deshmukh", "Irani Marine Exports", "Mirana Rao", "Samiha Irani", "Kiran Patel"]
    },
    {
        "jurisdiction": "SYRIA",
        "tokens": ["Syria", "Damascus", "Levant"],
        "benign_names": ["Surya Narayanan", "Suriya Logistics", "Syriac Thomas", "Suraj Sharma", "Surya Prakash"]
    },
    {
        "jurisdiction": "CUBA",
        "tokens": ["Cuba", "Havana"],
        "benign_names": ["Yakub Merchant", "Kuber Enterprises", "Cuban Cafe Pune", "Akub Memon"]
    },
    {
        "jurisdiction": "NORTH_KOREA",
        "tokens": ["DPRK", "Pyongyang"],
        "benign_names": ["Koryo Traders", "Parkash Kim", "Pyo Industries"]
    },
    {
        "jurisdiction": "MYANMAR_SANCTIONED",
        "tokens": ["Myanmar", "Yangon", "Burma"],
        "benign_names": ["Mayur Marine", "Burman Brothers", "Yashwant Gore"]
    }
]

CHANNELS = ["UPI", "NEFT", "RTGS", "IMPS", "SWIFT", "POS_MERCHANT", "CASH_DEPOSIT"]
REGIONS = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Dubai", "London"]

def shuffle_dict(d: dict) -> dict:
    """Randomizes key order so the model cannot memorize fixed structural offsets."""
    items = list(d.items())
    random.shuffle(items)
    return dict(items)

# -------------------------------------------------------------
# 2. DYNAMIC EVIDENCE SYNTHESIZERS
# -------------------------------------------------------------
def compose_structuring_evidence(count, ceiling, total, time_span):
    templates = [
        f"Detected {count} successive outbound transfers hovering within 5% beneath regulatory threshold of INR {ceiling:,}.",
        f"Cumulative value of INR {total:,} split across {count} tranches to evade currency monitoring limits ({ceiling:,}).",
        f"Smurfing pattern flagged: {count} sub-threshold transfers executed rapidly in {time_span} minutes.",
        f"Deliberate tranche structuring identified: repetitive values near statutory reporting limit of INR {ceiling:,}."
    ]
    return random.sample(templates, k=random.randint(1, 2))

def compose_counter_evidence(scenario_type, context):
    if scenario_type == "mitigated":
        phrasings = [
            "Customer filed advance purchase notice for verified commercial capital expenditure.",
            "Counterparty is an audited Category-A registered corporate entity.",
            "Prior regulatory notice filed by account holder satisfies AML threshold exception.",
            "Transaction originated strictly from verified primary authentication token and domestic IP."
        ]
        return random.sample(phrasings, k=random.randint(1, 3))
    elif scenario_type == "benign_entity":
        name = context.get("flagged_name", "Individual")
        phrasings = [
            f"Entity resolution verifies '{name}' is an individual citizen with zero sanctioned ownership linkage.",
            "Domestic jurisdiction confirmed; transactional node routes solely through authorized domestic clearing.",
            "Watchlist match determined to be a phonetic false-positive following national ID verification."
        ]
        return random.sample(phrasings, k=random.randint(1, 2))
    return []

# -------------------------------------------------------------
# 3. CORE SCENARIO BUILDERS
# -------------------------------------------------------------
def build_dynamic_normal(idx):
    median = int(np.random.choice([1500, 3500, 7500, 16000, 38000, 85000]))
    home = random.choice(REGIONS[:8])
    device = f"DEV-{fake.hexify('^^^^^^^^')}"
    
    account = {
        "account_id": f"ACC-{idx:05d}",
        "customer": fake.name(),
        "historical_median": median,
        "typical_bracket": [int(median * 0.15), int(median * 2.3)],
        "registered_city": home,
        "primary_device": device
    }

    tx_count = random.randint(1, 4)
    txs = []
    for _ in range(tx_count):
        amt = int(np.random.uniform(account["typical_bracket"][0], account["typical_bracket"][1]))
        txs.append({
            "tx_id": f"TXN-{fake.hexify('^^^^^^')}",
            "amount": amt,
            "rail": random.choice(CHANNELS),
            "recipient": fake.name(),
            "location": home if random.random() > 0.15 else random.choice(REGIONS),
            "device": device if random.random() > 0.10 else f"DEV-{fake.hexify('^^^^^^^^')}",
            "time": f"{random.randint(7, 22):02d}:{random.randint(0, 59):02d}"
        })

    context = shuffle_dict({
        "account_baseline": shuffle_dict(account),
        "activity_batch": txs,
        "active_rules": ["POL-DEFAULT-MONITOR: Flag transactions breaching 3.5x baseline."]
    })

    target = shuffle_dict({
        "risk_level": "LOW",
        "primary_typology": "NORMAL_ROUTINE",
        "supporting_evidence": ["All transactions conform strictly to customer historical baseline and velocity parameters."],
        "counter_evidence": ["Standard temporal intervals and verified primary device authentication observed."],
        "recommended_action": "AUTO_CLEAR"
    })
    return context, target

def build_dynamic_structuring(idx):
    ceiling = int(random.choice([20000, 50000, 100000, 250000, 500000]))
    count = random.randint(3, 6)
    time_span = random.randint(3, 20)
    
    # Values hover 0.8% to 4.5% below the statutory limit
    amounts = [ceiling - int(ceiling * random.uniform(0.008, 0.045)) for _ in range(count)]
    total = sum(amounts)
    
    acct_id = f"ACC-{idx:05d}"
    beneficiary = f"BENEF-{random.randint(100, 999)}"
    
    txs = []
    for i, amt in enumerate(amounts):
        txs.append({
            "tx_id": f"TXN-{fake.hexify('^^^^^^')}",
            "amount": amt,
            "rail": random.choice(["IMPS", "UPI", "NEFT"]),
            "recipient": beneficiary,
            "device": f"DEV-UNMAPPED-{random.randint(10, 99)}",
            "time": f"03:{random.randint(10, 40) + (i * 2):02d}:00"
        })

    context = shuffle_dict({
        "subject_account": acct_id,
        "batch_records": txs,
        "statutory_limit": ceiling,
        "network_edges": [f"{acct_id} -> {beneficiary}"],
        "applied_policy": f"POL-STRUC-REG: Mandatory escalation for deliberate threshold avoidance near INR {ceiling:,}."
    })

    target = shuffle_dict({
        "risk_level": "HIGH",
        "primary_typology": "STRUCTURING_SMURFING",
        "supporting_evidence": compose_structuring_evidence(count, ceiling, total, time_span),
        "counter_evidence": [],
        "recommended_action": "ESCALATE_TO_FIU"
    })
    return context, target

def build_dynamic_adversarial(idx):
    cfg = random.choice(SANCTIONED_TARGETS)
    is_malicious = random.random() < 0.5
    
    if is_malicious:
        entity = f"{fake.company()} {random.choice(cfg['tokens'])} Forwarders"
        jurisdiction = cfg["jurisdiction"]
        risk = "CRITICAL"
        action = "BLOCK_IMMEDIATELY"
        typology = "SANCTIONS_BREACH"
        indicators = [f"Direct transshipment or funds transfer linked to restricted jurisdiction: {jurisdiction}."]
        mitigations = []
    else:
        entity = f"{random.choice(cfg['benign_names'])}"
        jurisdiction = "INDIA"
        risk = "LOW"
        action = "DISMISS_FALSE_ALARM"
        typology = "BENIGN_PHONETIC_MATCH"
        indicators = [f"Lexical overlap detected with restricted jurisdiction keyword '{random.choice(cfg['tokens'])}'."]
        mitigations = compose_counter_evidence("benign_entity", {"flagged_name": entity})

    tx = {
        "tx_id": f"TXN-{fake.hexify('^^^^^^')}",
        "amount": random.randint(25000, 850000),
        "recipient": entity,
        "jurisdiction": jurisdiction,
        "rail": "SWIFT" if is_malicious else random.choice(CHANNELS)
    }

    context = shuffle_dict({
        "audit_account": f"ACC-{idx:05d}",
        "transaction_event": tx,
        "compliance_policy": "SANCTIONS-SCREEN-01: Freeze funds destined for high-risk embargoed jurisdictions."
    })

    target = shuffle_dict({
        "risk_level": risk,
        "primary_typology": typology,
        "supporting_evidence": indicators,
        "counter_evidence": mitigations,
        "recommended_action": action
    })
    return context, target

def build_dynamic_mitigated(idx):
    median = int(np.random.choice([2500, 5500, 12000, 30000]))
    mult = random.randint(14, 30)
    spike_amt = median * mult
    vendor = f"{fake.company()} Heavy Industries Ltd"
    home = random.choice(REGIONS[:6])
    device = f"DEV-{fake.hexify('^^^^^^^^')}"

    acct = {
        "account_id": f"ACC-{idx:05d}",
        "customer": fake.name(),
        "median_tx_amount": median,
        "registered_city": home,
        "primary_device": device
    }

    tx = {
        "tx_id": f"TXN-{fake.hexify('^^^^^^')}",
        "amount": spike_amt,
        "recipient": vendor,
        "rail": "RTGS",
        "location": home,
        "device": device,
        "time": "14:15:00"
    }

    context = shuffle_dict({
        "account_baseline": acct,
        "transaction_event": tx,
        "documented_justifications": [
            "Advance tax invoice and statutory declaration filed prior to execution.",
            "Counterparty is an audited Category-A verified corporate entity."
        ],
        "applied_policy": "POL-SPIKE-102: Flag isolated transactions breaching 10x median baseline."
    })

    target = shuffle_dict({
        "risk_level": "MEDIUM",
        "primary_typology": "EXPLAINABLE_HIGH_VALUE",
        "supporting_evidence": [f"Single wire of INR {spike_amt:,} deviates {mult}x from historical median."],
        "counter_evidence": compose_counter_evidence("mitigated", {}),
        "recommended_action": "VERIFY_DOCUMENTATION"
    })
    return context, target

# -------------------------------------------------------------
# 4. MASTER 50K COMPILATION
# -------------------------------------------------------------
def generate_dataset(samples=50000):
    os.makedirs("data", exist_ok=True)
    target_path = "data/financial_risk_corpus.jsonl"
    print(f"Synthesizing {samples:,} high-entropy financial scenarios...")

    generators = [
        (build_dynamic_normal, 0.35),
        (build_dynamic_structuring, 0.25),
        (build_dynamic_adversarial, 0.25),
        (build_dynamic_mitigated, 0.15)
    ]
    funcs, weights = zip(*generators)

    with open(target_path, "w", encoding="utf-8") as f:
        for idx in range(samples):
            fn = random.choices(funcs, weights=weights)[0]
            ctx, tgt = fn(idx)
            record = {
                "scenario_id": f"SCN-{idx:06d}",
                "input_context": ctx,
                "investigation_target": tgt
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generation complete: {target_path} successfully saved ({samples:,} records).")

if __name__ == "__main__":
    generate_dataset(50000)