"""
service/rules.py — Deterministic Heuristic Compliance Rules
"""
from typing import List
from service.schema import NormalizedTransaction, FeatureSet, RuleResult

def evaluate_rules(
        transactions: List[NormalizedTransaction],
        features: FeatureSet,
        statutory_limit: float = 50000.0
) -> List[RuleResult]:
    """Evaluates auditable deterministic rules against calculated features"""
    results = []

    #Structuring and Smurfing Detection Rule
    structuring_triggered = features.near_threshold_count >= 2
    results.append(RuleResult(
        rule_id="RULE_STRUCT_001",
        rule_name="Near-Threshold Transactions Clustering",
        triggered=structuring_triggered,
        severity="HIGH" if structuring_triggered else "LOW",
        reason=f"Identified {features.near_threshold_count} transactions clustering immediately below statutory limit ({statutory_limit}).",
        evidence={
            "near_threshold_count": features.near_threshold_count,
            "statutory_limit": statutory_limit
        }
    ))

    #Velocity and Burst Activity Rule
    velocity_triggered = features.burst_flag or features.velocity_tx_per_hour > 8.0
    results.append(RuleResult(
        rule_id="RULE_VEL_002",
        rule_name="High Transaction Velocity or Burst Activity",
        triggered=velocity_triggered,
        severity="HIGH" if velocity_triggered else "LOW",
        reason=f"Observed velocity of {features.velocity_tx_per_hour} tx/hr (Burst flag: {features.burst_flag}).",
        evidence={
            "velocity_tx_per_hour": features.velocity_tx_per_hour,
            "burst_flag": features.burst_flag
        }
    ))

    #3 Sanctions and Ristricted counterparty match
    sanction_keywords = ["OFAC", "SANCTION", "PROHIBITED", "TERROR", "BLOCKLIST"]
    matched_recipients = [
        tx.recipient for tx in transactions
        if any(keyword in tx.recipient.upper() for keyword in sanction_keywords)
    ]
    sanctions_triggered = len(matched_recipients) > 0
    results.append(RuleResult(
        rule_id="RULE_SANCT_003",
        rule_name="Restricted Counterparty Detection",
        triggered=sanctions_triggered,
        severity="CRITICAL" if sanctions_triggered else "LOW",
        reason=f"Matched {len(matched_recipients)} restricted counterparty patterns.",
        evidence={"matched_recipients": matched_recipients}
    ))

    # 4. Behavioral Baseline Spike Rule
    deviation_triggered = (
        features.baseline_deviation_ratio >= 3.5 and 
        features.window_total_amount > (statutory_limit * 1.5)
    )
    results.append(RuleResult(
        rule_id="RULE_DEV_004",
        rule_name="Significant Baseline Outlier",
        triggered=deviation_triggered,
        severity="MEDIUM" if deviation_triggered else "LOW",
        reason=f"Average transaction value is {features.baseline_deviation_ratio}x higher than historical baseline median.",
        evidence={"deviation_ratio": features.baseline_deviation_ratio}
    ))

    return results
