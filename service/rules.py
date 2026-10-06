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
    tx_lines = [
        f"Tx {tx.tx_id}: INR {tx.amount:,.2f} via {tx.rail} to '{tx.recipient}'"
        for tx in transactions
    ]

    # 1. Structuring and Smurfing Detection Rule
    structuring_triggered = features.near_threshold_count >= 2
    near_txs = [tx for tx in transactions if (statutory_limit * 0.7 <= tx.amount <= statutory_limit)]
    if not near_txs and structuring_triggered:
        near_txs = transactions
    avg_near = (sum(tx.amount for tx in near_txs) / len(near_txs)) if near_txs else (features.window_avg_amount if features.near_threshold_count else 0.0)
    near_tx_lines = [f"Tx {tx.tx_id}: INR {tx.amount:,.2f} via {tx.rail} to '{tx.recipient}'" for tx in near_txs]

    struct_reason = (
        f"{features.near_threshold_count} structured transactions closely clustered below INR {statutory_limit:,.0f} threshold (avg INR {avg_near:,.2f})."
        if structuring_triggered
        else f"Identified {features.near_threshold_count} transactions clustering immediately below statutory limit (INR {statutory_limit:,.0f})."
    )
    results.append(RuleResult(
        rule_id="RULE_STRUCT_001",
        rule_name="Near-Threshold Transactions Clustering",
        triggered=structuring_triggered,
        severity="HIGH" if structuring_triggered else "LOW",
        reason=struct_reason,
        evidence={
            "near_threshold_count": features.near_threshold_count,
            "statutory_limit": statutory_limit,
            "average_near_threshold_amount": round(avg_near, 2),
            "flagged_transactions": near_tx_lines or tx_lines[:3]
        }
    ))

    # 2. Velocity and Burst Activity Rule
    velocity_triggered = features.burst_flag or features.velocity_tx_per_hour > 8.0
    vel_reason = (
        f"Observed velocity of {features.velocity_tx_per_hour:.1f} tx/hr (Total burst volume: INR {features.window_total_amount:,.2f} across {features.window_tx_count} transactions; Burst flag: {features.burst_flag})."
        if velocity_triggered
        else f"Observed velocity of {features.velocity_tx_per_hour:.1f} tx/hr (Burst flag: {features.burst_flag})."
    )
    results.append(RuleResult(
        rule_id="RULE_VEL_002",
        rule_name="High Transaction Velocity or Burst Activity",
        triggered=velocity_triggered,
        severity="HIGH" if velocity_triggered else "LOW",
        reason=vel_reason,
        evidence={
            "velocity_tx_per_hour": features.velocity_tx_per_hour,
            "burst_flag": features.burst_flag,
            "total_burst_volume": features.window_total_amount,
            "window_tx_count": features.window_tx_count,
            "flagged_transactions": tx_lines
        }
    ))

    # 3. Sanctions and Restricted counterparty match
    sanction_keywords = ["OFAC", "SANCTION", "PROHIBITED", "TERROR", "BLOCKLIST"]
    matched_txs = [
        tx for tx in transactions
        if any(keyword in tx.recipient.upper() for keyword in sanction_keywords)
    ]
    matched_recipients = [tx.recipient for tx in matched_txs]
    sanctions_triggered = len(matched_recipients) > 0
    sanct_tx_lines = [f"Tx {tx.tx_id}: INR {tx.amount:,.2f} via {tx.rail} to '{tx.recipient}'" for tx in matched_txs]

    if sanctions_triggered:
        first_m = matched_txs[0]
        sanct_reason = f"Counterparty '{first_m.recipient}' flagged in High-Risk/Sanctions watchlist (Tx {first_m.tx_id}: INR {first_m.amount:,.2f} via {first_m.rail})."
    else:
        sanct_reason = f"Matched {len(matched_recipients)} restricted counterparty patterns."

    results.append(RuleResult(
        rule_id="RULE_SANCT_003",
        rule_name="Restricted Counterparty Detection",
        triggered=sanctions_triggered,
        severity="CRITICAL" if sanctions_triggered else "LOW",
        reason=sanct_reason,
        evidence={
            "matched_recipients": matched_recipients,
            "matched_count": len(matched_recipients),
            "flagged_transactions": sanct_tx_lines
        }
    ))

    # 4. Behavioral Baseline Spike Rule
    deviation_triggered = (
        features.baseline_deviation_ratio >= 3.5 and 
        features.window_total_amount > (statutory_limit * 1.5)
    )
    dev_reason = (
        f"Average transaction value (INR {features.window_avg_amount:,.2f}) is {features.baseline_deviation_ratio:.1f}x higher than historical baseline median."
        if deviation_triggered
        else f"Average transaction value is {features.baseline_deviation_ratio:.1f}x historical baseline median."
    )
    results.append(RuleResult(
        rule_id="RULE_DEV_004",
        rule_name="Significant Baseline Outlier",
        triggered=deviation_triggered,
        severity="MEDIUM" if deviation_triggered else "LOW",
        reason=dev_reason,
        evidence={
            "deviation_ratio": features.baseline_deviation_ratio,
            "window_avg_amount": features.window_avg_amount,
            "window_total_amount": features.window_total_amount,
            "flagged_transactions": tx_lines
        }
    ))

    return results
