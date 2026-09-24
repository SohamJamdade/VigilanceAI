
"""deterministic financial and behavioural feature engine """
from typing import List
from datetime import datetime, timezone
import numpy as np
from service.schema import NormalizedTransaction, BehavioralBaseline, FeatureSet

def compute_features(
    account_id: str,
    transactions: List[NormalizedTransaction],
    baseline: BehavioralBaseline,
    statutory_limit: float = 50000.0
) -> FeatureSet:
    """Extracts explainable mathematical features from normalized transactions."""
    if not transactions:
        return FeatureSet(
            account_id=account_id,
            window_tx_count=0,
            window_total_amount=0.0,
            window_avg_amount=0.0,
            window_median_amount=0.0,
            velocity_tx_per_hour=0.0,
            burst_flag=False,
            near_threshold_count=0,
            baseline_deviation_ratio=1.0,
            unique_recipients_count=0,
            new_recipient_ratio=0.0,
            unique_devices_count=0,
            night_activity_ratio=0.0,
            primary_rail_ratio=1.0
        )
    amounts = np.array([tx.amount for tx in transactions], dtype=float)
    count = len(transactions)
    total_amount = float(np.sum(amounts))
    avg_amount = float(np.mean(amounts))
    median_amount = float(np.median(amounts))

# Time calculations (robust to tz-aware / tz-naive mixes)
    timestamps = sorted([
        tx.timestamp.replace(tzinfo=timezone.utc) if tx.timestamp.tzinfo is None else tx.timestamp 
        for tx in transactions
    ])
    time_span_seconds = max((timestamps[-1] - timestamps[0]).total_seconds(), 1.0)
    time_span_hours = time_span_seconds / 3600.0

    # Minimum divisor of 0.25h (15 min) avoids division-by-zero spikes on instant transfers
    velocity = count / max(time_span_hours, 0.25)
    burst_flag = bool(count >= 3 and time_span_seconds <= 900)  # >= 3 txs within 15 minutes

    # Structuring markers: 90% to 99.9% of statutory limit (e.g., 45,000 to 49,999)
    near_thresh = sum(1 for a in amounts if (0.90 * statutory_limit) <= a < statutory_limit)

    # Dynamic baseline comparison
    base_med = baseline.historical_median if baseline.historical_median > 0 else avg_amount
    deviation_ratio = round(avg_amount / base_med, 2) if base_med > 0 else 1.0

    # Diversity metrics
    recipients = [tx.recipient for tx in transactions]
    devices = [tx.device_id for tx in transactions]
    rails = [tx.rail for tx in transactions]

    unique_recipients = len(set(recipients))
    unique_devices = len(set(devices))

    # Night transfers (between 23:00 and 05:00 UTC)
    night_count = sum(1 for t in timestamps if t.hour >= 23 or t.hour <= 5)
    night_ratio = round(night_count / count, 2)

    # Rail dominance
    rail_counts: dict[str, int] = {}
    for r in rails:
        rail_counts[r] = rail_counts.get(r, 0) + 1
    primary_rail_ratio = round(max(rail_counts.values()) / count, 2) if rail_counts else 1.0

    return FeatureSet(
        account_id=account_id,
        window_tx_count=count,
        window_total_amount=round(total_amount, 2),
        window_avg_amount=round(avg_amount, 2),
        window_median_amount=round(median_amount, 2),
        velocity_tx_per_hour=round(velocity, 2),
        burst_flag=burst_flag,
        near_threshold_count=near_thresh,
        baseline_deviation_ratio=deviation_ratio,
        unique_recipients_count=unique_recipients,
        new_recipient_ratio=round(unique_recipients / count, 2),
        unique_devices_count=unique_devices,
        night_activity_ratio=night_ratio,
        primary_rail_ratio=primary_rail_ratio
    )     
    