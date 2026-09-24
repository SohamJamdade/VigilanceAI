"""
service/schema.py — Core Domain Data Models & Typed Contracts
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class NormalizedTransaction(BaseModel):
    """Canonical representation of a financial transaction across all payment rails."""
    tx_id: str
    account_id: str
    customer_id: Optional[str] = None
    amount: float
    currency: str = "INR"
    rail: str = "IMPS"
    recipient: str
    device_id: str = "DEV-UNKNOWN"
    timestamp: datetime
    location: Optional[str] = "DOMESTIC"
    source_hash: str

class BehavioralBaseline(BaseModel):
    """Dynamic historical profile of a customer derived from observed transactions."""
    account_id: str
    historical_median: float = 0.0
    typical_bracket: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    historical_tx_count: int = 0

class FeatureSet(BaseModel):
    """Deterministic mathematical and temporal metrics calculated by Python."""
    account_id: str
    window_tx_count: int
    window_total_amount: float
    window_avg_amount: float
    window_median_amount: float
    velocity_tx_per_hour: float
    burst_flag: bool
    near_threshold_count: int
    baseline_deviation_ratio: float
    unique_recipients_count: int
    new_recipient_ratio: float
    unique_devices_count: int
    night_activity_ratio: float
    primary_rail_ratio: float

class RuleResult(BaseModel):
    """Standardized output of an explainable compliance heuristic."""
    rule_id: str
    rule_name: str
    triggered: bool
    severity: str
    reason: str
    evidence: Dict[str, Any]

class EntityProfile(BaseModel):
    """Network relationships connecting accounts, hardware devices, and counterparties."""
    account_id: str
    associated_customers: List[str]
    associated_devices: List[str]
    frequent_counterparties: List[str]
    shared_device_accounts: List[str]

class RiskAssessment(BaseModel):
    """The final case audit bundle synthesized by the pipeline."""
    case_id: str
    account_id: str
    timestamp: datetime
    deterministic_risk_score: float
    rule_severities: List[str]
    triggered_rules: List[RuleResult]
    features: FeatureSet
    entity_profile: EntityProfile
    slm_reasoning: str
    primary_typology: str
    recommended_action: str
    confidence_level: str
    model_version: str = "15.2M-INT8-v2.0"
    reproducibility_hash: str