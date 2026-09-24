import sqlite3
import json
from typing import Dict, Any, List
from service.schema import RiskAssessment

DB_FILE = "audit_log.db"


def _safe_model_json(obj) -> str:
    # Pydantic v2/v1 compatible JSON serializer
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json()
    if hasattr(obj, "json"):
        return obj.json()
    return json.dumps(obj)


def _safe_model_dict(obj) -> dict:
    # Pydantic v2/v1 compatible dict serializer
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def init_case_storage(db_file: str = DB_FILE):
    with sqlite3.connect(db_file, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")

        # Legacy audit table for chat and backward compatibility
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screening_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                subject_account TEXT,
                risk_level TEXT,
                primary_typology TEXT,
                recommended_action TEXT,
                evidence_summary TEXT,
                raw_payload TEXT
            )
        """)

        # Structured case management table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_cases (
                case_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                account_id TEXT,
                risk_level TEXT,
                deterministic_score REAL,
                primary_typology TEXT,
                recommended_action TEXT,
                model_reasoning TEXT,
                features_json TEXT,
                triggered_rules_json TEXT,
                entity_profile_json TEXT,
                model_version TEXT,
                reproducibility_hash TEXT,
                case_status TEXT DEFAULT 'OPEN'
            )
        """)

        # Entity linkage cache for device/account resolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_account TEXT,
                entity_type TEXT,
                entity_val TEXT,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Upload batch history tracker
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS upload_history (
                batch_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                tx_count INTEGER DEFAULT 0,
                account_count INTEGER DEFAULT 0,
                account_ids_json TEXT
            )
        """)
        conn.commit()


def record_case(assessment: RiskAssessment, raw_payload: Dict[str, Any], db_file: str = DB_FILE):
    with sqlite3.connect(db_file, timeout=5.0) as conn:
        cursor = conn.cursor()

        # Safe-serialize features, rules, and entity profile
        features_json = _safe_model_json(assessment.features)
        rules_json = json.dumps([_safe_model_dict(r) for r in assessment.triggered_rules])
        entity_json = _safe_model_json(assessment.entity_profile)

        cursor.execute("""
            INSERT OR REPLACE INTO risk_cases (
                case_id, created_at, account_id, risk_level, deterministic_score,
                primary_typology, recommended_action, model_reasoning,
                features_json, triggered_rules_json, entity_profile_json,
                model_version, reproducibility_hash, case_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (
            assessment.case_id,
            assessment.timestamp.isoformat(),
            assessment.account_id,
            assessment.confidence_level,
            assessment.deterministic_risk_score,
            assessment.primary_typology,
            assessment.recommended_action,
            assessment.slm_reasoning,
            features_json,
            rules_json,
            entity_json,
            assessment.model_version,
            assessment.reproducibility_hash
        ))

        # Legacy audit record for /v1/chat compatibility
        evidence = [r.reason for r in assessment.triggered_rules if r.triggered] or ["Baseline conforming."]
        cursor.execute("""
            INSERT INTO screening_audit 
            (subject_account, risk_level, primary_typology, recommended_action, evidence_summary, raw_payload)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            assessment.account_id,
            assessment.confidence_level,
            assessment.primary_typology,
            assessment.recommended_action,
            json.dumps(evidence),
            json.dumps(raw_payload)
        ))

        # Persist device linkages for entity resolution
        for dev in assessment.entity_profile.associated_devices:
            cursor.execute(
                "INSERT INTO entity_links (subject_account, entity_type, entity_val) VALUES (?, 'DEVICE', ?)",
                (assessment.account_id, dev)
            )

        conn.commit()


def log_upload_batch(batch_id: str, filename: str, tx_count: int, account_ids: List[str], db_file: str = DB_FILE):
    # Logs uploaded file details and accounts evaluated in SQLite
    with sqlite3.connect(db_file, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO upload_history (batch_id, filename, uploaded_at, tx_count, account_count, account_ids_json)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
        """, (batch_id, filename, tx_count, len(account_ids), json.dumps(account_ids)))
        conn.commit()


def get_upload_history(db_file: str = DB_FILE) -> List[Dict[str, Any]]:
    # Retrieves all uploaded file batch records ordered by time
    with sqlite3.connect(db_file, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT batch_id, filename, uploaded_at, tx_count, account_count, account_ids_json
            FROM upload_history ORDER BY uploaded_at DESC
        """)
        rows = cursor.fetchall()
        result = []
        for r in rows:
            result.append({
                "batch_id": r["batch_id"],
                "filename": r["filename"],
                "uploaded_at": r["uploaded_at"],
                "tx_count": r["tx_count"],
                "account_count": r["account_count"],
                "account_ids": json.loads(r["account_ids_json"]) if r["account_ids_json"] else []
            })
        return result


def delete_upload_batch(batch_id: str, db_file: str = DB_FILE) -> Dict[str, Any]:
    # Deletes all cases and audit records corresponding to an upload batch
    with sqlite3.connect(db_file, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, account_ids_json FROM upload_history WHERE batch_id = ?", (batch_id,))
        row = cursor.fetchone()
        if not row:
            return {"status": "not_found", "message": "Batch not found"}
        filename, accs_json = row
        accounts = json.loads(accs_json) if accs_json else []

        for acc in accounts:
            cursor.execute("DELETE FROM risk_cases WHERE account_id = ?", (acc,))
            cursor.execute("DELETE FROM screening_audit WHERE UPPER(subject_account) = ?", (acc.upper(),))
            cursor.execute("DELETE FROM entity_links WHERE subject_account = ?", (acc,))

        cursor.execute("DELETE FROM upload_history WHERE batch_id = ?", (batch_id,))
        conn.commit()
        return {"status": "success", "filename": filename, "accounts_purged": len(accounts)}


def purge_all_records(db_file: str = DB_FILE) -> Dict[str, Any]:
    # Clears all test case and audit data from SQLite
    with sqlite3.connect(db_file, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM risk_cases")
        cursor.execute("DELETE FROM screening_audit")
        cursor.execute("DELETE FROM entity_links")
        cursor.execute("DELETE FROM upload_history")
        conn.commit()
        return {"status": "success", "message": "All records purged"}