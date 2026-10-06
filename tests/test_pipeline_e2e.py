import os
import sys
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

os.environ["SLM_VARIANT"] = "130m"

from service.cases import DB_FILE, init_case_storage
from service.db_poller import init_audit_db
from service.app import get_engine, screen_account_hybrid


def setup_test_databases():
    init_case_storage(DB_FILE)
    init_audit_db()


class TestVigilanceAIPipeline(unittest.TestCase):

    def setUp(self):
        os.environ["SLM_VARIANT"] = "130m"
        setup_test_databases()

    def test_01_engine_cache_initialization(self):
        engine = get_engine()
        self.assertIn("model", engine)
        self.assertIn("tokenizer", engine)
        self.assertIsNotNone(engine["model"])
        self.assertIsNotNone(engine["tokenizer"])

    def test_02_structuring_hybrid_screening(self):
        transactions = [
            {"tx_id": "TX-01", "amount": 49200.0, "recipient": "Mule Recipient", "rail": "IMPS"},
            {"tx_id": "TX-02", "amount": 48950.0, "recipient": "Mule Recipient", "rail": "IMPS"}
        ]
        result = screen_account_hybrid(account_id="ACC-TEST-99", transactions=transactions)

        required_keys = [
            "account_id", "risk_level", "primary_typology",
            "recommended_action", "narrative", "supporting_evidence"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing required key: {key}")

        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["primary_typology"], "STRUCTURING_SMURFING")
        self.assertEqual(result["recommended_action"], "ESCALATE_TO_FIU")
        self.assertNotIn("SLM fallback triggered", result["narrative"])

    def test_03_sanctions_critical_alert(self):
        transactions = [
            {"tx_id": "TX-SANCT-01", "amount": 25000.0, "recipient": "OFAC-BLOCKED-ENTITY", "rail": "SWIFT"}
        ]
        result = screen_account_hybrid(account_id="ACC-TEST-SANCT", transactions=transactions)

        self.assertEqual(result["risk_level"], "CRITICAL")
        self.assertEqual(result["recommended_action"], "BLOCK_IMMEDIATELY")
        self.assertEqual(result["primary_typology"], "SANCTIONS_BREACH")


if __name__ == "__main__":
    unittest.main()