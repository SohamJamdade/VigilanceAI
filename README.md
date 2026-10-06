# VigilanceAI: Automated, SLM-Driven Anti-Money Laundering (AML) Pipeline

VigilanceAI is an enterprise-grade, automated transaction monitoring and compliance screening engine driven by a quantized Financial Small Language Model (SLM) (130M INT8). It bridges deterministic heuristic rule engines with contextual deep-learning inference to classify risk, detect advanced typologies (such as structuring, velocity bursts, and sanctions breaches), and generate audit-ready compliance dossiers.

---

## 🧠 Comprehensive SLM Pipeline & Architecture Flow

The end-to-end lifecycle of financial transactions within VigilanceAI spans multiple synchronized background and foreground layers:

```
[ Data Ingestion / simulate_bank_db.py ]
               │
               ▼
   [ SQLite Core Banking DB ]
   (core_banking.db / WAL Mode + Busy Timeout)
               │
               ▼
[ Heuristic Rule Engine (service/rules.py) ]
  - Velocity & Aggregate Calculation
  - Structuring & Threshold Checks
  - Watchlist & Sanctions Matching (OFAC, etc.)
               │
               ▼
   [ Quantized SLM Inference (service/app.py) ]
    - Global Singleton Engine Cache (_ENGINE_CACHE)
    - 130M INT8 PyTorch Model (slm_130m_int8.pt)
    - Clamped Token Budget (max_new_tokens = 56)
    - Early Stop Sequences (<|endoftext|>, \n\n)
               │
               ▼
[ Dossier & Evidence Synthesizer (service/cases.py) ]
  - Itemized Transaction Breakdowns (IDs, Rails, Amounts)
  - Non-Duplicated Regulatory Narratives & Bulleted Evidence
               │
               ▼
[ Immutable Case Store & Streamlit UI (service/dashboard.py) ]
  - Case Manager & Risk Telemetry Analytics
  - Automated Database Polling Engine UI
  - LLM Compliance Assistant Chat Query Interface

```

---

## 🚀 Complete Local-First Runbook & Setup

Because VigilanceAI runs a multi-process stateful architecture with background watchers, SQLite Write-Ahead Logging (WAL) concurrency, and PyTorch model weights, it is designed for a robust, high-performance local run. Follow these instructions to spin it up from scratch.

### Step 1: System Prerequisites

* **Python 3.10 – 3.12** installed and added to your system `PATH`.
* **Git** and **Git LFS** (Large File Storage) installed.
```powershell
git lfs install

```



---

### Step 2: Clone Repository & Pull Large Model Checkpoints

```powershell
git clone https://github.com/SohamJamdade/VigilanceAI.git
cd VigilanceAI
git lfs pull

```

> *Verify that `checkpoints/slm_130m_int8.pt` is fully downloaded (~133 MB on disk, not a text pointer).*

---

### Step 3: Create Virtual Environment & Install Dependencies

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

---

### Step 4: Run Automated Smoke Tests

Validate database schemas, BPE tokenizers, and the 130M INT8 model configuration before starting services:

```powershell
python -m unittest tests/test_pipeline_e2e.py

```

*Expected output:* `Ran 3 tests in ...s — OK`

---

### Step 5: Launch the Multi-Process Services

Open **3 split terminal panes** in your workspace, activate your virtual environment (`.\venv\Scripts\Activate.ps1`) in each:

* **Terminal 1 — Ingestion Drop Watcher:**
```powershell
python service/drop_watcher.py

```


* **Terminal 2 — 130M Screening Poller:**
```powershell
$env:SLM_VARIANT="130m"
python service/db_poller.py

```


* **Terminal 3 — Compliance Workstation Dashboard (Streamlit):**
```powershell
python -m streamlit run service/dashboard.py

```



---

### Step 6: Simulate Traffic & Test Automated DB Sync UI

1. **Inject Simulated Banking Traffic (Terminal 4):**
```powershell
python simulate_bank_db.py

```


Let it run for 10–15 seconds to stream records into `core_banking.db`, then press `Ctrl + C`.
2. **Test Automated Database Polling UI:**
* Open **`http://localhost:8501`** in your browser.
* Navigate to the **Auto-DB Sync** tab.
* Confirm the database path points to `core_banking.db` and table to `transactions`.
* Click **⚡ Run Single Scan Now** to watch the UI poller process pending batches, update telemetry counters instantly, and render live risk records.



---

### Step 7: Explore the Compliance Workstation

* **Case Manager Tab:** Inspect severity tags (`CRITICAL`, `HIGH`, `LOW`), typology classifications (`STRUCTURING_SMURFING`, `SANCTIONS_BREACH`, `HIGH_VELOCITY_BURST`), and recommended actions (`BLOCK_IMMEDIATELY`, `ESCALATE_TO_FIU`).
* **Compliance Assistant Tab:** Enter natural-language queries like `why is account ACC-10001 flagged?` to view rich, audit-ready dossier cards featuring itemized transaction breakdowns and concise SLM regulatory narratives.
