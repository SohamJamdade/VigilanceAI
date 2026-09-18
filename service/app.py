import os
import sys
import json
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

runtime_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(4)
    tok_path = "tokenizer/financial_bpe.json"
    ckpt_path = "checkpoints/slm_15m_int8.pt"

    if not os.path.exists(tok_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing required assets: {tok_path} or {ckpt_path}")

    tokenizer = Tokenizer.from_file(tok_path)
    base_model = FinancialSLM(
        vocab_size=2048, d_model=384, n_layers=8, n_heads=12, max_seq_len=512
    ).to("cpu")

    quantized_model = torch.ao.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    quantized_model.eval()

    runtime_state["tokenizer"] = tokenizer
    runtime_state["model"] = quantized_model
    runtime_state["target_end_id"] = tokenizer.token_to_id("<|target_end|>")
    yield
    runtime_state.clear()

app = FastAPI(
    title="VigilanceAI Dynamic AML Engine",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/v1/screen", status_code=status.HTTP_200_OK)
async def screen_transaction(request: Request):
    """
    Accepts ANY valid JSON scenario dynamically, converts it to tokens,
    runs the INT8 quantized model, and returns the decision.
    """
    try:
        scenario_dict = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload received.")

    model = runtime_state["model"]
    tokenizer = runtime_state["tokenizer"]
    target_end_id = runtime_state["target_end_id"]

    # Wrap dynamic payload into model prompt boundaries
    prompt_str = f"<|context_start|>{json.dumps(scenario_dict)}<|context_end|><|target_start|>"
    input_ids = tokenizer.encode(prompt_str).ids
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

    start_time = time.time()

    with torch.no_grad():
        for _ in range(140):
            idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            logits, _ = model(idx_window)
            logits = logits[:, -1, :] / 0.1
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if target_end_id is not None and next_token.item() == target_end_id:
                break
            curr_ids = torch.cat((curr_ids, next_token), dim=1)

    latency_ms = (time.time() - start_time) * 1000
    decoded = tokenizer.decode(curr_ids[0].tolist())

    try:
        decision_part = decoded.split("<|target_start|>")[1]
        if "<|target_end|>" in decision_part:
            decision_part = decision_part.split("<|target_end|>")[0]
        parsed_output = json.loads(decision_part.strip())
    except Exception:
        parsed_output = {
            "error": "Failed to parse generation as JSON",
            "raw_text": decoded
        }

    return {
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "decision": parsed_output
    }

