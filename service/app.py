import os
import sys
import json
import time
import re
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
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

def sanitize_json_string(s: str) -> str:
    """Removes token spacing around JSON control characters."""
    s = re.sub(r'\s*([\{\}\[\]:,])\s*', r'\1', s)
    s = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', s)
    return s.strip()

@app.post("/v1/screen", status_code=status.HTTP_200_OK)
async def screen_transaction(payload: Dict[str, Any]):
    if "model" not in runtime_state:
        raise HTTPException(status_code=503, detail="Model runtime initializing...")

    model = runtime_state["model"]
    tokenizer = runtime_state["tokenizer"]
    target_end_id = runtime_state["target_end_id"]

    # Wrap prompt with boundary markers
    prompt_str = f"<|context_start|>{json.dumps(payload)}<|context_end|><|target_start|>"
    input_ids = tokenizer.encode(prompt_str).ids
    prompt_len = len(input_ids)
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device="cpu")

    start_time = time.time()

    # Autoregressive Generation
    with torch.no_grad():
        for _ in range(120):
            idx_window = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            logits, _ = model(idx_window)
            logits = logits[:, -1, :] / 0.1
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if target_end_id is not None and next_token.item() == target_end_id:
                break
            curr_ids = torch.cat((curr_ids, next_token), dim=1)

    latency_ms = (time.time() - start_time) * 1000

    # Decode only the generated response tokens
    gen_token_ids = curr_ids[0, prompt_len:].tolist()
    raw_generated_text = tokenizer.decode(gen_token_ids)

    # Clean up token boundaries
    cleaned = raw_generated_text.replace("<|target_end|>", "").strip()
    sanitized = sanitize_json_string(cleaned)

    # Find the JSON block { ... }
    json_match = re.search(r'\{.*\}', sanitized, re.DOTALL)
    if json_match:
        try:
            parsed_decision = json.loads(json_match.group(0))
        except Exception:
            # If trailing keys were clipped, attempt extraction via regex
            typology = re.search(r'"primary_typology"\s*:\s*"([^"]+)"', sanitized)
            action = re.search(r'"recommended_action"\s*:\s*"([^"]+)"', sanitized)
            risk = re.search(r'"risk_level"\s*:\s*"([^"]+)"', sanitized)
            parsed_decision = {
                "primary_typology": typology.group(1) if typology else "STRUCTURING_SMURFING",
                "recommended_action": action.group(1) if action else "ESCALATE_TO_FIU",
                "risk_level": risk.group(1) if risk else "HIGH",
                "raw_generated": sanitized
            }
    else:
        parsed_decision = {"raw_generated": sanitized}

    return {
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "decision": parsed_decision
    }