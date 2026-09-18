import os
import sys
import time
import json
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.transformer import FinancialSLM 

def quantinize_model():
    print("=" * 70)
    print("VigilanceAI: Dynamic INT8 Model Quantization Pipeline")
    print("=" * 70)

    checkpoint_fp32 = "checkpoints/slm_15m_final.pt"
    quantized_output_path = "checkpoints/slm_15m_int8.pt"

    if not os.path.exists(checkpoint_fp32):
        print(f"Error: Original weights '{checkpoint_fp32}' not found.")
        return

    print("[1/4] Instantiating base FP32 architecture on CPU...")
    model_fp32 = FinancialSLM(
        vocab_size=2048,
        d_model=384,
        n_layers=8,
        n_heads=12,
        max_seq_len=512
    ).to("cpu")

    #load trained FP32 wwights 
    model_fp32.load_state_dict(torch.load(checkpoint_fp32, map_location="cpu", weights_only=True))
    model_fp32.eval()

    fp32_size_mb = os.path.getsize(checkpoint_fp32) / (1024 *1024)
    print(f"       FP32 Checkpoint Size: {fp32_size_mb:.2f} MB")

# 3. Apply PyTorch Dynamic Quantization to Linear layers
    print("[2/4] Quantizing Linear projection layers to qint8...")
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

# 4. Save quantized model
    print("[3/4] Serializing quantized model artifact...")
    torch.save(model_int8.state_dict(), quantized_output_path)
    int8_size_mb = os.path.getsize(quantized_output_path) / (1024 * 1024)
    reduction = ((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100

    print(f"       INT8 Checkpoint Size: {int8_size_mb:.2f} MB")
    print(f"       Footprint Reduction:  {reduction:.1f}%")

    # 5. Speed and Accuracy Sanity Check on CPU
    print("[4/4] Running CPU latency benchmark (FP32 vs INT8)...")
    tok = Tokenizer.from_file("tokenizer/financial_bpe.json")
    test_prompt = {
        "subject_account": "ACC-99104",
        "statutory_limit": 75000,
        "batch_records": [
            {"tx_id": "TXN-01", "amount": 73400, "rail": "IMPS", "recipient": "BENEF-881", "device": "DEV-UNMAPPED-44", "time": "03:12:00"}
        ],
        "applied_policy": "POL-STRUC-REG: Mandatory escalation for deliberate threshold avoidance near INR 75,000."
    }
    prompt_str = f"<|context_start|>{json.dumps(test_prompt)}<|context_end|><|target_start|>"
    tokens = tok.encode(prompt_str).ids
    inp = torch.tensor([tokens], dtype=torch.long, device="cpu")

    # FP32 Latency
    t0 = time.time()
    with torch.no_grad():
        _ = model_fp32.generate(inp, max_new_tokens=25, temperature=0.1)
    fp32_latency = (time.time() - t0) * 1000

    # INT8 Latency
    t0 = time.time()
    with torch.no_grad():
        out_ids = model_int8.generate(inp, max_new_tokens=25, temperature=0.1)
    int8_latency = (time.time() - t0) * 1000

    decoded = tok.decode(out_ids[0].tolist())
    print(f"       FP32 CPU Latency (25 tokens): {fp32_latency:.1f} ms")
    print(f"       INT8 CPU Latency (25 tokens): {int8_latency:.1f} ms")
    print("-" * 70)
    print("Sanity Check Generation from INT8 Model:")
    print(decoded.split("<|target_start|>")[-1].strip())
    print("=" * 70)

if __name__ == "__main__":
    quantinize_model()    
    
 