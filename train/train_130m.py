"""
train/train_130m.py — Training & Quantization Pipeline for 130M FinancialSLM
Optimized for Google Colab GPU (T4/A100) with mixed precision and dynamic INT8 export.
"""
import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

# Hyperparameters
CONFIG_130M = {
    "vocab_size": 2048,
    "d_model": 768,
    "n_layers": 16,
    "n_heads": 12,
    "max_seq_len": 512
}

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

TOK_PATH = "tokenizer/financial_bpe.json"
OUT_DIR = "checkpoints"
FP32_CKPT = os.path.join(OUT_DIR, "slm_130m_final.pt")
INT8_CKPT = os.path.join(OUT_DIR, "slm_130m_int8.pt")

os.makedirs(OUT_DIR, exist_ok=True)


class SyntheticFinancialDataset(Dataset):
    """Generates synthetic contextual training pairs for multi-sentence AML reasoning."""
    def __init__(self, tokenizer, num_samples=3000):
        self.tokenizer = tokenizer
        self.samples = []
        typologies = [
            ("SANCTIONS_BREACH", "CRITICAL", "Immediate OFAC SDN list match identified on destination routing. Account frozen pending legal review."),
            ("STRUCTURING_SMURFING", "HIGH", "Repeated sub-threshold transfers executed within narrow temporal intervals to evade reporting ceilings."),
            ("MULE_BURST", "HIGH", "High-velocity inbound and outbound flow with zero retention period across unmapped mobile devices."),
            ("NORMAL_ROUTINE", "LOW", "Standard consumer purchasing pattern consistent with documented operational baseline and verified devices.")
        ]

        for i in range(num_samples):
            typ, risk, reasoning = typologies[i % len(typologies)]
            context = {
                "account_id": f"ACC-{i:05d}",
                "features": {
                    "tx_count": (i % 10) + 1,
                    "total_val": float((i % 50 + 1) * 15000),
                    "velocity": float((i % 8) + 0.5),
                    "burst": (typ == "MULE_BURST"),
                    "near_threshold": (4 if typ == "STRUCTURING_SMURFING" else 0),
                    "deviation_ratio": (4.5 if "HIGH" in risk or "CRITICAL" in risk else 1.0)
                },
                "triggered_rules": [f"RULE_{typ[:6]}_01"] if risk != "LOW" else [],
                "shared_devices": 2 if typ == "MULE_BURST" else 0
            }
            target = {
                "risk_level": risk,
                "primary_typology": typ,
                "recommended_action": "BLOCK_IMMEDIATELY" if risk == "CRITICAL" else ("ESCALATE_TO_L2" if risk == "HIGH" else "AUTO_CLEAR"),
                "supporting_evidence": [reasoning],
                "counter_evidence": ["Verified secondary KYC details"] if risk == "LOW" else []
            }
            
            prompt = f"<|context_start|>{json.dumps(context)}<|context_end|><|target_start|>{json.dumps(target)}<|target_end|>"
            self.samples.append(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text).ids[:CONFIG_130M["max_seq_len"]]
        # Pad sequence to max length
        pad_id = self.tokenizer.token_to_id("<|pad|>") or 0
        pad_len = CONFIG_130M["max_seq_len"] - len(tokens)
        input_ids = tokens + [pad_id] * pad_len
        labels = tokens[1:] + [pad_id] * (pad_len + 1)
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing 130M model training on: {device}")
    
    tokenizer = Tokenizer.from_file(TOK_PATH)
    model = FinancialSLM(**CONFIG_130M).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable Parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    dataset = SyntheticFinancialDataset(tokenizer, num_samples=3200)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{step+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--> Epoch {epoch+1} Completed | Mean Loss: {avg_loss:.4f}")

    print(f"[*] Training finished in {(time.time() - start_time) / 60:.2f} minutes.")
    
    # Save unquantized FP32/FP16 model
    torch.save(model.state_dict(), FP32_CKPT)
    print(f"[+] Saved FP32 weights to {FP32_CKPT}")

    # Quantize to dynamic INT8 for local deployment
    print("[*] Quantizing 130M model to INT8 on CPU...")
    model_cpu = FinancialSLM(**CONFIG_130M).to("cpu")
    model_cpu.load_state_dict(torch.load(FP32_CKPT, map_location="cpu"))
    quantized_130m = torch.ao.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_130m.state_dict(), INT8_CKPT)
    print(f"[+] Quantized checkpoint saved to {INT8_CKPT} (~{os.path.getsize(INT8_CKPT) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    train()