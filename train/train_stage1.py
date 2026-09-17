import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer import FinancialSLM

class FinancialDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=512):
        self.examples = []
        pad_id = tokenizer.token_to_id("<|pad|>")
        self.pad_token_id = pad_id if pad_id is not None else 0

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                input_str = json.dumps(record["input_context"])
                target_str = json.dumps(record["investigation_target"])
                
                full_text = f"<|context_start|>{input_str}<|context_end|><|target_start|>{target_str}<|target_end|>"
                encoded = tokenizer.encode(full_text).ids

                if len(encoded) > max_seq_len:
                    encoded = encoded[:max_seq_len]
                else:
                    encoded = encoded + [self.pad_token_id] * (max_seq_len - len(encoded))
                
                self.examples.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        targets = tokens.clone()
        targets[targets == self.pad_token_id] = -100
        return tokens, targets

def get_cosine_lr(step, total_steps, warmup_steps, max_lr=8e-4, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def run_stage1():
    print("=" * 70)
    print("VigilanceAI: Stage-1 Pretraining (15.2M Parameter SLM - 50k Corpus)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Hardware: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    tok = Tokenizer.from_file("tokenizer/financial_bpe.json")
    dataset = FinancialDataset("data/financial_risk_corpus.jsonl", tok, max_seq_len=512)
    
    batch_size = 32 if device.type == "cuda" else 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=2)

    # 15.2M Architecture: d_model=384, n_layers=8, n_heads=12
    model = FinancialSLM(vocab_size=2048, d_model=384, n_layers=8, n_heads=12, max_seq_len=512).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Verified Model Parameters : {total_params:,} (~{total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    
    # 50k samples at batch size 32 = 1,562 steps per epoch. 3 epochs = ~4,687 steps (optimal convergence)
    epochs = 3
    total_steps = epochs * len(loader)
    warmup_steps = int(0.04 * total_steps)

    print(f"Dataset Records           : {len(dataset):,}")
    print(f"Batch Size                : {batch_size}")
    print(f"Total Steps               : {total_steps} ({epochs} Epochs)")
    print(f"Warmup Steps              : {warmup_steps}")
    print("-" * 70)

    model.train()
    step = 0
    total_tokens = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            lr = get_cosine_lr(step, total_steps, warmup_steps, max_lr=8e-4, min_lr=1e-5)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            total_tokens += x.numel()
            epoch_loss += loss.item()

            if step == 1 or step % 100 == 0 or step == total_steps:
                elapsed = time.time() - start_time
                tok_s = total_tokens / elapsed
                vram = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0
                print(f"Epoch {epoch+1}/{epochs} | Step {step:04d}/{total_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_s:.0f} tok/s | VRAM: {vram:.1f} MB")

        avg_loss = epoch_loss / len(loader)
        print(f">>> Epoch {epoch+1} Completed | Mean Loss: {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/slm_15m_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved 15M checkpoint to {ckpt_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_stage1()