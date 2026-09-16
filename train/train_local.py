import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# Project root path resolution
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

def run_pipeline():
    print("=" * 60)
    print("VigilanceAI: Training & Verification Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device     : {device}")
    if device.type == "cuda":
        print(f"GPU Model           : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM          : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    tokenizer_path = "tokenizer/financial_bpe.json"
    data_path = "data/financial_risk_corpus.jsonl"

    tokenizer = Tokenizer.from_file(tokenizer_path)
    max_seq_len = 512
    batch_size = 8 if device.type == "cpu" else 16

    print(f"Loading data (max_seq_len={max_seq_len}, batch_size={batch_size})...")
    dataset = FinancialDataset(data_path, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    print(f"Loaded Records      : {len(dataset):,}")

    model = FinancialSLM(vocab_size=2048, d_model=256, n_layers=6, n_heads=8, max_seq_len=max_seq_len)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters    : {total_params:,} (~{total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    print("\nExecuting verification steps...")
    model.train()
    step = 0
    max_steps = 30
    start_time = time.time()
    total_tokens = 0
    first_loss = None
    final_loss = None

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert grad_norm > 0, "Gradients are zero!"
        optimizer.step()

        step += 1
        total_tokens += x.numel()

        if step == 1:
            first_loss = loss.item()
            print(f"Step {step:02d} | Loss: {first_loss:.4f} | Grad Norm: {grad_norm:.4f}")
        elif step % 10 == 0:
            current_loss = loss.item()
            final_loss = current_loss
            elapsed = time.time() - start_time
            tok_s = total_tokens / elapsed
            vram_info = ""
            if device.type == "cuda":
                vram_info = f"| VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB"
            print(f"Step {step:02d} | Loss: {current_loss:.4f} | Speed: {tok_s:.0f} tok/s {vram_info}")

        if step >= max_steps:
            break

    # Checkpoint test
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/slm_local_sanity.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint successfully saved to {ckpt_path}")

    # Generation sanity test
    test_prompt = '<|context_start|>{"amount": 49500}'
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt).ids], dtype=torch.long, device=device)
    out_ids = model.generate(prompt_ids, max_new_tokens=16, temperature=0.8)
    decoded = tokenizer.decode(out_ids[0].tolist())
    print(f"Sample Output: {decoded}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()