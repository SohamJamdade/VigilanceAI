import json
import os
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

def prepare_raw_text(jsonl_path="data/financial_risk_corpus.jsonl", txt_path="data/corpus_for_tokenizer.txt"):
    """Extracts string representations of inputs and targets for tokenization."""
    print("Extracting text stream from synthetic JSONL...")
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(txt_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line.strip())
            input_text = json.dumps(record["input_context"])
            target_text = json.dumps(record["investigation_target"])
            
            # Format as sequence pairs with clear boundary tokens
            formatted_sample = f"<|context_start|>{input_text}<|context_end|><|target_start|>{target_text}<|target_end|>\n"
            fout.write(formatted_sample)
    print(f"Extracted corpus saved to {txt_path}")
    return txt_path

def train_custom_bpe(corpus_path, vocab_size=2048):
    os.makedirs("tokenizer", exist_ok=True)
    save_path = "tokenizer/financial_bpe.json"
    
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFKC()

    # Split on whitespace and isolate every digit (0-9) individually
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=True)
    ])

    special_tokens = [
        "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
        "<|context_start|>", "<|context_end|>",
        "<|target_start|>", "<|target_end|>",
        "<|risk_low|>", "<|risk_medium|>", "<|risk_high|>", "<|risk_critical|>",
        "<|entity_person|>", "<|entity_org|>", "<|entity_geo|>"
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    print(f"Training BPE Tokenizer (vocab_size={vocab_size})...")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer successfully trained and saved to {save_path}")

    # Sanity check encode/decode
    test_sample = '<|context_start|>{"amount": 49500, "recipient": "Kiran Deshmukh"}<|context_end|>'
    encoded = tokenizer.encode(test_sample)
    print("\nSanity Check Tokenization:")
    print("Tokens:", encoded.tokens[:15], "...")
    print("Token IDs:", encoded.ids[:15], "...")

if __name__ == "__main__":
    text_corpus = prepare_raw_text()
    train_custom_bpe(text_corpus, vocab_size=2048)
