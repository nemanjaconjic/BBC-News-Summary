from pathlib import Path
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

MODEL = "distilbert-base-uncased"
BATCH = 32
MAX_LENGTH = 64

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        x = self.tokenizer(self.texts[i], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        return x["input_ids"].squeeze(0), x["attention_mask"].squeeze(0)

def main():
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data/processed/clean_sentence_dataset.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    loader = DataLoader(TextDataset(df["sentence"].astype(str).tolist(), tokenizer), batch_size=BATCH)
    embeddings = []

    with torch.no_grad():
        for i, (ids, mask) in enumerate(loader, 1):
            x = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            embeddings.append(x.cpu())
            if i % 100 == 0: print(f"Batch {i}/{len(loader)}")

    embeddings = torch.cat(embeddings)
    out = root / "data/processed/distilbert_embeddings.pt"
    torch.save({"embeddings": embeddings, "labels": torch.tensor(df["label"].values), "category": df["category"].tolist(), "file_name": df["file_name"].tolist()}, out)
    print(f"Saved: {out}")
    print(f"Shape: {embeddings.shape}")

if __name__ == "__main__":
    main()