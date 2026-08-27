from collections import Counter
from pathlib import Path
import re

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str) -> list[str]:

    text = text.lower()

    tokens = re.findall(r"\b\w+\b", text)

    return tokens


def build_vocabulary(
    texts: list[str],
    min_frequency: int = 2,
) -> dict[str, int]:

    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, frequency in counter.items():
        if frequency >= min_frequency:
            vocab[word] = len(vocab)

    return vocab

def encode_text(
    text: str,
    vocab: dict[str, int],
    max_length: int,
) -> list[int]:

    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    token_ids = token_ids[:max_length]

    padding_length = max_length - len(token_ids)

    if padding_length > 0:
        token_ids.extend(
            [vocab[PAD_TOKEN]] * padding_length
        )

    return token_ids

class BBCSentenceDataset(Dataset):
    """
    PyTorch Dataset za BBC sentence classification.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocab: dict[str, int],
        max_length: int = 60,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        encoded_sentence = encode_text(
            row["sentence"],
            self.vocab,
            self.max_length,
        )

        input_ids = torch.tensor(
            encoded_sentence,
            dtype=torch.long,
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.float32,
        )

        return input_ids, label

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "clean_sentence_dataset.pkl"
    )

    print("Loading dataset...")

    df = pd.read_pickle(dataset_path)

    print("Records:", len(df))

    vocab = build_vocabulary(
        df["sentence"].tolist(),
        min_frequency=2,
    )

    print("Vocabulary size:", len(vocab))

    dataset = BBCSentenceDataset(
        dataframe=df,
        vocab=vocab,
        max_length=60,
    )

    print("Dataset size:", len(dataset))

    input_ids, label = dataset[0]

    print("\nFirst encoded sentence:")
    print(input_ids)

    print("\nTensor shape:")
    print(input_ids.shape)

    print("\nLabel:")
    print(label)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    batch_inputs, batch_labels = next(iter(dataloader))

    print("\nBatch input shape:")
    print(batch_inputs.shape)

    print("\nBatch labels shape:")
    print(batch_labels.shape)


if __name__ == "__main__":
    main()