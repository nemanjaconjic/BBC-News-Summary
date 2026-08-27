from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from torch.utils.data import DataLoader

from dataset import (
    BBCSentenceDataset,
    build_vocabulary,
)
from model import BaselineModel


SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
MAX_LENGTH = 60
N_SPLITS = 5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_groups(df: pd.DataFrame) -> pd.Series:
    return (
        df["category"].astype(str)
        + "_"
        + df["file_name"].astype(str)
    )


def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for input_ids, labels in dataloader:

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * input_ids.size(0)
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "precision": precision_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
    }


def train_fold(
    fold,
    train_df,
    val_df,
    device,
    project_root,
):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold}")
    print(f"{'=' * 60}")

    set_seed(SEED + fold)

    vocab = build_vocabulary(
        train_df["sentence"].tolist(),
        min_frequency=2,
    )

    print(f"Vocabulary size: {len(vocab)}")

    train_dataset = BBCSentenceDataset(
        dataframe=train_df,
        vocab=vocab,
        max_length=MAX_LENGTH,
    )

    val_dataset = BBCSentenceDataset(
        dataframe=val_df,
        vocab=vocab,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BaselineModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        dropout=0.2,
    ).to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0

    fold_history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train_loss = 0.0

        for input_ids, labels in train_loader:

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_train_loss += (
                loss.item()
                * input_ids.size(0)
            )

        train_loss = (
            total_train_loss
            / len(train_loader.dataset)
        )

        metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        fold_history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {metrics['loss']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            best_epoch = epoch

    training_time = time.time() - start_time

    result = {
        "fold": fold,
        "vocab_size": len(vocab),
        "parameters": num_parameters,
        "best_epoch": best_epoch,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "val_loss": best_metrics["loss"],
        "training_time_seconds": training_time,
    }

    return result, fold_history


def main() -> None:
    set_seed()

    project_root = Path(__file__).resolve().parent.parent

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "clean_sentence_dataset.pkl"
    )

    results_dir = (
        project_root
        / "results"
        / "metrics"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    df = pd.read_pickle(dataset_path)

    print(f"Records: {len(df)}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    groups = create_groups(df)

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_results = []
    all_history = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            X=df,
            y=df["label"],
            groups=groups,
        ),
        start=1,
    ):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        result, history = train_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            device=device,
            project_root=project_root,
        )

        fold_results.append(result)
        all_history.extend(history)

    results_df = pd.DataFrame(
        fold_results
    )

    history_df = pd.DataFrame(
        all_history
    )

    results_path = (
        results_dir
        / "baseline_cross_validation.csv"
    )

    history_path = (
        results_dir
        / "baseline_cv_history.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df[
            [
                "fold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "training_time_seconds",
            ]
        ]
    )

    print("\nMean metrics:")

    print(
        f"Accuracy: "
        f"{results_df['accuracy'].mean():.4f} "
        f"± {results_df['accuracy'].std():.4f}"
    )

    print(
        f"Precision: "
        f"{results_df['precision'].mean():.4f} "
        f"± {results_df['precision'].std():.4f}"
    )

    print(
        f"Recall: "
        f"{results_df['recall'].mean():.4f} "
        f"± {results_df['recall'].std():.4f}"
    )

    print(
        f"F1: "
        f"{results_df['f1'].mean():.4f} "
        f"± {results_df['f1'].std():.4f}"
    )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )

    print(
        f"History saved to: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()