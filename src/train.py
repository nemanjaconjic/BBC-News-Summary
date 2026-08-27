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
    """
    Fix random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_groups(df: pd.DataFrame) -> pd.Series:
    """
    Create unique article identifiers.

    All sentences from the same article must stay
    in the same cross-validation fold.
    """

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
    """
    Evaluate the model on a validation dataset.
    """

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

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def measure_inference_time(
    model,
    dataloader,
    device,
):
    """
    Measure inference time on the validation dataset.
    """

    model.eval()

    start_time = time.time()

    total_samples = 0

    with torch.no_grad():

        for input_ids, _ in dataloader:

            input_ids = input_ids.to(device)

            _ = model(input_ids)

            total_samples += input_ids.size(0)

    elapsed_time = time.time() - start_time

    time_per_sample = (
        elapsed_time / total_samples
    )

    return elapsed_time, time_per_sample


def main() -> None:

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed()

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

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

    models_dir = (
        project_root
        / "results"
        / "models"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("Loading dataset...")

    df = pd.read_pickle(dataset_path)

    print(f"Records: {len(df)}")

    # --------------------------------------------------
    # Create article groups
    # --------------------------------------------------

    groups = create_groups(df)

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    # --------------------------------------------------
    # Use only Fold 1 for baseline testing
    # --------------------------------------------------

    train_idx, val_idx = next(
        group_kfold.split(
            X=df,
            y=df["label"],
            groups=groups,
        )
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    print(
        f"Train records: {len(train_df)}"
    )

    print(
        f"Validation records: {len(val_df)}"
    )

    # --------------------------------------------------
    # Build vocabulary using TRAINING DATA ONLY
    # --------------------------------------------------

    print("\nBuilding vocabulary...")

    vocab = build_vocabulary(
        train_df["sentence"].tolist(),
        min_frequency=2,
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    # --------------------------------------------------
    # Create PyTorch datasets
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = BaselineModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        dropout=0.2,
    )

    model = model.to(device)

    # --------------------------------------------------
    # Count trainable parameters
    # --------------------------------------------------

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    # --------------------------------------------------
    # Best model tracking
    # --------------------------------------------------

    best_f1 = -1.0
    best_epoch = 0

    best_model_path = (
        models_dir
        / "baseline_fold1_best.pt"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\nStarting training...\n")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train_loss = 0.0

        for input_ids, labels in train_loader:

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Clear gradients from previous batch
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)

            # Calculate loss
            loss = criterion(
                logits,
                labels,
            )

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_train_loss += (
                loss.item()
                * input_ids.size(0)
            )

        # Average training loss
        train_loss = (
            total_train_loss
            / len(train_loader.dataset)
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        validation_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        history["epoch"].append(epoch)

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            validation_metrics["loss"]
        )

        history["accuracy"].append(
            validation_metrics["accuracy"]
        )

        history["precision"].append(
            validation_metrics["precision"]
        )

        history["recall"].append(
            validation_metrics["recall"]
        )

        history["f1"].append(
            validation_metrics["f1"]
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if validation_metrics["f1"] > best_f1:

            best_f1 = validation_metrics["f1"]
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        # --------------------------------------------------
        # Print epoch results
        # --------------------------------------------------

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"  Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"  Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"  F1: "
            f"{validation_metrics['f1']:.4f}"
        )

        print("-" * 50)

    # --------------------------------------------------
    # Total training time
    # --------------------------------------------------

    training_time = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------
    # Best epoch information
    # --------------------------------------------------

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best model saved to: "
        f"{best_model_path}"
    )

    # --------------------------------------------------
    # Restore best model before measuring inference
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
        )
    )

    # --------------------------------------------------
    # Measure inference time
    # --------------------------------------------------

    inference_time, inference_per_sample = (
        measure_inference_time(
            model=model,
            dataloader=val_loader,
            device=device,
        )
    )

    print(
        f"Inference time for validation set: "
        f"{inference_time:.4f} seconds"
    )

    print(
        f"Inference time per sentence: "
        f"{inference_per_sample * 1000:.4f} ms"
    )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = (
        results_dir
        / "baseline_fold1_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print(
        f"Training history saved to: "
        f"{history_path}"
    )

    # --------------------------------------------------
    # Save experiment summary
    # --------------------------------------------------

    summary = {
        "model": "BaselineModel",
        "fold": 1,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "max_length": MAX_LENGTH,
        "vocabulary_size": len(vocab),
        "trainable_parameters": num_parameters,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "inference_ms_per_sentence": (
            inference_per_sample * 1000
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = (
        results_dir
        / "baseline_fold1_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Experiment summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()