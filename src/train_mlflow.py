from pathlib import Path
import random
import time

import mlflow
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
DROPOUT = 0.2
MIN_FREQUENCY = 2

EXPERIMENT_NAME = "BBC-News-Summary"
MODEL_NAME = "BaselineModel"


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
    Create a unique identifier for every article.

    All sentences from one article must remain
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
    Evaluate model and return validation metrics.
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
    Measure total inference time and
    inference time per sentence.
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


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device,
    project_root: Path,
):
    """
    Train and log one cross-validation fold.
    """

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    fold_seed = SEED + fold
    set_seed(fold_seed)

    print("Building vocabulary...")

    vocab = build_vocabulary(
        train_df["sentence"].tolist(),
        min_frequency=MIN_FREQUENCY,
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

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
        dropout=DROPOUT,
    ).to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_metrics = None

    models_dir = (
        project_root
        / "results"
        / "models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / f"baseline_fold{fold}_best.pt"
    )

    run_name = f"baseline_fold_{fold}"

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "fold": fold,
                "seed": fold_seed,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "embedding_dim": EMBEDDING_DIM,
                "dropout": DROPOUT,
                "max_length": MAX_LENGTH,
                "min_frequency": MIN_FREQUENCY,
                "vocab_size": len(vocab),
                "optimizer": "Adam",
                "loss_function": "BCEWithLogitsLoss",
                "train_records": len(train_df),
                "validation_records": len(val_df),
                "device": str(device),
            }
        )

        mlflow.log_metric(
            "trainable_parameters",
            num_parameters,
        )

        print(
            f"Trainable parameters: "
            f"{num_parameters:,}"
        )

        start_time = time.time()

        for epoch in range(
            1,
            EPOCHS + 1
        ):

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

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                metrics["loss"],
                step=epoch,
            )

            mlflow.log_metric(
                "accuracy",
                metrics["accuracy"],
                step=epoch,
            )

            mlflow.log_metric(
                "precision",
                metrics["precision"],
                step=epoch,
            )

            mlflow.log_metric(
                "recall",
                metrics["recall"],
                step=epoch,
            )

            mlflow.log_metric(
                "f1",
                metrics["f1"],
                step=epoch,
            )

            if metrics["f1"] > best_f1:

                best_f1 = metrics["f1"]
                best_epoch = epoch
                best_metrics = metrics.copy()

                torch.save(
                    model.state_dict(),
                    model_path,
                )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

        training_time = (
            time.time()
            - start_time
        )

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        inference_time, inference_per_sample = (
            measure_inference_time(
                model=model,
                dataloader=val_loader,
                device=device,
            )
        )

        mlflow.log_metrics(
            {
                "best_f1": best_f1,
                "best_accuracy": best_metrics["accuracy"],
                "best_precision": best_metrics["precision"],
                "best_recall": best_metrics["recall"],
                "best_val_loss": best_metrics["loss"],
                "best_epoch": best_epoch,
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
                "inference_ms_per_sentence": (
                    inference_per_sample * 1000
                ),
            }
        )

        mlflow.log_artifact(
            str(model_path)
        )

        print(
            f"\nBest epoch: {best_epoch}"
        )

        print(
            f"Best F1: {best_f1:.4f}"
        )

        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )

        print(
            f"Inference time: "
            f"{inference_time:.4f} seconds"
        )

        print(
            f"Inference per sentence: "
            f"{inference_per_sample * 1000:.4f} ms"
        )

    return {
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
        "inference_time_seconds": inference_time,
        "inference_ms_per_sentence": (
            inference_per_sample * 1000
        ),
    }


def main() -> None:

    set_seed()

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    dataset_path = (
    project_root
    / "data"
    / "processed"
    / "clean_sentence_dataset.csv"
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

    mlflow_db = (
        project_root
        / "mlflow.db"
    )

    tracking_uri = (
        f"sqlite:///{mlflow_db.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow tracking URI: {tracking_uri}"
    )

    print("Loading dataset...")

    df = pd.read_csv(
    dataset_path
    )

    print("\nData types:")
    print(df.dtypes)

    print(
        f"Records: {len(df)}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    groups = create_groups(df)

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_results = []

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        group_kfold.split(
            X=df,
            y=df["label"],
            groups=groups,
        ),
        start=1,
    ):

        train_df = (
            df.iloc[train_idx]
            .copy()
        )

        val_df = (
            df.iloc[val_idx]
            .copy()
        )

        result = train_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            device=device,
            project_root=project_root,
        )

        fold_results.append(
            result
        )

    results_df = pd.DataFrame(
        fold_results
    )

    results_path = (
        results_dir
        / "baseline_mlflow_cv_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL CROSS-VALIDATION RESULTS")
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

    print("\nMean ± standard deviation")

    mean_accuracy = (
        results_df["accuracy"].mean()
    )

    std_accuracy = (
        results_df["accuracy"].std()
    )

    mean_precision = (
        results_df["precision"].mean()
    )

    std_precision = (
        results_df["precision"].std()
    )

    mean_recall = (
        results_df["recall"].mean()
    )

    std_recall = (
        results_df["recall"].std()
    )

    mean_f1 = (
        results_df["f1"].mean()
    )

    std_f1 = (
        results_df["f1"].std()
    )

    print(
        f"Accuracy: "
        f"{mean_accuracy:.4f} "
        f"± {std_accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{mean_precision:.4f} "
        f"± {std_precision:.4f}"
    )

    print(
        f"Recall: "
        f"{mean_recall:.4f} "
        f"± {std_recall:.4f}"
    )

    print(
        f"F1: "
        f"{mean_f1:.4f} "
        f"± {std_f1:.4f}"
    )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()