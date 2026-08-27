from pathlib import Path
import random
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader

from dataset import BBCSentenceDataset, build_vocabulary
from model import BaselineModel


SEED = 42

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.0005

EMBEDDING_DIM = 200
MAX_LENGTH = 80
DROPOUT = 0.5

MIN_FREQUENCY = 2
N_SPLITS = 5

EXPERIMENT_NAME = "BBC-News-Summary"
MODEL_NAME = "BaselineModel_V4"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_groups(df):
    return (
        df["category"].astype(str)
        + "_"
        + df["file_name"].astype(str)
    )


def evaluate_model(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs).view(-1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy().astype(int)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss / len(data_loader)
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
    data_loader,
    device,
):
    model.eval()

    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            _ = model(inputs)

            total_samples += inputs.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time

    milliseconds_per_sentence = (
        total_time / total_samples
    ) * 1000

    return (
        total_time,
        milliseconds_per_sentence,
    )


def train_fold(
    fold,
    train_df,
    val_df,
    project_root,
    device,
):
    print("\n" + "=" * 60)
    print(f"MODEL 4 - FOLD {fold}")
    print("=" * 60)

    fold_seed = SEED + fold

    set_seed(fold_seed)

    print("Building vocabulary...")

    vocabulary = build_vocabulary(
        train_df["sentence"].tolist(),
        min_frequency=MIN_FREQUENCY,
    )

    vocab_size = len(vocabulary)

    print(f"Vocabulary size: {vocab_size}")

    train_dataset = BBCSentenceDataset(
        train_df,
        vocabulary,
        max_length=MAX_LENGTH,
    )

    val_dataset = BBCSentenceDataset(
        val_df,
        vocabulary,
        max_length=MAX_LENGTH,
    )

    generator = torch.Generator()
    generator.manual_seed(fold_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    model_directory = (
        project_root
        / "results"
        / "models"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        model_directory
        / f"model4_fold{fold}_best.pt"
    )

    with mlflow.start_run(
        run_name=f"model4_fold_{fold}"
    ):
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "model_version": 4,
                "fold": fold,
                "seed": fold_seed,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "embedding_dim": EMBEDDING_DIM,
                "dropout": DROPOUT,
                "max_length": MAX_LENGTH,
                "min_frequency": MIN_FREQUENCY,
                "vocab_size": vocab_size,
                "optimizer": "Adam",
                "loss_function": "BCEWithLogitsLoss",
                "train_records": len(train_df),
                "validation_records": len(val_df),
                "device": str(device),
            }
        )

        mlflow.log_metric(
            "trainable_parameters",
            trainable_parameters,
        )

        best_f1 = -1.0
        best_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_val_loss = float("inf")
        best_epoch = 0

        training_start = time.perf_counter()

        for epoch in range(1, EPOCHS + 1):
            model.train()

            total_train_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                outputs = model(inputs).view(-1)

                loss = criterion(
                    outputs,
                    labels,
                )

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            average_train_loss = (
                total_train_loss
                / len(train_loader)
            )

            validation_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
                device,
            )

            val_loss = validation_metrics["loss"]
            accuracy = validation_metrics["accuracy"]
            precision = validation_metrics["precision"]
            recall = validation_metrics["recall"]
            f1 = validation_metrics["f1"]

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {average_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {accuracy:.4f} | "
                f"Prec: {precision:.4f} | "
                f"Rec: {recall:.4f} | "
                f"F1: {f1:.4f}"
            )

            mlflow.log_metric(
                "train_loss",
                average_train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "accuracy",
                accuracy,
                step=epoch,
            )

            mlflow.log_metric(
                "precision",
                precision,
                step=epoch,
            )

            mlflow.log_metric(
                "recall",
                recall,
                step=epoch,
            )

            mlflow.log_metric(
                "f1",
                f1,
                step=epoch,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_val_loss = val_loss
                best_epoch = epoch

                torch.save(
                    model.state_dict(),
                    best_model_path,
                )

        training_end = time.perf_counter()

        training_time = (
            training_end - training_start
        )

        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
                weights_only=True,
            )
        )

        (
            inference_time,
            inference_ms_per_sentence,
        ) = measure_inference_time(
            model,
            val_loader,
            device,
        )

        mlflow.log_metric(
            "best_f1",
            best_f1,
        )

        mlflow.log_metric(
            "best_accuracy",
            best_accuracy,
        )

        mlflow.log_metric(
            "best_precision",
            best_precision,
        )

        mlflow.log_metric(
            "best_recall",
            best_recall,
        )

        mlflow.log_metric(
            "best_val_loss",
            best_val_loss,
        )

        mlflow.log_metric(
            "best_epoch",
            best_epoch,
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        mlflow.log_metric(
            "inference_time_seconds",
            inference_time,
        )

        mlflow.log_metric(
            "inference_ms_per_sentence",
            inference_ms_per_sentence,
        )

        mlflow.log_artifact(
            str(best_model_path)
        )

    print("\nFold completed:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best precision: {best_precision:.4f}")
    print(f"Best recall: {best_recall:.4f}")
    print(f"Training time: {training_time:.2f} sec")
    print(f"Inference time: {inference_time:.4f} sec")
    print(
        f"Inference per sentence: "
        f"{inference_ms_per_sentence:.4f} ms"
    )

    return {
        "model": "Model 4",
        "fold": fold,
        "best_epoch": best_epoch,
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "val_loss": best_val_loss,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "inference_ms_per_sentence":
            inference_ms_per_sentence,
        "trainable_parameters":
            trainable_parameters,
        "vocab_size": vocab_size,
    }


def main():
    set_seed(SEED)

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

    mlflow_db = (
        project_root
        / "mlflow.db"
    )

    tracking_uri = (
        f"sqlite:///"
        f"{mlflow_db.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow tracking URI: "
        f"{tracking_uri}"
    )

    print("\nLoading dataset...")

    df = pd.read_csv(
        dataset_path
    )

    print(f"Records: {len(df)}")

    print("\nData types:")
    print(df.dtypes)

    print("\nLabel distribution:")
    print(
        df["label"].value_counts()
    )

    invalid_labels = (
        ~df["label"].isin([0, 1])
    ).sum()

    if invalid_labels > 0:
        raise ValueError(
            f"Dataset contains "
            f"{invalid_labels} invalid labels."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    groups = create_groups(df)

    print(
        f"Unique articles: "
        f"{groups.nunique()}"
    )

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_results = []

    for fold, (
        train_indices,
        val_indices,
    ) in enumerate(
        group_kfold.split(
            df,
            df["label"],
            groups=groups,
        ),
        start=1,
    ):
        train_df = (
            df
            .iloc[train_indices]
            .reset_index(drop=True)
        )

        val_df = (
            df
            .iloc[val_indices]
            .reset_index(drop=True)
        )

        train_groups = set(
            create_groups(train_df)
        )

        val_groups = set(
            create_groups(val_df)
        )

        overlap = (
            train_groups
            .intersection(val_groups)
        )

        if overlap:
            raise RuntimeError(
                "Data leakage detected: "
                "articles appear in both "
                "training and validation sets."
            )

        print(
            f"\nFold {fold}: "
            f"Train={len(train_df)}, "
            f"Validation={len(val_df)}, "
            f"Article overlap={len(overlap)}"
        )

        result = train_fold(
            fold,
            train_df,
            val_df,
            project_root,
            device,
        )

        fold_results.append(
            result
        )

    results_df = pd.DataFrame(
        fold_results
    )

    metrics_directory = (
        project_root
        / "results"
        / "metrics"
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        metrics_directory
        / "model4_mlflow_cv_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    print("\n" + "=" * 60)
    print("MODEL 4 - CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df[
            [
                "fold",
                "best_epoch",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    print("\nMean results:")

    for metric in metric_columns:
        mean_value = (
            results_df[metric].mean()
        )

        std_value = (
            results_df[metric].std()
        )

        print(
            f"{metric.capitalize()}: "
            f"{mean_value:.4f} "
            f"± {std_value:.4f}"
        )

    print(
        "\nAverage training time: "
        f"{results_df['training_time_seconds'].mean():.2f} sec"
    )

    print(
        "Average inference time per sentence: "
        f"{results_df['inference_ms_per_sentence'].mean():.4f} ms"
    )

    print(
        f"\nResults saved to:\n"
        f"{results_path}"
    )


if __name__ == "__main__":
    main()