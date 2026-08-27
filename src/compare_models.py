from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path, model_name):
    df = pd.read_csv(path)
    df["model"] = model_name
    return df


def calculate_summary(results):
    rows = []

    for model_name, df in results.items():
        row = {
            "model": model_name,
            "accuracy_mean": df["accuracy"].mean(),
            "accuracy_std": df["accuracy"].std(),
            "precision_mean": df["precision"].mean(),
            "precision_std": df["precision"].std(),
            "recall_mean": df["recall"].mean(),
            "recall_std": df["recall"].std(),
            "f1_mean": df["f1"].mean(),
            "f1_std": df["f1"].std(),
            "training_time_mean": df["training_time_seconds"].mean(),
            "training_time_std": df["training_time_seconds"].std(),
            "inference_ms_mean": df["inference_ms_per_sentence"].mean(),
            "inference_ms_std": df["inference_ms_per_sentence"].std(),
        }

        if "trainable_parameters" in df.columns:
            row["trainable_parameters_mean"] = df[
                "trainable_parameters"
            ].mean()
        else:
            row["trainable_parameters_mean"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def plot_metrics(summary_df, output_directory):
    metrics = [
        ("accuracy_mean", "Accuracy"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
        ("f1_mean", "F1"),
    ]

    x = np.arange(len(summary_df))
    width = 0.18

    plt.figure(figsize=(12, 7))

    for index, (column, label) in enumerate(metrics):
        offset = (
            index - (len(metrics) - 1) / 2
        ) * width

        plt.bar(
            x + offset,
            summary_df[column],
            width,
            label=label,
        )

    plt.xticks(
        x,
        summary_df["model"],
    )

    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Metric value")
    plt.title("Comparison of classification metrics")
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()

    output_path = (
        output_directory
        / "model_metrics_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_f1(summary_df, output_directory):
    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["model"],
        summary_df["f1_mean"],
        yerr=summary_df["f1_std"],
        capsize=5,
    )

    plt.xlabel("Model")
    plt.ylabel("F1 score")
    plt.title("Mean F1 score with standard deviation")
    plt.ylim(0, 1)
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()

    output_path = (
        output_directory
        / "f1_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_training_time(
    summary_df,
    output_directory,
):
    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["model"],
        summary_df["training_time_mean"],
    )

    plt.xlabel("Model")
    plt.ylabel("Training time (seconds)")
    plt.title("Average training time")
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()

    output_path = (
        output_directory
        / "training_time_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_inference_time(
    summary_df,
    output_directory,
):
    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["model"],
        summary_df["inference_ms_mean"],
    )

    plt.xlabel("Model")
    plt.ylabel(
        "Inference time per sentence (ms)"
    )
    plt.title("Average inference time")
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()

    output_path = (
        output_directory
        / "inference_time_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_f1_per_fold(
    results,
    output_directory,
):
    plt.figure(figsize=(11, 7))

    for model_name, df in results.items():
        plt.plot(
            df["fold"],
            df["f1"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Fold")
    plt.ylabel("F1 score")
    plt.title("F1 score across cross-validation folds")
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = (
        output_directory
        / "f1_per_fold_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def main():
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    metrics_directory = (
        project_root
        / "results"
        / "metrics"
    )

    plots_directory = (
        project_root
        / "results"
        / "plots"
    )

    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_files = {
        "Model 1":
            metrics_directory
            / "baseline_mlflow_cv_results.csv",
        "Model 2":
            metrics_directory
            / "model2_mlflow_cv_results.csv",
        "Model 3":
            metrics_directory
            / "model3_mlflow_cv_results.csv",
        "Model 4":
            metrics_directory
            / "model4_mlflow_cv_results.csv",
        "Model 5":
            metrics_directory
            / "model5_mlflow_cv_results.csv",
    }

    results = {}

    for model_name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing results file: {path}"
            )

        results[model_name] = load_results(
            path,
            model_name,
        )

    summary_df = calculate_summary(
        results
    )

    summary_df = summary_df.sort_values(
        by="f1_mean",
        ascending=False,
    ).reset_index(drop=True)

    summary_path = (
        metrics_directory
        / "model_comparison_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    plot_metrics(
        summary_df,
        plots_directory,
    )

    plot_f1(
        summary_df,
        plots_directory,
    )

    plot_training_time(
        summary_df,
        plots_directory,
    )

    plot_inference_time(
        summary_df,
        plots_directory,
    )

    plot_f1_per_fold(
        results,
        plots_directory,
    )

    print("\nMODEL COMPARISON")
    print("=" * 100)

    display_columns = [
        "model",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "f1_std",
        "training_time_mean",
        "inference_ms_mean",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_model = summary_df.iloc[0]

    print("\nBEST MODEL")
    print("=" * 100)

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"Accuracy: "
        f"{best_model['accuracy_mean']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_model['precision_mean']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_model['recall_mean']:.4f}"
    )

    print(
        f"F1: "
        f"{best_model['f1_mean']:.4f}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_path}"
    )

    print(
        f"\nPlots saved to:\n"
        f"{plots_directory}"
    )


if __name__ == "__main__":
    main()