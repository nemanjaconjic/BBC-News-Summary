from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


N_SPLITS = 5


def create_groups(df: pd.DataFrame) -> pd.Series:
   
    return (
        df["category"].astype(str)
        + "_"
        + df["file_name"].astype(str)
    )


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

    print(f"Sentence records: {len(df)}")

    groups = create_groups(df)

    print(f"Unique articles: {groups.nunique()}")

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    print(f"\nCreating {N_SPLITS} folds...\n")

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            X=df,
            y=df["label"],
            groups=groups,
        ),
        start=1,
    ):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_groups = set(
            groups.iloc[train_idx]
        )

        val_groups = set(
            groups.iloc[val_idx]
        )

        overlap = train_groups.intersection(
            val_groups
        )

        print(f"Fold {fold}")
        print(f"  Train sentences: {len(train_df)}")
        print(f"  Validation sentences: {len(val_df)}")

        print(
            f"  Train articles: "
            f"{len(train_groups)}"
        )

        print(
            f"  Validation articles: "
            f"{len(val_groups)}"
        )

        print(
            f"  Train positive ratio: "
            f"{train_df['label'].mean():.4f}"
        )

        print(
            f"  Validation positive ratio: "
            f"{val_df['label'].mean():.4f}"
        )

        print(
            f"  Article overlap: "
            f"{len(overlap)}"
        )

        print("-" * 50)


if __name__ == "__main__":
    main()