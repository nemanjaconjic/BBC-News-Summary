from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentence_dataset.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_sentence_dataset.pkl"
)


def clean_text(text: str) -> str:
    """Remove unnecessary whitespace from text."""
    return " ".join(str(text).split())


def main() -> None:
    print("Loading dataset...")

    df = pd.read_pickle(INPUT_FILE)

    print(f"Original records: {len(df)}")

    text_columns = ["sentence", "summary"]

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    print("Text cleaning completed.")

    before = len(df)

    df = df[df["sentence"].str.len() > 0].copy()

    removed_empty = before - len(df)

    print(f"Removed empty sentences: {removed_empty}")

    before = len(df)

    df = df.drop_duplicates().copy()

    removed_duplicates = before - len(df)

    print(f"Removed duplicate rows: {removed_duplicates}")

    valid_labels = {0, 1}

    invalid_labels = ~df["label"].isin(valid_labels)

    print(
        f"Invalid labels found: {invalid_labels.sum()}"
    )

    if invalid_labels.any():
        df = df[~invalid_labels].copy()

    invalid_similarity = (
        (df["similarity_score"] < 0)
        | (df["similarity_score"] > 1)
    )

    print(
        f"Invalid similarity scores found: "
        f"{invalid_similarity.sum()}"
    )

    if invalid_similarity.any():
        df = df[~invalid_similarity].copy()

    df = df.reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_pickle(OUTPUT_FILE)

    print("\nPreprocessing completed.")
    print(f"Final records: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()