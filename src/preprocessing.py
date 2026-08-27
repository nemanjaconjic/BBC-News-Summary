from pathlib import Path
import re

import pandas as pd

CATEGORIES = [
    "business",
    "entertainment",
    "politics",
    "sport",
    "tech",
]

def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and remove leading/trailing whitespace."""
    return path.read_text(
        encoding="utf-8",
        errors="replace"
    ).strip()


def clean_text(text: str) -> str:
    """Basic deterministic text cleaning."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Load all article-summary pairs from the BBC dataset."""

    dataset_dirs = list(data_dir.rglob("News Articles"))

    if not dataset_dirs:
        raise FileNotFoundError(
            "Folder 'News Articles' not found."
        )

    dataset_root = dataset_dirs[0].parent

    articles_dir = dataset_root / "News Articles"
    summaries_dir = dataset_root / "Summaries"

    records = []

    for category in CATEGORIES:
        category_articles = articles_dir / category
        category_summaries = summaries_dir / category

        for article_path in sorted(category_articles.glob("*.txt")):
            summary_path = category_summaries / article_path.name

            if not summary_path.exists():
                continue

            article = read_text_file(article_path)
            summary = read_text_file(summary_path)

            records.append(
                {
                    "category": category,
                    "file_name": article_path.name,
                    "article": article,
                    "summary": summary,
                }
            )

    return pd.DataFrame(records)

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic preprocessing to the dataset."""

    df = df.copy()

    df["article"] = df["article"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)

    df["sentences"] = df["article"].apply(split_sentences)

    df["num_sentences"] = df["sentences"].apply(len)

    return df

def main() -> None:
    """Run the complete preprocessing pipeline."""

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading dataset...")

    df = load_dataset(data_dir)

    print(f"Loaded records: {len(df)}")

    print("Preprocessing dataset...")

    df = preprocess_dataset(df)

    output_path = output_dir / "preprocessed_dataset.pkl"

    df.to_pickle(output_path)

    print(f"Saved preprocessed dataset to: {output_path}")
    print(f"Final records: {len(df)}")


if __name__ == "__main__":
    main()