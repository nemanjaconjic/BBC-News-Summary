from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:

    project_root = Path(__file__).resolve().parent.parent

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "sentence_dataset.pkl"
    )

    df = pd.read_pickle(dataset_path)

    print("Total sentence records:", len(df))

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nLabel percentages:")
    print(df["label"].value_counts(normalize=True))

    print("\nExample records:")
    print(
        df[
            [
                "category",
                "sentence_index",
                "sentence",
                "similarity_score",
                "label",
            ]
        ].head(10).to_string(index=False)
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["similarity_score"],
        bins=30
    )

    plt.axvline(
        0.15,
        linestyle="--",
        label="Threshold = 0.15"
    )

    plt.xlabel("TF-IDF cosine similarity")
    plt.ylabel("Number of sentences")
    plt.title("Distribution of sentence-summary similarity")

    plt.legend()
    plt.tight_layout()
    plt.show()

    positive_per_article = (
        df.groupby(["category", "file_name"])["label"]
        .sum()
    )

    print("\nPositive sentences per article:")
    print(positive_per_article.describe())

    max_positive = positive_per_article.idxmax()
    max_count = positive_per_article.max()

    print("\nArticle with the most positive sentences:")
    print("Category:", max_positive[0])
    print("File:", max_positive[1])
    print("Positive sentences:", max_count)

    problematic_article = df[
        (df["category"] == max_positive[0])
        & (df["file_name"] == max_positive[1])
    ]

    print("\nSentences:")
    print(
        problematic_article[
            [
                "sentence_index",
                "similarity_score",
                "label",
                "sentence"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()