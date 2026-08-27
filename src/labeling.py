from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_sentence_scores(
    sentences: list[str],
    summary: str,
) -> list[float]:
    """
    Calculate cosine similarity between each article sentence
    and the reference summary.
    """

    if not sentences:
        return []

    texts = sentences + [summary]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)

    sentence_vectors = matrix[:-1]
    summary_vector = matrix[-1]

    scores = cosine_similarity(
        sentence_vectors,
        summary_vector
    ).flatten()

    return scores.tolist()

def assign_labels(
    scores: list[float],
    threshold: float = 0.15,
) -> list[int]:
    """
    Convert similarity scores into binary labels.

    1 = sentence is considered relevant for the summary
    0 = sentence is not considered relevant.
    """

    return [
        int(score >= threshold)
        for score in scores
    ]

def create_sentence_dataset(
    df: pd.DataFrame,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Convert article-level data into sentence-level training data.
    """

    records = []

    for _, row in df.iterrows():

        sentences = row["sentences"]

        scores = calculate_sentence_scores(
            sentences,
            row["summary"]
        )

        labels = assign_labels(
            scores,
            threshold=threshold
        )

        for index, sentence in enumerate(sentences):

            records.append(
                {
                    "category": row["category"],
                    "file_name": row["file_name"],
                    "sentence_index": index,
                    "sentence": sentence,
                    "summary": row["summary"],
                    "similarity_score": scores[index],
                    "label": labels[index],
                }
            )

    return pd.DataFrame(records)

def main() -> None:

    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "data"
        / "processed"
        / "preprocessed_dataset.pkl"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "sentence_dataset.pkl"
    )

    print("Loading preprocessed dataset...")

    df = pd.read_pickle(input_path)

    print(f"Articles loaded: {len(df)}")

    print("Creating sentence-level labels...")

    sentence_df = create_sentence_dataset(
        df,
        threshold=0.15
    )

    sentence_df.to_pickle(output_path)

    print(
        f"Sentence-level records: {len(sentence_df)}"
    )

    print(
        f"Positive labels: "
        f"{sentence_df['label'].sum()}"
    )

    print(
        f"Positive label ratio: "
        f"{sentence_df['label'].mean():.2%}"
    )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()