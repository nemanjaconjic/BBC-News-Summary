from pathlib import Path

import pandas as pd

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "preprocessed_dataset.pkl"
    )

    df = pd.read_pickle(dataset_path)

    print("Records:", len(df))
    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst article:")
    print(df.iloc[0]["article"][:500])

    print("\nFirst sentences:")
    for sentence in df.iloc[0]["sentences"][:5]:
        print("-", sentence)

    print("\nNumber of sentences:")
    print(df.iloc[0]["num_sentences"])

if __name__ == "__main__":
    main()