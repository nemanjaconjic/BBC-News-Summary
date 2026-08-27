from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentence_dataset.pkl"
)


def main() -> None:
    print("Loading sentence dataset...")

    df = pd.read_pickle(INPUT_FILE)

    print("Number of records:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()