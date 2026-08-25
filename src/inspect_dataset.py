from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def print_directory_tree(path: Path, level: int = 0):
    indent = "    " * level

    for item in sorted(path.iterdir()):
        print(f"{indent}{item.name}")

        if item.is_dir():
            print_directory_tree(item, level + 1)


if __name__ == "__main__":
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nRaw data directory:")
    print(DATA_DIR)

    print("\nDataset structure:")

    if DATA_DIR.exists():
        print_directory_tree(DATA_DIR)
    else:
        print("ERROR: data/raw directory does not exist.")