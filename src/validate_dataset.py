from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

CATEGORIES = [
    "business",
    "entertainment",
    "politics",
    "sport",
    "tech",
]


def get_txt_files(directory: Path):
    return sorted(directory.glob("*.txt"))


def main():

    dataset_dirs = list(DATA_DIR.rglob("News Articles"))

    if not dataset_dirs:
        print("ERROR: 'News Articles' not found.")
        return

    dataset_root = dataset_dirs[0].parent

    articles_root = dataset_root / "News Articles"
    summaries_root = dataset_root / "Summaries"

    print(f"Dataset root: {dataset_root}")
    print()

    total_articles = 0
    total_summaries = 0

    all_valid = True

    for category in CATEGORIES:
        articles_dir = articles_root / category
        summaries_dir = summaries_root / category

        article_files = get_txt_files(articles_dir)
        summary_files = get_txt_files(summaries_dir)

        article_names = {file.name for file in article_files}
        summary_names = {file.name for file in summary_files}

        missing_summaries = article_names - summary_names
        missing_articles = summary_names - article_names

        total_articles += len(article_files)
        total_summaries += len(summary_files)

        print(f"Category: {category}")
        print(f"  Articles:  {len(article_files)}")
        print(f"  Summaries: {len(summary_files)}")

        if missing_summaries:
            all_valid = False
            print(f"  Missing summaries: {len(missing_summaries)}")

        if missing_articles:
            all_valid = False
            print(f"  Missing articles: {len(missing_articles)}")

        if not missing_summaries and not missing_articles:
            print("  Pairing: OK")

        print()

    print("=" * 50)
    print(f"Total articles:  {total_articles}")
    print(f"Total summaries: {total_summaries}")

    if all_valid and total_articles == total_summaries:
        print("Dataset validation: PASSED")
    else:
        print("Dataset validation: FAILED")


if __name__ == "__main__":
    main()