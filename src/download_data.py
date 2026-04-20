import argparse
import logging
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:  # pragma: no cover
    kagglehub = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(
    competition: str,
    dest_dir: Path,
    force: bool = False,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing_csv_files = list(dest_dir.rglob("*.csv"))
    if existing_csv_files and not force:
        logger.info(
            "Kaggle raw data already exists at %s (%d csv files).",
            dest_dir,
            len(existing_csv_files),
        )
        return dest_dir

    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is not installed in the current environment. "
            "Install it with: pip install kagglehub"
        )

    logger.info("Downloading Kaggle competition data for '%s' ...", competition)
    kaggle_cache_path = Path(kagglehub.competition_download(competition))

    for src_file in kaggle_cache_path.rglob("*"):
        if src_file.is_file():
            relative = src_file.relative_to(kaggle_cache_path)
            dst_file = dest_dir / relative
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    logger.info("Kaggle data synced to %s", dest_dir)
    return dest_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle competition data to local raw folder.")
    parser.add_argument(
        "--competition",
        type=str,
        default="ieee-fraud-detection",
        help="Kaggle competition slug",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="data/raw",
        help="Destination directory for downloaded raw files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download / overwrite local raw files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = download_kaggle_dataset(
        competition=args.competition,
        dest_dir=Path(args.dest_dir),
        force=args.force,
    )

    csv_files = list(destination.rglob("*.csv"))
    logger.info("Download process completed. Found %d CSV files in %s", len(csv_files), destination)


if __name__ == "__main__":
    main()
