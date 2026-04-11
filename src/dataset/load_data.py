"""
This module contains utilities for downloading, extracting, and registering the CIFAR-10 dataset.
The script is designed to be used as a DVC pipeline stage.
"""

import argparse
import hashlib
import logging
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import CFG_PATH, get_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_file_md5(file_path: Union[str, Path]) -> str:
    file_path = Path(file_path)
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_dataset(url: str, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192

        with output_path.open("wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info("Dataset downloaded successfully and saved to %s", output_path)
    except requests.exceptions.RequestException as e:
        logger.error("An error occurred while downloading the dataset: %s", e)
        raise
    except IOError as e:
        logger.error("An error occurred while saving the dataset: %s", e)
        raise


def extract_dataset(
    archive_path: Union[str, Path],
    extract_to: Union[str, Path],
    keep_archive: bool = False,
) -> None:
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    def identity_filter(tarinfo, path=None):
        return tarinfo

    try:
        logger.info("Starting extraction of archive: %s", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=extract_to, filter=identity_filter)

        extracted_items = list(extract_to.iterdir())
        if extracted_items and extracted_items[0].is_dir():
            root_folder = extracted_items[0]
            logger.info(
                "Found single root folder: %s. Moving contents to %s",
                root_folder.name,
                extract_to,
            )

            for item in root_folder.iterdir():
                dest_path = extract_to / item.name
                if dest_path.exists():
                    if dest_path.is_dir():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()
                shutil.move(str(item), str(dest_path))

            root_folder.rmdir()
            logger.info(
                "Successfully moved contents from %s to %s",
                root_folder.name,
                extract_to,
            )

        logger.info("Dataset extracted successfully to %s", extract_to)
        if not keep_archive:
            os.remove(archive_path)
            logger.info("Original archive %s has been removed.", archive_path)
        else:
            logger.info("Original archive %s retained.", archive_path)
    except tarfile.ReadError as e:
        logger.error("The provided file is not a valid tar.gz archive: %s", e)
        raise
    except IOError as e:
        logger.error("An error occurred while extracting the dataset: %s", e)
        raise


def create_dataset_registry(
    config: dict,
    output_dir: Union[str, Path],
    registry_path: Union[str, Path],
    archive_file: Union[str, Path],
    archive_hash: str,
) -> None:
    output_dir = Path(output_dir)
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry = {
        "dataset_name": "cifar10",
        "data_dir": str(output_dir),
        "source_url": config["data_url"],
        "archive_name": config["archive_name"],
        "archive_path": str(Path(archive_file)),
        "archive_md5": archive_hash,
        "downloaded_at": datetime.now(tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }

    pd.DataFrame([registry]).to_csv(registry_path, index=False)
    logger.info("Dataset registry written to %s", registry_path)


def get_dataset(
    url: str,
    cfg_path: Union[str, Path],
    archive_name: str,
    output_path: Union[str, Path] = None,
    keep_archive: bool = False,
    registry_path: Union[str, Path] = None,
) -> None:
    config = get_cfg(cfg_path)
    output_dir = Path(output_path or config["data_dir"])
    archive_name = archive_name or config["archive_name"]
    registry_path = registry_path or config["dataset_registry_path"]

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name

    data_files_exist = (
        all((output_dir / f"data_batch_{i}").exists() for i in range(1, 6))
        and (output_dir / "test_batch").exists()
    )

    if data_files_exist:
        logger.info(
            "Dataset already exists at %s, skipping download/extract.", output_dir
        )
    else:
        download_dataset(url, archive_path)
        extract_dataset(archive_path, output_dir, keep_archive)

    archive_hash = compute_file_md5(archive_path) if archive_path.exists() else ""
    create_dataset_registry(
        config, output_dir, registry_path, archive_path, archive_hash
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract CIFAR-10 dataset."
    )
    parser.add_argument(
        "--url", type=str, default=None, help="Dataset URL to download."
    )
    parser.add_argument(
        "--archive", type=str, default=None, help="Archive filename to download."
    )
    parser.add_argument(
        "--config", type=str, default=str(CFG_PATH), help="Path to the parameter file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for extracted dataset.",
    )
    parser.add_argument(
        "--keep-archive", action="store_true", help="Keep the archive after extraction."
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to save the dataset registry CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg(args.config)
    get_dataset(
        url=args.url or cfg["data_url"],
        cfg_path=args.config,
        archive_name=args.archive or cfg["archive_name"],
        output_path=args.output,
        keep_archive=args.keep_archive or cfg.get("download_keep_archive", False),
        registry_path=args.registry,
    )
