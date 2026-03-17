"""
This module contains functional utils for downloading the archived dataset and extracting it
into the data directory. The dataset is downloaded via the hyperlink provided.
"""

import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm

from src.utils import CFG_PATH, get_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset(url: str, output_path: Union[str, Path]) -> None:
    """
    Downloads the dataset from the provided URL and saves it to the specified output path.

    Args:
        url (str): The URL of the dataset to download.
        output_path (Union[str, Path]): The path where the downloaded dataset will be saved.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
        IOError: If there is an issue saving the file to disk.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Dataset downloaded successfully and saved to {output_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while downloading the dataset: {e}")
    except IOError as e:
        logger.error(f"An error occurred while saving the dataset: {e}")


def extract_dataset(
    archive_path: Union[str, Path],
    extract_to: Union[str, Path],
    keep_archive: bool = False,
) -> None:
    """
    Extracts the dataset from the specified archive path to the given directory.
    If the archive contains a single root folder, moves its contents to extract_to.

    Args:
        archive_path (Union[str, Path]): The path to the archived dataset file (tar.gz format).
        extract_to (Union[str, Path]): The directory where the dataset will be extracted.
        keep_archive (bool): Whether to keep the original archive file.

    Raises:
        IOError: If there is an issue with reading the archive or writing the extracted files.
        tarfile.ReadError: If the archive file is not a valid tar.gz file.
    """
    extract_to = Path(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    try:
        logger.info(f"Starting extraction of archive: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=extract_to)

        # Check if all extracted files are in a single root folder
        extracted_items = list(extract_to.iterdir())
        if extracted_items[0].is_dir():
            root_folder = extracted_items[0]
            logger.info(
                f"Found single root folder: {root_folder.name}. Moving contents to {extract_to}"
            )

            # Move all contents from the root folder to extract_to
            for item in root_folder.iterdir():
                dest_path = extract_to / item.name
                # Remove destination if it already exists
                if dest_path.exists():
                    if dest_path.is_dir():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()
                shutil.move(str(item), str(dest_path))

            # Remove the now-empty root folder
            root_folder.rmdir()
            logger.info(
                f"Successfully moved all contents from {root_folder.name} to {extract_to}"
            )

        logger.info(f"Dataset extracted successfully to {extract_to}")
        if not keep_archive:
            os.remove(archive_path)
            logger.info(f"Original archive {archive_path} has been removed.")
        else:
            logger.info(f"Original archive {archive_path} retained.")
    except tarfile.ReadError as e:
        logger.error(f"The provided file is not a valid tar.gz archive: {e}")
    except IOError as e:
        logger.error(f"An error occurred while extracting the dataset: {e}")


def get_dataset(
    url: str,
    cfg_path: Union[str, Path],
    archive_path: Union[str, Path],
    keep_archive: bool = False,
) -> None:
    """
    Downloads and extracts the dataset from the provided URL.

    Args:
        url (str): The URL of the dataset to download.
        cfg_path (Union[str, Path]): The path to the configuration file.
        archive_path (Union[str, Path]): The path where the archived dataset will be saved.
        extract_to (Union[str, Path]): The directory where the dataset will be extracted.
        keep_archive (bool): Whether to keep the original archive file after extraction.
    """
    cfg = get_cfg(cfg_path)
    extract_to = Path(cfg["data_dir"])
    os.makedirs(extract_to, exist_ok=True)
    download_dataset(url, extract_to / archive_path)
    extract_dataset(extract_to / archive_path, extract_to, keep_archive)


if __name__ == "__main__":
    get_dataset(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        CFG_PATH,
        "cifar10.tar.gz",
        True,
    )
