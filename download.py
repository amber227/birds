#!/usr/bin/env python3
"""
Download Xeno-canto recordings shorter than 10 seconds via API v3.

Behavior:
- By default runs in TEST_MODE and downloads only a limited number of files
  (e.g. 50) so you can confirm it works.
- You can later set TEST_MODE = False to let it run through all pages.
- File downloads are done in parallel using a thread pool.

Requirements:
    pip install requests
"""

import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ================== CONFIGURATION ==================

API_KEY = "66f02b95425dfb139e5accc3c43c9f83e78f73cf"  # <-- put your real key here

# Base API endpoint
API_URL = "https://xeno-canto.org/api/3/recordings"

# Query: all recordings with length < 10 seconds
QUERY = 'len:"<10"'

# Directory where audio files will be saved
OUTPUT_DIR = Path("downloads")

# Politeness / rate control
REQUEST_SLEEP_SEC = 1.0      # pause between API page requests
DOWNLOAD_SLEEP_SEC = 0.1     # pause between file downloads (per thread)

# Thread pool size for parallel downloads
DOWNLOAD_THREADS = 10       # you can increase/decrease this

# Test mode: only download a limited number of recordings
TEST_MODE = True
TEST_MAX_DOWNLOADS = 20      # "a few dozen"

# In full mode, optional hard upper bound as a safety
FULL_MAX_DOWNLOADS = None    # or e.g. 10000

# Per-page size (between 50 and 500 according to API docs)
PER_PAGE = 500

# ===================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def build_api_params(page: int) -> dict:
    """Build query parameters for the API request."""
    return {
        "query": QUERY,
        "key": API_KEY,
        "per_page": PER_PAGE,
        "page": page,
    }


def normalize_file_url(file_path: str) -> str:
    """Convert API file field to a full https URL if needed."""
    # API returns file like "//xeno-canto.org/694038/download"
    if file_path.startswith("//"):
        return "https:" + file_path
    if file_path.startswith("/"):
        return "https://xeno-canto.org" + file_path
    return file_path


def safe_filename(rec: dict) -> str:
    """
    Generate a safe local filename for a recording.
    Uses XC id and (optionally) the original filename extension.
    """
    xc_id = rec.get("id", "unknown")
    orig_name = rec.get("file-name", "")
    # Try to preserve the extension from the original file-name
    ext = ""
    if "." in orig_name:
        ext = "." + orig_name.rsplit(".", 1)[-1]
    if not ext:
        ext = ".mp3"
    return f"XC{xc_id}{ext}"


def download_one(recording: dict) -> bool:
    """
    Worker function: download a single recording.
    Returns True on success, False on failure or if skipped.
    """
    file_field = recording.get("file")
    if not file_field:
        return False

    url = normalize_file_url(file_field)
    filename = safe_filename(recording)
    dest = OUTPUT_DIR / filename

    if dest.exists():
        logging.info(f"[SKIP] Already exists: {dest.name}")
        return False

    xc_id = recording.get("id")
    logging.info(f"[DL] XC{xc_id} -> {dest.name}")

    try:
        # Use a separate session/request per thread for safety
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        time.sleep(DOWNLOAD_SLEEP_SEC)
        return True
    except Exception as e:
        logging.warning(f"[ERR] Failed to download {url}: {e}")
        return False


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Please set API_KEY to your own Xeno-canto API key.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # First request: find total number of pages & recordings
    page = 1
    params = build_api_params(page)
    logging.info(f"Requesting page {page} for initial metadata...")
    resp = session.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    num_recordings = int(data.get("numRecordings", "0"))
    num_pages = int(data.get("numPages", "0"))
    logging.info(
        f"Query: {QUERY}\n"
        f"Total recordings: {num_recordings}, total pages: {num_pages}"
    )

    if TEST_MODE:
        max_downloads = TEST_MAX_DOWNLOADS
        logging.info(
            f"TEST MODE is ON. Will download at most {max_downloads} recordings."
        )
    else:
        max_downloads = FULL_MAX_DOWNLOADS or num_recordings
        logging.info(
            f"FULL MODE. Will attempt to download up to {max_downloads} recordings."
        )

    downloaded = 0

    # Thread pool for downloads
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
        # Process first page (already fetched), then loop through the rest
        while True:
            logging.info(f"Processing page {page}/{num_pages}...")
            recordings = data.get("recordings", [])
            logging.info(f"  {len(recordings)} recordings on this page.")

            # Build a list of recordings to download from this page,
            # skipping ones already present and capping at remaining slots.
            to_download = []
            remaining_slots = max_downloads - downloaded
            if remaining_slots <= 0:
                logging.info(
                    f"Reached download limit ({max_downloads}). Stopping."
                )
                break

            for rec in recordings:
                if len(to_download) >= remaining_slots:
                    break

                file_field = rec.get("file")
                if not file_field:
                    continue

                filename = safe_filename(rec)
                dest = OUTPUT_DIR / filename
                if dest.exists():
                    continue  # will be logged as [SKIP] inside download_one if re-checked, but we skip work here

                to_download.append(rec)

            if not to_download:
                logging.info("  Nothing new to download on this page.")
            else:
                logging.info(
                    f"  Submitting {len(to_download)} downloads to thread pool "
                    f"(threads={DOWNLOAD_THREADS})..."
                )

                futures = [executor.submit(download_one, rec) for rec in to_download]

                # Collect results
                for fut in as_completed(futures):
                    if fut.result():
                        downloaded += 1

                logging.info(f"  Downloads so far: {downloaded}")

            # Check if we've hit the global limit
            if downloaded >= max_downloads:
                logging.info(
                    f"Reached download limit ({max_downloads}). Stopping."
                )
                break

            # Next page?
            page += 1
            if page > num_pages:
                logging.info("No more pages left.")
                break

            time.sleep(REQUEST_SLEEP_SEC)

            params = build_api_params(page)
            logging.info(f"Requesting page {page}...")
            resp = session.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

    logging.info(f"Finished. Total successfully downloaded: {downloaded}")


if __name__ == "__main__":
    main()
