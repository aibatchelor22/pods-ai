# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Common utilities for audio downloading and S3 operations.

This module contains shared functions used by both extract_training_samples.py
and download_wavs.py to avoid code duplication.
"""

from datetime import datetime
from typing import List
import http.client
import os
import time
import urllib.error

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import m3u8
import requests


# Simple in-memory cache for S3 folder listings keyed by "bucket::prefix"
_FOLDERS_CACHE = {}

# Number of times to retry a download on transient connection errors.
MAX_DOWNLOAD_RETRIES = 3

# Seconds to wait between download retry attempts.
DOWNLOAD_RETRY_DELAY_SECONDS = 2


def get_all_folders(bucket: str, prefix: str) -> List[str]:
    """
    Get all folder names from an S3 bucket with the given prefix.
    
    Parameters:
        bucket (str): Name of the S3 bucket.
        prefix (str): Prefix to filter objects.
    
    Returns:
        List[str]: List of folder names (without the prefix).
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}

    all_keys = []
    for page in paginator.paginate(**kwargs):
        try:
            common_prefixes = page["CommonPrefixes"]
            prefixes = [
                prefix["Prefix"].split("/")[-2] for prefix in common_prefixes
            ]
            all_keys.extend(prefixes)
        except KeyError:
            break

    return all_keys


def get_cached_folders(bucket: str, prefix: str) -> List[str]:
    """
    Return cached folder list for (bucket, prefix). If absent, call get_all_folders()
    to populate the cache, then return the cached value.
    """
    key = f"{bucket}::{prefix}"
    if key not in _FOLDERS_CACHE:
        _FOLDERS_CACHE[key] = get_all_folders(bucket, prefix)
    return _FOLDERS_CACHE[key]


def get_folders_between_timestamp(bucket_list: List[str], start_time: int, end_time: int) -> List[int]:
    """
    Filter bucket list to only include folders between start_time and end_time.
    
    Parameters:
        bucket_list (List[str]): List of folder names (as strings).
        start_time (int): Start unix timestamp.
        end_time (int): End unix timestamp.
    
    Returns:
        List[int]: Filtered list of folder names as integers.
    """
    bucket_list = [int(bucket) for bucket in bucket_list]
    start_index = 0
    end_index = len(bucket_list) - 1

    while start_index < len(bucket_list) and bucket_list[start_index] < start_time:
        start_index += 1

    while end_index >= 0 and bucket_list[end_index] > end_time:
        end_index -= 1

    # Include the folder before start_time to ensure we have data.
    return bucket_list[max(0, start_index - 1) : end_index + 1]


def get_difference_between_times_in_seconds(unix_time1: int, unix_time2: int) -> float:
    """
    Calculate the difference between two unix timestamps in seconds.
    
    Parameters:
        unix_time1 (int): First unix timestamp.
        unix_time2 (int): Second unix timestamp.
    
    Returns:
        float: Difference in seconds.
    """
    dt1 = datetime.fromtimestamp(int(unix_time1))
    dt2 = datetime.fromtimestamp(int(unix_time2))
    return (dt1 - dt2).total_seconds()


def download_from_url(dl_url: str, dl_dir: str):
    """
    Download a file from URL to a directory, with retry on transient connection errors.

    Retries up to MAX_DOWNLOAD_RETRIES times on ConnectionError or ChunkedEncodingError
    before re-raising the exception.

    Parameters:
        dl_url (str): URL to download from.
        dl_dir (str): Directory to save the file.
    """
    file_name = os.path.basename(dl_url)
    dl_path = os.path.join(dl_dir, file_name)

    if os.path.isfile(dl_path):
        return

    last_exception: Exception = RuntimeError("No attempts made")
    for attempt in range(MAX_DOWNLOAD_RETRIES + 1):
        try:
            response = requests.get(dl_url, timeout=30)
            response.raise_for_status()
            with open(dl_path, 'wb') as f:
                f.write(response.content)
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_exception = e
            if attempt < MAX_DOWNLOAD_RETRIES:
                print(f"  Retry {attempt + 1} of {MAX_DOWNLOAD_RETRIES} for {dl_url}: {e}")
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
    raise last_exception


def load_m3u8_with_retry(stream_url: str) -> m3u8.M3U8:
    """
    Load an m3u8 playlist from a URL, retrying on transient network errors.

    Retries up to MAX_DOWNLOAD_RETRIES times on IncompleteRead, URLError,
    or ConnectionError before re-raising the exception.

    Parameters:
        stream_url (str): URL of the m3u8 playlist to load.

    Returns:
        m3u8.M3U8: Parsed m3u8 object.
    """
    last_exception: Exception = RuntimeError("No attempts made")
    for attempt in range(MAX_DOWNLOAD_RETRIES + 1):
        try:
            return m3u8.load(stream_url)
        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError) as e:
            last_exception = e
            if attempt < MAX_DOWNLOAD_RETRIES:
                print(f"  Retry {attempt + 1} of {MAX_DOWNLOAD_RETRIES} for {stream_url}: {e}")
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
    raise last_exception
