# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import csv
import math
import os
import shutil
import sys
from tempfile import TemporaryDirectory

import ffmpeg
import m3u8
from pytz import timezone

from audio_utils import (
    download_60s_audio,
    get_cached_folders,
    get_folders_between_timestamp,
    get_difference_between_times_in_seconds,
    download_from_url,
    load_m3u8_with_retry
)

PACIFIC_TZ = timezone('US/Pacific')
N_SECONDS = 3  # Create 3-second wav files.
TESTING_WINDOW_SECONDS = 60
TESTING_CENTER_OFFSET_SECONDS = 30

@dataclass
class CSVRow:
    category: str
    node_name: str
    timestamp_pst: str
    uri: str
    description: str
    notes: str

# ============================================================================
# CSV Parsing
# ============================================================================



def parse_csv(csv_path: Path) -> List[CSVRow]:
    """
    Parse a CSV file (detections or training samples) and return a list of CSVRow objects.
    
    Parameters:
        csv_path (Path): Path to the CSV file.
    
    Returns:
        List[CSVRow]: List of parsed CSV rows.
    """
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip header
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 6:
                rows.append(CSVRow(
                    category=row[0],
                    node_name=row[1],
                    timestamp_pst=row[2],
                    uri=row[3],
                    description=row[4],
                    notes=row[5]
                ))
    return rows

def parse_timestamp_pst(timestamp_str: str) -> datetime:
    """
    Parse a PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.
    
    Parameters:
        timestamp_str (str): Timestamp string (e.g., "2025_12_24_17_51_23_PST").
    
    Returns:
        datetime: Parsed datetime object with Pacific timezone.
    """
    # Remove _PST suffix if present.
    timestamp_str = timestamp_str.replace('_PST', '')

    # Parse the datetime.
    dt_naive = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")

    # Localize to Pacific timezone.
    dt_aware = PACIFIC_TZ.localize(dt_naive)

    return dt_aware


def add_seconds_to_timestamp_pst(timestamp_str: str, seconds: int) -> str:
    """
    Add seconds to a PST timestamp string and return the same formatted representation.

    Parameters:
        timestamp_str (str): Timestamp string (e.g., "2025_12_24_17_51_23_PST").
        seconds (int): Number of seconds to add (or subtract if negative).

    Returns:
        str: Adjusted timestamp in the format YYYY_MM_DD_HH_MM_SS_PST.
    """
    adjusted = parse_timestamp_pst(timestamp_str) + timedelta(seconds=seconds)
    return adjusted.strftime("%Y_%m_%d_%H_%M_%S_PST")


def _training_window(row: CSVRow) -> tuple[datetime, datetime]:
    start = parse_timestamp_pst(row.timestamp_pst)
    return start, start + timedelta(seconds=N_SECONDS)


def _testing_window(row: CSVRow) -> tuple[datetime, datetime]:
    sample_time = parse_timestamp_pst(row.timestamp_pst)
    download_time = (
        sample_time
        if row.notes == "tp_human_only"
        else sample_time + timedelta(seconds=TESTING_CENTER_OFFSET_SECONDS)
    )

    # Mirror audio_utils.download_60s_audio() behavior: snap end time to the next 10-second boundary.
    snapped_sec = ((download_time.second + 9) // 10) * 10
    if snapped_sec == 60:
        download_time = download_time + timedelta(minutes=1)
        snapped_sec = 0
    end_time = download_time.replace(second=snapped_sec, microsecond=0)

    return end_time - timedelta(seconds=TESTING_WINDOW_SECONDS), end_time


def _find_overlaps(rows: list[CSVRow], window_fn, label: str) -> list[str]:
    overlaps = []
    by_node: dict[str, list[tuple[datetime, datetime, CSVRow]]] = {}
    for row in rows:
        start, end = window_fn(row)
        by_node.setdefault(row.node_name, []).append((start, end, row))

    for node_name, windows in by_node.items():
        windows.sort(key=lambda item: item[0])
        prev_start, prev_end, prev_row = windows[0]
        for curr_start, curr_end, curr_row in windows[1:]:
            if curr_start < prev_end:
                overlaps.append(
                    f"{label} overlap at node {node_name}: "
                    f"{prev_row.timestamp_pst} overlaps {curr_row.timestamp_pst}"
                )
            if curr_end > prev_end:
                prev_start, prev_end, prev_row = curr_start, curr_end, curr_row
    return overlaps


def _find_cross_overlaps(training_rows: list[CSVRow], testing_rows: list[CSVRow]) -> list[str]:
    overlaps = []
    train_by_node: dict[str, list[tuple[datetime, datetime, CSVRow]]] = {}
    test_by_node: dict[str, list[tuple[datetime, datetime, CSVRow]]] = {}

    for row in training_rows:
        start, end = _training_window(row)
        train_by_node.setdefault(row.node_name, []).append((start, end, row))
    for row in testing_rows:
        start, end = _testing_window(row)
        test_by_node.setdefault(row.node_name, []).append((start, end, row))

    for node_name in set(train_by_node.keys()) & set(test_by_node.keys()):
        train_windows = sorted(train_by_node[node_name], key=lambda item: item[0])
        test_windows = sorted(test_by_node[node_name], key=lambda item: item[0])
        i = 0
        j = 0
        while i < len(train_windows) and j < len(test_windows):
            train_start, train_end, train_row = train_windows[i]
            test_start, test_end, test_row = test_windows[j]
            if train_start < test_end and test_start < train_end:
                overlaps.append(
                    f"cross-file overlap at node {node_name}: "
                    f"training {train_row.timestamp_pst} overlaps testing {test_row.timestamp_pst}"
                )
            if train_end <= test_end:
                i += 1
            else:
                j += 1
    return overlaps


def _get_wav_filename(node_name: str, timestamp_pst: str) -> str:
    node_name_in_filename = node_name.replace("_", "-")
    return f"{node_name_in_filename}_{timestamp_pst}.wav"


def _get_relative_wav_path(row: CSVRow) -> Path:
    return Path(row.category) / _get_wav_filename(row.node_name, row.timestamp_pst)


def _copy_wav_from_cache_if_exists(expected_path: Path, output_root: Path, cache_root: Path | None) -> bool:
    """
    Copy a WAV file from cache_root into output_root if it exists there.

    Returns True when a cached file is copied, otherwise False.
    """
    if cache_root is None:
        return False
    try:
        relative_path = expected_path.relative_to(output_root)
    except ValueError:
        return False
    source_path = cache_root / relative_path
    if not source_path.exists():
        return False
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, expected_path)
    print(f"Copied from cache: {source_path} -> {expected_path}")
    return True


def delete_stale_wavs(output_root: Path, expected_relative_paths: set[Path]) -> None:
    """Delete WAV files under output_root that are not expected by the current CSV rows."""
    if not output_root.exists():
        return

    deleted_count = 0
    for wav_path in output_root.rglob("*.wav"):
        if wav_path.relative_to(output_root) not in expected_relative_paths:
            wav_path.unlink()
            print(f"Deleted stale wav: {wav_path}")
            deleted_count += 1

    for directory in sorted((path for path in output_root.rglob("*") if path.is_dir()), reverse=True):
        if directory == output_root:
            continue
        if not any(directory.iterdir()):
            directory.rmdir()

    if deleted_count:
        print(f"Deleted {deleted_count} stale wav file(s) from {output_root}")


def validate_no_overlaps(training_rows: list[CSVRow], testing_rows: list[CSVRow]) -> None:
    overlaps = []
    if training_rows:
        overlaps.extend(_find_overlaps(training_rows, _training_window, "training"))
    if testing_rows:
        overlaps.extend(_find_overlaps(testing_rows, _testing_window, "testing"))
    if training_rows and testing_rows:
        overlaps.extend(_find_cross_overlaps(training_rows, testing_rows))

    if overlaps:
        details = "\n".join(f"  - {overlap}" for overlap in overlaps)
        raise ValueError(f"Detected overlapping sample windows:\n{details}")


def download_audio_segment(
    category: str,
    node_name: str,
    timestamp_str: str,
    output_root: Path,
    cache_root: Path | None = None,
):
    """
    Download a 3-second audio segment for a detection and save it to the appropriate label directory.
    
    This function implements a simplified version of DateRangeHLSStream logic to download
    only a 3-second wav file instead of the full 60-second clip.
    
    Parameters:
        category (str): The label/category for the detection (e.g., "resident", "transient").
        node_name (str): The node name (e.g., "rpi_sunset_bay").
        timestamp_str (str): The detection timestamp in Pacific time.
        output_root (Path): Root directory where label subdirectories and audio files will be saved.
    """
    label_dir = output_root / category
    label_dir.mkdir(parents=True, exist_ok=True)
    timestamp_pst = parse_timestamp_pst(timestamp_str)
    
    # Check if the file already exists.
    wav_filename = _get_wav_filename(node_name, timestamp_str)
    clipname = wav_filename.removesuffix(".wav")
    expected_path = label_dir / wav_filename
    if expected_path.exists():
        print(f"Skipping (already exists): {expected_path}")
        return
    if _copy_wav_from_cache_if_exists(expected_path, output_root, cache_root):
        return
    
    # Set up S3 bucket and folder information.
    hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + node_name
    bucket_folder = hydrophone_stream_url.split("https://s3-us-west-2.amazonaws.com/")[1]
    tokens = bucket_folder.split("/")
    s3_bucket = tokens[0]
    folder_name = tokens[1]
    prefix = folder_name + "/hls/"
    
    # Convert timestamps to unix time.
    start_time = timestamp_pst
    end_time = start_time + timedelta(seconds=N_SECONDS)
    start_unix_time = int(start_time.timestamp())
    end_unix_time = int(end_time.timestamp())
    
    # Get all folders from S3 and filter by timestamp.
    try:
        # Use cached folders per node/bucket/prefix to avoid repeated S3 listing calls.
        all_hydrophone_folders = get_cached_folders(s3_bucket, prefix=prefix)
        print(f"Found {len(all_hydrophone_folders)} folders in total for {node_name}")
        
        valid_folders = get_folders_between_timestamp(all_hydrophone_folders, start_unix_time, end_unix_time)
        print(f"Found {len(valid_folders)} folders in date range")
        
        if not valid_folders:
            print(f"Warning: No folders found for timestamp {start_time}")
            return
        
        # Use the first valid folder.
        current_folder = int(valid_folders[0])
        
    except Exception as e:
        print(f"\nERROR: Failed to query S3 bucket.")
        print(f"Details: {e}")
        print(f"Hydrophone: {node_name}")
        print(f"Start time (unix): {start_unix_time}")
        print(f"End time (unix): {end_unix_time}")
        return
 
    # Read the m3u8 file for the current folder.
    stream_url = f"{hydrophone_stream_url}/hls/{current_folder}/live.m3u8"
    
    try:
        stream_obj = load_m3u8_with_retry(stream_url)
    except Exception as e:
        print(f"ERROR: Failed to load m3u8 file from {stream_url}")
        print(f"Details: {e}")
        return
    
    num_total_segments = len(stream_obj.segments)
    if num_total_segments == 0:
        print(f"ERROR: No segments found in m3u8 file")
        return
    
    # Calculate target duration (average segment duration).
    target_duration_exact = sum(item.duration for item in stream_obj.segments) / num_total_segments
    target_duration = round(target_duration_exact, 1)
    
    # Calculate number of segments needed for N_SECONDS.
    num_segments_needed = math.ceil(N_SECONDS / target_duration)
    
    # Calculate start and end indices based on time since folder start.
    # Don't apply a 2-second offset since it was already applied into the timestamps we have.
    time_since_folder_start_for_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)

    time_since_folder_start_for_end = get_difference_between_times_in_seconds(end_unix_time, current_folder)

    segment_start_index = max(0, math.floor(time_since_folder_start_for_start / target_duration))
    segment_end_index = min(num_total_segments, math.ceil(time_since_folder_start_for_end / target_duration))
    
    if segment_end_index > num_total_segments:
        print(f"ERROR: Not enough segments available. Need {segment_end_index}, but only {num_total_segments} available.")
        return
    
    # Download and process segments.
    try:
        with TemporaryDirectory() as tmp_path:
            os.makedirs(tmp_path, exist_ok=True)
            
            file_names = []
            for i in range(segment_start_index, segment_end_index):
                audio_segment = stream_obj.segments[i]
                base_path = audio_segment.base_uri
                file_name = audio_segment.uri
                audio_url = base_path + file_name
                download_from_url(audio_url, tmp_path)
                file_names.append(file_name)
            
            if not file_names:
                print("ERROR: No segments were successfully downloaded")
                return
            
            # Concatenate all .ts files.
            if len(file_names) > 1:
                hls_file = os.path.join(tmp_path, clipname + ".ts")
                with open(hls_file, "wb") as wfd:
                    for f in file_names:
                        with open(os.path.join(tmp_path, f), "rb") as fd:
                            shutil.copyfileobj(fd, wfd)
            else:
                hls_file = os.path.join(tmp_path, file_names[0])
            
            # Convert to wav using ffmpeg, but only extract N_SECONDS starting
            # at the requested timestamp offset inside the concatenated file.
            wav_file_path = os.path.join(label_dir, wav_filename)

            # Compute offset (seconds) into the concatenated .ts where the desired start occurs.
            # time_since_folder_start and target_duration are computed earlier in the function.
            ss_offset = time_since_folder_start_for_start - (segment_start_index * target_duration)
            if ss_offset < 0:
                ss_offset = 0.0

            # Use input seeking (ss on input) and limit duration with t on output.
            stream = ffmpeg.input(hls_file, ss=ss_offset)
            stream = ffmpeg.output(
                stream,
                wav_file_path,
                t=N_SECONDS,
                acodec="pcm_s16le",  # optional: force WAV PCM format
                ar=44100,            # optional: sample rate
                ac=1                 # optional: mono
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            print(f"Downloaded: {wav_file_path}")
            
    except Exception as e:
        print(f"\nWarning: Unable to retrieve audio clip.")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        print(f"Hydrophone: {node_name}")

def process_csv(csv_path: Path, output_root: Path, cache_root: Path | None = None):
    """
    Read the training samples CSV file and download corresponding WAV files.
    
    Parameters:
        csv_path (Path): Path to the training_3s_samples.csv file.
        output_root (Path): Root directory where audio files will be saved in label subdirectories.
    """
    rows = parse_csv(csv_path)
    
    print(f"Found {len(rows)} training samples to process")
    
    expected_relative_paths: set[Path] = set()
    for row in rows:
        expected_relative_paths.add(_get_relative_wav_path(row))
        print(f"Processing: {row.category} - {row.node_name} - {row.timestamp_pst}")
        download_audio_segment(
            row.category,
            row.node_name,
            row.timestamp_pst,
            output_root,
            cache_root=cache_root,
        )

    delete_stale_wavs(output_root, expected_relative_paths)


def download_testing_sample(row: CSVRow, output_root: Path, cache_root: Path | None = None):
    """
    Download audio for a testing sample.

    tp_human_only samples download a full 60-second clip.
    Other samples use the machine-detection segment logic.

    Args:
        row: Parsed CSV row describing one testing sample.
        output_root: Root directory where category subdirectories are created.

    Returns:
        None.
    """
    label_dir = output_root / row.category
    label_dir.mkdir(parents=True, exist_ok=True)
    wav_filename = _get_wav_filename(row.node_name, row.timestamp_pst)
    expected_path = label_dir / wav_filename
    if expected_path.exists():
        print(f"Skipping (already exists): {expected_path}")
        return
    if _copy_wav_from_cache_if_exists(expected_path, output_root, cache_root):
        return

    # For non-tp_human_only rows, shift by +30s so downloaded 60s clip is centered on row timestamp.
    download_timestamp = row.timestamp_pst
    if row.notes != "tp_human_only":
        download_timestamp = add_seconds_to_timestamp_pst(row.timestamp_pst, 30)

    with TemporaryDirectory() as tmp_dir:
        wav_path = download_60s_audio(row.node_name, download_timestamp, tmp_dir)
        if wav_path is None:
            print(f"Warning: Failed to download 60-second clip for {row.node_name} at {row.timestamp_pst}")
            return
        shutil.move(wav_path, expected_path)
        print(f"Downloaded: {expected_path}")


def process_testing_csv(csv_path: Path, output_root: Path, cache_root: Path | None = None):
    """
    Read the testing samples CSV file and download corresponding WAV files.

    Args:
        csv_path: Path to the testing_60s_samples.csv file.
        output_root: Root directory where testing WAV files are saved.
    """
    rows = parse_csv(csv_path)
    print(f"Found {len(rows)} testing samples to process")

    expected_relative_paths: set[Path] = set()
    for row in rows:
        expected_relative_paths.add(_get_relative_wav_path(row))
        print(f"Processing testing sample: {row.category} - {row.node_name} - {row.timestamp_pst} ({row.notes})")
        download_testing_sample(row, output_root, cache_root=cache_root)

    delete_stale_wavs(output_root, expected_relative_paths)


def print_usage():
    """
    Display usage information for this script.
    """
    print("Usage: python download_wavs.py [--validate-only]")
    print()
    print("This script downloads wav files for training and testing samples.")
    print("It reads from:")
    print("  - output/csv/training_3s_samples.csv")
    print("  - output/csv/testing_60s_samples.csv")
    print()
    print("And saves wav files to:")
    print("  - output/wav/ (training samples)")
    print("  - output/testing-wav/ (testing samples)")
    print()
    print("Optional argument:")
    print("  --validate-only: validate CSV overlap rules without downloading WAV files")
    print("Optional environment variables:")
    print("  - WAV_WORKTREE_DIR: root directory containing output/ (default: current directory)")
    print("  - WAV_CACHE_DIR: root directory to copy existing wav files from before downloading")


def run_download_wavs(validate_only: bool = False) -> None:
    training_csv_path = Path("output/csv/training_3s_samples.csv")
    testing_csv_path = Path("output/csv/testing_60s_samples.csv")

    worktree_root = Path(os.getenv("WAV_WORKTREE_DIR", "."))
    training_output_root = worktree_root / "output/wav"
    testing_output_root = worktree_root / "output/testing-wav"

    cache_root_env = os.getenv("WAV_CACHE_DIR")
    training_cache_root = None
    testing_cache_root = None
    if cache_root_env:
        cache_root = Path(cache_root_env)
        training_cache_root = cache_root / "output/wav"
        testing_cache_root = cache_root / "output/testing-wav"

    if not training_csv_path.exists():
        print(f"Error: CSV file not found at {training_csv_path}")
        print("Please update output/csv/training_3s_samples.csv before running download_wavs.py.")
        sys.exit(1)

    training_rows = parse_csv(training_csv_path)
    testing_rows: list[CSVRow] = []
    if not testing_csv_path.exists():
        print(f"Warning: CSV file not found at {testing_csv_path}")
        print("Skipping testing WAV downloads. Update output/csv/testing_60s_samples.csv to enable testing downloads.")
    else:
        testing_rows = parse_csv(testing_csv_path)

    validate_no_overlaps(training_rows, testing_rows)

    if validate_only:
        print("Overlap validation completed successfully.")
        return

    process_csv(training_csv_path, training_output_root, cache_root=training_cache_root)

    if testing_rows:
        process_testing_csv(testing_csv_path, testing_output_root, cache_root=testing_cache_root)


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--validate-only"):
        print_usage()
        sys.exit(1)

    run_download_wavs(validate_only=(len(sys.argv) == 2))
