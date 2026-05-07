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

from extract_training_samples import download_60s_audio
from audio_utils import (
    get_cached_folders,
    get_folders_between_timestamp,
    get_difference_between_times_in_seconds,
    download_from_url,
    load_m3u8_with_retry
)

PACIFIC_TZ = timezone('US/Pacific')
N_SECONDS = 3  # Create 3-second wav files.

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
    with open(csv_path, 'r', newline='') as csvfile:
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


def download_audio_segment(
    category: str,
    node_name: str,
    timestamp_str: str,
    output_root: Path,
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
    node_name_in_filename = node_name.replace("_", "-")
    clipname = f"{node_name_in_filename}_{timestamp_str}"
    wav_filename = f"{clipname}.wav"
    expected_path = label_dir / wav_filename
    if expected_path.exists():
        print(f"Skipping (already exists): {expected_path}")
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

def process_csv(csv_path: Path, output_root: Path):
    """
    Read the training samples CSV file and download corresponding WAV files.
    
    Parameters:
        csv_path (Path): Path to the training_samples.csv file.
        output_root (Path): Root directory where audio files will be saved in label subdirectories.
    """
    rows = parse_csv(csv_path)
    
    print(f"Found {len(rows)} training samples to process")
    
    for row in rows:
        print(f"Processing: {row.category} - {row.node_name} - {row.timestamp_pst}")
        download_audio_segment(row.category, row.node_name, row.timestamp_pst, output_root)


def download_testing_sample(row: CSVRow, output_root: Path):
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
    node_name_in_filename = row.node_name.replace("_", "-")
    wav_filename = f"{node_name_in_filename}_{row.timestamp_pst}.wav"
    expected_path = label_dir / wav_filename
    if expected_path.exists():
        print(f"Skipping (already exists): {expected_path}")
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


def process_testing_csv(csv_path: Path, output_root: Path):
    """
    Read the testing samples CSV file and download corresponding WAV files.

    Args:
        csv_path: Path to the testing_samples.csv file.
        output_root: Root directory where testing WAV files are saved.
    """
    rows = parse_csv(csv_path)
    print(f"Found {len(rows)} testing samples to process")

    for row in rows:
        print(f"Processing testing sample: {row.category} - {row.node_name} - {row.timestamp_pst} ({row.notes})")
        download_testing_sample(row, output_root)

if __name__ == "__main__":
    training_csv_path = Path("output/csv/training_samples.csv")
    training_output_root = Path("output/wav")
    testing_csv_path = Path("output/csv/testing_samples.csv")
    testing_output_root = Path("output/testing-wav")

    if not training_csv_path.exists():
        print(f"Error: CSV file not found at {training_csv_path}")
        print("Please run extract_training_samples.py first to generate the training_samples.csv file.")
        sys.exit(1)

    process_csv(training_csv_path, training_output_root)

    if not testing_csv_path.exists():
        print(f"Warning: CSV file not found at {testing_csv_path}")
        print("Skipping testing WAV downloads. Run extract_training_samples.py to generate testing_samples.csv.")
    else:
        process_testing_csv(testing_csv_path, testing_output_root)
