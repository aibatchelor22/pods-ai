#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Extract initial training and testing samples from detections.csv with the following constraints:
- At least 30 samples per category (or all available if < 30)
- At least one sample per category-node combination
- Prefer tp_machine_only or fp_machine_only notes
- Spread samples evenly across nodes per category
- Minimize total rows while meeting constraints

Timestamp correction and manual-sample merging now happen in merge_training_samples.py.
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from pytz import timezone
import os
import math
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
import glob

import ffmpeg
import m3u8
from audio_utils import (
    get_cached_folders,
    get_folders_between_timestamp,
    get_difference_between_times_in_seconds,
    download_from_url,
    load_m3u8_with_retry
)

# Get repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Offset, in seconds, between the detection end time and the start of the downloaded audio.
AUDIO_OFFSET_SECONDS: int = 2

PACIFIC_TZ = timezone('US/Pacific')
UTC_TZ = timezone('UTC')
PREFERRED_NOTES = {'tp_machine_only', 'fp_machine_only'}
QUALITY_FILTER_TERMS = {'faint', 'distant', 'quiet', 'noise'}
# Maximum number of detections.csv rows to use per category for training (can supplement with unlimited manual_samples.csv).
MAX_TRAINING_DETECTIONS_PER_CATEGORY = 30
MIN_TRAINING_SAMPLES_PER_CATEGORY = 30
# Minimum number of testing samples selected per category.
MIN_TESTING_SAMPLES_PER_CATEGORY = 10
# Maximum number of testing samples selected per category.
MAX_TESTING_SAMPLES_PER_CATEGORY = 30
# Categories where tp_human_only detections are excluded from testing samples.
NEGATIVE_CATEGORIES = {'water', 'human', 'vessel', 'jingle'}

# Count existing humpback signal files.
OTHER_HUMPBACK_SAMPLES = len(glob.glob(str(REPO_ROOT / 'output' / 'wav' / 'humpback' / 'signals-humpback_*.wav')))
SEGMENT_DURATION_SECONDS = 3  # Default duration of each audio segment for model inference.

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse a PST timestamp string to datetime object."""
    dt = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S_PST')
    return PACIFIC_TZ.localize(dt)


def format_timestamp(dt: datetime) -> str:
    """Format a datetime object to PST timestamp string."""
    return dt.strftime('%Y_%m_%d_%H_%M_%S_PST')


def subtract_segment_duration(timestamp_str: str, segment_duration: int = SEGMENT_DURATION_SECONDS) -> str:
    """Subtract segment_duration seconds from a PST timestamp string."""
    dt = parse_timestamp(timestamp_str)
    dt_minus_offset = dt - timedelta(seconds=segment_duration)
    return format_timestamp(dt_minus_offset)


def generate_uri(original_uri: str, timestamp_str: str) -> str:
    """
    Generate a URI for the Orcasound bouts interface from a PST timestamp.

    Args:
        original_uri: The original URI
        timestamp_str: PST timestamp string in format 'YYYY_MM_DD_HH_MM_SS_PST'

    Returns:
        URI in format "https://live.orcasound.net/bouts/new/{node}?time={utc_time}"
    """
    # Parse PST timestamp and convert to UTC.
    dt = parse_timestamp(timestamp_str)
    utc_dt = dt.astimezone(UTC_TZ)

    # Format as ISO 8601 with milliseconds and Z suffix.
    time_str = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # URL encode the time parameter.
    time_encoded = quote(time_str, safe='')

    base_uri = original_uri.split("?", 1)[0] # keep everything before the first "?".
    return f"{base_uri}?time={time_encoded}"


def load_detections(csv_path: Path) -> list[dict]:
    """Load all detections from CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def organize_by_category_node(detections: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """
    Organize detections by category and node.
    Returns: {category: {node: [list of detections]}}
    """
    organized = defaultdict(lambda: defaultdict(list))
    for det in detections:
        category = det['Category']
        node = det['NodeName']
        organized[category][node].append(det)
    return organized


def sort_by_preference(detections: list[dict], manual_confidences: dict[str, str]) -> list[dict]:
    """
    Sort detections by preference:
    1. Preferred notes (tp_machine_only, fp_machine_only) first
    2. Manual timestamps with 100.0 confidence preferred over no manual entry
    3. Descriptions without quality issues (e.g., faint, distant) preferred
    4. Then by timestamp (oldest first)

    Args:
        detections: List of detection dictionaries to sort
        manual_confidences: Dictionary mapping URIs to confidence strings (0.0-100.0).
            Detections with 100.0 confidence are prioritized. Detections with 0.0
            confidence should already have been removed before calling this function.

    Returns:
        list[dict]: Sorted list of detection dictionaries, ordered by preference
            (preferred notes first, 100.0 confidence before no-entry,
            quality issues deprioritized within each tier, oldest timestamps first)
    """
    def sort_key(det):
        has_preferred_note = det['Notes'] in PREFERRED_NOTES
        description_lower = det.get('Description', '').lower()
        has_quality_issue = any(term in description_lower for term in QUALITY_FILTER_TERMS)

        # Check if this detection has a manual override with 100.0 confidence.
        has_full_confidence = False
        if det['URI'] in manual_confidences:
            try:
                conf = float(manual_confidences[det['URI']])
                has_full_confidence = (conf == 100.0)
            except (ValueError, TypeError):
                pass

        timestamp = det['Timestamp']
        # Return tuple: (not preferred note, not full confidence, has quality issue, timestamp).
        # This puts preferred notes first, then 100.0-confidence entries, then by timestamp.
        return (not has_preferred_note, not has_full_confidence, has_quality_issue, timestamp)

    return sorted(detections, key=sort_key)


def validate_category_sample_counts(
    organized_data: dict[str, dict[str, list[dict]]],
    manual_sample_counts: dict[str, int]
) -> dict[str, dict[str, int]]:
    """
    Validate that each category has enough detections for training and testing.

    Args:
        organized_data: Nested dictionary with structure {category: {node: [detections]}} from detections.csv only.
        manual_sample_counts: Dictionary mapping category to count of manual samples.

    Returns:
        Dictionary mapping category to stats.
    """
    category_stats = {}

    for category in sorted(organized_data.keys()):
        # Count from detections.csv only.
        detections_available = sum(len(nodes) for nodes in organized_data[category].values())

        # Add manual samples to get total available.
        manual_count = manual_sample_counts.get(category, 0)
        total_available = detections_available + manual_count

        # For humpback, account for existing signal files.
        if category == 'humpback':
            required_training = max(0, MIN_TRAINING_SAMPLES_PER_CATEGORY - OTHER_HUMPBACK_SAMPLES)
        else:
            required_training = MIN_TRAINING_SAMPLES_PER_CATEGORY

        required_testing = MIN_TESTING_SAMPLES_PER_CATEGORY
        required_total = required_training + required_testing

        shortage = max(0, required_total - total_available)

        # Calculate training target from detections.csv:
        # Use at most MAX_TRAINING_DETECTIONS_PER_CATEGORY from detections.csv,
        # but reserve MIN_TESTING_SAMPLES_PER_CATEGORY for testing.
        # Formula: min(MAX_TRAINING, detections_available - MIN_TESTING)
        if detections_available >= MIN_TESTING_SAMPLES_PER_CATEGORY:
            # Reserve testing samples, take up to MAX for training.
            available_for_training = detections_available - MIN_TESTING_SAMPLES_PER_CATEGORY
            adjusted_training = min(MAX_TRAINING_DETECTIONS_PER_CATEGORY, available_for_training)
        else:
            # Not enough for testing minimum - take what we can.
            adjusted_training = max(0, detections_available - MIN_TESTING_SAMPLES_PER_CATEGORY)

        category_stats[category] = {
            'available': total_available,
            'detections_available': detections_available,
            'manual_count': manual_count,
            'required': required_total,
            'required_training': required_training,
            'required_testing': required_testing,
            'shortage': shortage,
            'adjusted_training_target': adjusted_training
        }

    return category_stats


def select_training_samples(
    organized_data: dict[str, dict[str, list[dict]]],
    manual_confidences: dict[str, str],
    category_stats: dict[str, dict[str, int]]
) -> list[dict]:
    """
    Select training samples according to requirements:
    - At least 30 samples per category (or all if < 30)
    - At least one sample per category-node combination
    - Spread evenly across nodes
    - Prefer certain note types
    - Minimize total rows

    Args:
        organized_data: Nested dictionary with structure {category: {node: [detections]}}.
            Each detection is a dictionary with keys: Category, NodeName, Timestamp, URI, etc.
        manual_confidences: Dictionary mapping URIs to confidence strings (0.0-100.0).
            Used to prioritize detections with 100.0 confidence during sorting.
        category_stats: Statistics about available vs required samples per category.

    Returns:
        List[Dict]: Selected training sample dictionaries, each containing:
            - Category: Detection category (resident, transient, humpback, water, vessel, jingle, human)
            - NodeName: Hydrophone node name
            - Timestamp: Detection timestamp (PST format)
            - URI: Orcasound bouts interface URI
            - Description: Human-readable description
            - Notes: Detection type (tp_machine_only, tp_human_only, etc.)
            - Confidence: Confidence score (if available)
        Samples are distributed evenly across nodes per category and sorted by preference.
    """
    selected = []

    for category in sorted(organized_data.keys()):
        nodes_data = organized_data[category]

        # Sort and prepare detections for each node.
        node_detections = {}
        for node in nodes_data.keys():
            node_detections[node] = sort_by_preference(nodes_data[node], manual_confidences)

        # Get adjusted target count for this category.
        stats = category_stats.get(category, {})
        target_count = stats.get('adjusted_training_target', MIN_TRAINING_SAMPLES_PER_CATEGORY)

        if stats.get('shortage', 0) > 0:
            print(f"  {category}: Adjusting training target to {target_count} (shortage of {stats['shortage']} total samples)")

        # Track how many samples selected per node.
        node_counts = defaultdict(int)
        category_samples = []

        # Round-robin selection across nodes to ensure even distribution.
        # First, ensure at least one sample per node (respecting target_count).
        for node in sorted(node_detections.keys()):
            if len(category_samples) >= target_count:
                break  # Stop if we've already reached the target.
            if node_detections[node]:
                category_samples.append(node_detections[node][0])
                node_counts[node] = 1

        # Continue round-robin to reach target count.
        while len(category_samples) < target_count:
            added_this_round = False

            # Sort nodes by how many samples they've contributed (fewest first).
            # This ensures even distribution.
            nodes_by_count = sorted(node_detections.keys(),
                                   key=lambda n: node_counts[n])

            for node in nodes_by_count:
                if len(category_samples) >= target_count:
                    break

                # Check if this node has more samples to give.
                if node_counts[node] < len(node_detections[node]):
                    idx = node_counts[node]
                    category_samples.append(node_detections[node][idx])
                    node_counts[node] += 1
                    added_this_round = True

            # If no node could contribute, we've exhausted all samples.
            if not added_this_round:
                break

        # Add all selected samples for this category.
        selected.extend(category_samples)

    return selected


def select_testing_samples(
    detections: list[dict],
    training_samples: list[dict],
    manual_confidences: dict[str, str]
) -> list[dict]:
    """
    Select testing samples according to requirements.

    Selection criteria:
    1. Exclude samples already selected for training.
    2. Exclude samples with manual confidence 0.0.
    3. Select up to MAX_TESTING_SAMPLES_PER_CATEGORY per category from eligible samples.
    4. Additionally select up to MAX_TESTING_SAMPLES_PER_CATEGORY tp_machine_only samples
       in resident category.
    5. Select up to MIN_TESTING_SAMPLES_PER_CATEGORY samples from negative categories
       (any note type).

    Args:
        detections: List of detection dictionaries.
        training_samples: List of selected training sample dictionaries.
        manual_confidences: Dictionary mapping URIs to confidence strings
            (string values representing floats in range 0.0-100.0).

    Returns:
        list[dict]: Selected testing sample dictionaries.
    """
    training_uris = {sample['URI'] for sample in training_samples if sample.get('URI', '')}

    # Separate detections into three groups.
    standard_eligible = []
    resident_tp_machine_only = []
    negative_category_samples = []  # All samples from negative categories.

    for detection in detections:
        uri = detection.get('URI', '')
        category = detection.get('Category', '')
        notes = detection.get('Notes', '')

        # Exclude training samples.
        if uri in training_uris:
            continue

        # Exclude samples with 0.0 confidence.
        if uri in manual_confidences:
            try:
                if float(manual_confidences[uri]) == 0.0:
                    continue
            except (ValueError, TypeError):
                pass

        # Categorize the detection.
        if notes == 'tp_machine_only' and category == 'resident':
            resident_tp_machine_only.append(detection)
        elif category in NEGATIVE_CATEGORIES:
            # Collect ALL negative category samples regardless of note type.
            negative_category_samples.append(detection)
        else:
            standard_eligible.append(detection)

    # Select from standard eligible samples.
    organized_standard = organize_by_category_node(standard_eligible)
    selected_testing_samples = []

    for category in sorted(organized_standard.keys()):
        category_detections = []
        for node_detections in organized_standard[category].values():
            category_detections.extend(node_detections)

        sorted_detections = sort_by_preference(category_detections, manual_confidences)
        selected_testing_samples.extend(sorted_detections[:MAX_TESTING_SAMPLES_PER_CATEGORY])

    # Add tp_machine_only resident samples.
    if resident_tp_machine_only:
        sorted_resident_machine = sort_by_preference(resident_tp_machine_only, manual_confidences)
        selected_testing_samples.extend(sorted_resident_machine[:MAX_TESTING_SAMPLES_PER_CATEGORY])

    # Add negative category samples (any note type).
    if negative_category_samples:
        # Organize by category to respect per-category limit.
        organized_negative = organize_by_category_node(negative_category_samples)
        for category in sorted(organized_negative.keys()):
            category_detections = []
            for node_detections in organized_negative[category].values():
                category_detections.extend(node_detections)

            sorted_detections = sort_by_preference(category_detections, manual_confidences)
            # Use MIN_TESTING_SAMPLES_PER_CATEGORY for negative categories to ensure minimum coverage.
            num_to_select = min(len(sorted_detections), MIN_TESTING_SAMPLES_PER_CATEGORY)
            selected_testing_samples.extend(sorted_detections[:num_to_select])

    return selected_testing_samples


def get_aligned_end_time(timestamp_str: str) -> datetime:
    """
    Calculate the aligned end time for 60-second audio download window.

    Snaps the given timestamp to the next 10-second boundary to align with
    HLS segment boundaries. This ensures we can download complete segments
    that cover the detection time.

    Args:
        timestamp_str: PST timestamp string in format 'YYYY_MM_DD_HH_MM_SS_PST'

    Returns:
        datetime: Aligned end time snapped to next 10-second boundary
            (rolls over to next minute if seconds become 60)
    """
    timestamp_pst = parse_timestamp(timestamp_str)

    # We need 60 seconds of audio ending at or shortly after the given timestamp.
    raw_end = timestamp_pst
    snapped_sec = ((raw_end.second + 9) // 10) * 10
    if snapped_sec == 60:
       # roll over to next minute.
       raw_end = raw_end + timedelta(minutes=1)
       snapped_sec = 0
    end_time = raw_end.replace(second=snapped_sec, microsecond=0)
    return end_time


def download_60s_audio(node_name: str, timestamp_str: str, tmp_dir: str) -> Optional[str]:
    """
    Download 60 seconds of audio, starting 51-60 seconds preceding the given timestamp.

    Args:
        node_name: The node name (e.g., "rpi_sunset_bay").
        timestamp_str: The detection timestamp in Pacific time.
        tmp_dir: Temporary directory to save the audio file.

    Returns:
        Path to the downloaded wav file, or None if download failed.
    """
    end_time = get_aligned_end_time(timestamp_str)
    start_time = end_time - timedelta(seconds=60)

    # Set up S3 bucket and folder information.
    hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + node_name
    bucket_folder = hydrophone_stream_url.split("https://s3-us-west-2.amazonaws.com/")[1]
    tokens = bucket_folder.split("/")
    s3_bucket = tokens[0]
    folder_name = tokens[1]
    prefix = folder_name + "/hls/"

    # Convert timestamps to unix time.
    start_unix_time = int(start_time.timestamp())
    end_unix_time = int(end_time.timestamp())

    try:
        # Use cached folders per node/bucket/prefix.
        all_hydrophone_folders = get_cached_folders(s3_bucket, prefix=prefix)
        print(f"  Found {len(all_hydrophone_folders)} folders in total for {node_name}")

        valid_folders = get_folders_between_timestamp(all_hydrophone_folders, start_unix_time, end_unix_time)
        print(f"  Found {len(valid_folders)} folders in date range")

        if not valid_folders:
            print(f"  Warning: No folders found for timestamp {start_time}")
            return None

        # Use the first valid folder.
        current_folder = int(valid_folders[0])

    except Exception as e:
        print(f"  ERROR: Failed to query S3 bucket: {e}")
        return None

    # Read the m3u8 file for the current folder.
    stream_url = f"{hydrophone_stream_url}/hls/{current_folder}/live.m3u8"

    try:
        stream_obj = load_m3u8_with_retry(stream_url)
    except Exception as e:
        print(f"  ERROR: Failed to load m3u8 file: {e}")
        return None

    num_total_segments = len(stream_obj.segments)
    if num_total_segments == 0:
        print(f"  ERROR: No segments found in m3u8 file")
        return None

    # Calculate target duration (average segment duration).
    target_duration_exact = sum(item.duration for item in stream_obj.segments) / num_total_segments
    target_duration = round(target_duration_exact, 1)

    # Calculate start and end indices based on time since folder start.
    # Note: there's typically a 2-second audio offset.
    audio_offset = 2
    time_since_folder_start_for_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)
    time_since_folder_start_for_start -= audio_offset

    time_since_folder_start_for_end = get_difference_between_times_in_seconds(end_unix_time, current_folder)
    time_since_folder_start_for_end -= audio_offset

    segment_start_index = max(0, math.floor(time_since_folder_start_for_start / target_duration))
    segment_end_index = min(num_total_segments, math.ceil(time_since_folder_start_for_end / target_duration))

    if segment_end_index > num_total_segments:
        print(f"  ERROR: Not enough segments available")
        return None

    # Download and process segments.
    try:
        file_names = []
        for i in range(segment_start_index, segment_end_index):
            audio_segment = stream_obj.segments[i]
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            download_from_url(audio_url, tmp_dir)
            file_names.append(file_name)

        if not file_names:
            print("  ERROR: No segments were successfully downloaded")
            return None

        # Concatenate all .ts files.
        clipname = f"temp_60s_{node_name}_{timestamp_str}"
        if len(file_names) > 1:
            hls_file = os.path.join(tmp_dir, clipname + ".ts")
            with open(hls_file, "wb") as wfd:
                for f in file_names:
                    with open(os.path.join(tmp_dir, f), "rb") as fd:
                        shutil.copyfileobj(fd, wfd)
        else:
            hls_file = os.path.join(tmp_dir, file_names[0])

        # Convert to wav using ffmpeg.
        wav_file_path = os.path.join(tmp_dir, f"{clipname}.wav")

        # Compute offset (seconds) into the concatenated .ts where the desired start occurs.
        ss_offset = time_since_folder_start_for_start - (segment_start_index * target_duration)
        if ss_offset < 0:
            ss_offset = 0.0

        # Use input seeking and limit duration to 60 seconds.
        stream = ffmpeg.input(hls_file, ss=ss_offset)
        stream = ffmpeg.output(
            stream,
            wav_file_path,
            t=60,  # 60 seconds.
            acodec="pcm_s16le",
            ar=44100,
            ac=1
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        print(f"  Downloaded 60s audio: {wav_file_path}")
        return wav_file_path

    except Exception as e:
        print(f"  Warning: Unable to retrieve audio clip: {e}")
        return None


def compute_correct_timestamp_for_tp_human_only(
    sample: dict,
    model_inference,
    tmp_dir: str,
    segment_duration: int = SEGMENT_DURATION_SECONDS
) -> tuple[str, float]:
    """
    Compute the correct timestamp for a tp_human_only detection.

    For tp_human_only detections, we download the preceding 50-60 seconds of audio,
    run the model on it to score each segment, find the highest scoring
    segment, and adjust the timestamp accordingly.

    Args:
        sample: Detection sample dictionary
        model_inference: Model inference instance
        tmp_dir: Temporary directory for audio files
        segment_duration: Duration of each audio segment in seconds

    Returns:
        Tuple of (corrected timestamp string, max confidence score in range 0.0-100.0)
    """
    node_name = sample['NodeName']
    timestamp_str = sample['Timestamp']

    print(f"Computing correct timestamp for tp_human_only: {node_name} - {timestamp_str}")

    # Download 60 seconds of audio.
    wav_path = download_60s_audio(node_name, timestamp_str, tmp_dir)

    if wav_path is None:
        print(f"  Failed to download audio, falling back to {segment_duration}-second offset")
        return subtract_segment_duration(timestamp_str, segment_duration), 0.0
    try:
        # Run model inference.
        print(f"  Running model inference...")
        prediction_results = model_inference.predict(wav_path)

        local_confidences = prediction_results["local_confidences"]

        # Get hop_duration from model output if available, otherwise infer it using corrected formula.
        # Models should return hop_duration and segment_duration to avoid inference errors.
        if "hop_duration" in prediction_results and "segment_duration" in prediction_results:
            hop_duration = prediction_results["hop_duration"]
            segment_duration_used = prediction_results["segment_duration"]
            print(f"  Model reported hop_duration={hop_duration}s, segment_duration={segment_duration_used}s")
        else:
            # Fallback: Infer hop duration using the corrected formula.
            # Corrected formula: hop = (audio_duration - segment_duration) / (num_positions - 1)
            # This accounts for the sliding window formula:
            # num_positions = floor((duration - segment) / hop) + 1
            try:
                import librosa
                audio, sr = librosa.load(wav_path, sr=None)
                audio_duration = len(audio) / sr
            except Exception as e:
                print(f"  Warning: Could not determine audio duration: {e}")
                audio_duration = 60.0  # Assume 60 seconds as fallback.

            # Calculate inferred hop duration using corrected formula.
            if len(local_confidences) > 1:
                # Corrected formula avoids overestimating hop duration.
                hop_duration = (audio_duration - segment_duration) / (len(local_confidences) - 1)
            elif len(local_confidences) == 1:
                hop_duration = audio_duration  # Single position covers entire audio.
            else:
                hop_duration = 1.0  # Default to 1 second if no confidences.

            print(f"  Warning: Model did not report hop_duration, inferred {hop_duration:.2f}s from {len(local_confidences)} positions")

        print(f"  Received {len(local_confidences)} confidence values")
        print(f"  Using hop duration: {hop_duration:.2f} seconds")

        # Print confidences with their corresponding time ranges.
        for (i, score) in enumerate(local_confidences):
            time_start = i * hop_duration
            time_end = time_start + hop_duration
            print(f"    Position {i} (time {time_start:.1f}-{time_end:.1f}s): Probability {score * 100:.2f}")

        if not local_confidences:
            print(f"  No confidences returned, falling back to {segment_duration}-second offset")
            return subtract_segment_duration(timestamp_str, segment_duration), 0.0

        # Find the index of the highest scoring segment.
        max_confidence_idx = local_confidences.index(max(local_confidences))
        max_confidence = local_confidences[max_confidence_idx]

        print(f"  Highest score: {max_confidence:.3f} at position {max_confidence_idx}")

        # Find whether it's better at the previous or next position.
        previous_confidence = local_confidences[max(0, max_confidence_idx - 1)]
        next_confidence = local_confidences[min(len(local_confidences) - 1, max_confidence_idx + 1)]
        if previous_confidence > next_confidence:
            max_confidence_idx = max(0, max_confidence_idx - 1)
            max_confidence = previous_confidence
            print(f"  Adjusted to previous position: {max_confidence:.3f} at position {max_confidence_idx}")

        # Convert position index to actual time offset in seconds.
        # max_confidence_idx represents a position, and each position covers hop_duration seconds.
        # The center of the position is at: max_confidence_idx * hop_duration + (hop_duration / 2).
        # For simplicity, we use the start of the position: max_confidence_idx * hop_duration.
        time_offset_seconds = max_confidence_idx * hop_duration

        print(f"  Position {max_confidence_idx} corresponds to time offset {time_offset_seconds:.1f}s")

        # max_confidence_idx represents a position in the downloaded 60-second WAV
        # We need to calculate what clock time that corresponds to
        # The WAV starts at (aligned_end - 60 - AUDIO_OFFSET_SECONDS) due to the 2-second offset in download.
        end_time = get_aligned_end_time(timestamp_str)
        wav_start_time = end_time - timedelta(seconds=60 + AUDIO_OFFSET_SECONDS)

        # The detected call is at wav_start_time + timedelta(seconds=time_offset_seconds).
        call_time = wav_start_time + timedelta(seconds=time_offset_seconds)
        print(f"  Call detected at: {call_time}")

        return format_timestamp(call_time), max_confidence * 100

    except Exception as e:
        print(f"  Error during inference: {e}")
        print(f"  Falling back to {segment_duration}-second offset")
        return subtract_segment_duration(timestamp_str, segment_duration), 0.0
    finally:
        # Clean up the downloaded audio file.
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except (OSError, PermissionError):
                # Ignore cleanup errors - file may already be deleted or inaccessible.
                pass


def process_sample(
    sample: dict,
    manual_timestamps: dict[str, str],
    manual_confidences: dict[str, str],
    model_inference=None,
    tmp_dir: Optional[str] = None,
    segment_duration: int = SEGMENT_DURATION_SECONDS
) -> dict:
    """
    Process a single sample, adjusting its timestamp, URI, and confidence.

    Applies one of four strategies based on the sample's Notes field and available data:
    1. Manual correction: if the sample's URI has a manual timestamp override.
    2. Model-based correction: if Notes is 'tp_human_only' and model_inference is provided.
    3. No offset: if sample came from manual_samples.csv (has _from_manual_samples marker).
    4. Fixed offset: subtract segment_duration seconds from the timestamp.

    Args:
        sample: Detection sample dictionary with keys Category, NodeName, Timestamp,
            URI, Description, Notes, Confidence. May also contain _from_manual_samples
            internal marker.
        manual_timestamps: Dictionary mapping URIs to corrected timestamp strings.
        manual_confidences: Dictionary mapping URIs to confidence strings (0.0-100.0).
        model_inference: Optional model inference instance for tp_human_only timestamp correction.
        tmp_dir: Temporary directory for audio downloads (required for tp_human_only).
        segment_duration: Duration of each audio segment in seconds.

    Returns:
        dict: Copy of sample with Timestamp, URI, and Confidence adjusted.
    """
    output_row = sample.copy()

    # Remove internal marker if present (don't write to CSV).
    is_from_manual_samples = output_row.pop('_from_manual_samples', False)

    # Check if there's a manual timestamp correction for this sample.
    if sample['URI'] in manual_timestamps:
        print(f"  Using manual timestamp correction for {sample['URI']}")
        output_row['Timestamp'] = manual_timestamps[sample['URI']]
        output_row['Confidence'] = manual_confidences[sample['URI']]

    # For tp_human_only detections, use model-based timestamp correction.
    elif sample['Notes'] == 'tp_human_only' and model_inference is not None:
        timestamp, confidence = compute_correct_timestamp_for_tp_human_only(
            sample, model_inference, tmp_dir or '', segment_duration
        )
        output_row['Timestamp'] = timestamp
        output_row['Confidence'] = f"{confidence:.1f}"

    # For samples from manual_samples.csv, keep timestamp as-is (already correct).
    elif is_from_manual_samples:
        print(f"  Keeping timestamp as-is for manual sample: {sample['Timestamp']}")
        # Timestamp remains unchanged from manual_samples.csv.
        confidence_str = sample.get('Confidence', '')
        if confidence_str:
            try:
                confidence = float(confidence_str)
                output_row['Confidence'] = f"{confidence:.1f}"
            except (ValueError, TypeError):
                output_row['Confidence'] = confidence_str  # Keep as-is if not a valid number.
        else:
            output_row['Confidence'] = ''

    else:
        # For all other detections, use the timestamp from segment_duration seconds earlier
        # since machine detection timestamps in the orcasite UI seem to be off by that much currently.
        output_row['Timestamp'] = subtract_segment_duration(sample['Timestamp'], segment_duration)
        confidence_str = sample.get('Confidence', '')
        if confidence_str:
            try:
                confidence = float(confidence_str)
                output_row['Confidence'] = f"{confidence:.1f}"
            except (ValueError, TypeError):
                output_row['Confidence'] = confidence_str  # Keep as-is if not a valid number.
        else:
            output_row['Confidence'] = ''

    # Update URI to match the new timestamp.
    output_row['URI'] = generate_uri(sample['URI'], output_row['Timestamp'])

    return output_row


def write_training_samples(
    samples: list[dict],
    output_path: Path,
    manual_timestamps: dict[str, str],
    manual_confidences: dict[str, str],
    model_inference=None,
    segment_duration: int = SEGMENT_DURATION_SECONDS
):
    """
    Write selected samples to training CSV with timestamps adjusted.

    Args:
        samples: List of sample dictionaries
        output_path: Path to output CSV file
        manual_timestamps: Dictionary mapping URIs to corrected timestamp strings
        manual_confidences: Dictionary mapping URIs to confidence strings (0.0-100.0)
        model_inference: Optional model inference instance for tp_human_only timestamp correction
        segment_duration: Duration of each audio segment in seconds

    Raises:
        ValueError: If duplicate output URIs are detected after processing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort samples by Category, then NodeName, then Timestamp.
    sorted_samples = sorted(samples, key=lambda s: (s['Category'], s['NodeName'], s['Timestamp']))

    # Track output URIs to detect duplicates.
    output_uris_seen = {}  # URI -> (Category, NodeName, Timestamp)
    processed_rows = []

    # Create a temporary directory for audio downloads.
    with TemporaryDirectory() as tmp_dir:
        total_samples = len(sorted_samples)
        print(f"\nProcessing {total_samples} samples...")

        for idx, sample in enumerate(sorted_samples, start=1):
            print(f"\n[{idx}/{total_samples}] Processing: {sample['Category']} - {sample['NodeName']} - {sample['Timestamp']}")

            output_row = process_sample(
                sample, manual_timestamps, manual_confidences, model_inference, tmp_dir, segment_duration
            )

            # Check for duplicate output URI.
            output_uri = output_row['URI']
            if output_uri in output_uris_seen:
                prev_info = output_uris_seen[output_uri]
                error_msg = (
                    f"\nError: Duplicate output URI detected after processing:\n"
                    f"  URI: {output_uri}\n"
                    f"  First occurrence: {prev_info[0]}, {prev_info[1]}, {prev_info[2]}\n"
                    f"  Second occurrence: {output_row['Category']}, {output_row['NodeName']}, {output_row['Timestamp']}\n"
                    f"\nEach training sample must have a unique output URI.\n"
                    f"This typically means the same timestamp appears twice, or timestamp correction\n"
                    f"produced the same result for different input samples."
                )
                raise ValueError(error_msg)

            output_uris_seen[output_uri] = (
                output_row['Category'],
                output_row['NodeName'],
                output_row['Timestamp']
            )
            processed_rows.append(output_row)

        # Write all rows to CSV after validation passes.
        print(f"\nValidated: All {len(processed_rows)} output samples have unique URIs")
        print(f"\nWriting to {output_path}...")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Use same columns as detections.csv.
            fieldnames = ['Category', 'NodeName', 'Timestamp', 'URI', 'Description', 'Notes', 'Confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

            for row in processed_rows:
                writer.writerow(row)


def write_testing_samples(samples: list[dict], output_path: Path):
    """
    Write selected testing samples to CSV using detections.csv format.

    Args:
        samples: List of sample dictionaries.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_samples = sorted(samples, key=lambda s: (s['Category'], s['NodeName'], s['Timestamp']))
    fieldnames = ['Category', 'NodeName', 'Timestamp', 'URI', 'Description', 'Notes', 'Confidence']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

        for sample in sorted_samples:
            writer.writerow({field: sample.get(field, '') for field in fieldnames})


def remove_zero_confidence_detections(
    detections: list[dict],
    manual_confidences: dict[str, str]
) -> list[dict]:
    """
    Remove from detections any entries whose URI has a manual_confidence of 0.

    Args:
        detections: List of detection dictionaries
        manual_confidences: Dictionary mapping URIs to confidence strings (0.0-100.0)

    Returns:
        list[dict]: Filtered list of detections with zero-confidence entries removed
    """
    filtered = []
    removed_count = 0

    for det in detections:
        uri = det.get('URI', '')
        if uri in manual_confidences:
            try:
                conf = float(manual_confidences[uri])
                if conf == 0.0:
                    removed_count += 1
                    continue  # Skip this detection.
            except (ValueError, TypeError):
                pass  # Keep detection if confidence can't be parsed.

        filtered.append(det)

    if removed_count > 0:
        print(f"  Removed {removed_count} detections with 0.0 confidence")

    return filtered


def load_manual_corrections(corrections_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load manual timestamp corrections and confidence values from CSV file.

    Args:
        corrections_path: Path to manual_timestamps.csv file

    Returns:
        Tuple of (manual_timestamps dict, manual_confidences dict)
        Returns empty dicts if file doesn't exist or has errors
    """
    manual_timestamps = {}
    manual_confidences = {}

    if not corrections_path.exists():
        return manual_timestamps, manual_confidences

    try:
        print(f"\nLoading manual timestamp corrections from {corrections_path}...")
        with open(corrections_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required column.
            if 'SampleURI' not in reader.fieldnames:
                print(f"  Warning: 'SampleURI' column not found in {corrections_path}. Skipping manual corrections.")
                return {}, {}

            for row_num, row in enumerate(reader, start=2):  # start=2 accounts for header.
                try:
                    uri = row.get('SampleURI', '').strip()
                    if not uri:
                        continue  # Skip rows without URI.

                    timestamp = row.get('Timestamp', '').strip()
                    confidence = row.get('Confidence', '').strip()

                    # Only store timestamp if provided.
                    if timestamp:
                        manual_timestamps[uri] = timestamp

                    # Store confidence (use provided value or default to 100.0).
                    manual_confidences[uri] = confidence if confidence else '100.0'

                except Exception as e:
                    print(f"  Warning: Skipping row {row_num} due to error: {e}")
                    continue

        if manual_timestamps:
            print(f"  Loaded {len(manual_timestamps)} manual timestamp corrections")
        if manual_confidences:
            print(f"  Loaded {len(manual_confidences)} confidence values")

    except Exception as e:
        print(f"  Warning: Failed to load manual corrections from {corrections_path}: {e}")
        return {}, {}

    return manual_timestamps, manual_confidences


def write_initial_training_samples(samples: list[dict], output_path: Path):
    """
    Write auto-selected training samples using detections.csv schema.

    Args:
        samples: List of selected initial training sample dictionaries.
        output_path: Path to output CSV file.
    """
    write_testing_samples(samples, output_path)


def main():
    """Main function to extract training samples."""
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Extract initial training and testing samples from detections CSV"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='output/csv/detections.csv',
        help='Path to input detections CSV file (default: output/csv/detections.csv)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=SEGMENT_DURATION_SECONDS,
        help=f'Duration of each audio segment in seconds (default: {SEGMENT_DURATION_SECONDS})'
    )
    args = parser.parse_args()

    # Paths.
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    initial_training_output_path = REPO_ROOT / 'output' / 'csv' / 'initial_training_samples.csv'
    testing_output_path = REPO_ROOT / 'output' / 'csv' / 'testing_samples.csv'

    # Load manual confidences for sorting.
    manual_corrections_path = REPO_ROOT / 'output' / 'csv' / 'manual_timestamps.csv'
    manual_timestamps, manual_confidences = load_manual_corrections(manual_corrections_path)

    print(f"Loading detections from {input_path}...")
    detections = load_detections(input_path)
    print(f"Loaded {len(detections)} detections")

    # Remove from detections any entries whose URI has a manual_confidence of 0.
    detections = remove_zero_confidence_detections(detections, manual_confidences)

    print("\nOrganizing detections by category and node...")
    organized_data = organize_by_category_node(detections)

    # Print summary.
    for category in sorted(organized_data.keys()):
        total = sum(len(nodes) for nodes in organized_data[category].values())
        print(f"  {category}: {total} detections across {len(organized_data[category])} nodes")

    # Validate category sample counts and adjust targets if needed.
    print("\nValidating sample counts per category...")
    category_stats = validate_category_sample_counts(organized_data, {})

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        print(f"  {category}: {stats['detections_available']} detections available")

    print("\nSelecting training samples...")
    samples = select_training_samples(organized_data, manual_confidences, category_stats)

    # Select testing samples from detections not already chosen for the initial training set.
    testing_samples = select_testing_samples(detections, samples, manual_confidences)

    print(f"\nSelected {len(samples)} initial training samples")

    print(f"\nSelected {len(testing_samples)} testing samples")

    print(f"\nWriting {len(samples)} initial training samples to {initial_training_output_path}...")
    write_initial_training_samples(samples, initial_training_output_path)

    print(f"Writing testing samples to {testing_output_path}...")
    write_testing_samples(testing_samples, testing_output_path)
    print("Done!")


if __name__ == "__main__":
    main()
