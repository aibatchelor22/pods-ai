#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Split a WAV file into 3-second segments and run inference on each segment.

Usage:
    # Node name and timestamp inferred from filename:
    python add_samples.py rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav

    # Override node name and/or timestamp explicitly:
    python add_samples.py recording.wav --node-name rpi_orcasound_lab \\
        --timestamp 2025_01_15_12_30_00_PST
    python add_samples.py recording.wav --node-name rpi_sunset_bay \\
        --timestamp 2025_01_15_12_30_00_PST \\
        --model-path /path/to/custom-model

Saves 3-second segments with a 2-second hop to the "new/" output directory
(configurable with --output-dir) using the same filename convention as
output/wav/humpback/ etc.: {node_name_with_hyphens}_{timestamp_pst}.wav.
The timestamp in each filename reflects the actual start time of that sample.

Runs inference using the PODS-AI model (podsai) and prints the predicted class
label for each segment.  The default model is davethaler/whale-call-detector on
HuggingFace Hub; override with --model-path.

Output is printed in manual_samples.csv format (can be copy-pasted directly):
Category,NodeName,Timestamp,URI,Description,Notes,Confidence

URI/Description/Notes Lookup:
- The script looks up the detection in detections.csv (default: output/csv/detections.csv)
  by matching NodeName and Timestamp, and uses the URI, Description, and Notes from that row
- If not found in detections.csv, generates a URI from the timestamp (with empty Description and Notes="manual")

If --node-name and --timestamp are omitted the script parses them from the
input filename.  The filename must follow the convention used by the
download_wavs.py outputs:
    {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
For example, rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav yields
node_name=rpi_orcasound_lab and timestamp=2025_12_17_22_34_03_PST.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import ffmpeg
from pytz import timezone

from model_inference import get_model_inference
from orcasite_feeds import get_orcasite_feeds, OrcasiteFeed

SEGMENT_DURATION = 3  # Duration of each segment in seconds.
HOP_DURATION = 2  # Hop size between segments in seconds.
DEFAULT_OUTPUT_DIR = "new"  # Default output directory for segments.
DEFAULT_MODEL_PATH = "davethaler/whale-call-detector"  # Default HuggingFace model ID.
DEFAULT_DETECTIONS_CSV = "output/csv/detections.csv"  # Default path to detections.csv
PACIFIC_TZ = timezone("US/Pacific")  # Pacific timezone for timestamp formatting.
UTC_TZ = timezone("UTC")  # UTC timezone for URI generation.

# Regex that matches filenames produced by download_wavs.py:
# {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
_FILENAME_PATTERN = re.compile(
    r"^(?P<node>.+?)_(?P<ts>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_PST)\.wav$",
    re.IGNORECASE,
)

# Cache for Orcasite feeds to avoid repeated API calls.
_FEEDS_CACHE: Optional[list[OrcasiteFeed]] = None


@dataclass
class DetectionInfo:
    """Information from a detection row in detections.csv."""
    uri: str
    description: str
    notes: str


def parse_node_and_timestamp_from_filename(wav_file: str) -> tuple[str, str]:
    """
    Parse the hydrophone node name and PST timestamp from a WAV filename.

    Expects filenames following the convention used by download_wavs.py:
        {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
    For example, "rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav" yields
    node_name="rpi_orcasound_lab" and timestamp="2025_12_17_22_34_03_PST".
    Hyphens in the node-name portion are converted back to underscores.

    Args:
        wav_file: Path (or bare filename) of the WAV file to parse.

    Returns:
        Tuple of (node_name, timestamp_str) where node_name uses underscores.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    stem = Path(wav_file).name
    match = _FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Cannot infer node name and timestamp from filename: {stem!r}. "
            "The filename must follow the convention "
            "{node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav "
            "(e.g., rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav). "
            "Use --node-name and --timestamp to provide them explicitly."
        )
    node_name = match.group("node").replace("-", "_")
    timestamp_str = match.group("ts")
    return node_name, timestamp_str


def parse_timestamp_pst(timestamp_str: str) -> datetime:
    """
    Parse a PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.

    Args:
        timestamp_str: Timestamp string (e.g., "2025_12_24_17_51_23_PST").

    Returns:
        Parsed datetime object localized to the Pacific timezone.
    """
    timestamp_str = timestamp_str.replace("_PST", "")
    dt_naive = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
    return PACIFIC_TZ.localize(dt_naive)


def format_timestamp_pst(dt: datetime) -> str:
    """
    Format a datetime as a PST timestamp string.

    Args:
        dt: Datetime object (should already be localized to Pacific timezone).

    Returns:
        Timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.
    """
    return dt.strftime("%Y_%m_%d_%H_%M_%S_PST")


def get_node_slug(node_name: str) -> str:
    """
    Look up the slug for a node_name from Orcasite feeds.

    Args:
        node_name: Internal node name (e.g., "rpi_orcasound_lab").

    Returns:
        URL slug for the node (e.g., "orcasound-lab").

    Raises:
        ValueError: If the node_name is not found in the Orcasite feeds.
    """
    global _FEEDS_CACHE

    # Load feeds from API if not already cached.
    if _FEEDS_CACHE is None:
        try:
            _FEEDS_CACHE = get_orcasite_feeds()
        except Exception as e:
            raise ValueError(f"Failed to fetch Orcasite feeds: {e}") from e

    # Look up the node_name in the feeds.
    for feed in _FEEDS_CACHE:
        if feed.node_name == node_name:
            return feed.slug

    # If not found, raise an error with available node names.
    available = [f.node_name for f in _FEEDS_CACHE]
    raise ValueError(
        f"Node name '{node_name}' not found in Orcasite feeds. "
        f"Available nodes: {', '.join(available)}"
    )


def generate_uri(node_name: str, timestamp_str: str) -> str:
    """
    Generate a URI for the Orcasound bouts interface from a node name and PST timestamp.

    Args:
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        timestamp_str: PST timestamp string in format 'YYYY_MM_DD_HH_MM_SS_PST'.

    Returns:
        URI in format "https://live.orcasound.net/bouts/new/{slug}?time={utc_time}".

    Raises:
        ValueError: If the node_name is not found in Orcasite feeds.
    """
    # Look up the slug for this node_name.
    slug = get_node_slug(node_name)

    # Parse timestamp and convert to UTC.
    dt = parse_timestamp_pst(timestamp_str)
    utc_dt = dt.astimezone(UTC_TZ)

    # Format as ISO 8601 with milliseconds and Z suffix.
    time_str = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # URL encode the time parameter.
    time_encoded = quote(time_str, safe='')

    return f"https://live.orcasound.net/bouts/new/{slug}?time={time_encoded}"


def lookup_detection_in_csv(node_name: str, timestamp_str: str, detections_csv: str) -> Optional[DetectionInfo]:
    """
    Look up detection info in detections.csv by matching NodeName and Timestamp.

    Args:
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        timestamp_str: PST timestamp string (e.g., "2023_10_28_07_33_52_PST").
        detections_csv: Path to detections.csv file.

    Returns:
        DetectionInfo with uri, description, and notes if found, None otherwise.
    """
    detections_path = Path(detections_csv)
    if not detections_path.exists():
        print(f"Note: detections.csv not found at {detections_path}, will generate URI from timestamp")
        return None

    try:
        with open(detections_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Match both NodeName and Timestamp
                if row.get('NodeName', '') == node_name and row.get('Timestamp', '') == timestamp_str:
                    uri = row.get('URI', '').strip()
                    description = row.get('Description', '').strip()
                    notes = row.get('Notes', '').strip()

                    if uri:
                        print(f"Found detection in detections.csv:")
                        print(f"  URI: {uri}")
                        print(f"  Description: {description if description else '(empty)'}")
                        print(f"  Notes: {notes if notes else '(empty)'}")
                        return DetectionInfo(uri=uri, description=description, notes=notes)
    except Exception as e:
        print(f"Warning: Failed to read detections.csv: {e}")
        return None

    print(f"Note: No matching detection found in {detections_path} for {node_name}/{timestamp_str}")
    return None


def split_wav_into_segments(
    wav_file: str,
    node_name: str,
    base_timestamp: str,
    output_dir: Path,
    segment_duration: int = SEGMENT_DURATION,
    hop_duration: int = HOP_DURATION,
) -> list[tuple[Path, str]]:
    """
    Split a WAV file into fixed-duration segments with a hop and save to output_dir.

    Uses the same filename convention as output/wav/humpback/ etc.:
    {node_name_with_hyphens}_{timestamp_pst}.wav, where the timestamp is the
    actual start time of each sample.

    Args:
        wav_file: Path to the input WAV file.
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        base_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
        output_dir: Directory to save the segment WAV files.
        segment_duration: Duration of each segment in seconds (default: 3).
        hop_duration: Hop size between segment start times in seconds (default: 2).

    Returns:
        List of (segment_path, timestamp_str) tuples for each saved segment,
        in order of increasing start time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe the audio file to get duration.
    try:
        probe = ffmpeg.probe(wav_file)
        duration = float(probe["format"]["duration"])
    except Exception as e:
        print(f"Error: Could not probe {wav_file}: {e}", file=sys.stderr)
        return []

    # Compute number of segment positions (sliding window).
    # Each position starts at pos_idx * hop_duration seconds.
    # The last position must start early enough that the full segment fits.
    num_positions = int((duration - segment_duration) // hop_duration) + 1
    if num_positions < 1:
        num_positions = 1

    # Parse base timestamp and build filename prefix.
    base_time = parse_timestamp_pst(base_timestamp)
    # Replace underscores with hyphens in the node name (matches download_wavs.py convention).
    node_name_in_filename = node_name.replace("_", "-")

    segments: list[tuple[Path, str]] = []
    for pos_idx in range(num_positions):
        start_offset = pos_idx * hop_duration
        seg_time = base_time + timedelta(seconds=start_offset)
        timestamp_str = format_timestamp_pst(seg_time)
        filename = f"{node_name_in_filename}_{timestamp_str}.wav"
        out_path = output_dir / filename

        if out_path.exists():
            print(f"Skipping (already exists): {out_path}")
            segments.append((out_path, timestamp_str))
            continue

        try:
            stream = ffmpeg.input(wav_file, ss=start_offset)
            stream = ffmpeg.output(
                stream,
                str(out_path),
                t=segment_duration,
                acodec="pcm_s16le",
                ar=44100,
                ac=1,
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            print(f"Saved: {out_path}")
            segments.append((out_path, timestamp_str))
        except Exception as e:
            print(
                f"Warning: Failed to extract segment at offset {start_offset}s: {e}",
                file=sys.stderr,
            )

    return segments


def get_segment_prediction(model: object, segment_path: Path) -> tuple[str, float]:
    """
    Run inference on a single segment WAV file and return the predicted class and confidence.

    Uses the PODS-AI (podsai) model output format: the predicted label is taken
    from the 'global_prediction_label' key and confidence from 'global_confidence'.

    Args:
        model: Loaded model inference object (from get_model_inference).
        segment_path: Path to the segment WAV file to score.

    Returns:
        Tuple of (predicted_class_label, confidence_score) where confidence is 0.0-1.0.
        Returns ("unknown", 0.0) if inference fails.
    """
    try:
        result = model.predict(str(segment_path))
    except Exception as e:
        print(f"Warning: Inference failed for {segment_path}: {e}", file=sys.stderr)
        return "unknown", 0.0

    label = result.get("global_prediction_label", "unknown")
    confidence = float(result.get("global_confidence", 0.0))
    return label, confidence


def add_samples(
    wav_file: str,
    node_name: Optional[str] = None,
    base_timestamp: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    detections_csv: str = DEFAULT_DETECTIONS_CSV,
) -> list[dict]:
    """
    Split a WAV file into 3-second segments, save them, and run inference on each.

    Saves segments to output_dir using the filename convention
    {node_name_with_hyphens}_{timestamp_pst}.wav and returns a list of
    dictionaries with manual_samples.csv fields.  Inference always uses the
    PODS-AI (podsai) model type.

    If node_name or base_timestamp are not provided they are inferred from the
    wav_file filename, which must follow the convention used by download_wavs.py:
    {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
    (e.g., rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav).

    Args:
        wav_file: Path to the input WAV file.
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
            Inferred from wav_file filename if not provided.
        base_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
            Inferred from wav_file filename if not provided.
        output_dir: Directory to save segments (default: "new").
        model_path: HuggingFace Hub model ID or path to a local model directory
            (default: "davethaler/whale-call-detector").
        detections_csv: Path to detections.csv for detection lookup (default: "output/csv/detections.csv").

    Returns:
        List of dictionaries with keys matching manual_samples.csv format:
        Category, NodeName, Timestamp, URI, Description, Notes, Confidence.

    Raises:
        ValueError: If node_name or base_timestamp cannot be inferred and are
            not provided.
    """
    if node_name is None or base_timestamp is None:
        inferred_node, inferred_ts = parse_node_and_timestamp_from_filename(wav_file)
        if node_name is None:
            node_name = inferred_node
        if base_timestamp is None:
            base_timestamp = inferred_ts
    out_dir = Path(output_dir)

    # Try to look up detection info in detections.csv.
    detection_info = lookup_detection_in_csv(node_name, base_timestamp, detections_csv)
    if detection_info:
        # Use Description and Notes from detections.csv.
        shared_description = detection_info.description
        shared_notes = detection_info.notes
    else:
        shared_description = ""
        shared_notes = "manual"

    # Split the WAV and save segments.
    segments = split_wav_into_segments(wav_file, node_name, base_timestamp, out_dir)
    if not segments:
        return []

    # Load the model once and run inference on each segment.
    print(f"\nLoading podsai model from {model_path}...")
    model = get_model_inference(model_type="podsai", model_path=model_path)

    results: list[dict] = []
    print("\nSegments in manual_samples.csv format:")
    print("Category,NodeName,Timestamp,URI,Description,Notes,Confidence")

    for seg_path, timestamp_str in segments:
        label, confidence = get_segment_prediction(model, seg_path)

        # Convert confidence from 0.0-1.0 to 0.0-100.0.
        confidence_pct = confidence * 100

        # Generate URI matching this segment's timestamp.
        segment_uri = generate_uri(node_name, timestamp_str)

        # Create row dict matching manual_samples.csv format.
        row = {
            "Category": label,
            "NodeName": node_name,
            "Timestamp": timestamp_str,
            "URI": segment_uri,
            "Description": shared_description,
            "Notes": shared_notes,
            "Confidence": f"{confidence_pct:.1f}",
        }
        results.append(row)

        # Print in CSV format (ready to copy-paste).
        print(f"{label},{node_name},{timestamp_str},{segment_uri},{shared_description},{shared_notes},{row['Confidence']}")

    return results


def main() -> int:
    """Entry point for the add_samples CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Split a WAV file into 3-second segments with 2-second hop, "
            "save to the output directory using the standard filename convention "
            "({node_name_with_hyphens}_{timestamp_pst}.wav), "
            "and run inference on each segment. "
            "Output is printed in manual_samples.csv format for easy copy-paste."
        )
    )
    parser.add_argument(
        "wav_file",
        help="Path to the input WAV file to segment.",
    )
    parser.add_argument(
        "--node-name",
        default=None,
        help=(
            "Hydrophone node name (e.g., 'rpi_orcasound_lab'). "
            "Used in output filenames (underscores are replaced with hyphens). "
            "Inferred from the input filename if not provided."
        ),
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "PST timestamp of the start of the recording "
            "(e.g., '2025_01_15_12_30_00_PST'). "
            "Each segment filename encodes the actual start time of that sample. "
            "Inferred from the input filename if not provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save segments (default: {DEFAULT_OUTPUT_DIR!r}).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "HuggingFace Hub model ID or path to a local podsai model directory "
            f"(default: {DEFAULT_MODEL_PATH!r})."
        ),
    )
    parser.add_argument(
        "--detections-csv",
        default=DEFAULT_DETECTIONS_CSV,
        help=(
            f"Path to detections.csv for detection lookup (default: {DEFAULT_DETECTIONS_CSV!r})."
        ),
    )

    args = parser.parse_args()

    if not Path(args.wav_file).exists():
        print(f"Error: WAV file not found: {args.wav_file}", file=sys.stderr)
        return 1

    try:
        results = add_samples(
            wav_file=args.wav_file,
            node_name=args.node_name,
            base_timestamp=args.timestamp,
            output_dir=args.output_dir,
            model_path=args.model_path,
            detections_csv=args.detections_csv,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1

    if not results:
        print("No segments were processed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
