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

If --node-name and --timestamp are omitted the script parses them from the
input filename.  The filename must follow the convention used by the
download_wavs.py outputs:
    {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
For example, rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav yields
node_name=rpi_orcasound_lab and timestamp=2025_12_17_22_34_03_PST.
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ffmpeg
from pytz import timezone

from model_inference import get_model_inference

SEGMENT_DURATION = 3  # Duration of each segment in seconds.
HOP_DURATION = 2  # Hop size between segments in seconds.
DEFAULT_OUTPUT_DIR = "new"  # Default output directory for segments.
DEFAULT_MODEL_PATH = "davethaler/whale-call-detector"  # Default HuggingFace model ID.
PACIFIC_TZ = timezone("US/Pacific")  # Pacific timezone for timestamp formatting.

# Regex that matches filenames produced by download_wavs.py:
# {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
_FILENAME_PATTERN = re.compile(
    r"^(?P<node>.+?)_(?P<ts>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_PST)\.wav$",
    re.IGNORECASE,
)


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


def get_segment_prediction(model: object, segment_path: Path) -> str:
    """
    Run inference on a single segment WAV file and return the predicted class label.

    Uses the PODS-AI (podsai) model output format: the predicted label is taken
    from the 'global_prediction_label' key in the result dict.

    Args:
        model: Loaded model inference object (from get_model_inference).
        segment_path: Path to the segment WAV file to score.

    Returns:
        Predicted class label string (e.g., "resident", "humpback", "other").
        Returns "unknown" if inference fails.
    """
    try:
        result = model.predict(str(segment_path))
    except Exception as e:
        print(f"Warning: Inference failed for {segment_path}: {e}", file=sys.stderr)
        return "unknown"

    return result.get("global_prediction_label", "unknown")


def add_samples(
    wav_file: str,
    node_name: Optional[str] = None,
    base_timestamp: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
) -> list[tuple[str, str]]:
    """
    Split a WAV file into 3-second segments, save them, and run inference on each.

    Saves segments to output_dir using the filename convention
    {node_name_with_hyphens}_{timestamp_pst}.wav and returns a list of
    (filename, predicted_class) pairs.  Inference always uses the PODS-AI
    (podsai) model type.

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

    Returns:
        List of (filepath, predicted_class) tuples, one per segment.

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

    # Split the WAV and save segments.
    segments = split_wav_into_segments(wav_file, node_name, base_timestamp, out_dir)
    if not segments:
        return []

    # Load the model once and run inference on each segment.
    print(f"\nLoading podsai model from {model_path}...")
    model = get_model_inference(model_type="podsai", model_path=model_path)

    results: list[tuple[str, str]] = []
    print("\nSegment predictions:")
    for seg_path, _timestamp_str in segments:
        label = get_segment_prediction(model, seg_path)
        results.append((str(seg_path), label))
        print(f"  {seg_path.name}: {label}")

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
            "and run inference on each segment to output its predicted class."
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
