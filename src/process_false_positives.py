#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Process rejected OrcaHello resident detections into corrected manual samples.

For each rejected OrcaHello detection in the selected timeframe, this script:
1. Downloads the corresponding 60-second WAV file.
2. Runs PODS-AI inference on the full WAV.
3. Infers the corrected class from the OrcaHello comments.
4. Runs add_samples.py on the WAV file.
5. Appends whale-predicted segments that do not already match the corrected class
   to new_manual_samples.csv with the corrected class, avoiding duplicates.
"""

import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from add_samples import DEFAULT_DETECTIONS_CSV, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, add_samples
from extract_training_samples import download_60s_audio
from manual_samples_utils import append_manual_samples, load_existing_uris
from make_csv import (
    SKIP_TERMS,
    format_timestamp_pst,
    get_orcahello_detections,
    parse_pst_timestamp,
)
from model_inference import get_model_inference
from orcasite_feeds import get_orcasite_feeds_with_retry

DEFAULT_MANUAL_SAMPLES_CSV = "output/csv/new_manual_samples.csv"
RESIDENT_TERMS = ("resident", "pod")
TRANSIENT_TERMS = ("bigg", "transient")
HUMAN_TERMS = ("human", "radio")
VESSEL_TERMS = ("vessel", "ship", "boat", "train")
WHALE_CLASSES = {"resident", "transient", "humpback"}


def get_corrected_class(comments: str) -> Optional[str]:
    """Infer the corrected class from OrcaHello moderation comments."""
    text = (comments or "").lower()
    if not text or any(term in text for term in SKIP_TERMS):
        return None

    if any(term in text for term in RESIDENT_TERMS):
        return "resident"
    if any(term in text for term in TRANSIENT_TERMS):
        return "transient"
    if "humpback" in text:
        return "humpback"
    if any(term in text for term in HUMAN_TERMS):
        return "human"
    if any(term in text for term in VESSEL_TERMS):
        return "vessel"
    if "jingl" in text:
        return "jingle"
    if "water" in text:
        return "water"

    return None


def process_false_positives(
    manual_samples_path: Path,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    detections_csv: str = DEFAULT_DETECTIONS_CSV,
    feed_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> dict[str, int]:
    """Process rejected OrcaHello resident detections in the selected timeframe."""
    summary = {
        "rejected": 0,
        "download_failed": 0,
        "not_false_positive": 0,
        "processing_failed": 0,
        "unknown_class": 0,
        "whale_mismatch_segments": 0,
        "appended": 0,
        "duplicates": 0,
    }
    feeds = get_orcasite_feeds_with_retry()
    if not feeds:
        print("No Orcasite feeds available; nothing to process.")
        return summary
    if feed_filter:
        feeds = [feed for feed in feeds if feed.node_name == feed_filter]
        if not feeds:
            print(f"No feed found with node_name '{feed_filter}'")
            return summary

    existing_uris = load_existing_uris(manual_samples_path)
    print(f"Loading podsai model from {model_path}...")
    model = get_model_inference(model_type="podsai", model_path=model_path)

    for feed in feeds:
        print(f"Processing feed {feed.node_name}")
        for detection in get_orcahello_detections(feed):
            if detection.status.lower() != "rejected" or detection.timestamp is None:
                continue
            # OrcaHello detections are returned in descending timestamp order.
            # Once we are older than the requested start time, the remaining
            # detections for this feed will also be too old.
            if start_time is not None and detection.timestamp < start_time:
                break
            if end_time is not None and detection.timestamp > end_time:
                continue

            summary["rejected"] += 1
            timestamp_str = format_timestamp_pst(detection.timestamp)
            print(f"Checking rejected OrcaHello detection at {timestamp_str}")

            with TemporaryDirectory() as temp_dir:
                wav_path = download_60s_audio(feed.node_name, timestamp_str, temp_dir)
                if wav_path is None:
                    print(f"Skipping {feed.node_name} {timestamp_str}: failed to download audio.")
                    summary["download_failed"] += 1
                    continue

                try:
                    inference = model.predict(wav_path)
                    if inference.get("global_prediction_label") != "resident":
                        print(
                            f"Continuing with {feed.node_name} {timestamp_str}: "
                            "PODS-AI global prediction is not resident."
                        )
                        summary["not_false_positive"] += 1

                    corrected_class = get_corrected_class(detection.comments)
                    if corrected_class is None:
                        print(f"Skipping {feed.node_name} {timestamp_str}: could not determine corrected class from comments.")
                        summary["unknown_class"] += 1
                        continue

                    print(
                        f"Running add_samples.py for {feed.node_name} {timestamp_str} "
                        f"with corrected class '{corrected_class}'."
                    )

                    segment_rows = add_samples(
                        wav_file=wav_path,
                        node_name=feed.node_name,
                        base_timestamp=timestamp_str,
                        output_dir=str(output_dir),
                        model_path=model_path,
                        detections_csv=detections_csv,
                        model=model,
                        corrected_class=corrected_class,
                        fallback_description=detection.comments,
                        fallback_notes="fp_machine",
                    )
                except Exception as exc:
                    print(f"Skipping {feed.node_name} {timestamp_str}: processing failed ({exc}).")
                    summary["processing_failed"] += 1
                    continue

            mismatched_whale_rows = []
            for row in segment_rows:
                row_category = row.get("Category")
                if row_category not in WHALE_CLASSES or row_category == corrected_class:
                    continue
                updated_row = dict(row)
                updated_row["Category"] = corrected_class
                mismatched_whale_rows.append(updated_row)

            summary["whale_mismatch_segments"] += len(mismatched_whale_rows)
            appended, duplicates = append_manual_samples(
                manual_samples_path,
                mismatched_whale_rows,
                existing_uris,
            )
            summary["appended"] += appended
            summary["duplicates"] += duplicates

    return summary


def main() -> int:
    """Run the false-positive processing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Process rejected OrcaHello resident detections, re-run PODS-AI on the "
            "60-second WAV, and append mismatched whale-class sub-segments to "
            "new_manual_samples.csv with a corrected class."
        )
    )
    parser.add_argument(
        "--feed",
        type=str,
        help="Process only this feed (by node_name, e.g., rpi_sunset_bay).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY_MM_DD_HH_MM_SS_PST",
        help="Include only detections with timestamp >= this value.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026_03_17_00_00_00_PST",
        metavar="YYYY_MM_DD_HH_MM_SS_PST|now",
        help=(
            "Include only detections with timestamp <= this value. "
            "Use 'now' to remove the upper bound."
        ),
    )
    parser.add_argument(
        "--manual-samples-csv",
        default=DEFAULT_MANUAL_SAMPLES_CSV,
        help="Path to new_manual_samples.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where add_samples.py should write segment WAV files.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="PODS-AI model path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--detections-csv",
        default=DEFAULT_DETECTIONS_CSV,
        help="Path to detections.csv for add_samples.py metadata lookups.",
    )
    args = parser.parse_args()

    start_time = parse_pst_timestamp(args.start) if args.start else None
    end_time = None if (args.end or "").lower() == "now" else parse_pst_timestamp(args.end)

    summary = process_false_positives(
        manual_samples_path=Path(args.manual_samples_csv),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        detections_csv=args.detections_csv,
        feed_filter=args.feed,
        start_time=start_time,
        end_time=end_time,
    )

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
