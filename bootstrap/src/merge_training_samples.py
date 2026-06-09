#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Merge initial and manual training samples into training_samples.csv."""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from bootstrap.src.extract_training_samples import (
    REPO_ROOT,
    SEGMENT_DURATION_SECONDS,
    load_detections,
    load_manual_corrections,
    subtract_segment_duration,
    write_training_samples,
)


def load_manual_samples(manual_samples_path: Path) -> list[dict]:
    """
    Load manually-specified training samples from CSV file.

    Args:
        manual_samples_path: Path to manual_samples.csv file.

    Returns:
        List of manual sample dictionaries.
    """
    manual_samples = []

    if not manual_samples_path.exists():
        return manual_samples

    try:
        print(f"\nLoading manual training samples from {manual_samples_path}...")
        with open(manual_samples_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            required_fields = {'Category', 'NodeName', 'Timestamp', 'URI'}
            if not required_fields.issubset(set(reader.fieldnames or [])):
                missing = required_fields - set(reader.fieldnames or [])
                print(f"  Warning: Required columns missing in {manual_samples_path}: {missing}")
                print("  Skipping manual samples.")
                return []

            for row_num, row in enumerate(reader, start=2):
                try:
                    if not all(row.get(field, '').strip() for field in required_fields):
                        print(f"  Warning: Skipping row {row_num} - required fields are empty")
                        continue

                    manual_samples.append({
                        'Category': row.get('Category', '').strip(),
                        'NodeName': row.get('NodeName', '').strip(),
                        'Timestamp': row.get('Timestamp', '').strip(),
                        'URI': row.get('URI', '').strip(),
                        'Description': row.get('Description', '').strip(),
                        'Notes': row.get('Notes', '').strip(),
                        'Confidence': row.get('Confidence', '').strip(),
                        '_from_manual_samples': True,
                    })
                except Exception as e:
                    print(f"  Warning: Skipping row {row_num} due to error: {e}")

        if manual_samples:
            print(f"  Loaded {len(manual_samples)} manual training samples")

    except Exception as e:
        print(f"  Warning: Failed to load manual samples from {manual_samples_path}: {e}")
        return []

    return manual_samples


def predict_output_timestamp(
    sample: dict,
    manual_timestamps: dict[str, str],
    segment_duration: int = SEGMENT_DURATION_SECONDS,
) -> str:
    """
    Predict the output timestamp after write_training_samples processing.

    Args:
        sample: Training sample dictionary.
        manual_timestamps: Dictionary mapping URIs to corrected timestamp strings.
        segment_duration: Duration of each audio segment in seconds.

    Returns:
        Predicted output timestamp string.
    """
    if sample['URI'] in manual_timestamps:
        return manual_timestamps[sample['URI']]

    if sample['Notes'] == 'tp_human_only':
        return sample['Timestamp']

    if sample.get('_from_manual_samples', False):
        return sample['Timestamp']

    return subtract_segment_duration(sample['Timestamp'], segment_duration)


def merge_manual_samples(
    selected_samples: list[dict],
    manual_samples: list[dict],
    manual_timestamps: dict[str, str],
    segment_duration: int = SEGMENT_DURATION_SECONDS,
) -> list[dict]:
    """
    Merge manual samples with automatically selected samples.

    Args:
        selected_samples: Automatically selected training samples.
        manual_samples: Manually-specified training samples.
        manual_timestamps: Dictionary mapping URIs to corrected timestamp strings.
        segment_duration: Duration of each audio segment in seconds.

    Returns:
        Combined list of samples with manual samples replacing duplicates.
    """
    if not manual_samples:
        return selected_samples

    manual_keys = set()
    for sample in manual_samples:
        output_ts = predict_output_timestamp(sample, manual_timestamps, segment_duration)
        manual_keys.add((sample['Category'], sample['NodeName'], output_ts))

    filtered_selected = []
    replaced_count = 0
    for sample in selected_samples:
        output_ts = predict_output_timestamp(sample, manual_timestamps, segment_duration)
        key = (sample['Category'], sample['NodeName'], output_ts)
        if key in manual_keys:
            replaced_count += 1
            print(
                "  Replacing auto-selected sample "
                f"(URI: {sample['URI']}, timestamp: {sample['Timestamp']} -> {output_ts}) "
                "with manual sample"
            )
            continue
        filtered_selected.append(sample)

    merged = filtered_selected + list(manual_samples)

    print(f"\nSelected {len(merged)} training samples")

    if manual_samples:
        print(f"\n  Added {len(manual_samples)} manual samples to training set")
    if replaced_count > 0:
        print(f"  Replaced {replaced_count} auto-selected samples with manual samples")

    return merged


def print_training_sample_breakdown(samples: list[dict]):
    """
    Print category and note breakdowns for merged training samples.

    Args:
        samples: Training sample dictionaries to summarize.
    """
    category_counts = defaultdict(int)
    category_node_counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        category_counts[sample['Category']] += 1
        category_node_counts[sample['Category']][sample['NodeName']] += 1

    for category in sorted(category_counts.keys()):
        print(f"  {category}: {category_counts[category]} samples")
        for node in sorted(category_node_counts[category].keys()):
            print(f"    {node}: {category_node_counts[category][node]}")

    type_counts = defaultdict(int)
    type_node_counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        type_counts[sample['Notes']] += 1
        type_node_counts[sample['Notes']][sample['NodeName']] += 1

    for sample_type in sorted(type_counts.keys()):
        print(f"  {sample_type}: {type_counts[sample_type]} samples")
        for node in sorted(type_node_counts[sample_type].keys()):
            print(f"    {node}: {type_node_counts[sample_type][node]}")


def main():
    """Main function to merge initial and manual training samples."""
    parser = argparse.ArgumentParser(
        description="Merge initial and manual training samples into training_samples.csv"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='bootstrap/csv/initial_training_samples.csv',
        help='Path to input initial training samples CSV file (default: bootstrap/csv/initial_training_samples.csv)',
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=SEGMENT_DURATION_SECONDS,
        help=f'Duration of each audio segment in seconds (default: {SEGMENT_DURATION_SECONDS})',
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    output_path = REPO_ROOT / 'bootstrap' / 'csv' / 'training_samples.csv'
    manual_samples_path = REPO_ROOT / 'bootstrap' / 'csv' / 'manual_samples.csv'
    manual_corrections_path = REPO_ROOT / 'bootstrap' / 'csv' / 'manual_timestamps.csv'

    manual_timestamps, manual_confidences = load_manual_corrections(manual_corrections_path)

    print(f"Loading initial training samples from {input_path}...")
    selected_samples = load_detections(input_path)
    print(f"Loaded {len(selected_samples)} initial training samples")

    manual_samples = load_manual_samples(manual_samples_path)
    merged_samples = merge_manual_samples(
        selected_samples,
        manual_samples,
        manual_timestamps,
        args.duration,
    )
    print_training_sample_breakdown(merged_samples)

    print("\nInitializing model inference for tp_human_only timestamp correction...")

    from src.model_inference import get_model_inference

    model_type = os.environ.get("MODEL_TYPE", "fastai")
    model_path = os.environ.get("MODEL_PATH", "./model")
    model_url = os.environ.get("MODEL_URL", None)
    auto_download_default = "true" if model_type == "fastai" else "false"
    auto_download = os.environ.get("MODEL_AUTO_DOWNLOAD", auto_download_default).lower() == "true"

    print(f"  Model type: {model_type}")
    if model_type == "fastai":
        print(f"  Model path: {model_path}")
        print(f"  Auto download: {auto_download}")
        if model_url:
            print(f"  Model URL: {model_url}")
        print("  Note: FastAI is the default model type.")
        print("  To customize, set environment variables:")
        print("    MODEL_TYPE=fastai (default)")
        print("    MODEL_PATH=./model (default)")
        print("    MODEL_AUTO_DOWNLOAD=true (default for fastai)")
        print("    MODEL_URL=<custom-url> (optional, to use a specific model version)")

    try:
        model_inference = get_model_inference(
            model_path=model_path if model_type == "fastai" else None,
            model_type=model_type,
            auto_download=auto_download,
            model_url=model_url,
        )
    except Exception as e:
        print(f"  Error: Failed to initialize model inference: {e}", file=sys.stderr)
        print("  Cannot proceed without model for tp_human_only timestamp correction.", file=sys.stderr)
        sys.exit(1)

    print(f"\nWriting {len(merged_samples)} training samples to {output_path}...")
    try:
        write_training_samples(
            merged_samples,
            output_path,
            manual_timestamps,
            manual_confidences,
            model_inference,
            args.duration,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
