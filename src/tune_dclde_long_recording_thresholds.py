#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Tune long-recording KW event settings from cached window predictions.

Run evaluate_dclde_long_recordings.py once on a validation set, then use this
script to search thresholds without repeating neural-network inference.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

import evaluate_dclde_long_recordings as evaluation
from multispecies_train_model import ECOTYPE_ID2LABEL, ECOTYPE_LABELS


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_OUTPUT = "/kaggle/working/dclde_threshold_tuning"
DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80,0.90"
ECOTYPES = ("SRKW", "NRKW", "TKW", "OKW", "SAR")


@dataclass(frozen=True)
class CachedRecording:
    recording: evaluation.Recording
    starts: np.ndarray
    ends: np.ndarray
    kw_probabilities: np.ndarray
    ecotype_probabilities: np.ndarray


def parse_number_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        values = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return sorted(set(values))


def failed_soundfiles(path: Optional[Path]) -> set[str]:
    if path is None or not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            (row.get("Soundfile") or "").strip()
            for row in csv.DictReader(file)
            if (row.get("Soundfile") or "").strip()
        }


def load_cached_windows(
    path: Path,
    excluded_soundfiles: set[str],
) -> tuple[list[CachedRecording], dict[str, Any]]:
    required = {
        "Soundfile",
        "window_start_sec",
        "window_end_sec",
        "Provider",
        "Dataset",
        "kw_probability",
        *{f"ecotype_probability_{label}" for label in ECOTYPES},
    }
    builders: dict[str, dict[str, Any]] = {}
    row_count = 0
    excluded_rows = 0
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            soundfile = (row.get("Soundfile") or "").strip()
            if not soundfile:
                raise ValueError(f"Empty Soundfile at CSV row {row_number}")
            if soundfile in excluded_soundfiles:
                excluded_rows += 1
                continue
            try:
                start = float(row["window_start_sec"])
                end = float(row["window_end_sec"])
                kw_probability = float(row["kw_probability"])
                ecotype_probabilities = [
                    float(row[f"ecotype_probability_{label}"])
                    for label in ECOTYPES
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value at CSV row {row_number}") from error
            if end <= start or not 0 <= kw_probability <= 1:
                raise ValueError(f"Invalid window or KW probability at CSV row {row_number}")
            builder = builders.setdefault(
                soundfile,
                {
                    "provider": (row.get("Provider") or "").strip(),
                    "dataset": (row.get("Dataset") or "").strip(),
                    "starts": [],
                    "ends": [],
                    "kw": [],
                    "ecotype": [],
                },
            )
            builder["starts"].append(start)
            builder["ends"].append(end)
            builder["kw"].append(kw_probability)
            builder["ecotype"].append(ecotype_probabilities)
            row_count += 1

    cached = []
    duplicate_windows = []
    for soundfile, builder in builders.items():
        starts = np.asarray(builder["starts"], dtype=np.float64)
        order = np.argsort(starts, kind="stable")
        starts = starts[order]
        ends = np.asarray(builder["ends"], dtype=np.float64)[order]
        kw = np.asarray(builder["kw"], dtype=np.float32)[order]
        ecotype = np.asarray(builder["ecotype"], dtype=np.float32)[order]
        if len(np.unique(starts)) != len(starts):
            duplicate_windows.append(soundfile)
            continue
        duration = float(np.max(ends))
        recording = evaluation.Recording(
            soundfile,
            str(path),
            builder["provider"],
            builder["dataset"],
            duration,
        )
        cached.append(CachedRecording(recording, starts, ends, kw, ecotype))
    if duplicate_windows:
        examples = ", ".join(duplicate_windows[:10])
        raise ValueError(
            f"Duplicate window start times found in {len(duplicate_windows)} recordings: {examples}"
        )
    return cached, {
        "window_rows": row_count,
        "recordings": len(cached),
        "excluded_failed_soundfiles": len(excluded_soundfiles),
        "excluded_failed_window_rows": excluded_rows,
    }


def form_events(
    cached: CachedRecording,
    threshold: float,
    merge_gap_sec: float,
    minimum_positive_windows: int,
) -> list[evaluation.PredictedEvent]:
    positive = np.flatnonzero(cached.kw_probabilities >= threshold)
    if not len(positive):
        return []
    groups: list[list[int]] = []
    current = [int(positive[0])]
    current_end = float(cached.ends[positive[0]])
    for raw_index in positive[1:]:
        index = int(raw_index)
        if float(cached.starts[index]) - current_end <= merge_gap_sec:
            current.append(index)
            current_end = max(current_end, float(cached.ends[index]))
        else:
            groups.append(current)
            current = [index]
            current_end = float(cached.ends[index])
    groups.append(current)

    events = []
    for group in groups:
        if len(group) < minimum_positive_windows:
            continue
        ecotype_mean = np.mean(cached.ecotype_probabilities[group], axis=0)
        ecotype_index = int(np.argmax(ecotype_mean))
        kw_values = cached.kw_probabilities[group]
        events.append(
            evaluation.PredictedEvent(
                event_id=f"pred_{cached.recording.soundfile}_{len(events)}",
                soundfile=cached.recording.soundfile,
                start_sec=float(cached.starts[group[0]]),
                end_sec=float(np.max(cached.ends[group])),
                max_kw_probability=float(np.max(kw_values)),
                mean_kw_probability=float(np.mean(kw_values)),
                number_of_windows=len(group),
                predicted_ecotype=ECOTYPES[ecotype_index],
                ecotype_confidence=float(ecotype_mean[ecotype_index]),
                ecotype_probabilities=tuple(float(value) for value in ecotype_mean),
                provider=cached.recording.provider,
                dataset=cached.recording.dataset,
            )
        )
    return events


def ecotype_results(
    matches: list[evaluation.EventMatch],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], list[dict[str, Any]]]:
    pairs = [
        (match.true_ecotype, match.predicted_ecotype)
        for match in matches
        if match.status == "TP"
        and not match.ambiguous_ecotype_overlap
        and match.true_ecotype in ECOTYPE_LABELS
    ]
    matrix, rows, overall = evaluation.multiclass_metrics(pairs, ECOTYPES)
    return overall, matrix, rows


def evaluate_setting(
    cached_recordings: list[CachedRecording],
    truths_by_file: dict[str, list[evaluation.GroundTruthEvent]],
    threshold: float,
    merge_gap_sec: float,
    minimum_positive_windows: int,
    iou_threshold: float,
    audio_hours: float,
) -> tuple[dict[str, Any], list[evaluation.PredictedEvent], list[evaluation.EventMatch]]:
    predictions = []
    matches = []
    for cached in cached_recordings:
        file_predictions = form_events(
            cached, threshold, merge_gap_sec, minimum_positive_windows
        )
        file_matches = evaluation.match_events(
            file_predictions,
            truths_by_file.get(cached.recording.soundfile, []),
            iou_threshold,
        )
        predictions.extend(file_predictions)
        matches.extend(file_matches)
    detection = evaluation.event_detection_metrics(matches, audio_hours)
    ecotype, _, _ = ecotype_results(matches)
    row = {
        "kw_threshold": threshold,
        "event_merge_gap_sec": merge_gap_sec,
        "minimum_positive_windows": minimum_positive_windows,
        "ground_truth_kw_events": detection["true_positives"] + detection["false_negatives"],
        "predicted_kw_events": detection["true_positives"] + detection["false_positives"],
        **detection,
        "ecotype_evaluated": ecotype["evaluated"],
        "ecotype_accuracy": ecotype["accuracy"],
        "ecotype_macro_f1": ecotype["macro_f1"],
    }
    return row, predictions, matches


def sortable(value: Optional[float], missing: float = -1.0) -> float:
    return missing if value is None else float(value)


def select_best(
    rows: list[dict[str, Any]], max_fp_per_hour: Optional[float]
) -> tuple[dict[str, Any], bool]:
    eligible = rows
    constraint_met = True
    if max_fp_per_hour is not None:
        eligible = [
            row
            for row in rows
            if row["false_positives_per_hour"] is not None
            and row["false_positives_per_hour"] <= max_fp_per_hour
        ]
        if not eligible:
            constraint_met = False
            eligible = rows
    if constraint_met:
        best = max(
            eligible,
            key=lambda row: (
                sortable(row["f1"]),
                sortable(row["recall"]),
                -sortable(row["false_positives_per_hour"], math.inf),
                -row["kw_threshold"],
            ),
        )
    else:
        best = min(
            eligible,
            key=lambda row: (
                sortable(row["false_positives_per_hour"], math.inf),
                -sortable(row["recall"]),
                -sortable(row["f1"]),
            ),
        )
    return best, constraint_met


def write_rows(path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=evaluation.DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Randomly select at most this many cached recordings.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kw-thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--event-merge-gaps", default="0,1,2,3")
    parser.add_argument("--minimum-positive-windows", default="1,2,3")
    parser.add_argument("--event-iou-threshold", type=float, default=0.1)
    parser.add_argument(
        "--max-fp-per-hour",
        type=float,
        default=None,
        help="Optional operational constraint used when selecting the best setting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window_path = Path(args.window_predictions)
    if not window_path.is_file():
        raise FileNotFoundError(f"Window predictions not found: {window_path}")
    if not 0 <= args.event_iou_threshold <= 1:
        raise ValueError("--event-iou-threshold must be between 0 and 1")
    if args.max_fp_per_hour is not None and args.max_fp_per_hour < 0:
        raise ValueError("--max-fp-per-hour cannot be negative")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")

    thresholds = parse_number_list(args.kw_thresholds, float, "KW thresholds")
    gaps = parse_number_list(args.event_merge_gaps, float, "event merge gaps")
    minimum_windows = parse_number_list(
        args.minimum_positive_windows, int, "minimum positive windows"
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Every KW threshold must be between 0 and 1")
    if any(value < 0 for value in gaps) or any(value < 1 for value in minimum_windows):
        raise ValueError("Merge gaps cannot be negative and minimum windows must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_path = (
        Path(args.failed_files)
        if args.failed_files
        else window_path.parent / "failed_files.csv"
    )
    cached, cache_sanity = load_cached_windows(
        window_path, failed_soundfiles(failed_path)
    )
    if not cached:
        raise ValueError("No complete cached recordings were found")
    available_recordings = len(cached)
    if args.max_files is not None:
        cached = random.Random(args.seed).sample(
            cached,
            k=min(args.max_files, len(cached)),
        )
        cache_sanity["available_recordings_before_random_subset"] = available_recordings
        cache_sanity["random_subset_recordings"] = len(cached)
        cache_sanity["random_subset_seed"] = args.seed
    recordings = [item.recording for item in cached]
    annotation_table = evaluation.read_csv(args.annotations)
    truths_by_file, annotation_sanity = evaluation.load_annotations(
        args.annotations, recordings, annotation_table
    )
    audio_hours = sum(item.recording.duration_sec for item in cached) / 3600
    grid = list(itertools.product(thresholds, gaps, minimum_windows))
    print(f"Cached recordings: {len(cached):,}")
    print(f"Cached windows:    {cache_sanity['window_rows']:,}")
    print(f"Audio hours:       {audio_hours:.3f}")
    print(f"Grid settings:     {len(grid):,}")

    grid_rows = []
    for index, (threshold, gap, minimum) in enumerate(grid, start=1):
        row, _, _ = evaluate_setting(
            cached,
            truths_by_file,
            threshold,
            gap,
            minimum,
            args.event_iou_threshold,
            audio_hours,
        )
        grid_rows.append(row)
        if index % 10 == 0 or index == len(grid):
            print(f"Evaluated {index:,}/{len(grid):,} settings")
    grid_rows.sort(
        key=lambda row: (
            -sortable(row["f1"]),
            sortable(row["false_positives_per_hour"], math.inf),
            -sortable(row["recall"]),
        )
    )
    write_rows(output_dir / "threshold_grid_results.csv", grid_rows)

    best, constraint_met = select_best(grid_rows, args.max_fp_per_hour)
    best_row, best_predictions, best_matches = evaluate_setting(
        cached,
        truths_by_file,
        best["kw_threshold"],
        best["event_merge_gap_sec"],
        int(best["minimum_positive_windows"]),
        args.event_iou_threshold,
        audio_hours,
    )
    ecotype_overall, ecotype_matrix, ecotype_rows = ecotype_results(best_matches)
    provider_rows = evaluation.grouped_metrics(recordings, best_matches, "provider")
    dataset_rows = evaluation.grouped_metrics(recordings, best_matches, "dataset")

    prediction_rows = []
    for event in best_predictions:
        row = asdict(event)
        probabilities = row.pop("ecotype_probabilities")
        row.update(
            {
                f"ecotype_probability_{ECOTYPES[index]}": probabilities[index]
                for index in range(len(ECOTYPES))
            }
        )
        prediction_rows.append(row)
    prediction_fields = [
        field
        for field in evaluation.PredictedEvent.__dataclass_fields__
        if field != "ecotype_probabilities"
    ] + [f"ecotype_probability_{label}" for label in ECOTYPES]
    write_rows(
        output_dir / "best_predicted_events.csv",
        prediction_rows,
        prediction_fields,
    )
    write_rows(
        output_dir / "best_event_matches.csv",
        [asdict(match) for match in best_matches],
        list(evaluation.EventMatch.__dataclass_fields__),
    )
    write_rows(output_dir / "best_provider_metrics.csv", provider_rows)
    write_rows(output_dir / "best_dataset_metrics.csv", dataset_rows)
    write_rows(output_dir / "best_ecotype_metrics.csv", ecotype_rows)
    evaluation.write_matrix(
        output_dir / "best_ecotype_confusion_matrix.csv", ecotype_matrix, ECOTYPES
    )

    report = {
        "selection": {
            "maximum_false_positives_per_hour": args.max_fp_per_hour,
            "constraint_met": constraint_met,
            "ranking": "maximum event F1, then recall, then lower FP/hour",
        },
        "best_setting": best_row,
        "best_ecotype_metrics": ecotype_overall,
        "audio_hours": audio_hours,
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "grid": {
            "kw_thresholds": thresholds,
            "event_merge_gaps_sec": gaps,
            "minimum_positive_windows": minimum_windows,
            "event_iou_threshold": args.event_iou_threshold,
        },
        "arguments": vars(args),
    }
    with (output_dir / "best_threshold.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nBest validation setting")
    print("=" * 60)
    print(f"KW threshold:             {best_row['kw_threshold']}")
    print(f"Event merge gap:          {best_row['event_merge_gap_sec']} s")
    print(f"Minimum positive windows: {best_row['minimum_positive_windows']}")
    print(f"Event precision:          {best_row['precision']}")
    print(f"Event recall:             {best_row['recall']}")
    print(f"Event F1:                 {best_row['f1']}")
    print(f"False positives/hour:     {best_row['false_positives_per_hour']}")
    print(f"Ecotype macro F1:         {best_row['ecotype_macro_f1']}")
    if args.max_fp_per_hour is not None and not constraint_met:
        print("WARNING: no grid setting met the requested FP/hour constraint.")
    print(f"\nReports saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
