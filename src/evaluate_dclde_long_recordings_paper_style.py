#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Paper-style continuous KW evaluation from cached window predictions.

This script does not rerun the model. It converts the KW score time series in
``window_predictions.csv`` into events after an optional centered moving
average, sweeps a single detection threshold, and reports event precision-
recall curves plus recall at fixed false-positive/hour budgets.

High-confidence KW annotations are scored as ground truth. Annotations with
``KW_certain=0`` are treated as ignore intervals: predictions overlapping them
are reported but are neither true nor false positives. Missing confidence is
configurable and defaults to confirmed for compatibility with providers that
do not populate ``KW_certain``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_ANNOTATIONS = (
    "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/"
    "dclde_2027_killer_whales/Annotations.csv"
)
DEFAULT_OUTPUT = "/kaggle/working/dclde_long_paper_style"
SCORE_SOURCES = ("binary_kw", "species_kw", "max_ecotype_composite")
SPECIES = ("background", "KW", "HW", "AB")
ECOTYPES = ("SRKW", "NRKW", "TKW", "OKW", "SAR")


@dataclass(frozen=True)
class IgnoreInterval:
    soundfile: str
    annotation_id: str
    start_sec: float
    end_sec: float
    provider: str
    dataset: str
    reason: str = "KW_certain=0"


@dataclass(frozen=True)
class ContinuousEvent:
    soundfile: str
    event_id: str
    score_source: str
    moving_average_windows: int
    threshold: float
    first_window_index: int
    last_window_index: int
    peak_window_index: int
    peak_time_sec: float
    start_sec: float
    end_sec: float
    peak_score: float
    active_windows: int
    predicted_ecotype: str
    ecotype_confidence: float
    provider: str
    dataset: str


@dataclass(frozen=True)
class ContinuousMatch:
    soundfile: str
    status: str
    event_id: str
    true_event_id: str
    peak_time_sec: Optional[float]
    true_center_sec: Optional[float]
    absolute_distance_sec: Optional[float]
    event_score: Optional[float]
    predicted_ecotype: str
    true_ecotype: str
    provider: str
    dataset: str
    ignore_annotation_id: str = ""
    ignore_reason: str = ""


def parse_csv_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        values = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return list(dict.fromkeys(values))


def parse_score_sources(value: str) -> list[str]:
    values = parse_csv_list(value, str, "score sources")
    unknown = sorted(set(values) - set(SCORE_SOURCES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown score sources: {unknown}")
    return values


def score_vector(cached: Any, source: str) -> np.ndarray:
    if source == "binary_kw":
        return np.asarray(cached.kw_probabilities, dtype=np.float64)
    species_kw = np.asarray(
        cached.species_probabilities[:, SPECIES.index("KW")], dtype=np.float64
    )
    if source == "species_kw":
        return species_kw
    if source == "max_ecotype_composite":
        return species_kw * np.max(cached.ecotype_probabilities, axis=1)
    raise ValueError(f"Unknown score source: {source}")


def centered_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Return an edge-normalized centered moving average."""
    values = np.asarray(values, dtype=np.float64)
    if width < 1:
        raise ValueError("Moving-average width must be positive")
    if width == 1 or not len(values):
        return values.copy()
    left = (width - 1) // 2
    right = width // 2
    padded = np.pad(values, (left, right), mode="constant", constant_values=0.0)
    valid = np.pad(
        np.ones(len(values), dtype=np.float64),
        (left, right),
        mode="constant",
        constant_values=0.0,
    )
    kernel = np.ones(width, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid") / np.convolve(
        valid, kernel, mode="valid"
    )


def active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, len(indices) - 1]
    return [(int(indices[start]), int(indices[end])) for start, end in zip(starts, ends)]


def form_continuous_events(
    cached: Any,
    smoothed_scores: np.ndarray,
    score_source: str,
    moving_average_windows: int,
    threshold: float,
    ecotype_top_k: int,
    ecotype_threshold: float,
) -> list[ContinuousEvent]:
    scores = np.asarray(smoothed_scores, dtype=np.float64)
    events = []
    centers = (cached.starts + cached.ends) / 2.0
    for first, last in active_runs(scores >= threshold):
        run_indices = np.arange(first, last + 1)
        peak_index = int(run_indices[np.argmax(scores[run_indices])])
        top_count = min(ecotype_top_k, len(run_indices))
        top_indices = run_indices[
            np.argsort(scores[run_indices], kind="stable")[-top_count:]
        ]
        ecotype_mean = np.mean(cached.ecotype_probabilities[top_indices], axis=0)
        ecotype_index = int(np.argmax(ecotype_mean))
        ecotype_confidence = float(ecotype_mean[ecotype_index])
        predicted_ecotype = (
            ECOTYPES[ecotype_index]
            if ecotype_confidence >= ecotype_threshold
            else "unknown"
        )
        events.append(
            ContinuousEvent(
                soundfile=cached.recording.soundfile,
                event_id=f"continuous_{cached.recording.soundfile}_{first}_{last}",
                score_source=score_source,
                moving_average_windows=moving_average_windows,
                threshold=threshold,
                first_window_index=first,
                last_window_index=last,
                peak_window_index=peak_index,
                peak_time_sec=float(centers[peak_index]),
                start_sec=float(centers[first]),
                end_sec=float(centers[last]),
                peak_score=float(scores[peak_index]),
                active_windows=int(last - first + 1),
                predicted_ecotype=predicted_ecotype,
                ecotype_confidence=ecotype_confidence,
                provider=cached.recording.provider,
                dataset=cached.recording.dataset,
            )
        )
    return events


def confidence_state(value: Any, missing_mode: str) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"1", "1.0", "true", "yes", "y"}:
        return "confirmed"
    if normalized in {"0", "0.0", "false", "no", "n"}:
        return "ignore"
    if normalized in {"", "nan", "none", "null", "na", "n/a"}:
        return missing_mode
    return "unknown"


def load_confidence_annotations(
    base: Any,
    annotation_table: tuple[list[str], list[dict[str, str]]],
    recordings: list[Any],
    missing_confidence_mode: str,
) -> tuple[dict[str, list[Any]], dict[str, list[IgnoreInterval]], dict[str, Any]]:
    fields, rows = annotation_table
    sound_col = base.find_column(fields, ("Soundfile", "soundfile", "filename", "file"))
    start_col = base.find_column(
        fields, ("FileBeginSec", "Start", "start", "start_sec", "start_time_sec")
    )
    end_col = base.find_column(
        fields, ("FileEndSec", "End", "end", "end_sec", "end_time_sec")
    )
    species_col = base.find_column(fields, ("ClassSpecies", "class_species", "species"))
    confidence_col = base.find_column(
        fields, ("KW_certain", "kw_certain", "KWCertain"), required=False
    )
    ecotype_col = base.find_column(fields, ("Ecotype", "ecotype"), required=False)
    provider_col = base.find_column(fields, ("Provider", "provider"), required=False)
    dataset_col = base.find_column(fields, ("Dataset", "dataset"), required=False)
    recording_by_name = {recording.soundfile: recording for recording in recordings}
    confirmed: dict[str, list[Any]] = defaultdict(list)
    ignored: dict[str, list[IgnoreInterval]] = defaultdict(list)
    seen_confirmed: set[tuple[Any, ...]] = set()
    seen_ignored: set[tuple[Any, ...]] = set()
    counts = Counter()

    for row_index, row in enumerate(rows):
        if not base.is_kw_species(row.get(species_col)):
            continue
        soundfile = base.normalize_text(row.get(sound_col))
        recording = recording_by_name.get(soundfile)
        if recording is None:
            continue
        try:
            start = float(row[start_col])
            end = float(row[end_col])
        except (TypeError, ValueError):
            counts["malformed"] += 1
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            counts["malformed"] += 1
            continue
        start = max(0.0, start)
        if recording.duration_sec > 0:
            end = min(float(recording.duration_sec), end)
        if end <= start:
            counts["outside_recording"] += 1
            continue
        state = (
            confidence_state(row.get(confidence_col), missing_confidence_mode)
            if confidence_col
            else missing_confidence_mode
        )
        if state == "unknown":
            counts["unknown_confidence"] += 1
            state = missing_confidence_mode
        if state == "exclude":
            counts["missing_confidence_excluded"] += 1
            continue
        ecotype = base.normalize_text(row.get(ecotype_col)).upper() if ecotype_col else ""
        if ecotype not in ECOTYPES:
            ecotype = ""
        provider = (
            base.normalize_text(row.get(provider_col)) if provider_col else recording.provider
        ) or recording.provider
        dataset = (
            base.normalize_text(row.get(dataset_col)) if dataset_col else recording.dataset
        ) or recording.dataset
        duplicate_key = (soundfile, round(start, 6), round(end, 6), ecotype)
        if state == "ignore":
            if duplicate_key in seen_ignored:
                counts["duplicate_ignore"] += 1
                continue
            seen_ignored.add(duplicate_key)
            ignored[soundfile].append(
                IgnoreInterval(
                    soundfile,
                    f"ignore_{row_index}",
                    start,
                    end,
                    provider,
                    dataset,
                )
            )
            counts["ignore"] += 1
        else:
            if duplicate_key in seen_confirmed:
                counts["duplicate_confirmed"] += 1
                continue
            seen_confirmed.add(duplicate_key)
            confirmed[soundfile].append(
                base.GroundTruthEvent(
                    f"true_{row_index}",
                    soundfile,
                    start,
                    end,
                    ecotype,
                    provider,
                    dataset,
                )
            )
            counts["confirmed"] += 1
    for values in confirmed.values():
        values.sort(key=lambda item: (item.start_sec, item.end_sec))
    for values in ignored.values():
        values.sort(key=lambda item: (item.start_sec, item.end_sec))
    return confirmed, ignored, {
        "kw_confidence_column": confidence_col,
        "missing_confidence_mode": missing_confidence_mode,
        **dict(counts),
    }


def match_events(
    predictions: list[ContinuousEvent],
    truths: list[Any],
    ignore_intervals: list[IgnoreInterval],
    collar_sec: float,
    ignore_collar_sec: float,
) -> list[ContinuousMatch]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            center = (truth.start_sec + truth.end_sec) / 2.0
            distance = abs(prediction.peak_time_sec - center)
            if distance <= collar_sec:
                candidates.append(
                    (distance, -prediction.peak_score, pred_index, truth_index)
                )
    candidates.sort()
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches = []
    for distance, _, pred_index, truth_index in candidates:
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        prediction = predictions[pred_index]
        truth = truths[truth_index]
        matches.append(
            ContinuousMatch(
                prediction.soundfile,
                "TP",
                prediction.event_id,
                truth.event_id,
                prediction.peak_time_sec,
                (truth.start_sec + truth.end_sec) / 2.0,
                distance,
                prediction.peak_score,
                prediction.predicted_ecotype,
                truth.ecotype,
                prediction.provider,
                prediction.dataset,
            )
        )
        used_predictions.add(pred_index)
        used_truths.add(truth_index)

    for pred_index, prediction in enumerate(predictions):
        if pred_index in used_predictions:
            continue
        ignored_by = next(
            (
                interval
                for interval in ignore_intervals
                if interval.start_sec - ignore_collar_sec
                <= prediction.peak_time_sec
                <= interval.end_sec + ignore_collar_sec
            ),
            None,
        )
        status = "IGNORED" if ignored_by else "FP"
        matches.append(
            ContinuousMatch(
                prediction.soundfile,
                status,
                prediction.event_id,
                "",
                prediction.peak_time_sec,
                None,
                None,
                prediction.peak_score,
                prediction.predicted_ecotype,
                "",
                prediction.provider,
                prediction.dataset,
                ignored_by.annotation_id if ignored_by else "",
                ignored_by.reason if ignored_by else "",
            )
        )
    for truth_index, truth in enumerate(truths):
        if truth_index in used_truths:
            continue
        matches.append(
            ContinuousMatch(
                truth.soundfile,
                "FN",
                "",
                truth.event_id,
                None,
                (truth.start_sec + truth.end_sec) / 2.0,
                None,
                None,
                "",
                truth.ecotype,
                truth.provider,
                truth.dataset,
            )
        )
    return matches


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def detection_metrics(
    matches: list[ContinuousMatch], audio_hours: float
) -> dict[str, Any]:
    counts = Counter(match.status for match in matches)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    distances = np.asarray(
        [
            match.absolute_distance_sec
            for match in matches
            if match.status == "TP" and match.absolute_distance_sec is not None
        ],
        dtype=np.float64,
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "ignored_predictions": counts["IGNORED"],
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "false_positives_per_hour": safe_divide(fp, audio_hours),
        "median_absolute_timing_error_sec": (
            float(np.median(distances)) if len(distances) else None
        ),
    }


def ecotype_metrics(matches: list[ContinuousMatch]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    pairs = [
        (match.true_ecotype, match.predicted_ecotype)
        for match in matches
        if match.status == "TP" and match.true_ecotype in ECOTYPES
    ]
    columns = (*ECOTYPES, "unknown")
    matrix = np.zeros((len(ECOTYPES), len(columns)), dtype=np.int64)
    for actual, predicted in pairs:
        column = columns.index(predicted) if predicted in columns else len(ECOTYPES)
        matrix[ECOTYPES.index(actual), column] += 1
    rows = []
    f1_values = []
    for index, label in enumerate(ECOTYPES):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
        support = tp + fn
        if support and f1 is not None:
            f1_values.append(f1)
        rows.append(
            {
                "ecotype": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": f1,
            }
        )
    correct = int(sum(matrix[index, index] for index in range(len(ECOTYPES))))
    return (
        {
            "evaluated": len(pairs),
            "accuracy": safe_divide(correct, len(pairs)),
            "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        },
        rows,
        matrix,
    )


def evaluate_threshold(
    recordings: list[Any],
    smoothed_by_file: dict[str, np.ndarray],
    truths_by_file: dict[str, list[Any]],
    ignore_by_file: dict[str, list[IgnoreInterval]],
    score_source: str,
    moving_average_windows: int,
    threshold: float,
    collar_sec: float,
    ignore_collar_sec: float,
    ecotype_top_k: int,
    ecotype_threshold: float,
    audio_hours: float,
) -> tuple[dict[str, Any], list[ContinuousEvent], list[ContinuousMatch]]:
    events = []
    matches = []
    for cached in recordings:
        soundfile = cached.recording.soundfile
        file_events = form_continuous_events(
            cached,
            smoothed_by_file[soundfile],
            score_source,
            moving_average_windows,
            threshold,
            ecotype_top_k,
            ecotype_threshold,
        )
        file_matches = match_events(
            file_events,
            truths_by_file.get(soundfile, []),
            ignore_by_file.get(soundfile, []),
            collar_sec,
            ignore_collar_sec,
        )
        events.extend(file_events)
        matches.extend(file_matches)
    detection = detection_metrics(matches, audio_hours)
    ecotype, _, _ = ecotype_metrics(matches)
    return (
        {
            "score_source": score_source,
            "moving_average_windows": moving_average_windows,
            "threshold": threshold,
            "ground_truth_events": detection["true_positives"] + detection["false_negatives"],
            "predicted_events": detection["true_positives"] + detection["false_positives"],
            **detection,
            "ecotype_evaluated": ecotype["evaluated"],
            "ecotype_accuracy": ecotype["accuracy"],
            "ecotype_macro_f1": ecotype["macro_f1"],
        },
        events,
        matches,
    )


def precision_recall_auc(rows: list[dict[str, Any]]) -> Optional[float]:
    by_recall: dict[float, float] = {0.0: 1.0}
    for row in rows:
        recall, precision = row.get("recall"), row.get("precision")
        if recall is None or precision is None:
            continue
        by_recall[float(recall)] = max(float(precision), by_recall.get(float(recall), 0.0))
    if len(by_recall) < 2:
        return None
    recalls = np.asarray(sorted(by_recall), dtype=np.float64)
    precisions = np.asarray([by_recall[value] for value in recalls], dtype=np.float64)
    # A monotone precision envelope makes the area independent of small
    # threshold-to-threshold reversals in event matching.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(precisions, recalls))


def select_operating_point(
    rows: list[dict[str, Any]], fp_budget: float
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["false_positives_per_hour"] is not None
        and row["false_positives_per_hour"] <= fp_budget
    ]
    constraint_met = bool(eligible)
    if not eligible:
        eligible = rows
    if constraint_met:
        selected = max(
            eligible,
            key=lambda row: (
                -1.0 if row["recall"] is None else row["recall"],
                -1.0 if row["precision"] is None else row["precision"],
                -row["false_positives_per_hour"],
                -row["threshold"],
            ),
        )
    else:
        selected = min(
            eligible,
            key=lambda row: (
                math.inf
                if row["false_positives_per_hour"] is None
                else row["false_positives_per_hour"],
                -(-1.0 if row["recall"] is None else row["recall"]),
            ),
        )
    return {
        "fp_per_hour_budget": fp_budget,
        "budget_constraint_met": constraint_met,
        **selected,
    }


def write_rows(
    path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None
) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ecotype_matrix(path: Path, matrix: np.ndarray) -> None:
    columns = (*ECOTYPES, "unknown")
    write_rows(
        path,
        [
            {
                "actual_ecotype": actual,
                **{
                    label: int(matrix[row, column])
                    for column, label in enumerate(columns)
                },
            }
            for row, actual in enumerate(ECOTYPES)
        ],
        ["actual_ecotype", *columns],
    )


def grouped_metrics(
    recordings: list[Any], matches: list[ContinuousMatch], attribute: str
) -> list[dict[str, Any]]:
    hours: dict[str, float] = defaultdict(float)
    for cached in recordings:
        key = getattr(cached.recording, attribute) or "unknown"
        hours[key] += cached.recording.duration_sec / 3600.0
    rows = []
    for key in sorted(hours):
        selected = [match for match in matches if (getattr(match, attribute) or "unknown") == key]
        detection = detection_metrics(selected, hours[key])
        ecotype, _, _ = ecotype_metrics(selected)
        rows.append(
            {
                attribute: key,
                "audio_hours": hours[key],
                **detection,
                "ecotype_evaluated": ecotype["evaluated"],
                "ecotype_accuracy": ecotype["accuracy"],
                "ecotype_macro_f1": ecotype["macro_f1"],
            }
        )
    return rows


def recording_metrics(
    recordings: list[Any], matches: list[ContinuousMatch]
) -> list[dict[str, Any]]:
    by_file: dict[str, list[ContinuousMatch]] = defaultdict(list)
    for match in matches:
        by_file[match.soundfile].append(match)
    rows = []
    for cached in recordings:
        recording = cached.recording
        hours = recording.duration_sec / 3600.0
        rows.append(
            {
                "soundfile": recording.soundfile,
                "provider": recording.provider,
                "dataset": recording.dataset,
                "audio_hours": hours,
                **detection_metrics(by_file.get(recording.soundfile, []), hours),
            }
        )
    return rows


def bootstrap_intervals(
    recordings: list[Any],
    matches: list[ContinuousMatch],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    if replicates < 1:
        return []
    by_file: dict[str, list[ContinuousMatch]] = defaultdict(list)
    hours_by_file = {}
    for cached in recordings:
        soundfile = cached.recording.soundfile
        hours_by_file[soundfile] = cached.recording.duration_sec / 3600.0
    for match in matches:
        by_file[match.soundfile].append(match)
    names = list(hours_by_file)
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = [rng.choice(names) for _ in names]
        sampled_matches = [match for name in sampled for match in by_file.get(name, [])]
        sampled_hours = sum(hours_by_file[name] for name in sampled)
        metrics = detection_metrics(sampled_matches, sampled_hours)
        for metric in ("precision", "recall", "f1", "false_positives_per_hour"):
            if metrics[metric] is not None:
                values[metric].append(float(metrics[metric]))
    point = detection_metrics(matches, sum(hours_by_file.values()))
    return [
        {
            "metric": metric,
            "estimate": point[metric],
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
            "bootstrap_replicates": len(samples),
        }
        for metric, samples in values.items()
        if samples
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-sources", default=",".join(SCORE_SOURCES))
    parser.add_argument("--moving-average-windows", default="1,2,3")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=1.00)
    parser.add_argument("--threshold-count", type=int, default=50)
    parser.add_argument("--fp-per-hour-budgets", default="5,20")
    parser.add_argument("--primary-fp-budget", type=float, default=20.0)
    parser.add_argument("--collar-sec", type=float, default=1.5)
    parser.add_argument("--ignore-collar-sec", type=float, default=None)
    parser.add_argument(
        "--missing-kw-confidence",
        choices=("confirmed", "ignore", "exclude"),
        default="confirmed",
    )
    parser.add_argument("--ecotype-top-k", type=int, default=2)
    parser.add_argument("--ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--locked-score-source", choices=SCORE_SOURCES, default=None)
    parser.add_argument("--locked-moving-average-windows", type=int, default=None)
    parser.add_argument("--locked-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.threshold_count < 2 and args.thresholds is None:
        raise ValueError("--threshold-count must be at least 2")
    if not 0 <= args.threshold_min <= args.threshold_max <= 1:
        raise ValueError("Threshold range must be within 0..1")
    if args.collar_sec < 0:
        raise ValueError("--collar-sec cannot be negative")
    ignore_collar_sec = (
        args.collar_sec if args.ignore_collar_sec is None else args.ignore_collar_sec
    )
    if ignore_collar_sec < 0:
        raise ValueError("--ignore-collar-sec cannot be negative")
    if args.ecotype_top_k < 1 or not 0 <= args.ecotype_threshold <= 1:
        raise ValueError("Ecotype top-k must be positive and threshold must be within 0..1")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates cannot be negative")

    score_sources = parse_score_sources(args.score_sources)
    moving_widths = parse_csv_list(
        args.moving_average_windows, int, "moving-average windows"
    )
    if min(moving_widths) < 1:
        raise ValueError("Moving-average widths must be positive")
    budgets = parse_csv_list(args.fp_per_hour_budgets, float, "FP/hour budgets")
    if min(budgets) < 0 or args.primary_fp_budget < 0:
        raise ValueError("FP/hour budgets cannot be negative")
    thresholds = (
        parse_csv_list(args.thresholds, float, "thresholds")
        if args.thresholds
        else np.linspace(args.threshold_min, args.threshold_max, args.threshold_count).tolist()
    )
    if any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("All thresholds must be within 0..1")
    locked_values = (
        args.locked_score_source,
        args.locked_moving_average_windows,
        args.locked_threshold,
    )
    if any(value is not None for value in locked_values) and not all(
        value is not None for value in locked_values
    ):
        raise ValueError("Specify all three --locked-* arguments or none")
    if args.locked_score_source and args.locked_score_source not in score_sources:
        score_sources.append(args.locked_score_source)
    if (
        args.locked_moving_average_windows
        and args.locked_moving_average_windows not in moving_widths
    ):
        moving_widths.append(args.locked_moving_average_windows)
    if args.locked_threshold is not None and args.locked_threshold not in thresholds:
        thresholds.append(args.locked_threshold)
    thresholds = sorted(set(float(value) for value in thresholds))

    window_path = Path(args.window_predictions)
    if not window_path.is_file():
        raise FileNotFoundError(f"Window predictions not found: {window_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import evaluate_dclde_long_recordings as base
    import evaluate_dclde_long_recordings_improved as diagnostic

    failed_path = (
        Path(args.failed_files) if args.failed_files else window_path.parent / "failed_files.csv"
    )
    recordings, cache_sanity = diagnostic.load_cached_recordings(
        window_path, diagnostic.load_failed_soundfiles(failed_path)
    )
    available_recordings = len(recordings)
    if args.max_files is not None and args.max_files < len(recordings):
        recordings = random.Random(args.seed).sample(recordings, args.max_files)
        cache_sanity["available_recordings_before_random_subset"] = available_recordings
        cache_sanity["random_subset_recordings"] = len(recordings)
        cache_sanity["random_subset_seed"] = args.seed
    if not recordings:
        raise ValueError("No cached recordings remain")
    annotation_table = base.read_csv(args.annotations)
    truths_by_file, ignore_by_file, annotation_sanity = load_confidence_annotations(
        base,
        annotation_table,
        [cached.recording for cached in recordings],
        args.missing_kw_confidence,
    )
    audio_hours = sum(cached.recording.duration_sec for cached in recordings) / 3600.0
    cached_windows = sum(len(cached.starts) for cached in recordings)

    print(f"Cached recordings:       {len(recordings):,}")
    print(f"Cached windows:          {cached_windows:,}")
    print(f"Audio hours:             {audio_hours:.3f}")
    print(f"Confirmed KW events:     {annotation_sanity.get('confirmed', 0):,}")
    print(f"KW ignore intervals:     {annotation_sanity.get('ignore', 0):,}")
    print(
        f"Curve configurations:    {len(score_sources) * len(moving_widths):,} "
        f"({len(thresholds)} thresholds each)"
    )

    results = []
    curve_summary = []
    operating_points = []
    smoothed_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for source in score_sources:
        raw_by_file = {
            cached.recording.soundfile: score_vector(cached, source)
            for cached in recordings
        }
        for width in moving_widths:
            smoothed = {
                soundfile: centered_moving_average(values, width)
                for soundfile, values in raw_by_file.items()
            }
            smoothed_cache[(source, width)] = smoothed
            curve_rows = []
            for threshold in thresholds:
                row, _, _ = evaluate_threshold(
                    recordings,
                    smoothed,
                    truths_by_file,
                    ignore_by_file,
                    source,
                    width,
                    threshold,
                    args.collar_sec,
                    ignore_collar_sec,
                    args.ecotype_top_k,
                    args.ecotype_threshold,
                    audio_hours,
                )
                results.append(row)
                curve_rows.append(row)
            auc = precision_recall_auc(curve_rows)
            curve_summary.append(
                {
                    "score_source": source,
                    "moving_average_windows": width,
                    "event_pr_auc": auc,
                    "threshold_count": len(curve_rows),
                }
            )
            for budget in budgets:
                operating_points.append(select_operating_point(curve_rows, budget))
            print(
                f"Completed source={source}, moving_average_windows={width}, "
                f"event PR AUC={auc}"
            )

    write_rows(output_dir / "continuous_threshold_results.csv", results)
    write_rows(output_dir / "curve_summary.csv", curve_summary)
    write_rows(output_dir / "operating_points.csv", operating_points)

    if args.locked_score_source:
        selected = next(
            row
            for row in results
            if row["score_source"] == args.locked_score_source
            and row["moving_average_windows"] == args.locked_moving_average_windows
            and math.isclose(row["threshold"], args.locked_threshold, abs_tol=1e-12)
        )
        selection_mode = "locked_configuration"
    else:
        primary_points = [
            select_operating_point(
                [
                    row
                    for row in results
                    if row["score_source"] == source
                    and row["moving_average_windows"] == width
                ],
                args.primary_fp_budget,
            )
            for source in score_sources
            for width in moving_widths
        ]
        selected = max(
            primary_points,
            key=lambda row: (
                -1.0 if row["recall"] is None else row["recall"],
                -1.0 if row["precision"] is None else row["precision"],
                -row["false_positives_per_hour"],
                -row["threshold"],
            ),
        )
        selection_mode = f"maximum recall at <= {args.primary_fp_budget:g} FP/hour"

    selected_row, selected_events, selected_matches = evaluate_threshold(
        recordings,
        smoothed_cache[(selected["score_source"], selected["moving_average_windows"])],
        truths_by_file,
        ignore_by_file,
        selected["score_source"],
        selected["moving_average_windows"],
        selected["threshold"],
        args.collar_sec,
        ignore_collar_sec,
        args.ecotype_top_k,
        args.ecotype_threshold,
        audio_hours,
    )
    ecotype_overall, ecotype_rows, ecotype_matrix = ecotype_metrics(selected_matches)
    write_rows(
        output_dir / "selected_continuous_events.csv",
        [asdict(event) for event in selected_events],
        list(ContinuousEvent.__dataclass_fields__),
    )
    write_rows(
        output_dir / "selected_event_matches.csv",
        [asdict(match) for match in selected_matches],
        list(ContinuousMatch.__dataclass_fields__),
    )
    write_rows(output_dir / "selected_ecotype_metrics.csv", ecotype_rows)
    write_ecotype_matrix(
        output_dir / "selected_ecotype_confusion_matrix.csv", ecotype_matrix
    )
    write_rows(
        output_dir / "selected_provider_metrics.csv",
        grouped_metrics(recordings, selected_matches, "provider"),
    )
    write_rows(
        output_dir / "selected_dataset_metrics.csv",
        grouped_metrics(recordings, selected_matches, "dataset"),
    )
    write_rows(
        output_dir / "selected_recording_metrics.csv",
        recording_metrics(recordings, selected_matches),
    )
    confidence_intervals = bootstrap_intervals(
        recordings, selected_matches, args.bootstrap_replicates, args.seed
    )
    write_rows(output_dir / "selected_bootstrap_confidence_intervals.csv", confidence_intervals)

    summary = {
        "method": (
            "Centered moving average, single-threshold continuous event formation, "
            "one-to-one collar matching, and low-confidence annotation ignore regions"
        ),
        "selection_mode": selection_mode,
        "selected_configuration": selected_row,
        "selected_ecotype_metrics": ecotype_overall,
        "audio_hours": audio_hours,
        "recordings": len(recordings),
        "cached_windows": cached_windows,
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "curve_summary": curve_summary,
        "arguments": vars(args),
    }
    (output_dir / "paper_style_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nSelected paper-style operating point")
    print("=" * 72)
    print(f"Selection:                     {selection_mode}")
    print(f"Score source:                  {selected_row['score_source']}")
    print(f"Moving-average windows:        {selected_row['moving_average_windows']}")
    print(f"Threshold:                     {selected_row['threshold']}")
    print(f"Event precision:               {selected_row['precision']}")
    print(f"Event recall:                  {selected_row['recall']}")
    print(f"Event F1:                      {selected_row['f1']}")
    print(f"False positives/hour:          {selected_row['false_positives_per_hour']}")
    print(f"Ignored predictions:           {selected_row['ignored_predictions']}")
    print(f"Ecotype accuracy:              {selected_row['ecotype_accuracy']}")
    print(f"Ecotype macro F1:              {selected_row['ecotype_macro_f1']}")
    print("\nRecall at operational FP/hour budgets")
    for budget in budgets:
        eligible = [
            row for row in operating_points if row["fp_per_hour_budget"] == budget
        ]
        best = max(
            eligible,
            key=lambda row: (
                -1.0 if row["recall"] is None else row["recall"],
                -1.0 if row["precision"] is None else row["precision"],
            ),
        )
        print(
            f"  <= {budget:g} FP/h: recall={best['recall']}, "
            f"actual FP/h={best['false_positives_per_hour']}, "
            f"source={best['score_source']}, MA={best['moving_average_windows']}, "
            f"threshold={best['threshold']}"
        )
    print(f"\nReports saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
