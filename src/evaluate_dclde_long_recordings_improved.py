#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Diagnose long-recording performance from cached 3-second predictions.

This script does not rerun the neural network. It reads window_predictions.csv
from evaluate_dclde_long_recordings.py and evaluates the same probabilities in
three complementary ways:

1. exact non-overlapping 60-second top-k aggregation;
2. 3-second window-overlap detection with average precision;
3. local-peak call detection with one-to-one temporal-collar matching.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

import evaluate_dclde_long_recordings as base
from multispecies_train_model import ECOTYPE_LABELS


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_OUTPUT = "/kaggle/working/dclde_long_improved_evaluation"
SPECIES = ("background", "KW", "HW", "AB")
ECOTYPES = ("SRKW", "NRKW", "TKW", "OKW", "SAR")
ALL_CLASSES = ("background", "HW", "AB", *ECOTYPES)
LEGACY_CLASSES = ("humpback", "resident", "transient", "other/background")
SCORE_SOURCES = ("binary_kw", "species_kw", "max_ecotype_composite")


@dataclass(frozen=True)
class CachedRecording:
    recording: base.Recording
    starts: np.ndarray
    ends: np.ndarray
    kw_probabilities: np.ndarray
    species_probabilities: np.ndarray
    ecotype_probabilities: np.ndarray


@dataclass(frozen=True)
class PeakPrediction:
    soundfile: str
    peak_id: str
    time_sec: float
    window_start_sec: float
    window_end_sec: float
    score: float
    predicted_ecotype: str
    ecotype_confidence: float
    provider: str
    dataset: str


@dataclass(frozen=True)
class CollarMatch:
    soundfile: str
    status: str
    peak_id: str
    true_event_id: str
    peak_time_sec: Optional[float]
    true_center_sec: Optional[float]
    absolute_distance_sec: Optional[float]
    score: Optional[float]
    predicted_ecotype: str
    true_ecotype: str
    provider: str
    dataset: str


def parse_number_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        values = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return sorted(set(values))


def load_failed_soundfiles(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            (row.get("Soundfile") or "").strip()
            for row in csv.DictReader(file)
            if (row.get("Soundfile") or "").strip()
        }


def load_cached_recordings(
    path: Path,
    failed: set[str],
) -> tuple[list[CachedRecording], dict[str, Any]]:
    probability_columns = [
        *[f"species_probability_{label}" for label in SPECIES],
        *[f"ecotype_probability_{label}" for label in ECOTYPES],
    ]
    required = {
        "Soundfile", "window_start_sec", "window_end_sec", "Provider", "Dataset",
        "kw_probability", *probability_columns,
    }
    builders: dict[str, dict[str, Any]] = {}
    rows_read = 0
    rows_excluded = 0
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            soundfile = (row.get("Soundfile") or "").strip()
            if not soundfile:
                raise ValueError(f"Empty Soundfile at row {row_number}")
            if soundfile in failed:
                rows_excluded += 1
                continue
            try:
                start = float(row["window_start_sec"])
                end = float(row["window_end_sec"])
                kw = float(row["kw_probability"])
                species = [float(row[f"species_probability_{label}"]) for label in SPECIES]
                ecotypes = [float(row[f"ecotype_probability_{label}"]) for label in ECOTYPES]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value at row {row_number}") from error
            builder = builders.setdefault(
                soundfile,
                {
                    "provider": (row.get("Provider") or "").strip(),
                    "dataset": (row.get("Dataset") or "").strip(),
                    "starts": [], "ends": [], "kw": [], "species": [], "ecotype": [],
                },
            )
            builder["starts"].append(start)
            builder["ends"].append(end)
            builder["kw"].append(kw)
            builder["species"].append(species)
            builder["ecotype"].append(ecotypes)
            rows_read += 1

    recordings = []
    for soundfile, builder in builders.items():
        starts = np.asarray(builder["starts"], dtype=np.float64)
        order = np.argsort(starts, kind="stable")
        starts = starts[order]
        if len(starts) != len(np.unique(starts)):
            raise ValueError(f"Duplicate window starts found for {soundfile}")
        ends = np.asarray(builder["ends"], dtype=np.float64)[order]
        recording = base.Recording(
            soundfile,
            str(path),
            builder["provider"],
            builder["dataset"],
            float(np.max(ends)),
        )
        recordings.append(
            CachedRecording(
                recording,
                starts,
                ends,
                np.asarray(builder["kw"], dtype=np.float32)[order],
                np.asarray(builder["species"], dtype=np.float32)[order],
                np.asarray(builder["ecotype"], dtype=np.float32)[order],
            )
        )
    return recordings, {
        "window_rows": rows_read,
        "recordings": len(recordings),
        "failed_soundfiles_excluded": len(failed),
        "failed_window_rows_excluded": rows_excluded,
    }


def subset_recordings(
    recordings: list[CachedRecording], max_files: Optional[int], seed: int
) -> list[CachedRecording]:
    if max_files is None or max_files >= len(recordings):
        return recordings
    return random.Random(seed).sample(recordings, max_files)


def smooth_rows(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 3:
        return scores.copy()
    result = scores.copy()
    result[1:-1] = (scores[:-2] + scores[1:-1]) / 2.0
    return result


def comparison_score_matrix(cached: CachedRecording) -> dict[str, np.ndarray]:
    species = {label: cached.species_probabilities[:, index] for index, label in enumerate(SPECIES)}
    ecotype = {label: cached.ecotype_probabilities[:, index] for index, label in enumerate(ECOTYPES)}
    result = {
        "HW": species["HW"],
        "AB": species["AB"],
    }
    result.update({label: species["KW"] * ecotype[label] for label in ECOTYPES})
    return result


def topk_aggregate(
    score_rows: dict[str, np.ndarray],
    indices: np.ndarray,
    thresholds: dict[str, float],
    minimum_windows: dict[str, int],
    top_k: int,
    smoothing: bool,
) -> tuple[str, float, dict[str, int]]:
    candidates: dict[str, tuple[float, int]] = {}
    positive_counts = {}
    for label, all_scores in score_rows.items():
        values = all_scores[indices]
        if smoothing:
            values = smooth_rows(values)
        positive = values[values >= thresholds[label]]
        positive_counts[label] = int(len(positive))
        if len(positive) < minimum_windows[label]:
            continue
        top_values = np.sort(positive)[-top_k:]
        candidates[label] = (float(np.mean(top_values)), int(len(positive)))
    if not candidates:
        return "background", 0.0, positive_counts
    label = max(candidates, key=lambda item: (candidates[item][0], candidates[item][1]))
    return label, candidates[label][0], positive_counts


def annotation_class(event: base.LabeledAnnotation) -> Optional[str]:
    if event.species == "KW":
        return event.ecotype if event.ecotype in ECOTYPE_LABELS else None
    if event.species in {"HW", "AB"}:
        return event.species
    return None


def block_ground_truth(
    block_start: float,
    block_end: float,
    annotations: list[base.LabeledAnnotation],
) -> tuple[str, bool, str]:
    labels = sorted(
        {
            label
            for event in annotations
            if base.interval_overlap(block_start, block_end, event.start_sec, event.end_sec) > 0
            for label in [annotation_class(event)]
            if label is not None
        }
    )
    if not labels:
        return "background", False, ""
    if len(labels) == 1:
        return labels[0], False, labels[0]
    return labels[0], True, ";".join(labels)


def legacy_label(label: str) -> str:
    return {
        "HW": "humpback",
        "SRKW": "resident",
        "TKW": "transient",
    }.get(label, "other/background")


def evaluate_60_second_blocks(
    recordings: list[CachedRecording],
    annotations_by_file: dict[str, list[base.LabeledAnnotation]],
    block_sec: float,
    thresholds: dict[str, float],
    minimum_windows: dict[str, int],
    top_k: int,
    smoothing: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]], dict[str, Any]]:
    rows = []
    pairs = []
    legacy_pairs = []
    ambiguous = 0
    skipped_partial = 0
    skipped_wrong_window_count = 0
    expected_windows = int(math.floor((block_sec - 3.0) / 2.0)) + 1
    if expected_windows < 1:
        raise ValueError("Block duration must be at least 3 seconds")
    for cached in recordings:
        duration = cached.recording.duration_sec
        number_of_blocks = int(math.floor(duration / block_sec))
        skipped_partial += int(duration - number_of_blocks * block_sec > 1e-6)
        score_rows = comparison_score_matrix(cached)
        for block_index in range(number_of_blocks):
            start = block_index * block_sec
            end = start + block_sec
            relative = cached.starts - start
            indices = np.flatnonzero(
                (cached.starts >= start - 1e-6)
                & (cached.ends <= end + 1e-6)
                & np.isclose(np.mod(relative, 2.0), 0.0, atol=1e-4)
            )
            if not len(indices):
                continue
            if len(indices) != expected_windows:
                skipped_wrong_window_count += 1
                continue
            predicted, confidence, positive_counts = topk_aggregate(
                score_rows, indices, thresholds, minimum_windows, top_k, smoothing
            )
            actual, is_ambiguous, overlapping_labels = block_ground_truth(
                start, end, annotations_by_file.get(cached.recording.soundfile, [])
            )
            if is_ambiguous:
                ambiguous += 1
            else:
                pairs.append((actual, predicted))
                if actual in {"background", "HW", "SRKW", "TKW"}:
                    legacy_pairs.append((legacy_label(actual), legacy_label(predicted)))
            row = {
                "Soundfile": cached.recording.soundfile,
                "Provider": cached.recording.provider,
                "Dataset": cached.recording.dataset,
                "block_index": block_index,
                "block_start_sec": start,
                "block_end_sec": end,
                "number_of_windows": len(indices),
                "actual_label": actual,
                "predicted_label": predicted,
                "confidence": confidence,
                "ambiguous_ground_truth": is_ambiguous,
                "overlapping_labels": overlapping_labels,
                "correct": (not is_ambiguous and actual == predicted),
            }
            row.update({f"positive_windows_{label}": positive_counts[label] for label in score_rows})
            rows.append(row)
    return rows, pairs, legacy_pairs, {
        "ambiguous_blocks_excluded": ambiguous,
        "recordings_with_partial_tail_skipped": skipped_partial,
        "expected_windows_per_block": expected_windows,
        "blocks_skipped_wrong_window_count": skipped_wrong_window_count,
    }


def binary_counts(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = base.safe_divide(tp, tp + fp)
    recall = base.safe_divide(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / f1_denominator if f1_denominator else None,
        "accuracy": base.safe_divide(tp + tn, tp + fp + fn + tn),
    }


def average_precision(actual: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(np.sum(actual))
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked_actual = actual[order].astype(np.int64)
    cumulative_tp = np.cumsum(ranked_actual)
    ranks = np.arange(1, len(actual) + 1)
    return float(np.sum((cumulative_tp / ranks) * ranked_actual) / positives)


def score_vector(cached: CachedRecording, source: str) -> np.ndarray:
    if source == "binary_kw":
        return cached.kw_probabilities
    species_kw = cached.species_probabilities[:, SPECIES.index("KW")]
    if source == "species_kw":
        return species_kw
    if source == "max_ecotype_composite":
        return species_kw * np.max(cached.ecotype_probabilities, axis=1)
    raise ValueError(f"Unknown score source: {source}")


def window_ground_truth(
    cached: CachedRecording,
    kw_truths: list[base.GroundTruthEvent],
) -> np.ndarray:
    return np.asarray(
        [base.window_has_kw(start, end, kw_truths) for start, end in zip(cached.starts, cached.ends)],
        dtype=bool,
    )


def evaluate_windows(
    recordings: list[CachedRecording],
    kw_truths_by_file: dict[str, list[base.GroundTruthEvent]],
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_parts = []
    score_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for cached in recordings:
        actual_parts.append(window_ground_truth(cached, kw_truths_by_file.get(cached.recording.soundfile, [])))
        for source in SCORE_SOURCES:
            score_parts[source].append(score_vector(cached, source))
    actual = np.concatenate(actual_parts)
    rows = []
    summaries = {}
    for source in SCORE_SOURCES:
        scores = np.concatenate(score_parts[source])
        ap = average_precision(actual, scores)
        source_rows = []
        for threshold in thresholds:
            row = {
                "score_source": source,
                "threshold": threshold,
                **binary_counts(actual, scores >= threshold),
                "average_precision": ap,
            }
            rows.append(row)
            source_rows.append(row)
        best = max(source_rows, key=lambda row: (-1 if row["f1"] is None else row["f1"], row["recall"]))
        summaries[source] = {"average_precision": ap, "best_threshold_result": best}
    return rows, summaries


def find_peaks(
    cached: CachedRecording,
    source: str,
    threshold: float,
    minimum_distance_sec: float,
) -> list[PeakPrediction]:
    scores = score_vector(cached, source)
    if not len(scores):
        return []
    candidate_indices = [
        index
        for index, score in enumerate(scores)
        if score >= threshold
        and (index == 0 or score >= scores[index - 1])
        and (index == len(scores) - 1 or score >= scores[index + 1])
    ]
    selected = []
    centers = (cached.starts + cached.ends) / 2.0
    for index in sorted(candidate_indices, key=lambda item: float(scores[item]), reverse=True):
        if all(abs(float(centers[index] - centers[other])) >= minimum_distance_sec for other in selected):
            selected.append(index)
    selected.sort(key=lambda index: float(centers[index]))
    predictions = []
    for peak_number, index in enumerate(selected):
        ecotype_index = int(np.argmax(cached.ecotype_probabilities[index]))
        predictions.append(
            PeakPrediction(
                cached.recording.soundfile,
                f"peak_{cached.recording.soundfile}_{peak_number}",
                float(centers[index]),
                float(cached.starts[index]),
                float(cached.ends[index]),
                float(scores[index]),
                ECOTYPES[ecotype_index],
                float(cached.ecotype_probabilities[index, ecotype_index]),
                cached.recording.provider,
                cached.recording.dataset,
            )
        )
    return predictions


def collar_match(
    predictions: list[PeakPrediction],
    truths: list[base.GroundTruthEvent],
    collar_sec: float,
) -> list[CollarMatch]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            center = (truth.start_sec + truth.end_sec) / 2.0
            distance = abs(prediction.time_sec - center)
            if distance <= collar_sec:
                candidates.append((distance, -prediction.score, pred_index, truth_index))
    candidates.sort()
    used_predictions = set()
    used_truths = set()
    matches = []
    for distance, _, pred_index, truth_index in candidates:
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        prediction = predictions[pred_index]
        truth = truths[truth_index]
        matches.append(
            CollarMatch(
                prediction.soundfile, "TP", prediction.peak_id, truth.event_id,
                prediction.time_sec, (truth.start_sec + truth.end_sec) / 2.0,
                distance, prediction.score, prediction.predicted_ecotype, truth.ecotype,
                prediction.provider, prediction.dataset,
            )
        )
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
    for index, prediction in enumerate(predictions):
        if index not in used_predictions:
            matches.append(
                CollarMatch(
                    prediction.soundfile, "FP", prediction.peak_id, "", prediction.time_sec,
                    None, None, prediction.score, prediction.predicted_ecotype, "",
                    prediction.provider, prediction.dataset,
                )
            )
    for index, truth in enumerate(truths):
        if index not in used_truths:
            matches.append(
                CollarMatch(
                    truth.soundfile, "FN", "", truth.event_id, None,
                    (truth.start_sec + truth.end_sec) / 2.0, None, None, "", truth.ecotype,
                    truth.provider, truth.dataset,
                )
            )
    return matches


def collar_metrics(matches: list[CollarMatch], audio_hours: float) -> dict[str, Any]:
    counts = Counter(match.status for match in matches)
    precision = base.safe_divide(counts["TP"], counts["TP"] + counts["FP"])
    recall = base.safe_divide(counts["TP"], counts["TP"] + counts["FN"])
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return {
        "true_positives": counts["TP"],
        "false_positives": counts["FP"],
        "false_negatives": counts["FN"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * counts["TP"] / denominator if denominator else None,
        "false_positives_per_hour": base.safe_divide(counts["FP"], audio_hours),
    }


def evaluate_peak_grid(
    recordings: list[CachedRecording],
    truths_by_file: dict[str, list[base.GroundTruthEvent]],
    thresholds: list[float],
    minimum_distances: list[float],
    collar_sec: float,
    audio_hours: float,
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], list[PeakPrediction], list[CollarMatch]]]]:
    rows = []
    best_by_source = {}
    for source, threshold, distance in itertools.product(SCORE_SOURCES, thresholds, minimum_distances):
        predictions = []
        matches = []
        for cached in recordings:
            file_predictions = find_peaks(cached, source, threshold, distance)
            file_matches = collar_match(
                file_predictions,
                truths_by_file.get(cached.recording.soundfile, []),
                collar_sec,
            )
            predictions.extend(file_predictions)
            matches.extend(file_matches)
        metrics = collar_metrics(matches, audio_hours)
        ecotype_pairs = [
            (match.true_ecotype, match.predicted_ecotype)
            for match in matches
            if match.status == "TP" and match.true_ecotype in ECOTYPE_LABELS
        ]
        _, _, ecotype = base.multiclass_metrics(ecotype_pairs, ECOTYPES)
        row = {
            "score_source": source,
            "threshold": threshold,
            "minimum_peak_distance_sec": distance,
            "collar_sec": collar_sec,
            **metrics,
            "ecotype_evaluated": ecotype["evaluated"],
            "ecotype_accuracy": ecotype["accuracy"],
            "ecotype_macro_f1": ecotype["macro_f1"],
        }
        rows.append(row)
        previous = best_by_source.get(source)
        rank = (-1 if row["f1"] is None else row["f1"], row["recall"], -row["false_positives_per_hour"])
        if previous is None or rank > previous[0]:
            best_by_source[source] = (rank, row, predictions, matches)
    return rows, {
        source: (value[1], value[2], value[3]) for source, value in best_by_source.items()
    }


def write_rows(path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_multiclass_reports(
    output_dir: Path,
    prefix: str,
    pairs: list[tuple[str, str]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    matrix, rows, overall = base.multiclass_metrics(pairs, labels)
    write_rows(output_dir / f"{prefix}_metrics.csv", rows)
    base.write_matrix(output_dir / f"{prefix}_confusion_matrix.csv", matrix, labels)
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=base.DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-sec", type=float, default=60.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--smoothing", action="store_true")
    parser.add_argument("--base-threshold", type=float, default=0.25)
    parser.add_argument("--humpback-threshold", type=float, default=0.475)
    parser.add_argument("--srkw-threshold", type=float, default=0.05)
    parser.add_argument("--tkw-threshold", type=float, default=0.20)
    parser.add_argument("--humpback-min-windows", type=int, default=2)
    parser.add_argument("--srkw-min-windows", type=int, default=2)
    parser.add_argument("--tkw-min-windows", type=int, default=3)
    parser.add_argument("--other-min-windows", type=int, default=3)
    parser.add_argument(
        "--window-thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument(
        "--peak-thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument("--peak-min-distances", default="1,2,3,5")
    parser.add_argument("--collar-sec", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window_path = Path(args.window_predictions)
    if not window_path.is_file():
        raise FileNotFoundError(f"Window predictions not found: {window_path}")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.block_sec <= 0 or args.top_k < 1 or args.collar_sec < 0:
        raise ValueError("Block size/top-k must be positive and collar cannot be negative")
    threshold_values = [
        args.base_threshold, args.humpback_threshold, args.srkw_threshold, args.tkw_threshold
    ]
    if any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("Class thresholds must be between 0 and 1")

    failed_path = Path(args.failed_files) if args.failed_files else window_path.parent / "failed_files.csv"
    recordings, cache_sanity = load_cached_recordings(
        window_path, load_failed_soundfiles(failed_path)
    )
    recordings = subset_recordings(recordings, args.max_files, args.seed)
    if not recordings:
        raise ValueError("No cached recordings remain")
    plain_recordings = [item.recording for item in recordings]
    annotation_table = base.read_csv(args.annotations)
    kw_truths_by_file, annotation_sanity = base.load_annotations(
        args.annotations, plain_recordings, annotation_table
    )
    labeled_by_file = base.load_all_labeled_annotations(annotation_table, plain_recordings)
    audio_hours = sum(item.recording.duration_sec for item in recordings) / 3600
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "HW": args.humpback_threshold,
        "AB": args.base_threshold,
        "SRKW": args.srkw_threshold,
        "NRKW": args.base_threshold,
        "TKW": args.tkw_threshold,
        "OKW": args.base_threshold,
        "SAR": args.base_threshold,
    }
    minimum_windows = {
        "HW": args.humpback_min_windows,
        "AB": args.other_min_windows,
        "SRKW": args.srkw_min_windows,
        "NRKW": args.other_min_windows,
        "TKW": args.tkw_min_windows,
        "OKW": args.other_min_windows,
        "SAR": args.other_min_windows,
    }

    block_rows, block_pairs, legacy_pairs, block_sanity = evaluate_60_second_blocks(
        recordings, labeled_by_file, args.block_sec, thresholds, minimum_windows,
        args.top_k, args.smoothing,
    )
    write_rows(output_dir / "blocks_60s_predictions.csv", block_rows)
    block_metrics = write_multiclass_reports(
        output_dir, "blocks_60s_all_classes", block_pairs, ALL_CLASSES
    )
    legacy_metrics = write_multiclass_reports(
        output_dir, "blocks_60s_legacy", legacy_pairs, LEGACY_CLASSES
    )

    window_thresholds = parse_number_list(args.window_thresholds, float, "window thresholds")
    window_rows, window_summary = evaluate_windows(
        recordings, kw_truths_by_file, window_thresholds
    )
    write_rows(output_dir / "window_score_sweep.csv", window_rows)

    peak_thresholds = parse_number_list(args.peak_thresholds, float, "peak thresholds")
    peak_distances = parse_number_list(
        args.peak_min_distances, float, "peak minimum distances"
    )
    peak_rows, peak_best = evaluate_peak_grid(
        recordings, kw_truths_by_file, peak_thresholds, peak_distances,
        args.collar_sec, audio_hours,
    )
    write_rows(output_dir / "peak_collar_grid.csv", peak_rows)
    peak_summary = {}
    for source, (best_row, predictions, matches) in peak_best.items():
        write_rows(
            output_dir / f"peak_{source}_best_predictions.csv",
            [asdict(item) for item in predictions],
            list(PeakPrediction.__dataclass_fields__),
        )
        write_rows(
            output_dir / f"peak_{source}_best_matches.csv",
            [asdict(item) for item in matches],
            list(CollarMatch.__dataclass_fields__),
        )
        peak_summary[source] = best_row

    summary = {
        "audio_hours": audio_hours,
        "selected_recordings": len(recordings),
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "block_sanity": block_sanity,
        "blocks_60s_all_classes": block_metrics,
        "blocks_60s_legacy": legacy_metrics,
        "window_detection": window_summary,
        "peak_collar_best": peak_summary,
        "settings": vars(args),
        "class_thresholds": thresholds,
        "class_minimum_windows": minimum_windows,
    }
    with (output_dir / "improved_evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nImproved long-recording diagnostic")
    print("=" * 72)
    print(f"Recordings:                {len(recordings):,}")
    print(f"Audio hours:               {audio_hours:.3f}")
    print(f"60s all-class accuracy:    {block_metrics['accuracy']}")
    print(f"60s all-class macro F1:    {block_metrics['macro_f1']}")
    print(f"60s legacy accuracy:       {legacy_metrics['accuracy']}")
    print(f"60s legacy macro F1:       {legacy_metrics['macro_f1']}")
    for source in SCORE_SOURCES:
        window_best = window_summary[source]["best_threshold_result"]
        peak = peak_summary[source]
        print(f"\n{source}:")
        print(f"  Window AP:               {window_summary[source]['average_precision']}")
        print(f"  Best window F1:          {window_best['f1']} @ {window_best['threshold']}")
        print(f"  Best peak/collar F1:     {peak['f1']} @ {peak['threshold']}")
        print(f"  Peak/collar recall:      {peak['recall']}")
        print(f"  Peak/collar FP/hour:     {peak['false_positives_per_hour']}")
    print(f"\nReports saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
