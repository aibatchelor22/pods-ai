#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Tune stable KW events from cached long-recording window probabilities.

This script does not rerun the neural network. It adds peak-centered temporal
post-processing to ``window_predictions.csv``: a high start threshold, local
support, nonmaximum peak suppression, lower continuation threshold, short gap
tolerance, top-k event confidence, and event-level ecotype aggregation.
Predictions are matched one-to-one to annotation centers using a time collar.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_ANNOTATIONS = (
    "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/"
    "dclde_2027_killer_whales/Annotations.csv"
)
DEFAULT_OUTPUT = "/kaggle/working/dclde_long_stable_events"
SCORE_SOURCES = ("binary_kw", "species_kw", "max_ecotype_composite")
SPECIES = ("background", "KW", "HW", "AB")
ECOTYPES = ("SRKW", "NRKW", "TKW", "OKW", "SAR")


@dataclass(frozen=True)
class EventConfig:
    run: int
    score_source: str
    start_threshold: float
    support_threshold: float
    continuation_threshold: float
    minimum_support_windows: int
    support_radius_windows: int
    maximum_gap_windows: int
    peak_suppression_windows: int
    event_top_k: int
    event_score_threshold: float


@dataclass(frozen=True)
class StableEvent:
    soundfile: str
    event_id: str
    score_source: str
    peak_window_index: int
    peak_time_sec: float
    start_sec: float
    end_sec: float
    peak_score: float
    event_topk_mean: float
    supporting_windows: int
    envelope_windows: int
    predicted_ecotype: str
    ecotype_confidence: float
    provider: str
    dataset: str


@dataclass(frozen=True)
class StableMatch:
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


def parse_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        values = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return sorted(set(values))


def parse_score_sources(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(SCORE_SOURCES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown score sources: {unknown}")
    if not values:
        raise argparse.ArgumentTypeError("Score sources cannot be empty")
    return list(dict.fromkeys(values))


def build_grid(args: argparse.Namespace) -> list[EventConfig]:
    dimensions = (
        parse_score_sources(args.score_sources),
        parse_list(args.start_thresholds, float, "start thresholds"),
        parse_list(args.support_thresholds, float, "support thresholds"),
        parse_list(args.continuation_thresholds, float, "continuation thresholds"),
        parse_list(args.minimum_support_windows, int, "minimum support windows"),
        parse_list(args.support_radius_windows, int, "support radii"),
        parse_list(args.maximum_gap_windows, int, "maximum gaps"),
        parse_list(args.peak_suppression_windows, int, "peak suppression windows"),
        parse_list(args.event_top_k_values, int, "event top-k values"),
        parse_list(args.event_score_thresholds, float, "event score thresholds"),
    )
    configs = []
    for values in itertools.product(*dimensions):
        (
            source, start, support, continuation, minimum_support, radius,
            maximum_gap, suppression, top_k, event_threshold,
        ) = values
        if not continuation <= support <= start:
            continue
        if minimum_support > 2 * radius + 1:
            continue
        configs.append(
            EventConfig(
                len(configs) + 1,
                source,
                start,
                support,
                continuation,
                minimum_support,
                radius,
                maximum_gap,
                suppression,
                top_k,
                event_threshold,
            )
        )
    return configs


def score_vector(cached: Any, source: str) -> np.ndarray:
    if source == "binary_kw":
        return cached.kw_probabilities
    species_kw = cached.species_probabilities[:, SPECIES.index("KW")]
    if source == "species_kw":
        return species_kw
    if source == "max_ecotype_composite":
        return species_kw * np.max(cached.ecotype_probabilities, axis=1)
    raise ValueError(f"Unknown score source: {source}")


def local_peak_indices(scores: np.ndarray, threshold: float) -> list[int]:
    return [
        index
        for index, score in enumerate(scores)
        if score >= threshold
        and (index == 0 or score >= scores[index - 1])
        and (index == len(scores) - 1 or score >= scores[index + 1])
    ]


def supported_peaks(
    scores: np.ndarray,
    start_threshold: float,
    support_threshold: float,
    minimum_support_windows: int,
    support_radius_windows: int,
    peak_suppression_windows: int,
) -> list[int]:
    candidates = []
    for index in local_peak_indices(scores, start_threshold):
        left = max(0, index - support_radius_windows)
        right = min(len(scores), index + support_radius_windows + 1)
        if int(np.sum(scores[left:right] >= support_threshold)) >= minimum_support_windows:
            candidates.append(index)
    selected = []
    for index in sorted(candidates, key=lambda item: (-float(scores[item]), item)):
        if all(abs(index - other) > peak_suppression_windows for other in selected):
            selected.append(index)
    return sorted(selected)


def extend_from_peak(
    scores: np.ndarray,
    peak_index: int,
    continuation_threshold: float,
    maximum_gap_windows: int,
) -> tuple[int, int]:
    left = peak_index
    gap = 0
    for index in range(peak_index - 1, -1, -1):
        if scores[index] >= continuation_threshold:
            left = index
            gap = 0
        else:
            gap += 1
            if gap > maximum_gap_windows:
                break
    right = peak_index
    gap = 0
    for index in range(peak_index + 1, len(scores)):
        if scores[index] >= continuation_threshold:
            right = index
            gap = 0
        else:
            gap += 1
            if gap > maximum_gap_windows:
                break
    return left, right


def form_stable_events(
    cached: Any,
    config: EventConfig,
    ecotype_threshold: float = 0.0,
) -> list[StableEvent]:
    scores = np.asarray(score_vector(cached, config.score_source), dtype=np.float64)
    if not len(scores):
        return []
    peaks = supported_peaks(
        scores,
        config.start_threshold,
        config.support_threshold,
        config.minimum_support_windows,
        config.support_radius_windows,
        config.peak_suppression_windows,
    )
    events = []
    for peak_index in peaks:
        left, right = extend_from_peak(
            scores,
            peak_index,
            config.continuation_threshold,
            config.maximum_gap_windows,
        )
        envelope = np.arange(left, right + 1)
        support_indices = envelope[scores[envelope] >= config.support_threshold]
        if len(support_indices) < config.minimum_support_windows:
            continue
        top_indices = support_indices[
            np.argsort(scores[support_indices], kind="stable")[-config.event_top_k:]
        ]
        event_score = float(np.mean(scores[top_indices]))
        if event_score < config.event_score_threshold:
            continue
        ecotype_mean = np.mean(cached.ecotype_probabilities[top_indices], axis=0)
        ecotype_index = int(np.argmax(ecotype_mean))
        ecotype_confidence = float(ecotype_mean[ecotype_index])
        predicted_ecotype = (
            ECOTYPES[ecotype_index]
            if ecotype_confidence >= ecotype_threshold
            else "unknown"
        )
        events.append(
            StableEvent(
                soundfile=cached.recording.soundfile,
                event_id=f"stable_{cached.recording.soundfile}_{peak_index}",
                score_source=config.score_source,
                peak_window_index=peak_index,
                peak_time_sec=float((cached.starts[peak_index] + cached.ends[peak_index]) / 2.0),
                start_sec=float(cached.starts[left]),
                end_sec=float(cached.ends[right]),
                peak_score=float(scores[peak_index]),
                event_topk_mean=event_score,
                supporting_windows=int(len(support_indices)),
                envelope_windows=int(len(envelope)),
                predicted_ecotype=predicted_ecotype,
                ecotype_confidence=ecotype_confidence,
                provider=cached.recording.provider,
                dataset=cached.recording.dataset,
            )
        )
    # Hysteresis envelopes from nearby retained peaks can overlap. Split the
    # overlap at the peak midpoint so the exported events remain disjoint.
    for index in range(1, len(events)):
        previous = events[index - 1]
        current = events[index]
        if previous.end_sec <= current.start_sec:
            continue
        boundary = (previous.peak_time_sec + current.peak_time_sec) / 2.0
        events[index - 1] = replace(
            previous,
            end_sec=max(previous.start_sec, min(previous.end_sec, boundary)),
        )
        events[index] = replace(
            current,
            start_sec=min(current.end_sec, max(current.start_sec, boundary)),
        )
    return events


def collar_match(
    predictions: list[StableEvent], truths: list[Any], collar_sec: float
) -> list[StableMatch]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            center = (truth.start_sec + truth.end_sec) / 2.0
            distance = abs(prediction.peak_time_sec - center)
            if distance <= collar_sec:
                candidates.append(
                    (distance, -prediction.event_topk_mean, pred_index, truth_index)
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
            StableMatch(
                prediction.soundfile,
                "TP",
                prediction.event_id,
                truth.event_id,
                prediction.peak_time_sec,
                (truth.start_sec + truth.end_sec) / 2.0,
                distance,
                prediction.event_topk_mean,
                prediction.predicted_ecotype,
                truth.ecotype,
                prediction.provider,
                prediction.dataset,
            )
        )
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
    for index, prediction in enumerate(predictions):
        if index not in used_predictions:
            matches.append(
                StableMatch(
                    prediction.soundfile,
                    "FP",
                    prediction.event_id,
                    "",
                    prediction.peak_time_sec,
                    None,
                    None,
                    prediction.event_topk_mean,
                    prediction.predicted_ecotype,
                    "",
                    prediction.provider,
                    prediction.dataset,
                )
            )
    for index, truth in enumerate(truths):
        if index not in used_truths:
            matches.append(
                StableMatch(
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


def safe_divide(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def detection_metrics(matches: list[StableMatch], audio_hours: float) -> dict[str, Any]:
    counts = Counter(match.status for match in matches)
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    distances = np.asarray(
        [match.absolute_distance_sec for match in matches if match.status == "TP"],
        dtype=np.float64,
    )
    return {
        "true_positives": counts["TP"],
        "false_positives": counts["FP"],
        "false_negatives": counts["FN"],
        "precision": safe_divide(counts["TP"], counts["TP"] + counts["FP"]),
        "recall": safe_divide(counts["TP"], counts["TP"] + counts["FN"]),
        "f1": 2 * counts["TP"] / denominator if denominator else None,
        "false_positives_per_hour": safe_divide(counts["FP"], audio_hours),
        "mean_absolute_timing_error_sec": float(np.mean(distances)) if len(distances) else None,
        "median_absolute_timing_error_sec": (
            float(np.median(distances)) if len(distances) else None
        ),
        "p90_absolute_timing_error_sec": (
            float(np.percentile(distances, 90)) if len(distances) else None
        ),
    }


def ecotype_metrics(matches: list[StableMatch]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    pairs = [
        (match.true_ecotype, match.predicted_ecotype)
        for match in matches
        if match.status == "TP" and match.true_ecotype in ECOTYPES
    ]
    predicted_labels = (*ECOTYPES, "unknown")
    matrix = np.zeros((len(ECOTYPES), len(predicted_labels)), dtype=np.int64)
    for actual, predicted in pairs:
        column = predicted_labels.index(predicted) if predicted in predicted_labels else len(ECOTYPES)
        matrix[ECOTYPES.index(actual), column] += 1
    rows = []
    f1_values = []
    for index, label in enumerate(ECOTYPES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        if f1 is not None and tp + fn:
            f1_values.append(f1)
        rows.append(
            {
                "ecotype": label,
                "support": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    correct = sum(matrix[index, index] for index in range(len(ECOTYPES)))
    return (
        {
            "evaluated": len(pairs),
            "accuracy": safe_divide(correct, len(pairs)),
            "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        },
        rows,
        matrix,
    )


def evaluate_config(
    recordings: list[Any],
    truths_by_file: dict[str, list[Any]],
    config: EventConfig,
    collar_sec: float,
    audio_hours: float,
    ecotype_threshold: float,
) -> tuple[dict[str, Any], list[StableEvent], list[StableMatch]]:
    events = []
    matches = []
    for cached in recordings:
        file_events = form_stable_events(cached, config, ecotype_threshold)
        file_matches = collar_match(
            file_events,
            truths_by_file.get(cached.recording.soundfile, []),
            collar_sec,
        )
        events.extend(file_events)
        matches.extend(file_matches)
    detection = detection_metrics(matches, audio_hours)
    ecotype, _, _ = ecotype_metrics(matches)
    return (
        {
            **asdict(config),
            "collar_sec": collar_sec,
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


def rank_value(value: Optional[float], missing: float = -math.inf) -> float:
    return missing if value is None else float(value)


def select_best(
    rows: list[dict[str, Any]], maximum_fp_per_hour: Optional[float]
) -> tuple[dict[str, Any], bool]:
    eligible = rows
    constraint_met = True
    if maximum_fp_per_hour is not None:
        eligible = [
            row
            for row in rows
            if row["false_positives_per_hour"] is not None
            and row["false_positives_per_hour"] <= maximum_fp_per_hour
        ]
        if not eligible:
            eligible = rows
            constraint_met = False
    if constraint_met:
        return (
            max(
                eligible,
                key=lambda row: (
                    rank_value(row["f1"]),
                    rank_value(row["recall"]),
                    -rank_value(row["false_positives_per_hour"], math.inf),
                    rank_value(row["ecotype_macro_f1"]),
                ),
            ),
            True,
        )
    return (
        min(
            eligible,
            key=lambda row: (
                rank_value(row["false_positives_per_hour"], math.inf),
                -rank_value(row["recall"]),
                -rank_value(row["f1"]),
            ),
        ),
        False,
    )


def group_metrics(
    recordings: list[Any], matches: list[StableMatch], attribute: str
) -> list[dict[str, Any]]:
    hours = {}
    for cached in recordings:
        key = getattr(cached.recording, attribute) or "unknown"
        hours[key] = hours.get(key, 0.0) + cached.recording.duration_sec / 3600.0
    rows = []
    for group in sorted(hours):
        selected = [match for match in matches if getattr(match, attribute) == group]
        metrics = detection_metrics(selected, hours[group])
        ecotype, _, _ = ecotype_metrics(selected)
        rows.append(
            {
                attribute: group,
                "recording_hours": hours[group],
                **metrics,
                "ecotype_evaluated": ecotype["evaluated"],
                "ecotype_accuracy": ecotype["accuracy"],
                "ecotype_macro_f1": ecotype["macro_f1"],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ecotype_matrix(path: Path, matrix: np.ndarray) -> None:
    columns = (*ECOTYPES, "unknown")
    write_rows(
        path,
        [
            {
                "actual_ecotype": actual,
                **{label: int(matrix[row, column]) for column, label in enumerate(columns)},
            }
            for row, actual in enumerate(ECOTYPES)
        ],
        ["actual_ecotype", *columns],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-sources", default="max_ecotype_composite")
    parser.add_argument("--start-thresholds", default="0.80,0.85,0.90")
    parser.add_argument("--support-thresholds", default="0.55,0.65")
    parser.add_argument("--continuation-thresholds", default="0.35,0.45")
    parser.add_argument("--minimum-support-windows", default="2,3")
    parser.add_argument("--support-radius-windows", default="1,2")
    parser.add_argument("--maximum-gap-windows", default="0,1")
    parser.add_argument(
        "--peak-suppression-windows",
        default="1,2,3",
        help="Suppress weaker peaks within this many window indices of a stronger peak.",
    )
    parser.add_argument("--event-top-k-values", default="2")
    parser.add_argument("--event-score-thresholds", default="0.65,0.75")
    parser.add_argument("--ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--collar-sec", type=float, default=1.5)
    parser.add_argument("--max-fp-per-hour", type=float, default=None)
    parser.add_argument("--top-results", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.collar_sec < 0:
        raise ValueError("--collar-sec cannot be negative")
    if args.max_fp_per_hour is not None and args.max_fp_per_hour < 0:
        raise ValueError("--max-fp-per-hour cannot be negative")
    if not 0 <= args.ecotype_threshold <= 1:
        raise ValueError("--ecotype-threshold must be between 0 and 1")
    configs = build_grid(args)
    if not configs:
        raise ValueError("The valid grid is empty")
    for config in configs:
        thresholds = (
            config.start_threshold,
            config.support_threshold,
            config.continuation_threshold,
            config.event_score_threshold,
        )
        if any(not 0 <= value <= 1 for value in thresholds):
            raise ValueError("All score thresholds must be between 0 and 1")
        integer_values = (
            config.minimum_support_windows,
            config.support_radius_windows,
            config.peak_suppression_windows,
            config.event_top_k,
        )
        if min(integer_values) < 1 or config.maximum_gap_windows < 0:
            raise ValueError("Window counts/top-k must be positive; maximum gap can be zero")

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
    plain_recordings = [cached.recording for cached in recordings]
    annotation_table = base.read_csv(args.annotations)
    truths_by_file, annotation_sanity = base.load_annotations(
        args.annotations, plain_recordings, annotation_table
    )
    audio_hours = sum(cached.recording.duration_sec for cached in recordings) / 3600.0
    print(f"Cached recordings: {len(recordings):,}")
    print(f"Cached windows:    {sum(len(item.starts) for item in recordings):,}")
    print(f"Audio hours:       {audio_hours:.3f}")
    print(f"Valid grid runs:   {len(configs):,}")

    results = []
    for index, config in enumerate(configs, start=1):
        row, _, _ = evaluate_config(
            recordings,
            truths_by_file,
            config,
            args.collar_sec,
            audio_hours,
            args.ecotype_threshold,
        )
        results.append(row)
        if index % 25 == 0 or index == len(configs):
            print(f"Evaluated {index:,}/{len(configs):,} event configurations")
    ranked = sorted(
        results,
        key=lambda row: (
            rank_value(row["f1"]),
            rank_value(row["recall"]),
            -rank_value(row["false_positives_per_hour"], math.inf),
            rank_value(row["ecotype_macro_f1"]),
        ),
        reverse=True,
    )
    write_rows(output_dir / "stable_event_grid_results.csv", results)
    write_rows(output_dir / "ranked_stable_event_grid_results.csv", ranked)

    best, constraint_met = select_best(results, args.max_fp_per_hour)
    best_config = next(config for config in configs if config.run == best["run"])
    best_row, best_events, best_matches = evaluate_config(
        recordings,
        truths_by_file,
        best_config,
        args.collar_sec,
        audio_hours,
        args.ecotype_threshold,
    )
    ecotype_overall, ecotype_rows, ecotype_matrix = ecotype_metrics(best_matches)
    write_rows(
        output_dir / "best_stable_events.csv",
        [asdict(event) for event in best_events],
        list(StableEvent.__dataclass_fields__),
    )
    write_rows(
        output_dir / "best_event_matches.csv",
        [asdict(match) for match in best_matches],
        list(StableMatch.__dataclass_fields__),
    )
    write_rows(output_dir / "best_ecotype_metrics.csv", ecotype_rows)
    write_ecotype_matrix(output_dir / "best_ecotype_confusion_matrix.csv", ecotype_matrix)
    write_rows(
        output_dir / "best_provider_metrics.csv",
        group_metrics(recordings, best_matches, "provider"),
    )
    write_rows(
        output_dir / "best_dataset_metrics.csv",
        group_metrics(recordings, best_matches, "dataset"),
    )
    summary = {
        "selection": {
            "maximum_false_positives_per_hour": args.max_fp_per_hour,
            "constraint_met": constraint_met,
            "ranking": "maximum event F1, then recall, then lower FP/hour, then ecotype F1",
        },
        "best_config": asdict(best_config),
        "best_detection_metrics": best_row,
        "best_ecotype_metrics": ecotype_overall,
        "audio_hours": audio_hours,
        "selected_recordings": len(recordings),
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "grid_runs": len(configs),
        "arguments": vars(args),
    }
    (output_dir / "best_stable_event_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nTop {min(args.top_results, len(ranked))} unconstrained configurations")
    print("=" * 110)
    for rank, row in enumerate(ranked[:args.top_results], start=1):
        print(
            f"{rank:>3}. run={row['run']:<4} F1={rank_value(row['f1']):.4f} "
            f"P={rank_value(row['precision']):.4f} R={rank_value(row['recall']):.4f} "
            f"FP/h={rank_value(row['false_positives_per_hour'], math.inf):.2f} "
            f"source={row['score_source']} start={row['start_threshold']} "
            f"support={row['support_threshold']}/{row['minimum_support_windows']} "
            f"continue={row['continuation_threshold']} gap={row['maximum_gap_windows']} "
            f"suppress={row['peak_suppression_windows']}"
        )
    print("\nSelected stable-event configuration")
    print("=" * 72)
    print(f"Score source:                 {best_config.score_source}")
    print(f"Start threshold:              {best_config.start_threshold}")
    print(f"Support threshold/windows:    {best_config.support_threshold} / {best_config.minimum_support_windows}")
    print(f"Continuation threshold:       {best_config.continuation_threshold}")
    print(f"Maximum gap windows:          {best_config.maximum_gap_windows}")
    print(f"Peak suppression windows:     {best_config.peak_suppression_windows}")
    print(f"Event top-k/threshold:        {best_config.event_top_k} / {best_config.event_score_threshold}")
    print(f"Event precision:              {best_row['precision']}")
    print(f"Event recall:                 {best_row['recall']}")
    print(f"Event F1:                     {best_row['f1']}")
    print(f"False positives/hour:         {best_row['false_positives_per_hour']}")
    print(f"Median timing error (seconds): {best_row['median_absolute_timing_error_sec']}")
    print(f"Ecotype accuracy:             {best_row['ecotype_accuracy']}")
    print(f"Ecotype macro F1:             {best_row['ecotype_macro_f1']}")
    if args.max_fp_per_hour is not None and not constraint_met:
        print("WARNING: No grid run met --max-fp-per-hour; lowest-FP run was selected.")
    print(f"\nReports saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
