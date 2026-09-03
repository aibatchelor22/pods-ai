#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Evaluate KW, humpback, and AB events in cached DCLDE long-audio scores.

This CPU-only evaluator consumes ``window_predictions.csv`` produced by
``evaluate_dclde_long_recordings.py``. It forms stable events independently
for KW, HW, and AB, matches them one-to-one to all DCLDE annotations, and
reports per-species recall, precision, F1, and false positives/hour. Ecotype
metrics are calculated only for correctly detected KW events.

Use a validation collection to tune the event parameters. Freeze the selected
parameters before running an independent test collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

import evaluate_dclde_long_recordings as base
import evaluate_dclde_long_recordings_improved as cache_tools
import evaluate_dclde_long_recordings_stable_events as stable


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_ANNOTATIONS = base.DEFAULT_ANNOTATIONS
DEFAULT_OUTPUT = "/kaggle/working/dclde_long_multispecies_events"
SPECIES = ("KW", "HW", "AB")
ECOTYPES = stable.ECOTYPES

SPECIES_ALIASES = {
    "kw": "KW",
    "killer whale": "KW",
    "killer_whale": "KW",
    "killerwhale": "KW",
    "orca": "KW",
    "hw": "HW",
    "humpback": "HW",
    "humpback whale": "HW",
    "humpback_whale": "HW",
    "ab": "AB",
}


@dataclass(frozen=True)
class SpeciesConfig:
    species: str
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
class TruthEvent:
    soundfile: str
    event_id: str
    species: str
    start_sec: float
    end_sec: float
    ecotype: str
    provider: str
    dataset: str

    @property
    def center_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(frozen=True)
class PredictedEvent:
    soundfile: str
    event_id: str
    species: str
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
class EventMatch:
    soundfile: str
    status: str
    predicted_event_id: str
    true_event_id: str
    predicted_species: str
    true_species: str
    predicted_peak_sec: Optional[float]
    true_center_sec: Optional[float]
    absolute_distance_sec: Optional[float]
    event_score: Optional[float]
    predicted_ecotype: str
    true_ecotype: str
    provider: str
    dataset: str


def normalize_species(value: Any) -> str:
    return SPECIES_ALIASES.get(base.normalize_text(value).casefold(), "")


def load_truth_events(
    annotation_table: tuple[list[str], list[dict[str, str]]],
    recordings: list[Any],
) -> tuple[dict[str, list[TruthEvent]], dict[str, Any]]:
    fields, rows = annotation_table
    sound_col = base.find_column(fields, ("Soundfile", "soundfile", "filename", "file"))
    start_col = base.find_column(
        fields, ("FileBeginSec", "Start", "start", "start_sec", "start_time_sec")
    )
    end_col = base.find_column(
        fields, ("FileEndSec", "End", "end", "end_sec", "end_time_sec")
    )
    species_col = base.find_column(fields, ("ClassSpecies", "class_species", "species"))
    ecotype_col = base.find_column(fields, ("Ecotype", "ecotype"), required=False)
    selected = {cached.recording.soundfile: cached.recording for cached in recordings}
    by_file: dict[str, list[TruthEvent]] = defaultdict(list)
    counts = Counter()
    malformed: list[str] = []
    unknown = Counter()
    duplicate_counts = Counter()

    for row_index, row in enumerate(rows, start=2):
        soundfile = base.normalize_text(row.get(sound_col))
        if soundfile not in selected:
            continue
        raw_species = base.normalize_text(row.get(species_col))
        species = normalize_species(raw_species)
        if not species:
            if raw_species:
                unknown[raw_species] += 1
            continue
        try:
            start = float(row[start_col])
            end = float(row[end_col])
        except (TypeError, ValueError):
            malformed.append(f"row {row_index}: invalid start/end")
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            malformed.append(f"row {row_index}: invalid interval {start}..{end}")
            continue
        recording = selected[soundfile]
        start = max(0.0, start)
        end = min(float(recording.duration_sec), end)
        if end <= start:
            continue
        ecotype = base.normalize_text(row.get(ecotype_col)).upper() if ecotype_col else ""
        if species != "KW" or ecotype not in ECOTYPES:
            ecotype = ""
        duplicate_key = (soundfile, species, round(start, 6), round(end, 6), ecotype)
        duplicate_counts[duplicate_key] += 1
        counts[species] += 1
        by_file[soundfile].append(
            TruthEvent(
                soundfile=soundfile,
                event_id=f"true_{row_index}",
                species=species,
                start_sec=start,
                end_sec=end,
                ecotype=ecotype,
                provider=recording.provider,
                dataset=recording.dataset,
            )
        )
    for events in by_file.values():
        events.sort(key=lambda event: (event.start_sec, event.end_sec, event.species))
    duplicate_keys = [key for key, value in duplicate_counts.items() if value > 1]
    return by_file, {
        "selected_annotation_rows": sum(counts.values()),
        "species_counts": dict(counts),
        "kw_ecotype_counts": dict(
            Counter(
                event.ecotype or "unknown"
                for events in by_file.values()
                for event in events
                if event.species == "KW"
            )
        ),
        "ignored_class_counts": dict(unknown),
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:25],
        "duplicate_annotation_count": len(duplicate_keys),
        "duplicate_annotation_examples": [str(key) for key in duplicate_keys[:25]],
    }


def class_scores(cached: Any, config: SpeciesConfig) -> np.ndarray:
    if config.species == "KW":
        if config.score_source == "binary_kw":
            return np.asarray(cached.kw_probabilities, dtype=np.float64)
        species_kw = cached.species_probabilities[:, cache_tools.SPECIES.index("KW")]
        if config.score_source == "species_kw":
            return np.asarray(species_kw, dtype=np.float64)
        if config.score_source == "max_ecotype_composite":
            return np.asarray(
                species_kw * np.max(cached.ecotype_probabilities, axis=1),
                dtype=np.float64,
            )
        raise ValueError(f"Unsupported KW score source: {config.score_source}")
    if config.score_source != "species":
        raise ValueError(f"{config.species} score source must be 'species'")
    index = cache_tools.SPECIES.index(config.species)
    return np.asarray(cached.species_probabilities[:, index], dtype=np.float64)


def form_events(cached: Any, config: SpeciesConfig, ecotype_threshold: float) -> list[PredictedEvent]:
    scores = class_scores(cached, config)
    if not len(scores):
        return []
    peaks = stable.supported_peaks(
        scores,
        config.start_threshold,
        config.support_threshold,
        config.minimum_support_windows,
        config.support_radius_windows,
        config.peak_suppression_windows,
    )
    events: list[PredictedEvent] = []
    for peak_index in peaks:
        left, right = stable.extend_from_peak(
            scores,
            peak_index,
            config.continuation_threshold,
            config.maximum_gap_windows,
        )
        envelope = np.arange(left, right + 1)
        supporting = envelope[scores[envelope] >= config.support_threshold]
        if len(supporting) < config.minimum_support_windows:
            continue
        top_indices = supporting[
            np.argsort(scores[supporting], kind="stable")[-config.event_top_k :]
        ]
        event_score = float(np.mean(scores[top_indices]))
        if event_score < config.event_score_threshold:
            continue
        predicted_ecotype = ""
        ecotype_confidence = 0.0
        if config.species == "KW":
            ecotype_mean = np.mean(cached.ecotype_probabilities[top_indices], axis=0)
            ecotype_index = int(np.argmax(ecotype_mean))
            ecotype_confidence = float(ecotype_mean[ecotype_index])
            predicted_ecotype = (
                ECOTYPES[ecotype_index]
                if ecotype_confidence >= ecotype_threshold
                else "unknown"
            )
        events.append(
            PredictedEvent(
                soundfile=cached.recording.soundfile,
                event_id=f"{config.species}_{cached.recording.soundfile}_{peak_index}",
                species=config.species,
                score_source=config.score_source,
                peak_window_index=peak_index,
                peak_time_sec=float((cached.starts[peak_index] + cached.ends[peak_index]) / 2),
                start_sec=float(cached.starts[left]),
                end_sec=float(cached.ends[right]),
                peak_score=float(scores[peak_index]),
                event_topk_mean=event_score,
                supporting_windows=int(len(supporting)),
                envelope_windows=int(len(envelope)),
                predicted_ecotype=predicted_ecotype,
                ecotype_confidence=ecotype_confidence,
                provider=cached.recording.provider,
                dataset=cached.recording.dataset,
            )
        )
    for index in range(1, len(events)):
        previous = events[index - 1]
        current = events[index]
        if previous.end_sec > current.start_sec:
            boundary = (previous.peak_time_sec + current.peak_time_sec) / 2
            events[index - 1] = replace(previous, end_sec=min(previous.end_sec, boundary))
            events[index] = replace(current, start_sec=max(current.start_sec, boundary))
    return events


def suppress_cross_class_events(
    events: list[PredictedEvent], radius_sec: float
) -> list[PredictedEvent]:
    if radius_sec <= 0:
        return sorted(events, key=lambda event: (event.peak_time_sec, event.species))
    selected: list[PredictedEvent] = []
    for event in sorted(events, key=lambda item: (-item.event_topk_mean, item.peak_time_sec)):
        if all(abs(event.peak_time_sec - other.peak_time_sec) > radius_sec for other in selected):
            selected.append(event)
    return sorted(selected, key=lambda event: (event.peak_time_sec, event.species))


def match_events(
    predictions: list[PredictedEvent], truths: list[TruthEvent], collar_sec: float
) -> list[EventMatch]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            distance = abs(prediction.peak_time_sec - truth.center_sec)
            if distance <= collar_sec:
                candidates.append((distance, -prediction.event_topk_mean, pred_index, truth_index))
    candidates.sort()
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches: list[EventMatch] = []
    for distance, _, pred_index, truth_index in candidates:
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        prediction = predictions[pred_index]
        truth = truths[truth_index]
        matches.append(
            EventMatch(
                soundfile=prediction.soundfile,
                status="TP" if prediction.species == truth.species else "MISCLASSIFIED",
                predicted_event_id=prediction.event_id,
                true_event_id=truth.event_id,
                predicted_species=prediction.species,
                true_species=truth.species,
                predicted_peak_sec=prediction.peak_time_sec,
                true_center_sec=truth.center_sec,
                absolute_distance_sec=distance,
                event_score=prediction.event_topk_mean,
                predicted_ecotype=prediction.predicted_ecotype,
                true_ecotype=truth.ecotype,
                provider=prediction.provider,
                dataset=prediction.dataset,
            )
        )
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
    for index, prediction in enumerate(predictions):
        if index not in used_predictions:
            matches.append(
                EventMatch(
                    prediction.soundfile, "FP", prediction.event_id, "",
                    prediction.species, "", prediction.peak_time_sec, None, None,
                    prediction.event_topk_mean, prediction.predicted_ecotype, "",
                    prediction.provider, prediction.dataset,
                )
            )
    for index, truth in enumerate(truths):
        if index not in used_truths:
            matches.append(
                EventMatch(
                    truth.soundfile, "FN", "", truth.event_id, "", truth.species,
                    None, truth.center_sec, None, None, "", truth.ecotype,
                    truth.provider, truth.dataset,
                )
            )
    return matches


def safe_divide(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def per_species_metrics(matches: list[EventMatch], audio_hours: float) -> list[dict[str, Any]]:
    rows = []
    for species in SPECIES:
        tp = sum(match.status == "TP" and match.true_species == species for match in matches)
        fp = sum(
            bool(match.predicted_species == species and match.status != "TP")
            for match in matches
        )
        fn = sum(bool(match.true_species == species and match.status != "TP") for match in matches)
        denominator = 2 * tp + fp + fn
        rows.append(
            {
                "species": species,
                "audio_hours": audio_hours,
                "ground_truth_events": tp + fn,
                "predicted_events": tp + fp,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": 2 * tp / denominator if denominator else None,
                "false_positives_per_hour": safe_divide(fp, audio_hours),
                "false_negatives_per_hour": safe_divide(fn, audio_hours),
            }
        )
    return rows


def overall_metrics(matches: list[EventMatch], audio_hours: float) -> dict[str, Any]:
    counts = Counter(match.status for match in matches)
    temporal_tp = counts["TP"] + counts["MISCLASSIFIED"]
    temporal_fp = counts["FP"]
    temporal_fn = counts["FN"]
    temporal_denominator = 2 * temporal_tp + temporal_fp + temporal_fn
    class_tp = counts["TP"]
    class_fp = counts["FP"] + counts["MISCLASSIFIED"]
    class_fn = counts["FN"] + counts["MISCLASSIFIED"]
    class_denominator = 2 * class_tp + class_fp + class_fn
    distances = [
        match.absolute_distance_sec
        for match in matches
        if match.status in {"TP", "MISCLASSIFIED"} and match.absolute_distance_sec is not None
    ]
    species_rows = per_species_metrics(matches, audio_hours)
    class_f1_values = [row["f1"] for row in species_rows if row["f1"] is not None]
    return {
        "audio_hours": audio_hours,
        "temporally_matched_events": temporal_tp,
        "unmatched_predictions": temporal_fp,
        "unmatched_truth_events": temporal_fn,
        "detection_precision_ignoring_species": safe_divide(temporal_tp, temporal_tp + temporal_fp),
        "detection_recall_ignoring_species": safe_divide(temporal_tp, temporal_tp + temporal_fn),
        "detection_f1_ignoring_species": (
            2 * temporal_tp / temporal_denominator if temporal_denominator else None
        ),
        "correct_species_events": class_tp,
        "misclassified_species_events": counts["MISCLASSIFIED"],
        "class_aware_micro_precision": safe_divide(class_tp, class_tp + class_fp),
        "class_aware_micro_recall": safe_divide(class_tp, class_tp + class_fn),
        "class_aware_micro_f1": 2 * class_tp / class_denominator if class_denominator else None,
        "class_aware_macro_f1": (
            float(np.mean(class_f1_values)) if class_f1_values else None
        ),
        "unmatched_false_positives_per_hour": safe_divide(temporal_fp, audio_hours),
        "total_false_positives_per_hour": safe_divide(class_fp, audio_hours),
        "median_absolute_timing_error_sec": float(np.median(distances)) if distances else None,
    }


def ecotype_metrics(
    matches: list[EventMatch],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    eligible = [
        match
        for match in matches
        if match.status == "TP"
        and match.true_species == "KW"
        and match.true_ecotype in ECOTYPES
    ]
    predicted_labels = (*ECOTYPES, "unknown")
    matrix = np.zeros((len(ECOTYPES), len(predicted_labels)), dtype=np.int64)
    for match in eligible:
        column = (
            predicted_labels.index(match.predicted_ecotype)
            if match.predicted_ecotype in predicted_labels
            else len(ECOTYPES)
        )
        matrix[ECOTYPES.index(match.true_ecotype), column] += 1
    rows = []
    f1_values = []
    for index, label in enumerate(ECOTYPES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else None
        if matrix[index, :].sum() and f1 is not None:
            f1_values.append(f1)
        rows.append(
            {
                "ecotype": label,
                "support": int(matrix[index, :].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": f1,
            }
        )
    correct = sum(int(matrix[index, index]) for index in range(len(ECOTYPES)))
    return (
        {
            "evaluated_correctly_detected_kw_events": len(eligible),
            "accuracy": safe_divide(correct, len(eligible)),
            "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        },
        rows,
        matrix,
    )


def species_confusion(matches: list[EventMatch]) -> tuple[list[str], list[str], np.ndarray]:
    actual = (*SPECIES, "background")
    predicted = (*SPECIES, "missed")
    matrix = np.zeros((len(actual), len(predicted)), dtype=np.int64)
    for match in matches:
        if match.status in {"TP", "MISCLASSIFIED"}:
            matrix[actual.index(match.true_species), predicted.index(match.predicted_species)] += 1
        elif match.status == "FP":
            matrix[actual.index("background"), predicted.index(match.predicted_species)] += 1
        elif match.status == "FN":
            matrix[actual.index(match.true_species), predicted.index("missed")] += 1
    return list(actual), list(predicted), matrix


def grouped_metrics(
    recordings: list[Any], matches: list[EventMatch], attribute: str
) -> list[dict[str, Any]]:
    hours: dict[str, float] = defaultdict(float)
    for cached in recordings:
        group = getattr(cached.recording, attribute) or "unknown"
        hours[group] += cached.recording.duration_sec / 3600
    output = []
    for group in sorted(hours):
        selected = [match for match in matches if (getattr(match, attribute) or "unknown") == group]
        for row in per_species_metrics(selected, hours[group]):
            output.append({attribute: group, **row})
    return output


def grouped_overall_metrics(
    recordings: list[Any], matches: list[EventMatch], attribute: str
) -> list[dict[str, Any]]:
    hours: dict[str, float] = defaultdict(float)
    for cached in recordings:
        group = getattr(cached.recording, attribute) or "unknown"
        hours[group] += cached.recording.duration_sec / 3600
    output = []
    for group in sorted(hours):
        selected = [match for match in matches if (getattr(match, attribute) or "unknown") == group]
        output.append({attribute: group, **overall_metrics(selected, hours[group])})
    return output


def write_rows(path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(
    path: Path, row_labels: list[str], column_labels: list[str], matrix: np.ndarray, row_name: str
) -> None:
    write_rows(
        path,
        [
            {row_name: label, **{column: int(matrix[i, j]) for j, column in enumerate(column_labels)}}
            for i, label in enumerate(row_labels)
        ],
        [row_name, *column_labels],
    )


def add_species_arguments(parser: argparse.ArgumentParser, prefix: str, defaults: dict[str, Any]) -> None:
    parser.add_argument(f"--{prefix}-start-threshold", type=float, default=defaults["start"])
    parser.add_argument(f"--{prefix}-support-threshold", type=float, default=defaults["support"])
    parser.add_argument(f"--{prefix}-continuation-threshold", type=float, default=defaults["continuation"])
    parser.add_argument(f"--{prefix}-minimum-support-windows", type=int, default=defaults["minimum_support"])
    parser.add_argument(f"--{prefix}-support-radius-windows", type=int, default=defaults["radius"])
    parser.add_argument(f"--{prefix}-maximum-gap-windows", type=int, default=defaults["gap"])
    parser.add_argument(f"--{prefix}-peak-suppression-windows", type=int, default=defaults["suppression"])
    parser.add_argument(f"--{prefix}-event-top-k", type=int, default=defaults["top_k"])
    parser.add_argument(f"--{prefix}-event-score-threshold", type=float, default=defaults["event_threshold"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kw-score-source",
        choices=("binary_kw", "species_kw", "max_ecotype_composite"),
        default="max_ecotype_composite",
    )
    parser.add_argument("--ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--collar-sec", type=float, default=1.5)
    parser.add_argument(
        "--cross-class-suppression-sec",
        type=float,
        default=0.0,
        help="Keep only the highest-scoring species event within this radius; zero disables it.",
    )
    defaults = {
        "start": 0.85,
        "support": 0.55,
        "continuation": 0.35,
        "minimum_support": 2,
        "radius": 1,
        "gap": 0,
        "suppression": 1,
        "top_k": 2,
        "event_threshold": 0.75,
    }
    add_species_arguments(parser, "kw", defaults)
    add_species_arguments(parser, "hw", defaults)
    add_species_arguments(parser, "ab", defaults)
    return parser.parse_args()


def build_config(args: argparse.Namespace, species: str) -> SpeciesConfig:
    prefix = species.casefold()
    return SpeciesConfig(
        species=species,
        score_source=args.kw_score_source if species == "KW" else "species",
        start_threshold=getattr(args, f"{prefix}_start_threshold"),
        support_threshold=getattr(args, f"{prefix}_support_threshold"),
        continuation_threshold=getattr(args, f"{prefix}_continuation_threshold"),
        minimum_support_windows=getattr(args, f"{prefix}_minimum_support_windows"),
        support_radius_windows=getattr(args, f"{prefix}_support_radius_windows"),
        maximum_gap_windows=getattr(args, f"{prefix}_maximum_gap_windows"),
        peak_suppression_windows=getattr(args, f"{prefix}_peak_suppression_windows"),
        event_top_k=getattr(args, f"{prefix}_event_top_k"),
        event_score_threshold=getattr(args, f"{prefix}_event_score_threshold"),
    )


def validate_args(args: argparse.Namespace, configs: list[SpeciesConfig]) -> None:
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.collar_sec < 0 or args.cross_class_suppression_sec < 0:
        raise ValueError("Collar and suppression durations cannot be negative")
    if not 0 <= args.ecotype_threshold <= 1:
        raise ValueError("--ecotype-threshold must be between 0 and 1")
    for config in configs:
        thresholds = (
            config.start_threshold,
            config.support_threshold,
            config.continuation_threshold,
            config.event_score_threshold,
        )
        if any(not 0 <= value <= 1 for value in thresholds):
            raise ValueError(f"{config.species} thresholds must be between 0 and 1")
        if not config.continuation_threshold <= config.support_threshold <= config.start_threshold:
            raise ValueError(
                f"{config.species} requires continuation <= support <= start threshold"
            )
        integers = (
            config.minimum_support_windows,
            config.support_radius_windows,
            config.peak_suppression_windows,
            config.event_top_k,
        )
        if min(integers) < 1 or config.maximum_gap_windows < 0:
            raise ValueError(f"Invalid window-count setting for {config.species}")
        if config.minimum_support_windows > 2 * config.support_radius_windows + 1:
            raise ValueError(f"{config.species} support requirement exceeds its radius")


def main() -> int:
    args = parse_args()
    configs = [build_config(args, species) for species in SPECIES]
    validate_args(args, configs)
    window_path = Path(args.window_predictions)
    if not window_path.is_file():
        raise FileNotFoundError(f"Window predictions not found: {window_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_path = Path(args.failed_files) if args.failed_files else window_path.parent / "failed_files.csv"
    recordings, cache_sanity = cache_tools.load_cached_recordings(
        window_path, cache_tools.load_failed_soundfiles(failed_path)
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
    truths_by_file, annotation_sanity = load_truth_events(annotation_table, recordings)
    audio_hours = sum(cached.recording.duration_sec for cached in recordings) / 3600
    all_events: list[PredictedEvent] = []
    all_truths: list[TruthEvent] = []
    all_matches: list[EventMatch] = []
    for cached in recordings:
        file_events = []
        for config in configs:
            file_events.extend(form_events(cached, config, args.ecotype_threshold))
        file_events = suppress_cross_class_events(file_events, args.cross_class_suppression_sec)
        file_truths = truths_by_file.get(cached.recording.soundfile, [])
        all_events.extend(file_events)
        all_truths.extend(file_truths)
        all_matches.extend(match_events(file_events, file_truths, args.collar_sec))

    species_rows = per_species_metrics(all_matches, audio_hours)
    overall = overall_metrics(all_matches, audio_hours)
    ecotype_overall, ecotype_rows, ecotype_matrix = ecotype_metrics(all_matches)
    actual_labels, predicted_labels, species_matrix = species_confusion(all_matches)

    write_rows(output_dir / "multispecies_events.csv", [asdict(event) for event in all_events])
    write_rows(output_dir / "event_matches.csv", [asdict(match) for match in all_matches])
    write_rows(output_dir / "species_event_metrics.csv", species_rows)
    write_rows(output_dir / "provider_species_event_metrics.csv", grouped_metrics(recordings, all_matches, "provider"))
    write_rows(output_dir / "dataset_species_event_metrics.csv", grouped_metrics(recordings, all_matches, "dataset"))
    write_rows(output_dir / "provider_overall_event_metrics.csv", grouped_overall_metrics(recordings, all_matches, "provider"))
    write_rows(output_dir / "dataset_overall_event_metrics.csv", grouped_overall_metrics(recordings, all_matches, "dataset"))
    write_matrix(
        output_dir / "species_event_confusion_matrix.csv",
        actual_labels,
        predicted_labels,
        species_matrix,
        "actual_species",
    )
    write_rows(output_dir / "ecotype_metrics.csv", ecotype_rows)
    write_matrix(
        output_dir / "ecotype_confusion_matrix.csv",
        list(ECOTYPES),
        [*ECOTYPES, "unknown"],
        ecotype_matrix,
        "actual_ecotype",
    )
    summary = {
        "recordings": len(recordings),
        "cached_windows": sum(len(cached.starts) for cached in recordings),
        "audio_hours": audio_hours,
        "ground_truth_events": len(all_truths),
        "predicted_events": len(all_events),
        "configuration": {config.species: asdict(config) for config in configs},
        "overall_metrics": overall,
        "species_metrics": species_rows,
        "ecotype_metrics": ecotype_overall,
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "arguments": vars(args),
    }
    (output_dir / "multispecies_event_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Recordings:       {len(recordings):,}")
    print(f"Cached windows:   {summary['cached_windows']:,}")
    print(f"Audio hours:      {audio_hours:.3f}")
    print(f"Truth events:     {len(all_truths):,}")
    print(f"Predicted events: {len(all_events):,}")
    print("\nPer-species stable-event performance")
    print("=" * 92)
    print(f"{'Species':<9} {'Truth':>8} {'Pred':>8} {'TP':>8} {'FP':>8} {'FN':>8} {'P':>8} {'R':>8} {'F1':>8} {'FP/h':>9}")
    for row in species_rows:
        def number(value: Optional[float]) -> str:
            return "None" if value is None else f"{value:.4f}"
        print(
            f"{row['species']:<9} {row['ground_truth_events']:>8,} {row['predicted_events']:>8,} "
            f"{row['true_positives']:>8,} {row['false_positives']:>8,} {row['false_negatives']:>8,} "
            f"{number(row['precision']):>8} {number(row['recall']):>8} {number(row['f1']):>8} "
            f"{number(row['false_positives_per_hour']):>9}"
        )
    print("\nCombined event performance")
    print("=" * 72)
    print(f"Detection F1 ignoring species: {overall['detection_f1_ignoring_species']}")
    print(f"Class-aware micro F1:          {overall['class_aware_micro_f1']}")
    print(f"Class-aware macro F1:          {overall['class_aware_macro_f1']}")
    print(f"Species misclassifications:    {overall['misclassified_species_events']}")
    print(f"Total class-aware FP/hour:     {overall['total_false_positives_per_hour']}")
    print(f"Median timing error (seconds): {overall['median_absolute_timing_error_sec']}")
    print("\nKW ecotype performance (correctly detected KW events only)")
    print("=" * 72)
    print(f"Evaluated events:              {ecotype_overall['evaluated_correctly_detected_kw_events']}")
    print(f"Ecotype accuracy:              {ecotype_overall['accuracy']}")
    print(f"Ecotype macro F1:              {ecotype_overall['macro_f1']}")
    print(f"\nReports saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
