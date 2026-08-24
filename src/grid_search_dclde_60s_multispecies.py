#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Vectorized aggregation grid search for DCLDE 60-second clips.

Neural-network inference is performed once per clip. The three raw comparison
score sequences (humpback, resident/SRKW, transient/TKW) are persisted to CSV
and reused for all threshold, top-k, minimum-window, and smoothing settings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

DEFAULT_MANIFEST = (
    "/kaggle/input/datasets/leonisviridis/orca-data-dclde-60s-clips/"
    "dclde_60s_validation_test/dclde_60s_validation_test_manifest_ready.csv"
)
DEFAULT_DATASET_ROOT = (
    "/kaggle/input/datasets/leonisviridis/orca-data-dclde-60s-clips"
)
DEFAULT_MODEL = (
    "aibatchelor22/"
    "multi_species_detector_epoch_10_shift_gain_mean_sub_high_pass_os_bkg_ovs_weighted"
)
DEFAULT_OUTPUT_DIR = "/kaggle/working/dclde_60s_grid_search"
BACKGROUND_COMPARISON_LABEL = "other/background"
LABELS = ("humpback", "resident", "transient", BACKGROUND_COMPARISON_LABEL)
WHALE_LABELS = ("humpback", "resident", "transient")
SCORE_LABELS = WHALE_LABELS
CACHE_FIELDS = (
    "clip_id", "actual_label", "window_index", "humpback_score",
    "resident_score", "transient_score",
)


@dataclass(frozen=True)
class GridConfig:
    run: int
    smoothing: bool
    top_k: int
    humpback_threshold: float
    resident_threshold: float
    transient_threshold: float
    humpback_min_windows: int
    resident_min_windows: int
    transient_min_windows: int


def parse_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        values = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return sorted(set(values))


def parse_smoothing(value: str) -> list[bool]:
    mapping = {"on": True, "true": True, "1": True, "off": False, "false": False, "0": False}
    result = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized not in mapping:
            raise argparse.ArgumentTypeError(
                "Smoothing values must be comma-separated on/off values"
            )
        if mapping[normalized] not in result:
            result.append(mapping[normalized])
    if not result:
        raise argparse.ArgumentTypeError("Smoothing values cannot be empty")
    return result


def build_configs(args: argparse.Namespace) -> list[GridConfig]:
    dimensions = [
        parse_smoothing(args.smoothing_values),
        parse_list(args.top_k_values, int, "top-k values"),
        parse_list(args.humpback_thresholds, float, "humpback thresholds"),
        parse_list(args.resident_thresholds, float, "resident thresholds"),
        parse_list(args.transient_thresholds, float, "transient thresholds"),
        parse_list(args.humpback_min_windows, int, "humpback minimum windows"),
        parse_list(args.resident_min_windows, int, "resident minimum windows"),
        parse_list(args.transient_min_windows, int, "transient minimum windows"),
    ]
    return [
        GridConfig(run, *values)
        for run, values in enumerate(itertools.product(*dimensions), start=1)
    ]


def select_rows(
    rows: list[dict[str, str]], max_samples: Optional[int], seed: int
) -> list[dict[str, str]]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    return random.Random(seed).sample(rows, max_samples)


def find_group_column(fields: list[str], requested: str) -> Optional[str]:
    if requested.lower() == "none":
        return None
    lookup = {field.casefold(): field for field in fields}
    if requested.lower() != "auto":
        if requested.casefold() not in lookup:
            raise ValueError(f"Manifest has no group column {requested!r}")
        return lookup[requested.casefold()]
    for candidate in ("Provider", "source_dataset", "Dataset", "provider", "dataset"):
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def read_cache(path: Path, expected_windows: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    grouped: dict[str, list[tuple[int, list[float]]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or []) != CACHE_FIELDS:
            raise ValueError(f"Existing cache has an incompatible schema: {path}")
        for row_number, row in enumerate(reader, start=2):
            try:
                item = (
                    int(row["window_index"]),
                    [
                        float(row["humpback_score"]),
                        float(row["resident_score"]),
                        float(row["transient_score"]),
                    ],
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid cache value at row {row_number}") from error
            grouped.setdefault(row["clip_id"], []).append(item)
    result = {}
    for clip_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        indices = [item[0] for item in items]
        if indices != list(range(expected_windows)):
            continue
        result[clip_id] = np.asarray([item[1] for item in items], dtype=np.float32)
    return result


def validate_cache_metadata(
    metadata_path: Path,
    cache_exists: bool,
    model_path: str,
    expected_windows: int,
) -> None:
    if not cache_exists:
        return
    if not metadata_path.is_file():
        raise ValueError(
            f"Cache exists without metadata: {metadata_path}. Use --refresh-cache to rebuild it."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_path") != model_path:
        raise ValueError(
            "Cached scores were produced by a different model. "
            "Use a different --output-dir or pass --refresh-cache."
        )
    if metadata.get("expected_windows") != expected_windows:
        raise ValueError("Cached window count differs from --expected-windows")


def write_cache_metadata(
    path: Path,
    model_path: str,
    expected_windows: int,
    manifest: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "model_path": model_path,
                "expected_windows": expected_windows,
                "manifest": str(manifest),
                "score_definitions": {
                    "humpback": "P(species=HW)",
                    "resident": "P(species=KW) * P(ecotype=SRKW)",
                    "transient": "P(species=KW) * P(ecotype=TKW)",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def cache_inference(
    rows: list[dict[str, str]],
    resolved_paths: dict[str, Path],
    cached: dict[str, np.ndarray],
    cache_path: Path,
    predictor: Any,
    expected_windows: int,
    log_every: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    write_header = not cache_path.is_file()
    failures = []
    pending = [row for row in rows if row["clip_id"] not in cached]
    with cache_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CACHE_FIELDS)
        if write_header:
            writer.writeheader()
        for index, row in enumerate(pending, start=1):
            clip_id = row["clip_id"]
            path = resolved_paths[clip_id]
            try:
                score_rows = predictor.predict_window_scores(path)
                if len(score_rows) != expected_windows:
                    raise ValueError(
                        f"expected {expected_windows} windows, received {len(score_rows)}"
                    )
                scores = np.asarray(
                    [
                        [score["humpback"], score["resident"], score["transient"]]
                        for score in score_rows
                    ],
                    dtype=np.float32,
                )
                for window_index, score in enumerate(scores):
                    writer.writerow(
                        {
                            "clip_id": clip_id,
                            "actual_label": row["comparison_label"],
                            "window_index": window_index,
                            "humpback_score": float(score[0]),
                            "resident_score": float(score[1]),
                            "transient_score": float(score[2]),
                        }
                    )
                file.flush()
                cached[clip_id] = scores
            except Exception as error:
                failures.append(
                    {
                        "clip_id": clip_id,
                        "clip_path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if index % log_every == 0 or index == len(pending):
                print(f"Inferred {index:,}/{len(pending):,} uncached clips")
    return cached, failures


def smooth_scores(scores: np.ndarray) -> np.ndarray:
    result = scores.copy()
    if scores.shape[1] >= 3:
        result[:, 1:-1, :] = (scores[:, :-2, :] + scores[:, 1:-1, :]) / 2.0
    return result


def candidate_array(
    class_scores: np.ndarray,
    threshold: float,
    top_k: int,
    minimum_windows: int,
) -> np.ndarray:
    positive = class_scores >= threshold
    counts = np.sum(positive, axis=1)
    masked = np.where(positive, class_scores, -np.inf)
    sorted_scores = np.sort(masked, axis=1)[:, ::-1]
    selected = sorted_scores[:, :top_k]
    finite = np.isfinite(selected)
    sums = np.sum(np.where(finite, selected, 0.0), axis=1)
    denominators = np.sum(finite, axis=1)
    means = np.divide(
        sums,
        denominators,
        out=np.full(len(class_scores), -np.inf, dtype=np.float64),
        where=denominators > 0,
    )
    means[counts < minimum_windows] = -np.inf
    return np.column_stack((means, counts))


def build_candidate_cache(
    scores_by_smoothing: dict[bool, np.ndarray],
    configs: list[GridConfig],
) -> dict[tuple[bool, int, float, int, int], np.ndarray]:
    cache = {}
    for config in configs:
        parameters = (
            (0, config.humpback_threshold, config.humpback_min_windows),
            (1, config.resident_threshold, config.resident_min_windows),
            (2, config.transient_threshold, config.transient_min_windows),
        )
        for class_index, threshold, minimum in parameters:
            key = (config.smoothing, class_index, threshold, config.top_k, minimum)
            if key not in cache:
                cache[key] = candidate_array(
                    scores_by_smoothing[config.smoothing][:, :, class_index],
                    threshold,
                    config.top_k,
                    minimum,
                )
    return cache


def predict_config(
    config: GridConfig,
    candidate_cache: dict[tuple[bool, int, float, int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    keys = (
        (config.smoothing, 0, config.humpback_threshold, config.top_k, config.humpback_min_windows),
        (config.smoothing, 1, config.resident_threshold, config.top_k, config.resident_min_windows),
        (config.smoothing, 2, config.transient_threshold, config.top_k, config.transient_min_windows),
    )
    candidate_values = [candidate_cache[key] for key in keys]
    candidates = np.column_stack([value[:, 0] for value in candidate_values])
    positive_counts = np.column_stack([value[:, 1] for value in candidate_values])
    confidence = np.max(candidates, axis=1)
    # Match MultiSpeciesWindowPredictor exactly: mean score wins first, then
    # the number of above-threshold windows breaks an equal-mean tie.
    tied_for_best = candidates == confidence[:, None]
    tie_break_counts = np.where(tied_for_best, positive_counts, -1)
    winner = np.argmax(tie_break_counts, axis=1)
    background = ~np.isfinite(confidence)
    winner = winner.astype(np.int64)
    winner[background] = 3
    confidence[background] = 0.0
    return winner, confidence


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def metrics_from_matrix(matrix: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = int(np.sum(matrix))
    correct = int(np.trace(matrix))
    rows = []
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        fn = int(np.sum(matrix[index, :]) - tp)
        fp = int(np.sum(matrix[:, index]) - tp)
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        denominator = 2 * tp + fp + fn
        rows.append(
            {
                "label": label,
                "support": tp + fn,
                "predicted_count": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * tp / denominator if denominator else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "false_negative_rate": fn / (tp + fn) if tp + fn else None,
            }
        )
    whale_f1 = [row["f1"] for row in rows[:3] if row["f1"] is not None]
    return {
        "evaluated": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "macro_whale_f1": float(np.mean(whale_f1)) if whale_f1 else None,
    }, rows


def group_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if int(np.sum(mask)) < minimum_samples:
            continue
        overall, _ = metrics_from_matrix(confusion_matrix(actual[mask], predicted[mask]))
        rows.append({"group": group, "samples": int(np.sum(mask)), **overall})
    return rows


def optional_mean(values: list[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return float(np.mean(available)) if available else None


def optional_min(values: list[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return min(available) if available else None


def flatten_result(
    config: GridConfig,
    matrix: np.ndarray,
    overall: dict[str, Any],
    per_class: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row = {
        **asdict(config),
        **overall,
        "mean_group_macro_whale_f1": optional_mean(
            [item["macro_whale_f1"] for item in group_rows]
        ),
        "minimum_group_macro_whale_f1": optional_min(
            [item["macro_whale_f1"] for item in group_rows]
        ),
        "groups_evaluated": len(group_rows),
    }
    for class_row in per_class:
        label = class_row["label"].replace("/", "_")
        for metric in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate"):
            row[f"{label}_{metric}"] = class_row[metric]
    for actual_index, actual_label in enumerate(LABELS):
        for predicted_index, predicted_label in enumerate(LABELS):
            row[f"{actual_label}_to_{predicted_label}".replace("/", "_")] = int(
                matrix[actual_index, predicted_index]
            )
    return row


def ranking_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return -math.inf
    return float(value)


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
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-windows", type=int, default=29)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--group-column", default="auto")
    parser.add_argument("--minimum-group-samples", type=int, default=20)
    parser.add_argument(
        "--ranking-metric",
        choices=(
            "macro_whale_f1",
            "mean_group_macro_whale_f1",
            "minimum_group_macro_whale_f1",
        ),
        default="mean_group_macro_whale_f1",
    )
    parser.add_argument("--top-results", type=int, default=25)
    parser.add_argument("--humpback-thresholds", default="0.40,0.45,0.475,0.50,0.55")
    parser.add_argument("--resident-thresholds", default="0.035,0.05,0.065,0.08")
    parser.add_argument("--transient-thresholds", default="0.15,0.20,0.25")
    parser.add_argument("--top-k-values", default="2,3")
    parser.add_argument("--humpback-min-windows", default="1,2")
    parser.add_argument("--resident-min-windows", default="2,3")
    parser.add_argument("--transient-min-windows", default="2,3,4")
    parser.add_argument("--smoothing-values", default="off")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.batch_size, args.expected_windows, args.log_every, args.minimum_group_samples, args.top_results) < 1:
        raise ValueError("Batch/window/log/group/top-result values must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    # Keep the model/audio stack out of utility-only imports and --help.
    import evaluate_dclde_60s_multispecies_kaggle as dclde
    import grid_search_compare_new_models_experimantal_2 as legacy_grid

    configs = build_configs(args)
    if not configs:
        raise ValueError("Grid is empty")
    for config in configs:
        thresholds = (
            config.humpback_threshold, config.resident_threshold, config.transient_threshold
        )
        if any(not 0 <= value <= 1 for value in thresholds):
            raise ValueError("All thresholds must be between 0 and 1")
        if min(
            config.top_k,
            config.humpback_min_windows,
            config.resident_min_windows,
            config.transient_min_windows,
        ) < 1:
            raise ValueError("Top-k and minimum-window values must be positive")

    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "window_score_cache.csv"
    metadata_path = output_dir / "window_score_cache_metadata.json"
    if args.refresh_cache:
        for path in (cache_path, metadata_path):
            if path.exists():
                path.unlink()

    manifest_fields, all_rows = dclde.read_manifest(manifest_path)
    rows = select_rows(all_rows, args.max_samples, args.seed)
    selected_ids = {row["clip_id"] for row in rows}
    group_column = find_group_column(manifest_fields, args.group_column)
    print(f"Manifest rows selected: {len(rows):,}")
    print(f"Grid configurations:    {len(configs):,}")
    print(f"Group column:           {group_column or 'none'}")

    validate_cache_metadata(
        metadata_path, cache_path.is_file(), args.model_path, args.expected_windows
    )
    cached = read_cache(cache_path, args.expected_windows)
    cached = {clip_id: scores for clip_id, scores in cached.items() if clip_id in selected_ids}
    pending_rows = [row for row in rows if row["clip_id"] not in cached]
    failures = []
    if pending_rows:
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        resolved = {
            row["clip_id"]: dclde.resolve_clip_path(row, dataset_root)
            for row in pending_rows
        }
        missing = [(clip_id, path) for clip_id, path in resolved.items() if not path.is_file()]
        if missing:
            examples = "\n".join(f"  {clip_id}: {path}" for clip_id, path in missing[:20])
            raise FileNotFoundError(
                f"Preflight found {len(missing):,} missing WAV files. First examples:\n{examples}"
            )
        predictor = legacy_grid.CachedWindowScorePredictor(
            model_path=args.model_path,
            threshold=0.25,
            class_thresholds={"humpback": 0.475, "resident": 0.05, "transient": 0.20},
            aggregation_mode="topk_mean",
            use_smoothing=False,
            top_k=2,
            class_min_windows={"humpback": 2, "resident": 2, "transient": 3},
            min_num_positive_calls_threshold=3,
            batch_size=args.batch_size,
            device=args.device,
        )
        write_cache_metadata(
            metadata_path, args.model_path, args.expected_windows, manifest_path
        )
        cached, failures = cache_inference(
            rows, resolved, cached, cache_path, predictor,
            args.expected_windows, args.log_every,
        )
    write_rows(
        output_dir / "inference_failures.csv",
        failures,
        ["clip_id", "clip_path", "error"],
    )

    successful_rows = [row for row in rows if row["clip_id"] in cached]
    if not successful_rows:
        raise RuntimeError("No selected clips have usable cached scores")
    scores = np.stack([cached[row["clip_id"]] for row in successful_rows])
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    actual = np.asarray(
        [label_to_id[row["comparison_label"]] for row in successful_rows], dtype=np.int64
    )
    groups = np.asarray(
        [
            ((row.get(group_column) or "unknown").strip() or "unknown")
            if group_column else "all"
            for row in successful_rows
        ],
        dtype=object,
    )
    scores_by_smoothing = {False: scores, True: smooth_scores(scores)}
    candidate_cache = build_candidate_cache(scores_by_smoothing, configs)

    started = time.perf_counter()
    results = []
    for index, config in enumerate(configs, start=1):
        predicted, confidence = predict_config(config, candidate_cache)
        matrix = confusion_matrix(actual, predicted)
        overall, per_class = metrics_from_matrix(matrix)
        group_rows = group_metrics(
            actual, predicted, groups, args.minimum_group_samples
        )
        result = flatten_result(config, matrix, overall, per_class, group_rows)
        results.append(result)
        if index % 250 == 0 or index == len(configs):
            print(f"Aggregated {index:,}/{len(configs):,} configurations")

    effective_ranking = args.ranking_metric
    if all(ranking_value(row, effective_ranking) == -math.inf for row in results):
        print(
            f"WARNING: {effective_ranking} is unavailable; falling back to macro_whale_f1",
            file=sys.stderr,
        )
        effective_ranking = "macro_whale_f1"
    ranked = sorted(
        results,
        key=lambda row: (
            ranking_value(row, effective_ranking),
            ranking_value(row, "macro_whale_f1"),
            ranking_value(row, "accuracy"),
        ),
        reverse=True,
    )
    write_rows(output_dir / "grid_results.csv", results)
    write_rows(output_dir / "ranked_grid_results.csv", ranked)

    best = ranked[0]
    best_config = next(config for config in configs if config.run == best["run"])
    best_predicted, best_confidence = predict_config(best_config, candidate_cache)
    best_matrix = confusion_matrix(actual, best_predicted)
    best_overall, best_per_class = metrics_from_matrix(best_matrix)
    best_groups = group_metrics(
        actual, best_predicted, groups, args.minimum_group_samples
    )
    write_rows(output_dir / "best_per_class_metrics.csv", best_per_class)
    write_rows(output_dir / "best_group_metrics.csv", best_groups)
    write_rows(
        output_dir / "best_predictions.csv",
        [
            {
                "clip_id": row["clip_id"],
                "group": groups[index],
                "actual_label": LABELS[actual[index]],
                "predicted_label": LABELS[best_predicted[index]],
                "confidence": float(best_confidence[index]),
                "correct": bool(actual[index] == best_predicted[index]),
            }
            for index, row in enumerate(successful_rows)
        ],
    )
    write_rows(
        output_dir / "best_confusion_matrix.csv",
        [
            {
                "actual_label": label,
                **{
                    predicted_label: int(best_matrix[actual_index, predicted_index])
                    for predicted_index, predicted_label in enumerate(LABELS)
                },
            }
            for actual_index, label in enumerate(LABELS)
        ],
        ["actual_label", *LABELS],
    )
    summary = {
        "selected_configuration": asdict(best_config),
        "selection_metric_requested": args.ranking_metric,
        "selection_metric_used": effective_ranking,
        "best_result": best,
        "best_overall_metrics": best_overall,
        "manifest": str(manifest_path),
        "model_path": args.model_path,
        "selected_manifest_rows": len(rows),
        "successfully_inferred_rows": len(successful_rows),
        "group_column": group_column,
        "minimum_group_samples": args.minimum_group_samples,
        "grid_configurations": len(configs),
        "arguments": vars(args),
    }
    (output_dir / "best_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nTop {min(args.top_results, len(ranked))} configurations")
    print("=" * 100)
    for rank, row in enumerate(ranked[: args.top_results], start=1):
        print(
            f"{rank:>3}. run={row['run']:<5} "
            f"rank={ranking_value(row, effective_ranking):.4f} "
            f"whale_F1={ranking_value(row, 'macro_whale_f1'):.4f} "
            f"accuracy={ranking_value(row, 'accuracy'):.4f} "
            f"smooth={'on' if row['smoothing'] else 'off'} top_k={row['top_k']} "
            f"thresholds=({row['humpback_threshold']}, {row['resident_threshold']}, "
            f"{row['transient_threshold']}) mins=({row['humpback_min_windows']}, "
            f"{row['resident_min_windows']}, {row['transient_min_windows']})"
        )
    print(f"\nUsable clips: {len(successful_rows):,}/{len(rows):,}")
    print(f"Aggregation time: {time.perf_counter() - started:.1f}s")
    print(f"Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
