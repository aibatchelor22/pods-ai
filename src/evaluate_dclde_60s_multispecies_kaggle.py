#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Evaluate one DCLDE multi-head model on the Kaggle 60-second manifest.

This is the multispecies-only inference path from
compare_new_models_experimantal_2.py, adapted to explicit DCLDE clip paths.
It rewrites stale /kaggle/working paths into the mounted Kaggle dataset,
preflights all WAVs, writes per-clip predictions, and reports a four-class
confusion matrix plus accuracy, precision, recall/TPR, FPR, FNR, and F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from compare_new_models_experimantal_2 import (
    BACKGROUND_COMPARISON_LABEL,
    MultiSpeciesWindowPredictor,
    normalize_multispecies_comparison_label,
)


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
DEFAULT_OUTPUT_DIR = "/kaggle/working/dclde_multispecies_evaluation"
STALE_KAGGLE_PREFIX = "/kaggle/working/"
LABELS = ("humpback", "resident", "transient", BACKGROUND_COMPARISON_LABEL)
PREDICTION_COLUMNS = (
    "eval_resolved_clip_path",
    "eval_actual_label",
    "eval_predicted_label",
    "eval_global_confidence",
    "eval_predict_time_seconds",
    "eval_correct",
    "eval_local_prediction_labels_json",
    "eval_local_confidences_json",
    "eval_status",
    "eval_error",
)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the DCLDE manifest."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        required = {"clip_id", "clip_path", "comparison_label"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        rows = list(reader)

    clip_ids = [row.get("clip_id", "").strip() for row in rows]
    if any(not clip_id for clip_id in clip_ids):
        raise ValueError("Manifest contains an empty clip_id")
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Manifest contains duplicate clip_id values")

    invalid_labels = sorted(
        {
            row.get("comparison_label", "").strip()
            for row in rows
            if row.get("comparison_label", "").strip() not in LABELS
        }
    )
    if invalid_labels:
        raise ValueError(f"Manifest contains unsupported comparison labels: {invalid_labels}")
    return fieldnames, rows


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Preserve candidate order while removing duplicate paths."""
    result = []
    for path in paths:
        if path not in result:
            result.append(path)
    return result


def clip_path_candidates(row: dict[str, str], dataset_root: Path) -> list[Path]:
    """Return explicit-rewrite and relative-path candidates for one clip."""
    raw_clip_path = (row.get("clip_path") or "").strip()
    relative_clip_path = (row.get("relative_clip_path") or "").strip()
    candidates: list[Path] = []

    if raw_clip_path:
        if raw_clip_path.startswith(STALE_KAGGLE_PREFIX):
            remainder = raw_clip_path[len(STALE_KAGGLE_PREFIX):].lstrip("/\\")
            candidates.append(dataset_root / remainder)
        else:
            raw_path = Path(raw_clip_path).expanduser()
            candidates.append(
                raw_path if raw_path.is_absolute() else dataset_root / raw_path
            )

    if relative_clip_path:
        candidates.extend(
            [
                dataset_root / "dclde_60s_validation_test" / relative_clip_path,
                dataset_root / relative_clip_path,
            ]
        )
    return unique_paths(candidates)


def resolve_clip_path(row: dict[str, str], dataset_root: Path) -> Path:
    """Resolve a clip to the first existing Kaggle path, or its primary candidate."""
    candidates = clip_path_candidates(row, dataset_root)
    if not candidates:
        raise ValueError(f"Clip {row.get('clip_id', '<unknown>')} has no usable path")
    return next((path for path in candidates if path.is_file()), candidates[0])


def safe_divide(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def harmonic_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def confusion_matrix(
    prediction_rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    """Build the fixed four-class actual-by-predicted matrix."""
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for row in prediction_rows:
        actual = row["eval_actual_label"]
        predicted = row["eval_predicted_label"]
        matrix[actual][predicted] += 1
    return matrix


def calculate_metrics(
    matrix: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calculate overall and one-vs-rest per-class metrics."""
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in LABELS)
    per_class = []

    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[actual][label] for actual in LABELS if actual != label)
        tn = total - tp - fn - fp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        false_positive_rate = safe_divide(fp, fp + tn)
        false_negative_rate = safe_divide(fn, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = harmonic_f1(precision, recall)
        per_class.append(
            {
                "label": label,
                "support": tp + fn,
                "predicted_count": tp + fp,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "true_negative_count": tn,
                "precision": precision,
                "recall_true_positive_rate": recall,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "specificity_true_negative_rate": specificity,
                "f1": f1,
            }
        )

    whale_rows = [row for row in per_class if row["label"] != BACKGROUND_COMPARISON_LABEL]
    available_f1 = [row["f1"] for row in per_class if row["f1"] is not None]
    whale_f1 = [row["f1"] for row in whale_rows if row["f1"] is not None]
    weighted_f1_numerator = sum(
        row["f1"] * row["support"]
        for row in per_class
        if row["f1"] is not None
    )
    weighted_f1_denominator = sum(
        row["support"] for row in per_class if row["f1"] is not None
    )
    overall = {
        "evaluated": total,
        "correct": correct,
        "accuracy": safe_divide(correct, total),
        "macro_f1_all_classes": (
            sum(available_f1) / len(available_f1) if available_f1 else None
        ),
        "macro_whale_f1": sum(whale_f1) / len(whale_f1) if whale_f1 else None,
        "weighted_f1_all_classes": safe_divide(
            weighted_f1_numerator,
            weighted_f1_denominator,
        ),
    }
    return overall, per_class


def format_rate(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}%}"


def print_report(
    matrix: dict[str, dict[str, int]],
    overall: dict[str, Any],
    per_class: list[dict[str, Any]],
    skipped: int,
) -> None:
    """Print an evaluation report suitable for a Kaggle notebook log."""
    print("\nOverall metrics")
    print("=" * 72)
    print(f"Evaluated:             {overall['evaluated']:,}")
    print(f"Skipped/errors:        {skipped:,}")
    print(f"Correct:               {overall['correct']:,}")
    print(f"Accuracy:              {format_rate(overall['accuracy'])}")
    print(f"Macro whale F1:        {overall['macro_whale_f1']:.4f}")
    print(f"Macro F1, all classes: {overall['macro_f1_all_classes']:.4f}")
    print(f"Weighted F1:           {overall['weighted_f1_all_classes']:.4f}")

    print("\nPer-class metrics")
    print("=" * 118)
    print(
        f"{'Label':<19}{'Support':>9}{'TP':>8}{'FP':>8}{'FN':>8}"
        f"{'Precision':>12}{'TP/Recall':>12}{'FP Rate':>11}{'FN Rate':>11}{'F1':>10}"
    )
    print("-" * 118)
    for row in per_class:
        f1_text = "N/A" if row["f1"] is None else f"{row['f1']:.4f}"
        print(
            f"{row['label']:<19}{row['support']:>9}{row['true_positive_count']:>8}"
            f"{row['false_positive_count']:>8}{row['false_negative_count']:>8}"
            f"{format_rate(row['precision']):>12}"
            f"{format_rate(row['recall_true_positive_rate']):>12}"
            f"{format_rate(row['false_positive_rate']):>11}"
            f"{format_rate(row['false_negative_rate']):>11}{f1_text:>10}"
        )

    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print("=" * 100)
    width = max(len(label) for label in LABELS) + 3
    print(f"{'actual':<{width}}" + "".join(f"{label:>{width}}" for label in LABELS))
    for actual in LABELS:
        print(
            f"{actual:<{width}}"
            + "".join(f"{matrix[actual][predicted]:>{width}}" for predicted in LABELS)
        )

    print("\nDefinitions")
    print("  TP/Recall = true positives / actual class support")
    print("  FP Rate   = false positives / all samples not in that class")
    print("  FN Rate   = false negatives / actual class support")
    print("  Whale F1  = unweighted macro F1 over humpback, resident, and transient")


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["actual_label", *LABELS])
        writer.writeheader()
        for actual in LABELS:
            writer.writerow({"actual_label": actual, **matrix[actual]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one DCLDE multispecies model on Kaggle 60-second clips."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-column", default="comparison_label")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--category", choices=LABELS, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--multispecies-batch-size", type=int, default=16)
    parser.add_argument("--multispecies-threshold", type=float, default=0.25)
    parser.add_argument("--multispecies-humpback-threshold", type=float, default=0.475)
    parser.add_argument("--multispecies-resident-threshold", type=float, default=0.05)
    parser.add_argument("--multispecies-transient-threshold", type=float, default=0.20)
    parser.add_argument(
        "--multispecies-aggregation-mode",
        choices=("vote", "topk_mean"),
        default="topk_mean",
    )
    parser.add_argument("--multispecies-top-k", type=int, default=2)
    parser.add_argument("--multispecies-min-positive-windows", type=int, default=3)
    parser.add_argument("--multispecies-humpback-min-windows", type=int, default=2)
    parser.add_argument("--multispecies-resident-min-windows", type=int, default=2)
    parser.add_argument("--multispecies-transient-min-windows", type=int, default=3)
    parser.add_argument(
        "--multispecies-no-smoothing",
        action="store_true",
        help="Disable adjacent-window score smoothing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    summary_path = output_dir / "summary.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Kaggle dataset root not found: {dataset_root}")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.log_every < 1 or args.multispecies_batch_size < 1:
        raise ValueError("--log-every and --multispecies-batch-size must be positive")
    for name, threshold in {
        "base": args.multispecies_threshold,
        "humpback": args.multispecies_humpback_threshold,
        "resident": args.multispecies_resident_threshold,
        "transient": args.multispecies_transient_threshold,
    }.items():
        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} threshold must be between 0 and 1")
    for name, value in {
        "top_k": args.multispecies_top_k,
        "min_positive_windows": args.multispecies_min_positive_windows,
        "humpback_min_windows": args.multispecies_humpback_min_windows,
        "resident_min_windows": args.multispecies_resident_min_windows,
        "transient_min_windows": args.multispecies_transient_min_windows,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")

    manifest_fields, rows = read_manifest(manifest_path)
    if args.label_column not in manifest_fields:
        raise ValueError(f"Manifest has no label column {args.label_column!r}")
    if args.category:
        rows = [row for row in rows if row[args.label_column].strip() == args.category]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No manifest rows remain after filtering")

    resolved_paths = {row["clip_id"]: resolve_clip_path(row, dataset_root) for row in rows}
    missing = [
        (row["clip_id"], resolved_paths[row["clip_id"]])
        for row in rows
        if not resolved_paths[row["clip_id"]].is_file()
    ]
    if missing:
        examples = "\n".join(f"  {clip_id}: {path}" for clip_id, path in missing[:20])
        raise FileNotFoundError(
            f"Preflight found {len(missing):,} missing WAV files. First examples:\n{examples}"
        )

    label_counts = {label: 0 for label in LABELS}
    for row in rows:
        label_counts[row[args.label_column].strip()] += 1
    print(f"Manifest: {manifest_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Validated {len(rows):,} WAV paths")
    print(f"Ground-truth counts: {label_counts}")
    print(f"Model: {args.model_path}")
    print(
        "Parameters: "
        f"HW={args.multispecies_humpback_threshold}, "
        f"resident={args.multispecies_resident_threshold}, "
        f"transient={args.multispecies_transient_threshold}, "
        f"mode={args.multispecies_aggregation_mode}, top_k={args.multispecies_top_k}, "
        f"min_windows=({args.multispecies_humpback_min_windows}, "
        f"{args.multispecies_resident_min_windows}, "
        f"{args.multispecies_transient_min_windows}), "
        f"smoothing={'off' if args.multispecies_no_smoothing else 'on'}"
    )

    output_fields = [*manifest_fields, *PREDICTION_COLUMNS]
    completed_ids: set[str] = set()
    if args.resume and predictions_path.is_file():
        with predictions_path.open(newline="", encoding="utf-8") as file:
            existing_reader = csv.DictReader(file)
            if list(existing_reader.fieldnames or []) != output_fields:
                raise ValueError("Existing predictions.csv schema does not match this script")
            completed_ids = {
                row["clip_id"] for row in existing_reader if row.get("clip_id")
            }
        print(f"Resume enabled: {len(completed_ids):,} clip IDs already recorded")
    else:
        for path in (predictions_path, per_class_path, confusion_path, summary_path):
            if path.exists():
                path.unlink()

    predictor = MultiSpeciesWindowPredictor(
        model_path=args.model_path,
        threshold=args.multispecies_threshold,
        class_thresholds={
            "humpback": args.multispecies_humpback_threshold,
            "resident": args.multispecies_resident_threshold,
            "transient": args.multispecies_transient_threshold,
        },
        aggregation_mode=args.multispecies_aggregation_mode,
        use_smoothing=not args.multispecies_no_smoothing,
        top_k=args.multispecies_top_k,
        class_min_windows={
            "humpback": args.multispecies_humpback_min_windows,
            "resident": args.multispecies_resident_min_windows,
            "transient": args.multispecies_transient_min_windows,
        },
        min_num_positive_calls_threshold=args.multispecies_min_positive_windows,
        batch_size=args.multispecies_batch_size,
        device=args.device,
    )

    write_header = not predictions_path.exists()
    with predictions_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        if write_header:
            writer.writeheader()

        pending_rows = [row for row in rows if row["clip_id"] not in completed_ids]
        for index, row in enumerate(pending_rows, start=1):
            clip_id = row["clip_id"]
            wav_path = resolved_paths[clip_id]
            actual = normalize_multispecies_comparison_label(
                row[args.label_column].strip()
            )
            output_row = dict(row)
            output_row.update(
                {
                    "eval_resolved_clip_path": str(wav_path),
                    "eval_actual_label": actual,
                    "eval_predicted_label": "",
                    "eval_global_confidence": "",
                    "eval_predict_time_seconds": "",
                    "eval_correct": "",
                    "eval_local_prediction_labels_json": "",
                    "eval_local_confidences_json": "",
                    "eval_status": "error",
                    "eval_error": "",
                }
            )
            try:
                result = predictor.predict(wav_path)
                predicted = normalize_multispecies_comparison_label(
                    str(result.get("global_prediction_label", ""))
                )
                output_row.update(
                    {
                        "eval_predicted_label": predicted,
                        "eval_global_confidence": float(
                            result.get("global_confidence", 0.0)
                        ),
                        "eval_predict_time_seconds": float(
                            result.get("predict_time", 0.0)
                        ),
                        "eval_correct": predicted == actual,
                        "eval_local_prediction_labels_json": json.dumps(
                            result.get("local_prediction_labels", [])
                        ),
                        "eval_local_confidences_json": json.dumps(
                            result.get("local_confidences", [])
                        ),
                        "eval_status": "ok",
                    }
                )
            except Exception as error:
                output_row["eval_error"] = f"{type(error).__name__}: {error}"
                print(f"ERROR {clip_id}: {output_row['eval_error']}", file=sys.stderr)

            writer.writerow(output_row)
            file.flush()
            if index % args.log_every == 0 or index == len(pending_rows):
                print(
                    f"Processed {index:,}/{len(pending_rows):,} pending clips; "
                    f"latest={clip_id}, status={output_row['eval_status']}"
                )

    selected_ids = {row["clip_id"] for row in rows}
    latest_rows: dict[str, dict[str, str]] = {}
    with predictions_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("clip_id") in selected_ids:
                latest_rows[row["clip_id"]] = row
    successful = [row for row in latest_rows.values() if row["eval_status"] == "ok"]
    skipped = len(rows) - len(successful)
    if not successful:
        raise RuntimeError("No clips were evaluated successfully")

    matrix = confusion_matrix(successful)
    overall, per_class = calculate_metrics(matrix)
    settings = {
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "model_path": args.model_path,
        "selected_rows": len(rows),
        "skipped_or_errors": skipped,
        "thresholds": {
            "humpback": args.multispecies_humpback_threshold,
            "resident": args.multispecies_resident_threshold,
            "transient": args.multispecies_transient_threshold,
        },
        "aggregation_mode": args.multispecies_aggregation_mode,
        "top_k": args.multispecies_top_k,
        "minimum_windows": {
            "humpback": args.multispecies_humpback_min_windows,
            "resident": args.multispecies_resident_min_windows,
            "transient": args.multispecies_transient_min_windows,
        },
        "smoothing": not args.multispecies_no_smoothing,
    }
    write_dict_rows(per_class_path, per_class)
    write_confusion_csv(confusion_path, matrix)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": settings,
                "overall_metrics": overall,
                "per_class_metrics": per_class,
                "confusion_matrix": matrix,
            },
            file,
            indent=2,
        )

    print_report(matrix, overall, per_class, skipped)
    print(f"\nPredictions:      {predictions_path}")
    print(f"Per-class report: {per_class_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"JSON summary:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
