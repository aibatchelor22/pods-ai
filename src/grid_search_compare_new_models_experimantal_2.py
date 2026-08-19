#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Run the requested multispecies parameter grid efficiently.

The model is loaded once and each WAV is inferred once. Raw window scores are
then reused for every aggregation configuration in GRID_RUNS.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import librosa
import torch

import compare_new_models_experimantal_2 as comparison
from multispecies_train_model import SAMPLE_RATE


DEFAULT_MODEL = (
    "aibatchelor22/"
    "multi_species_detector_epoch_10_shift_gain_mean_sub_high_pass_os_bkg_ovs_weighted"
)
REPORT_LABELS = (
    "humpback",
    "transient",
    "resident",
    comparison.BACKGROUND_COMPARISON_LABEL,
)


@dataclass(frozen=True)
class GridRun:
    run: int
    humpback_threshold: float
    resident_threshold: float
    transient_threshold: float
    top_k: int
    humpback_min_windows: int
    resident_min_windows: int
    transient_min_windows: int
    smoothing: bool
    purpose: str


GRID = {
    "humpback_threshold": [0.35, 0.45, 0.55, 0.65],
    "resident_threshold": [0.025, 0.05, 0.075, 0.10],
    "transient_threshold": [0.10, 0.20, 0.30],
    "top_k": [1, 2, 3],
    "humpback_min_windows": [1, 2, 3],
    "resident_min_windows": [1, 2, 3, 4],
    "transient_min_windows": [1, 2, 3, 4],
    "smoothing": [True],
}


def build_grid_runs() -> tuple[GridRun, ...]:
    """Expand GRID into a deterministic Cartesian product."""
    return tuple(
        GridRun(run_number, *values, purpose="Cartesian grid")
        for run_number, values in enumerate(product(*GRID.values()), start=1)
    )


GRID_RUNS = build_grid_runs()


class CachedWindowScorePredictor(comparison.MultiSpeciesWindowPredictor):
    """Expose pre-aggregation window scores for reuse across grid runs."""

    def predict_window_scores(self, wav_path: Path) -> list[dict[str, float]]:
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        windows = self._window_audio(audio)
        window_scores: list[dict[str, float]] = []

        with torch.inference_mode():
            for batch_start in range(0, len(windows), self.batch_size):
                batch_audio = windows[batch_start:batch_start + self.batch_size]
                inputs = self.feature_extractor(
                    batch_audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                _, species_logits, ecotype_logits = outputs["logits"]
                species_probs = torch.softmax(species_logits, dim=-1).cpu().numpy()
                ecotype_probs = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()
                window_scores.extend(
                    self._comparison_scores(species_row, ecotype_row)
                    for species_row, ecotype_row in zip(species_probs, ecotype_probs)
                )
        return window_scores


def configure_predictor(predictor: CachedWindowScorePredictor, grid_run: GridRun) -> None:
    """Apply one aggregation configuration."""
    predictor.class_thresholds = {
        "humpback": grid_run.humpback_threshold,
        "resident": grid_run.resident_threshold,
        "transient": grid_run.transient_threshold,
    }
    predictor.aggregation_mode = "topk_mean"
    predictor.use_smoothing = grid_run.smoothing
    predictor.top_k = grid_run.top_k
    predictor.class_min_windows = {
        "humpback": grid_run.humpback_min_windows,
        "resident": grid_run.resident_min_windows,
        "transient": grid_run.transient_min_windows,
    }


def evaluate_grid_run(
    predictor: CachedWindowScorePredictor,
    grid_run: GridRun,
    cached_samples: list[tuple[comparison.TestSample, list[dict[str, float]]]],
    total_samples: int,
) -> comparison.ModelResult:
    """Aggregate cached scores and build the normal comparison metrics."""
    configure_predictor(predictor, grid_run)
    result = comparison.ModelResult(
        model_type=f"run_{grid_run.run}",
        total=total_samples,
        skipped=total_samples - len(cached_samples),
    )

    for sample, window_scores in cached_samples:
        predicted_label, _, _, _ = predictor._aggregate(window_scores)
        predicted_label = comparison.normalize_multispecies_comparison_label(
            predicted_label
        )
        actual_label = comparison.normalize_multispecies_comparison_label(
            sample.category
        )
        expected_resident = sample.category == comparison.RESIDENT_LABEL
        predicted_resident = comparison.is_resident_prediction(
            predicted_label, "multispecies"
        )

        if comparison.is_correct_prediction(
            sample.category, predicted_label, "multispecies"
        ):
            result.correct += 1
        elif predicted_resident and not expected_resident:
            result.false_positives += 1
        elif expected_resident and not predicted_resident:
            result.false_negatives += 1

        predictions = result.confusion_matrix.setdefault(actual_label, {})
        predictions[predicted_label] = predictions.get(predicted_label, 0) + 1

    return result


def _optional_rate(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.8f}"


def _metric_value(row: dict[str, object], name: str) -> float:
    """Return a sortable metric value, treating unavailable metrics as worst."""
    value = row[name]
    return float(value) if value != "" else float("-inf")


def _display_metric(row: dict[str, object], name: str) -> str:
    value = row[name]
    return "N/A" if value == "" else f"{float(value):.4f}"


def result_row(grid_run: GridRun, result: comparison.ModelResult) -> dict[str, object]:
    """Flatten a run configuration and its metrics for CSV output."""
    row: dict[str, object] = {
        "run": grid_run.run,
        "purpose": grid_run.purpose,
        "smoothing": "on" if grid_run.smoothing else "off",
        "humpback_threshold": grid_run.humpback_threshold,
        "resident_threshold": grid_run.resident_threshold,
        "transient_threshold": grid_run.transient_threshold,
        "top_k": grid_run.top_k,
        "humpback_min_windows": grid_run.humpback_min_windows,
        "resident_min_windows": grid_run.resident_min_windows,
        "transient_min_windows": grid_run.transient_min_windows,
        "evaluated": result.evaluated,
        "skipped": result.skipped,
        "correct": result.correct,
        "accuracy": _optional_rate(result.accuracy),
        "macro_whale_f1": _optional_rate(result.whale_f1),
    }
    for label in ("resident", "transient", "humpback"):
        row[f"{label}_fp_rate"] = _optional_rate(
            result.false_positive_rate_for_label(label)
        )
        row[f"{label}_fn_rate"] = _optional_rate(
            result.false_negative_rate_for_label(label)
        )
    for actual in REPORT_LABELS:
        for predicted in REPORT_LABELS:
            column = f"{actual}_to_{predicted}".replace("/", "_")
            row[column] = result.confusion_matrix.get(actual, {}).get(predicted, 0)
    return row


def cache_window_scores(
    predictor: CachedWindowScorePredictor,
    samples: list[comparison.TestSample],
    wav_dir: Path,
) -> list[tuple[comparison.TestSample, list[dict[str, float]]]]:
    """Run model inference once per available sample."""
    cached = []
    for index, sample in enumerate(samples, start=1):
        wav_path = comparison.find_wav_file(sample, wav_dir)
        if wav_path is None:
            print(
                f"[{index}/{len(samples)}] Skipping {sample.node_name}/{sample.timestamp}: "
                "WAV not found"
            )
            continue
        try:
            scores = predictor.predict_window_scores(wav_path)
        except Exception as error:
            print(
                f"[{index}/{len(samples)}] Skipping {wav_path.name}: {error}",
                file=sys.stderr,
            )
            continue
        cached.append((sample, scores))
        print(
            f"[{index}/{len(samples)}] Cached {wav_path.name}: {len(scores)} windows"
        )
    return cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the requested parameter grid while "
            "performing model inference only once per WAV."
        )
    )
    parser.add_argument("--multispecies-model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--testing-csv",
        default="/content/pods-ai/output/csv/testing_60s_samples_old.csv",
    )
    parser.add_argument(
        "--wav-dir",
        default="/content/pods-ai/output/testing-wav",
    )
    parser.add_argument(
        "--output-csv",
        default="/content/pods-ai/output/csv/multispecies_grid_search.csv",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    testing_csv = Path(args.testing_csv)
    wav_dir = Path(args.wav_dir)
    output_csv = Path(args.output_csv)
    if not testing_csv.is_file():
        print(f"Error: testing CSV not found: {testing_csv}", file=sys.stderr)
        return 1
    if not wav_dir.is_dir():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        return 1
    if args.batch_size <= 0:
        print("Error: --batch-size must be positive", file=sys.stderr)
        return 1
    if args.max_samples is not None and args.max_samples <= 0:
        print("Error: --max-samples must be positive", file=sys.stderr)
        return 1

    samples = comparison.load_test_samples(
        testing_csv,
        max_samples=args.max_samples,
        category_filter=args.category,
    )
    if not samples:
        print("Error: no matching test samples found", file=sys.stderr)
        return 1

    control = GRID_RUNS[0]
    predictor = CachedWindowScorePredictor(
        model_path=args.multispecies_model_path,
        threshold=0.25,
        class_thresholds={
            "humpback": control.humpback_threshold,
            "resident": control.resident_threshold,
            "transient": control.transient_threshold,
        },
        aggregation_mode="topk_mean",
        use_smoothing=control.smoothing,
        top_k=control.top_k,
        class_min_windows={
            "humpback": control.humpback_min_windows,
            "resident": control.resident_min_windows,
            "transient": control.transient_min_windows,
        },
        min_num_positive_calls_threshold=3,
        batch_size=args.batch_size,
        device=args.device,
    )

    started = time.perf_counter()
    cached_samples = cache_window_scores(predictor, samples, wav_dir)
    if not cached_samples:
        print("Error: no samples were successfully inferred", file=sys.stderr)
        return 1
    print(f"\nApplying {len(GRID_RUNS)} grid configurations...")

    rows = []
    for grid_run in GRID_RUNS:
        result = evaluate_grid_run(
            predictor,
            grid_run,
            cached_samples,
            total_samples=len(samples),
        )
        row = result_row(grid_run, result)
        rows.append(row)
        print(
            f"Run {grid_run.run:>2}: accuracy={_display_metric(row, 'accuracy')} "
            f"F1={_display_metric(row, 'macro_whale_f1')} - {grid_run.purpose}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda row: (
            _metric_value(row, "macro_whale_f1"),
            _metric_value(row, "accuracy"),
        ),
        reverse=True,
    )
    print("\nRanking by macro whale F1, then accuracy:")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"  {rank:>2}. Run {row['run']}: "
            f"F1={_display_metric(row, 'macro_whale_f1')}, "
            f"accuracy={_display_metric(row, 'accuracy')} ({row['purpose']})"
        )
    print(f"\nGrid-search CSV saved to: {output_csv}")
    print(f"Elapsed time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
