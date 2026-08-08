#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Paired bootstrap comparison for two DCLDE multi-task AST whale detector models.

The script evaluates two models on the same 3-second validation manifest rows,
writes per-model prediction cache CSVs, then bootstraps paired row samples to
estimate uncertainty for challenger-vs-baseline combined_score differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from detailed_metrics import (
    add_prediction_columns,
    load_feature_extractor,
    load_manifest_frame,
    resolve_model_source,
    resolve_path,
    run_predictions,
)
from multispecies_train_model import (
    DEFAULT_MAX_DURATION,
    DCLDEAudioCollator,
    DCLDEAudioDataset,
    ECOTYPE_LABELS,
    IGNORE_INDEX,
    KW_LABELS,
    SAMPLE_RATE,
    SPECIES_LABELS,
    load_training_model,
)


DEFAULT_N_BOOTSTRAP = 10000
METRIC_NAMES = (
    "combined_score",
    "kw_f1",
    "species_macro_f1",
    "ecotype_srkw_tkw_f1",
)


def progress_iter(iterable, description: str, total: int | None = None):
    """Wrap an iterable in tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=description)


def safe_model_name(value: str) -> str:
    """Return a compact file-safe model identifier."""
    return (
        value.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def preprocessing_cache_suffix(
    mean_subtract: bool,
    high_pass_filter: bool,
    high_pass_cutoff_hz: float,
    high_pass_order: int,
) -> str:
    """Return a cache filename suffix describing waveform preprocessing."""
    parts = []
    if mean_subtract:
        parts.append("mean_subtract")
    if high_pass_filter:
        cutoff = f"{high_pass_cutoff_hz:g}".replace(".", "p")
        parts.append(f"hp{cutoff}hz_o{high_pass_order}")
    if not parts:
        return ""
    return "_" + "_".join(parts)


def f1_binary_positive(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> float:
    """Return binary positive-class F1 with zero_division=0 behavior."""
    true_positive = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    false_positive = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    false_negative = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float((2 * true_positive) / denominator)


def f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    """Return one-vs-rest F1 for one multiclass label."""
    true_positive = int(np.sum((y_true == label) & (y_pred == label)))
    false_positive = int(np.sum((y_true != label) & (y_pred == label)))
    false_negative = int(np.sum((y_true == label) & (y_pred != label)))
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float((2 * true_positive) / denominator)


def macro_f1_present_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return macro F1 over labels present in y_true or y_pred, matching sklearn default."""
    if len(y_true) == 0:
        return 0.0
    labels = np.union1d(y_true, y_pred)
    if len(labels) == 0:
        return 0.0
    return float(np.mean([f1_for_label(y_true, y_pred, int(label)) for label in labels]))


def macro_f1_fixed_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    """Return macro F1 over an explicit label set."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean([f1_for_label(y_true, y_pred, label) for label in labels]))


def combined_metrics_from_arrays(
    kw_true: np.ndarray,
    species_true: np.ndarray,
    ecotype_true: np.ndarray,
    kw_pred: np.ndarray,
    species_pred: np.ndarray,
    ecotype_pred: np.ndarray,
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the same combined_score components as multispecies_train_model.py."""
    if indices is not None:
        kw_true = kw_true[indices]
        species_true = species_true[indices]
        ecotype_true = ecotype_true[indices]
        kw_pred = kw_pred[indices]
        species_pred = species_pred[indices]
        ecotype_pred = ecotype_pred[indices]

    kw_f1 = f1_binary_positive(kw_true, kw_pred, KW_LABELS["kw"])
    species_macro_f1 = macro_f1_present_labels(species_true, species_pred)

    ecotype_mask = ecotype_true != IGNORE_INDEX
    if np.any(ecotype_mask):
        ecotype_srkw_tkw_f1 = macro_f1_fixed_labels(
            ecotype_true[ecotype_mask],
            ecotype_pred[ecotype_mask],
            labels=[ECOTYPE_LABELS["SRKW"], ECOTYPE_LABELS["TKW"]],
        )
    else:
        ecotype_srkw_tkw_f1 = 0.0

    combined_score = (
        0.4 * kw_f1
        + 0.3 * species_macro_f1
        + 0.3 * ecotype_srkw_tkw_f1
    )
    return {
        "combined_score": combined_score,
        "kw_f1": kw_f1,
        "species_macro_f1": species_macro_f1,
        "ecotype_srkw_tkw_f1": ecotype_srkw_tkw_f1,
    }


def make_dataset(frame: pd.DataFrame) -> DCLDEAudioDataset:
    """Build a DCLDEAudioDataset from a labeled manifest frame."""
    return DCLDEAudioDataset(frame.copy())


def predict_or_load_cache(
    *,
    model_label: str,
    model_name: str,
    manifest_frame: pd.DataFrame,
    cache_path: Path,
    batch_size: int,
    max_duration: float,
    dataloader_workers: int,
    device: str,
    reuse_cache: bool,
    mean_subtract: bool,
    high_pass_filter: bool,
    high_pass_cutoff_hz: float,
    high_pass_order: int,
) -> pd.DataFrame:
    """Load a prediction cache or run model inference and write one."""
    if reuse_cache and cache_path.exists():
        print(f"Loading {model_label} prediction cache: {cache_path}")
        cached = pd.read_csv(cache_path, low_memory=False)
        if len(cached) != len(manifest_frame):
            raise ValueError(
                f"{model_label} cache row count {len(cached):,} does not match "
                f"manifest row count {len(manifest_frame):,}: {cache_path}"
            )
        return cached

    model_source = resolve_model_source(model_name)
    print(f"\nLoading {model_label} feature extractor: {model_source}")
    feature_extractor = load_feature_extractor(model_source)
    print(f"Loading {model_label} model: {model_source}")
    model = load_training_model(
        model_name=model_source,
        dropout=0.0,
        kw_loss_weight=1.0,
        species_loss_weight=1.0,
        ecotype_loss_weight=1.0,
        freeze_backbone=False,
    )

    dataset = make_dataset(manifest_frame)
    collator = DCLDEAudioCollator(
        feature_extractor=feature_extractor,
        max_duration=max_duration,
        augmenter=None,
        mean_subtract=mean_subtract,
        high_pass_cutoff_hz=high_pass_cutoff_hz if high_pass_filter else None,
        high_pass_order=high_pass_order,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=dataloader_workers > 0,
    )

    print(f"Running {model_label} inference on {len(manifest_frame):,} validation clips...")
    predictions = run_predictions(model, dataloader, device)
    scored = add_prediction_columns(manifest_frame, predictions)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(cache_path, index=False)
    print(f"Saved {model_label} prediction cache: {cache_path}")
    return scored


def extract_prediction_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract label and prediction arrays from a scored cache frame."""
    required_columns = {
        "kw_labels",
        "species_labels",
        "ecotype_labels",
        "kw_pred",
        "species_pred",
        "ecotype_pred",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction cache missing required columns: {sorted(missing)}")
    return {
        "kw_true": frame["kw_labels"].to_numpy(dtype=int),
        "species_true": frame["species_labels"].to_numpy(dtype=int),
        "ecotype_true": frame["ecotype_labels"].to_numpy(dtype=int),
        "kw_pred": frame["kw_pred"].to_numpy(dtype=int),
        "species_pred": frame["species_pred"].to_numpy(dtype=int),
        "ecotype_pred": frame["ecotype_pred"].to_numpy(dtype=int),
    }


def bootstrap_deltas(
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run paired bootstrap row resampling and return metric deltas B - A."""
    n_rows = len(arrays_a["kw_true"])
    rng = np.random.default_rng(seed)
    deltas = {name: np.empty(n_bootstrap, dtype=np.float64) for name in METRIC_NAMES}

    iterator = range(n_bootstrap)
    for boot_idx in progress_iter(iterator, "Bootstrap samples", total=n_bootstrap):
        sample_indices = rng.integers(0, n_rows, size=n_rows)
        metrics_a = combined_metrics_from_arrays(**arrays_a, indices=sample_indices)
        metrics_b = combined_metrics_from_arrays(**arrays_b, indices=sample_indices)
        for name in METRIC_NAMES:
            deltas[name][boot_idx] = metrics_b[name] - metrics_a[name]
        if tqdm is None and (boot_idx + 1) % 1000 == 0:
            print(f"Processed {boot_idx + 1:,}/{n_bootstrap:,} bootstrap samples...")

    return deltas


def summarize_delta(delta: np.ndarray, observed_delta: float, confidence: float) -> dict[str, float]:
    """Return CI and paired-bootstrap directional statistics for one metric."""
    alpha = 1.0 - confidence
    lower = float(np.quantile(delta, alpha / 2.0))
    upper = float(np.quantile(delta, 1.0 - alpha / 2.0))
    probability_b_better = float(np.mean(delta > 0.0))
    probability_a_better = float(np.mean(delta < 0.0))
    two_sided_p = float(2.0 * min(probability_b_better, probability_a_better))
    two_sided_p = min(1.0, two_sided_p)
    return {
        "observed_delta": observed_delta,
        "ci_lower": lower,
        "ci_upper": upper,
        "probability_b_better": probability_b_better,
        "two_sided_p": two_sided_p,
    }


def print_metric_table(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    summaries: dict[str, dict[str, float]],
    confidence: float,
) -> None:
    """Print model metrics and paired-bootstrap uncertainty."""
    ci_label = f"{confidence * 100:.1f}% CI"
    print("\n" + "=" * 112)
    print("PAIRED BOOTSTRAP COMPARISON")
    print("=" * 112)
    print("Positive delta means challenger Model B is better than baseline Model A.")
    print()
    print(
        f"{'metric':24s} {'model_a':>10s} {'model_b':>10s} {'delta':>10s} "
        f"{ci_label:>25s} {'P(B>A)':>10s} {'p_two_sided':>12s}"
    )
    print("-" * 112)
    for name in METRIC_NAMES:
        summary = summaries[name]
        print(
            f"{name:24s} "
            f"{metrics_a[name]:10.4f} "
            f"{metrics_b[name]:10.4f} "
            f"{summary['observed_delta']:10.4f} "
            f"[{summary['ci_lower']:8.4f}, {summary['ci_upper']:8.4f}] "
            f"{summary['probability_b_better']:10.4f} "
            f"{summary['two_sided_p']:12.4f}"
        )
    print("=" * 112)


def save_bootstrap_summary(
    output_path: Path,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    summaries: dict[str, dict[str, float]],
) -> None:
    """Save one-row-per-metric bootstrap summary CSV."""
    rows = []
    for name in METRIC_NAMES:
        row = {
            "metric": name,
            "model_a": metrics_a[name],
            "model_b": metrics_b[name],
        }
        row.update(summaries[name])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved bootstrap summary: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two DCLDE multi-task AST models on paired 3-second validation rows "
            "using bootstrap confidence intervals for challenger-minus-baseline metrics."
        )
    )
    parser.add_argument("--model-a", "--model_a", required=True, help="Baseline model path or Hugging Face ID.")
    parser.add_argument("--model-b", "--model_b", required=True, help="Challenger model path or Hugging Face ID.")
    parser.add_argument("--manifest", required=True, help="Validation manifest CSV with clip_path labels.")
    parser.add_argument("--output-dir", "--output_dir", default="output/bootstrap_compare")
    parser.add_argument("--model-a-name", "--model_a_name", default="model_a")
    parser.add_argument("--model-b-name", "--model_b_name", default="model_b")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--dataloader-workers", "--dataloader_workers", type=int, default=2)
    parser.add_argument("--max-duration", "--max_duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument(
        "--mean-subtract",
        "--mean_subtract",
        action="store_true",
        help="Subtract each clip's waveform mean before AST feature extraction.",
    )
    parser.add_argument(
        "--high-pass-filter",
        "--high_pass_filter",
        action="store_true",
        help="Apply a Butterworth high-pass filter before AST feature extraction.",
    )
    parser.add_argument(
        "--high-pass-cutoff-hz",
        "--high_pass_cutoff_hz",
        type=float,
        default=50.0,
        help="High-pass cutoff frequency in Hz when --high-pass-filter is set (default: 50).",
    )
    parser.add_argument(
        "--high-pass-order",
        "--high_pass_order",
        type=int,
        default=4,
        help="Butterworth high-pass filter order when --high-pass-filter is set (default: 4).",
    )
    parser.add_argument("--n-bootstrap", "--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--drop-unknown-labels", "--drop_unknown_labels", action="store_true")
    parser.add_argument("--reuse-cache", "--reuse_cache", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.dataloader_workers < 0:
        raise ValueError("--dataloader-workers must be non-negative.")
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.high_pass_filter:
        nyquist = SAMPLE_RATE / 2.0
        if not 0.0 < args.high_pass_cutoff_hz < nyquist:
            raise ValueError(
                f"--high-pass-cutoff-hz must be between 0 and {nyquist}, "
                f"got {args.high_pass_cutoff_hz}"
            )
        if args.high_pass_order < 1:
            raise ValueError(f"--high-pass-order must be positive, got {args.high_pass_order}")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_path(args.manifest)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading validation manifest: {manifest_path}")
    manifest_frame = load_manifest_frame(
        str(manifest_path),
        drop_unknown_labels=args.drop_unknown_labels,
    )
    print(f"Validation rows: {len(manifest_frame):,}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Waveform mean subtraction: {'enabled' if args.mean_subtract else 'disabled'}")
    if args.high_pass_filter:
        print(
            "Waveform high-pass filter: "
            f"enabled, cutoff={args.high_pass_cutoff_hz:g} Hz, order={args.high_pass_order}"
        )
    else:
        print("Waveform high-pass filter: disabled")
    print(f"Bootstrap samples: {args.n_bootstrap:,}")

    cache_suffix = preprocessing_cache_suffix(
        mean_subtract=args.mean_subtract,
        high_pass_filter=args.high_pass_filter,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz,
        high_pass_order=args.high_pass_order,
    )
    cache_a = output_dir / f"{safe_model_name(args.model_a_name)}{cache_suffix}_prediction_cache.csv"
    cache_b = output_dir / f"{safe_model_name(args.model_b_name)}{cache_suffix}_prediction_cache.csv"

    scored_a = predict_or_load_cache(
        model_label=f"Model A ({args.model_a_name})",
        model_name=args.model_a,
        manifest_frame=manifest_frame,
        cache_path=cache_a,
        batch_size=args.batch_size,
        max_duration=args.max_duration,
        dataloader_workers=args.dataloader_workers,
        device=device,
        reuse_cache=args.reuse_cache,
        mean_subtract=args.mean_subtract,
        high_pass_filter=args.high_pass_filter,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz,
        high_pass_order=args.high_pass_order,
    )
    scored_b = predict_or_load_cache(
        model_label=f"Model B ({args.model_b_name})",
        model_name=args.model_b,
        manifest_frame=manifest_frame,
        cache_path=cache_b,
        batch_size=args.batch_size,
        max_duration=args.max_duration,
        dataloader_workers=args.dataloader_workers,
        device=device,
        reuse_cache=args.reuse_cache,
        mean_subtract=args.mean_subtract,
        high_pass_filter=args.high_pass_filter,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz,
        high_pass_order=args.high_pass_order,
    )

    arrays_a = extract_prediction_arrays(scored_a)
    arrays_b = extract_prediction_arrays(scored_b)

    for key in ("kw_true", "species_true", "ecotype_true"):
        if not np.array_equal(arrays_a[key], arrays_b[key]):
            raise ValueError(f"Model A and Model B caches are not label-aligned for {key}.")

    metrics_a = combined_metrics_from_arrays(**arrays_a)
    metrics_b = combined_metrics_from_arrays(**arrays_b)
    observed = {name: metrics_b[name] - metrics_a[name] for name in METRIC_NAMES}

    print("\nRunning paired bootstrap...")
    deltas = bootstrap_deltas(
        arrays_a=arrays_a,
        arrays_b=arrays_b,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    summaries = {
        name: summarize_delta(deltas[name], observed[name], args.confidence)
        for name in METRIC_NAMES
    }

    print_metric_table(metrics_a, metrics_b, summaries, args.confidence)
    save_bootstrap_summary(
        output_dir / "paired_bootstrap_summary.csv",
        metrics_a,
        metrics_b,
        summaries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
