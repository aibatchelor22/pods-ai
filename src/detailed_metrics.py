#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Print detailed validation metrics for a DCLDE multi-task AST checkpoint.

The script evaluates one 3-second clip per manifest row and prints:
  - overall KW/species/ecotype metrics using the same combined score as training
  - combined F1 grouped by selected manifest columns
  - BKG vs UndBio false-positive summaries
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from multispecies_train_model import (
    BACKGROUND_SPECIES,
    DEFAULT_MAX_DURATION,
    DCLDEAudioCollator,
    DCLDEAudioDataset,
    ECOTYPE_ID2LABEL,
    ECOTYPE_LABELS,
    IGNORE_INDEX,
    KNOWN_SPECIES,
    KW_ID2LABEL,
    KW_LABELS,
    REPO_ROOT,
    SPECIES_ID2LABEL,
    SPECIES_LABELS,
    load_multitask_checkpoint_files,
    load_training_model,
    map_ecotype_label,
    map_kw_label,
    map_species_label,
    normalize_label,
)


GROUP_COLUMNS = ["Provider", "Dataset", "Soundfile", "ClassSpecies", "Ecotype"]


def resolve_path(path: str) -> Path:
    """Resolve absolute paths as-is and repo-relative paths under REPO_ROOT."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def resolve_model_source(model_name: str) -> str:
    """Use local repo-relative model dirs when present, otherwise keep Hub IDs."""
    path_obj = Path(model_name)
    if path_obj.exists():
        return str(path_obj)
    repo_relative = resolve_path(model_name)
    if repo_relative.exists():
        return str(repo_relative)
    return model_name


def load_feature_extractor(model_name: str) -> Any:
    """Load feature extractor from checkpoint repo/dir, falling back to base AST."""
    try:
        return AutoFeatureExtractor.from_pretrained(model_name)
    except Exception as first_error:
        checkpoint = load_multitask_checkpoint_files(model_name)
        if checkpoint is None:
            raise first_error
        metadata, _ = checkpoint
        base_model = metadata.get("base_model")
        if not base_model:
            raise first_error
        print(f"Feature extractor not found in {model_name}; using base model {base_model}.")
        return AutoFeatureExtractor.from_pretrained(base_model)


def load_manifest_frame(manifest_path: str, drop_unknown_labels: bool = False) -> pd.DataFrame:
    """Load manifest rows and append numeric labels using training-script mappings."""
    path = resolve_path(manifest_path)
    df = pd.read_csv(path, low_memory=False)

    required_columns = {"clip_path", "ClassSpecies", "Ecotype"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ClassSpecies"] = df["ClassSpecies"].map(normalize_label)
    df["Ecotype"] = df["Ecotype"].map(normalize_label)

    known_species_mask = df["ClassSpecies"].isin(KNOWN_SPECIES)
    if not known_species_mask.all():
        unknown_counts = df.loc[~known_species_mask, "ClassSpecies"].value_counts(dropna=False)
        if not drop_unknown_labels:
            raise ValueError(
                "Unknown ClassSpecies values found. Pass --drop-unknown-labels to skip them: "
                f"{unknown_counts.to_dict()}"
            )
        print(f"Dropping unknown ClassSpecies rows: {unknown_counts.to_dict()}")
        df = df.loc[known_species_mask].copy()

    known_ecotype_mask = (
        (df["ClassSpecies"] != "KW")
        | (df["Ecotype"] == "")
        | df["Ecotype"].isin(ECOTYPE_LABELS)
    )
    if not known_ecotype_mask.all():
        unknown_counts = df.loc[~known_ecotype_mask, "Ecotype"].value_counts(dropna=False)
        if not drop_unknown_labels:
            raise ValueError(
                "Unknown KW Ecotype values found. Pass --drop-unknown-labels to skip them: "
                f"{unknown_counts.to_dict()}"
            )
        print(f"Dropping unknown KW Ecotype rows: {unknown_counts.to_dict()}")
        df = df.loc[known_ecotype_mask].copy()

    df["kw_labels"] = df["ClassSpecies"].map(map_kw_label).astype(int)
    df["species_labels"] = df["ClassSpecies"].map(map_species_label).astype(int)
    df["ecotype_labels"] = [
        map_ecotype_label(class_species, ecotype)
        for class_species, ecotype in zip(df["ClassSpecies"], df["Ecotype"])
    ]
    return df.reset_index(drop=True)


def f1_score_safe(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Return F1 with zero_division fixed to 0."""
    if len(y_true) == 0:
        return 0.0
    return float(
        precision_recall_fscore_support(
            y_true,
            y_pred,
            zero_division=0,
            **kwargs,
        )[2]
    )


def class_f1(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> float:
    """Return one-vs-rest F1 for one class."""
    if len(y_true) == 0:
        return 0.0
    scores = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[class_id],
        average=None,
        zero_division=0,
    )[2]
    return float(scores[0]) if len(scores) else 0.0


def compute_metric_dict(frame: pd.DataFrame) -> dict[str, float]:
    """Compute the same overall metrics used by multispecies_train_model.py."""
    kw_true = frame["kw_labels"].to_numpy(dtype=int)
    kw_pred = frame["kw_pred"].to_numpy(dtype=int)
    species_true = frame["species_labels"].to_numpy(dtype=int)
    species_pred = frame["species_pred"].to_numpy(dtype=int)
    ecotype_true_all = frame["ecotype_labels"].to_numpy(dtype=int)
    ecotype_pred_all = frame["ecotype_pred"].to_numpy(dtype=int)

    metrics: dict[str, float] = {
        "n": float(len(frame)),
        "kw_accuracy": float(accuracy_score(kw_true, kw_pred)) if len(frame) else 0.0,
        "kw_f1": f1_score_safe(
            kw_true,
            kw_pred,
            average="binary",
            pos_label=KW_LABELS["kw"],
        ),
        "species_accuracy": float(accuracy_score(species_true, species_pred)) if len(frame) else 0.0,
        "species_macro_f1": f1_score_safe(species_true, species_pred, average="macro"),
    }

    for class_id, class_name in SPECIES_ID2LABEL.items():
        metrics[f"species_f1_{class_name}"] = class_f1(species_true, species_pred, class_id)

    ecotype_mask = ecotype_true_all != IGNORE_INDEX
    if np.any(ecotype_mask):
        ecotype_true = ecotype_true_all[ecotype_mask]
        ecotype_pred = ecotype_pred_all[ecotype_mask]
        metrics["ecotype_n"] = float(len(ecotype_true))
        metrics["ecotype_accuracy"] = float(accuracy_score(ecotype_true, ecotype_pred))
        metrics["ecotype_macro_f1"] = f1_score_safe(ecotype_true, ecotype_pred, average="macro")
        for class_id, class_name in ECOTYPE_ID2LABEL.items():
            metrics[f"ecotype_f1_{class_name}"] = class_f1(ecotype_true, ecotype_pred, class_id)
        metrics["ecotype_srkw_tkw_f1"] = f1_score_safe(
            ecotype_true,
            ecotype_pred,
            average="macro",
            labels=[ECOTYPE_LABELS["SRKW"], ECOTYPE_LABELS["TKW"]],
        )
    else:
        metrics["ecotype_n"] = 0.0
        metrics["ecotype_accuracy"] = 0.0
        metrics["ecotype_macro_f1"] = 0.0
        metrics["ecotype_srkw_tkw_f1"] = 0.0

    metrics["combined_score"] = (
        0.4 * metrics["kw_f1"]
        + 0.3 * metrics["species_macro_f1"]
        + 0.3 * metrics["ecotype_srkw_tkw_f1"]
    )
    return metrics


def run_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict[str, np.ndarray]:
    """Run batched inference and collect predictions/probabilities for all rows."""
    model.to(device)
    model.eval()

    kw_pred: list[np.ndarray] = []
    species_pred: list[np.ndarray] = []
    ecotype_pred: list[np.ndarray] = []
    kw_probs: list[np.ndarray] = []
    species_probs: list[np.ndarray] = []
    ecotype_probs: list[np.ndarray] = []

    with torch.inference_mode():
        for step, batch in enumerate(dataloader, start=1):
            batch.pop("kw_labels")
            batch.pop("species_labels")
            batch.pop("ecotype_labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            kw_logits, species_logits, ecotype_logits = outputs["logits"]

            kw_prob = torch.softmax(kw_logits, dim=-1).cpu().numpy()
            species_prob = torch.softmax(species_logits, dim=-1).cpu().numpy()
            ecotype_prob = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()

            kw_probs.append(kw_prob)
            species_probs.append(species_prob)
            ecotype_probs.append(ecotype_prob)
            kw_pred.append(np.argmax(kw_prob, axis=1))
            species_pred.append(np.argmax(species_prob, axis=1))
            ecotype_pred.append(np.argmax(ecotype_prob, axis=1))

            if step % 25 == 0:
                print(f"Processed {step * dataloader.batch_size:,} validation rows...")

    return {
        "kw_pred": np.concatenate(kw_pred),
        "species_pred": np.concatenate(species_pred),
        "ecotype_pred": np.concatenate(ecotype_pred),
        "kw_probs": np.concatenate(kw_probs),
        "species_probs": np.concatenate(species_probs),
        "ecotype_probs": np.concatenate(ecotype_probs),
    }


def add_prediction_columns(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Attach prediction IDs, labels, and selected probabilities to the manifest frame."""
    output = frame.copy()
    output["kw_pred"] = predictions["kw_pred"].astype(int)
    output["species_pred"] = predictions["species_pred"].astype(int)
    output["ecotype_pred"] = predictions["ecotype_pred"].astype(int)
    output["kw_true_label"] = output["kw_labels"].map(KW_ID2LABEL)
    output["kw_pred_label"] = output["kw_pred"].map(KW_ID2LABEL)
    output["species_true_label"] = output["species_labels"].map(SPECIES_ID2LABEL)
    output["species_pred_label"] = output["species_pred"].map(SPECIES_ID2LABEL)
    output["ecotype_true_label"] = output["ecotype_labels"].map(
        lambda value: "" if int(value) == IGNORE_INDEX else ECOTYPE_ID2LABEL[int(value)]
    )
    output["ecotype_pred_label"] = output["ecotype_pred"].map(ECOTYPE_ID2LABEL)
    output["kw_prob_kw"] = predictions["kw_probs"][:, KW_LABELS["kw"]]
    output["species_prob_background"] = predictions["species_probs"][:, SPECIES_LABELS["background"]]
    output["species_prob_KW"] = predictions["species_probs"][:, SPECIES_LABELS["KW"]]
    output["species_prob_HW"] = predictions["species_probs"][:, SPECIES_LABELS["HW"]]
    output["species_prob_AB"] = predictions["species_probs"][:, SPECIES_LABELS["AB"]]
    output["ecotype_prob_SRKW"] = predictions["ecotype_probs"][:, ECOTYPE_LABELS["SRKW"]]
    output["ecotype_prob_TKW"] = predictions["ecotype_probs"][:, ECOTYPE_LABELS["TKW"]]
    return output


def format_rate(value: Optional[float]) -> str:
    """Format a metric rate for tables."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def print_overall_metrics(metrics: dict[str, float]) -> None:
    """Print overall metrics."""
    print("\n" + "=" * 78)
    print("OVERALL METRICS")
    print("=" * 78)
    for key in (
        "combined_score",
        "kw_accuracy",
        "kw_f1",
        "species_accuracy",
        "species_macro_f1",
        "ecotype_accuracy",
        "ecotype_macro_f1",
        "ecotype_srkw_tkw_f1",
    ):
        print(f"{key:24s}: {metrics[key]:.4f}")
    print(f"{'n':24s}: {int(metrics['n'])}")
    print(f"{'ecotype_n':24s}: {int(metrics['ecotype_n'])}")


def print_grouped_combined_f1(frame: pd.DataFrame, group_column: str, min_count: int) -> None:
    """Print combined score by a manifest column."""
    if group_column not in frame.columns:
        print(f"\n{group_column}: column not found; skipping.")
        return

    rows = []
    group_values = frame[group_column].fillna("").astype(str)
    for value in sorted(group_values.unique()):
        mask = group_values == value
        group_frame = frame.loc[mask]
        if len(group_frame) < min_count:
            continue
        metrics = compute_metric_dict(group_frame)
        rows.append(
            {
                "value": value if value else "<blank>",
                "n": int(metrics["n"]),
                "ecotype_n": int(metrics["ecotype_n"]),
                "combined_score": metrics["combined_score"],
                "kw_f1": metrics["kw_f1"],
                "species_macro_f1": metrics["species_macro_f1"],
                "ecotype_srkw_tkw_f1": metrics["ecotype_srkw_tkw_f1"],
            }
        )

    print("\n" + "=" * 78)
    print(f"COMBINED F1 BY {group_column}")
    print("=" * 78)
    if not rows:
        print(f"No groups with at least {min_count} rows.")
        return

    print(
        f"{group_column:<28} {'n':>7} {'eco_n':>7} {'combined':>10} "
        f"{'kw_f1':>8} {'species':>8} {'srkw_tkw':>9}"
    )
    print("-" * 78)
    for row in sorted(rows, key=lambda item: item["combined_score"], reverse=True):
        print(
            f"{row['value']:<28.28} {row['n']:>7} {row['ecotype_n']:>7} "
            f"{row['combined_score']:>10.4f} {row['kw_f1']:>8.4f} "
            f"{row['species_macro_f1']:>8.4f} {row['ecotype_srkw_tkw_f1']:>9.4f}"
        )


def print_bkg_undbio_false_positives(frame: pd.DataFrame) -> None:
    """Print false-positive summaries for BKG and UndBio rows."""
    print("\n" + "=" * 78)
    print("BKG vs UndBio FALSE POSITIVES")
    print("=" * 78)
    print(
        f"{'ClassSpecies':<14} {'n':>7} {'KW FP':>8} {'KW FP%':>8} "
        f"{'Species FP':>10} {'Species FP%':>11} {'Either FP':>10} {'Either FP%':>10}"
    )
    print("-" * 78)

    for label in ("BKG", "UndBio"):
        subset = frame.loc[frame["ClassSpecies"] == label]
        n = len(subset)
        if n == 0:
            print(f"{label:<14} {0:>7} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>10}")
            continue

        kw_fp = int((subset["kw_pred"] == KW_LABELS["kw"]).sum())
        species_fp = int((subset["species_pred"] != SPECIES_LABELS["background"]).sum())
        either_fp = int(
            (
                (subset["kw_pred"] == KW_LABELS["kw"])
                | (subset["species_pred"] != SPECIES_LABELS["background"])
            ).sum()
        )
        print(
            f"{label:<14} {n:>7} {kw_fp:>8} {kw_fp / n:>8.1%} "
            f"{species_fp:>10} {species_fp / n:>11.1%} {either_fp:>10} {either_fp / n:>10.1%}"
        )

        predicted_counts = Counter(subset["species_pred_label"])
        print(
            " " * 16
            + "species predictions: "
            + ", ".join(f"{name}={count}" for name, count in sorted(predicted_counts.items()))
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print detailed metrics for a multi-species AST checkpoint and validation manifest."
    )
    parser.add_argument("--model-name", required=True, help="Local checkpoint dir or Hugging Face model ID.")
    parser.add_argument("--manifest", required=True, help="Validation manifest CSV.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--drop-unknown-labels", action="store_true")
    parser.add_argument("--min-group-count", type=int, default=1)
    parser.add_argument(
        "--group-columns",
        default=",".join(GROUP_COLUMNS),
        help="Comma-separated manifest columns for grouped combined F1.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda, cpu. Defaults to cuda when available.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.dataloader_workers < 0:
        raise ValueError("--dataloader-workers must be non-negative")
    if args.min_group_count <= 0:
        raise ValueError("--min-group-count must be positive")

    model_source = resolve_model_source(args.model_name)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading manifest: {resolve_path(args.manifest)}")
    frame = load_manifest_frame(args.manifest, drop_unknown_labels=args.drop_unknown_labels)
    dataset = DCLDEAudioDataset(frame)
    print(f"Rows: {len(frame):,}")

    print(f"Loading feature extractor: {model_source}")
    feature_extractor = load_feature_extractor(model_source)
    collator = DCLDEAudioCollator(
        feature_extractor=feature_extractor,
        max_duration=args.max_duration,
        augmenter=None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.dataloader_workers > 0,
    )

    print(f"Loading model: {model_source}")
    model = load_training_model(
        model_name=model_source,
        dropout=0.0,
        kw_loss_weight=1.0,
        species_loss_weight=1.0,
        ecotype_loss_weight=1.0,
        freeze_backbone=False,
    )

    print(f"Running inference on {device}...")
    predictions = run_predictions(model, dataloader, device)
    scored_frame = add_prediction_columns(frame, predictions)

    print_overall_metrics(compute_metric_dict(scored_frame))
    for column in [column.strip() for column in args.group_columns.split(",") if column.strip()]:
        print_grouped_combined_f1(scored_frame, column, args.min_group_count)
    print_bkg_undbio_false_positives(scored_frame)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
