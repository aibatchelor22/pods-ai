#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Generate validation-set confusion matrices for the DCLDE multi-task AST model.

Outputs one confusion matrix per classification head:
  - KW detection: not_kw, kw
  - species: background, KW, HW, AB
  - ecotype: NRKW, SRKW, OKW, SAR, TKW

The ecotype matrix is computed only for validation rows with a real ecotype
label. Non-KW rows and KW rows with missing ecotype are ignored for that head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from multispecies_train_model import (
    DEFAULT_MAX_DURATION,
    DCLDEAudioCollator,
    ECOTYPE_ID2LABEL,
    IGNORE_INDEX,
    KW_ID2LABEL,
    REPO_ROOT,
    SPECIES_ID2LABEL,
    load_manifest,
    load_multitask_checkpoint_files,
    load_training_model,
)


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
    """Load feature extractor from the checkpoint repo/dir, falling back to base AST."""
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


def id2label_list(id2label: dict[int, str]) -> list[str]:
    """Return labels ordered by class ID."""
    return [id2label[class_id] for class_id in sorted(id2label)]


def save_matrix_csv(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    normalize: bool = False,
) -> None:
    """Save a confusion matrix as a labeled CSV."""
    values = matrix.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    frame = pd.DataFrame(values, index=labels, columns=labels)
    frame.index.name = "true_label"
    frame.to_csv(output_path)


def save_matrix_plot(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    """Save a confusion matrix plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG confusion matrix plots.")
        return

    values = matrix
    display_format = "d"
    if normalize:
        values = matrix.astype(float)
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
        display_format = ".2f"

    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=values, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format=display_format, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_head_outputs(
    output_dir: Path,
    head_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: dict[int, str],
) -> dict[str, Any]:
    """Save count/normalized confusion matrices and a classification report."""
    class_ids = sorted(id2label)
    labels = id2label_list(id2label)
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)

    counts_path = output_dir / f"{head_name}_confusion_matrix_counts.csv"
    normalized_path = output_dir / f"{head_name}_confusion_matrix_normalized.csv"
    plot_path = output_dir / f"{head_name}_confusion_matrix_counts.png"
    normalized_plot_path = output_dir / f"{head_name}_confusion_matrix_normalized.png"
    report_path = output_dir / f"{head_name}_classification_report.csv"

    save_matrix_csv(matrix, labels, counts_path)
    save_matrix_csv(matrix, labels, normalized_path, normalize=True)
    save_matrix_plot(matrix, labels, plot_path, f"{head_name} confusion matrix")
    save_matrix_plot(
        matrix,
        labels,
        normalized_plot_path,
        f"{head_name} normalized confusion matrix",
        normalize=True,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(report_path)

    return {
        "head": head_name,
        "n_samples": int(len(y_true)),
        "counts_csv": str(counts_path),
        "normalized_csv": str(normalized_path),
        "report_csv": str(report_path),
    }


def run_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict[str, np.ndarray]:
    """Run batched inference and collect labels/predictions for each head."""
    model.to(device)
    model.eval()

    kw_true: list[np.ndarray] = []
    kw_pred: list[np.ndarray] = []
    species_true: list[np.ndarray] = []
    species_pred: list[np.ndarray] = []
    ecotype_true: list[np.ndarray] = []
    ecotype_pred: list[np.ndarray] = []

    with torch.inference_mode():
        for step, batch in enumerate(dataloader, start=1):
            labels = {
                "kw": batch.pop("kw_labels"),
                "species": batch.pop("species_labels"),
                "ecotype": batch.pop("ecotype_labels"),
            }
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            kw_logits, species_logits, ecotype_logits = outputs["logits"]

            kw_true.append(labels["kw"].cpu().numpy())
            kw_pred.append(torch.argmax(kw_logits, dim=-1).cpu().numpy())
            species_true.append(labels["species"].cpu().numpy())
            species_pred.append(torch.argmax(species_logits, dim=-1).cpu().numpy())

            ecotype_labels = labels["ecotype"].cpu().numpy()
            ecotype_predictions = torch.argmax(ecotype_logits, dim=-1).cpu().numpy()
            ecotype_mask = ecotype_labels != IGNORE_INDEX
            if np.any(ecotype_mask):
                ecotype_true.append(ecotype_labels[ecotype_mask])
                ecotype_pred.append(ecotype_predictions[ecotype_mask])

            if step % 25 == 0:
                print(f"Processed {step * dataloader.batch_size:,} validation rows...")

    empty = np.array([], dtype=int)
    return {
        "kw_true": np.concatenate(kw_true) if kw_true else empty,
        "kw_pred": np.concatenate(kw_pred) if kw_pred else empty,
        "species_true": np.concatenate(species_true) if species_true else empty,
        "species_pred": np.concatenate(species_pred) if species_pred else empty,
        "ecotype_true": np.concatenate(ecotype_true) if ecotype_true else empty,
        "ecotype_pred": np.concatenate(ecotype_pred) if ecotype_pred else empty,
    }


def save_predictions_csv(
    predictions: dict[str, np.ndarray],
    clip_paths: list[str],
    output_path: Path,
) -> None:
    """Save per-row KW/species predictions and ecotype predictions for labeled ecotype rows."""
    row_count = len(predictions["kw_true"])
    frame = pd.DataFrame(
        {
            "clip_path": clip_paths[:row_count],
            "kw_true": predictions["kw_true"],
            "kw_pred": predictions["kw_pred"],
            "kw_true_label": [KW_ID2LABEL[int(value)] for value in predictions["kw_true"]],
            "kw_pred_label": [KW_ID2LABEL[int(value)] for value in predictions["kw_pred"]],
            "species_true": predictions["species_true"],
            "species_pred": predictions["species_pred"],
            "species_true_label": [
                SPECIES_ID2LABEL[int(value)] for value in predictions["species_true"]
            ],
            "species_pred_label": [
                SPECIES_ID2LABEL[int(value)] for value in predictions["species_pred"]
            ],
        }
    )
    frame.to_csv(output_path, index=False)

    ecotype_count = len(predictions["ecotype_true"])
    if ecotype_count != row_count:
        print(
            "Saved per-row predictions for KW/species. Ecotype predictions are summarized "
            "in the ecotype confusion matrix because unlabeled ecotype rows are ignored."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices for a DCLDE multi-task AST validation manifest."
    )
    parser.add_argument("--model-name", required=True, help="Local model dir or Hugging Face model ID.")
    parser.add_argument("--val-manifest", required=True, help="Validation manifest CSV.")
    parser.add_argument(
        "--output-dir",
        default="output/multispecies_confusion_matrices",
        help="Directory for confusion matrix CSV/PNG outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--drop-unknown-labels", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda, cpu. Defaults to cuda when available.",
    )
    args = parser.parse_args()

    model_source = resolve_model_source(args.model_name)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading validation manifest: {args.val_manifest}")
    dataset = load_manifest(args.val_manifest, drop_unknown_labels=args.drop_unknown_labels)
    print(f"Validation rows: {len(dataset):,}")

    print(f"Loading feature extractor from: {model_source}")
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

    print(f"Running validation inference on {device}...")
    predictions = run_predictions(model, dataloader, device)

    summaries = [
        save_head_outputs(
            output_dir,
            "kw",
            predictions["kw_true"],
            predictions["kw_pred"],
            KW_ID2LABEL,
        ),
        save_head_outputs(
            output_dir,
            "species",
            predictions["species_true"],
            predictions["species_pred"],
            SPECIES_ID2LABEL,
        ),
    ]
    if len(predictions["ecotype_true"]):
        summaries.append(
            save_head_outputs(
                output_dir,
                "ecotype",
                predictions["ecotype_true"],
                predictions["ecotype_pred"],
                ECOTYPE_ID2LABEL,
            )
        )
    else:
        print("No labeled ecotype rows found; skipping ecotype confusion matrix.")

    save_predictions_csv(predictions, dataset.clip_paths, output_dir / "validation_predictions.csv")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print(f"Saved confusion matrix outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
