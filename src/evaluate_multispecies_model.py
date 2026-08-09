#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Evaluate a saved DCLDE multi-task AST checkpoint on a validation manifest.

This script mirrors the validation metrics used by multispecies_train_model.py:
  - KW accuracy/F1
  - species accuracy/macro F1 and per-class F1
  - ecotype accuracy/macro F1 and per-class F1 for valid KW ecotype rows
  - combined_score

It evaluates one manifest row as one 3-second clip. Training augmentations are
not applied during evaluation, but deterministic preprocessing options such as
mean subtraction and high-pass filtering can be enabled to match training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, EvalPrediction

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from multispecies_train_model import (
    DEFAULT_DATALOADER_WORKERS,
    DEFAULT_MAX_DURATION,
    DCLDEAudioCollator,
    SAMPLE_RATE,
    compute_metrics,
    load_manifest,
    load_multitask_checkpoint_files,
    load_training_model,
    resolve_path,
)


def resolve_model_source(model_name: str) -> str:
    """Use local paths when present, otherwise leave Hugging Face IDs unchanged."""
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)
    repo_relative = resolve_path(model_name)
    if repo_relative.exists():
        return str(repo_relative)
    return model_name


def load_feature_extractor(model_name: str) -> Any:
    """Load a feature extractor from the checkpoint, falling back to its base AST."""
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


def progress_iter(dataloader: DataLoader, enabled: bool):
    """Wrap dataloader with tqdm when requested and installed."""
    if not enabled or tqdm is None:
        return dataloader
    return tqdm(dataloader, total=len(dataloader), desc="Validation", unit="batch")


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments that can otherwise fail late inside the dataloader."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.dataloader_workers < 0:
        raise ValueError("--dataloader-workers must be non-negative")
    if args.max_duration <= 0:
        raise ValueError("--max-duration must be positive")
    if args.high_pass_filter:
        nyquist = SAMPLE_RATE / 2.0
        if not 0.0 < args.high_pass_cutoff_hz < nyquist:
            raise ValueError(
                f"--high-pass-cutoff-hz must be between 0 and {nyquist:g}, "
                f"got {args.high_pass_cutoff_hz}"
            )
        if args.high_pass_order < 1:
            raise ValueError(f"--high-pass-order must be positive, got {args.high_pass_order}")


def run_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    show_progress: bool,
) -> tuple[dict[str, float], float]:
    """Run inference and return training-compatible metrics plus mean loss."""
    model.to(device)
    model.eval()

    kw_logits: list[np.ndarray] = []
    species_logits: list[np.ndarray] = []
    ecotype_logits: list[np.ndarray] = []
    kw_labels: list[np.ndarray] = []
    species_labels: list[np.ndarray] = []
    ecotype_labels: list[np.ndarray] = []
    loss_sum = 0.0
    example_count = 0

    with torch.inference_mode():
        for step, batch in enumerate(progress_iter(dataloader, show_progress), start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = int(batch["input_values"].shape[0])
            example_count += batch_size

            loss = outputs.get("loss")
            if loss is not None:
                loss_sum += float(loss.detach().cpu()) * batch_size

            batch_kw_logits, batch_species_logits, batch_ecotype_logits = outputs["logits"]
            kw_logits.append(batch_kw_logits.detach().cpu().numpy())
            species_logits.append(batch_species_logits.detach().cpu().numpy())
            ecotype_logits.append(batch_ecotype_logits.detach().cpu().numpy())
            kw_labels.append(batch["kw_labels"].detach().cpu().numpy())
            species_labels.append(batch["species_labels"].detach().cpu().numpy())
            ecotype_labels.append(batch["ecotype_labels"].detach().cpu().numpy())

            if (not show_progress or tqdm is None) and step % 25 == 0:
                print(f"Processed {example_count:,} validation rows...")

    eval_prediction = EvalPrediction(
        predictions=(
            np.concatenate(kw_logits, axis=0),
            np.concatenate(species_logits, axis=0),
            np.concatenate(ecotype_logits, axis=0),
        ),
        label_ids=(
            np.concatenate(kw_labels, axis=0),
            np.concatenate(species_labels, axis=0),
            np.concatenate(ecotype_labels, axis=0),
        ),
    )
    metrics = compute_metrics(eval_prediction)
    mean_loss = loss_sum / example_count if example_count else 0.0
    return metrics, mean_loss


def print_metrics(metrics: dict[str, float], eval_loss: float, row_count: int) -> None:
    """Print metrics in a compact validation-step style."""
    print("\n" + "=" * 72)
    print("VALIDATION METRICS")
    print("=" * 72)
    print(f"{'eval_loss':28s}: {eval_loss:.6f}")
    print(f"{'eval_rows':28s}: {row_count}")
    for key in sorted(metrics):
        print(f"{key:28s}: {metrics[key]:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a DCLDE multi-task AST checkpoint on a validation manifest."
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        required=True,
        help="Local checkpoint directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--val-manifest",
        "--val_manifest",
        "--manifest",
        required=True,
        help="Validation manifest CSV with clip_path, ClassSpecies, and Ecotype columns.",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument(
        "--dataloader-workers",
        "--dataloader_workers",
        type=int,
        default=DEFAULT_DATALOADER_WORKERS,
        help="Worker processes for lazy audio loading.",
    )
    parser.add_argument("--max-duration", "--max_duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--drop-unknown-labels", "--drop_unknown_labels", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda or cpu. Defaults to cuda when available.",
    )
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
        help="High-pass cutoff frequency in Hz when --high-pass-filter is set.",
    )
    parser.add_argument(
        "--high-pass-order",
        "--high_pass_order",
        type=int,
        default=4,
        help="Butterworth high-pass filter order when --high-pass-filter is set.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    args = parser.parse_args()
    validate_args(args)

    model_source = resolve_model_source(args.model_name)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading validation manifest: {resolve_path(args.val_manifest)}")
    validation_dataset = load_manifest(
        args.val_manifest,
        drop_unknown_labels=args.drop_unknown_labels,
    )
    print(f"Validation rows: {len(validation_dataset):,}")

    print(f"Loading feature extractor: {model_source}")
    feature_extractor = load_feature_extractor(model_source)

    print(
        "Preprocessing: "
        f"mean_subtract={'enabled' if args.mean_subtract else 'disabled'}, "
        f"high_pass_filter={'enabled' if args.high_pass_filter else 'disabled'}"
    )
    if args.high_pass_filter:
        print(
            "High-pass settings: "
            f"cutoff={args.high_pass_cutoff_hz:g} Hz, order={args.high_pass_order}"
        )

    collator = DCLDEAudioCollator(
        feature_extractor=feature_extractor,
        max_duration=args.max_duration,
        augmenter=None,
        mean_subtract=args.mean_subtract,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz if args.high_pass_filter else None,
        high_pass_order=args.high_pass_order,
    )
    dataloader = DataLoader(
        validation_dataset,
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

    print(f"Running validation on {device}...")
    metrics, eval_loss = run_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        show_progress=not args.no_progress,
    )
    print_metrics(metrics, eval_loss, len(validation_dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
