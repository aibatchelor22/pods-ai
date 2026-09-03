#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Train frozen-backbone trigger, species, and ecotype heads for DCLDE.

The script loads a strictly validated ``MultiTaskASTForDCLDE`` checkpoint,
freezes its AST backbone, extracts one pooled embedding for every 3-second
clip, and trains three lightweight output branches:

* any target cetacean: background vs any of KW/HW/AB (configurable)
* species: background, KW, HW, AB
* killer-whale ecotype: NRKW, SRKW, OKW, SAR, TKW

Species and ecotype branches initialize from the supplied checkpoint. The new
trigger branch is initialized randomly. Cached embeddings make additional head
epochs inexpensive and ensure that this experiment cannot alter the backbone.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from transformers import AutoFeatureExtractor

from multispecies_train_model import (
    DCLDEAudioCollator,
    ECOTYPE_ID2LABEL,
    ECOTYPE_LABELS,
    IGNORE_INDEX,
    SPECIES_ID2LABEL,
    SPECIES_LABELS,
    load_manifest,
    load_multitask_checkpoint_files,
    load_training_model,
    parse_class_weights,
)


HEAD_LABELS = {"background": 0, "any_target_cetacean": 1}
HEAD_ID2LABEL = {value: key for key, value in HEAD_LABELS.items()}


class FrozenBackboneMultitaskHead(nn.Module):
    """Three classification branches operating on a pooled AST embedding."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.trigger_classifier = nn.Linear(hidden_size, len(HEAD_LABELS))
        self.species_classifier = nn.Linear(hidden_size, len(SPECIES_LABELS))
        self.ecotype_classifier = nn.Linear(hidden_size, len(ECOTYPE_LABELS))

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.dropout(embeddings)
        return (
            self.trigger_classifier(features),
            self.species_classifier(features),
            self.ecotype_classifier(features),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an any-target-cetacean trigger plus species/ecotype heads "
            "on cached embeddings from a frozen DCLDE AST checkpoint."
        )
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="../output/any_whale_multitask_head")
    parser.add_argument("--train-dataset-root")
    parser.add_argument("--val-dataset-root")
    parser.add_argument("--drop-unknown-labels", action="store_true")
    parser.add_argument(
        "--trigger-species",
        nargs="+",
        choices=["KW", "HW", "AB"],
        default=["KW", "HW", "AB"],
        help="Species treated as positive by the binary trigger.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--existing-head-learning-rate",
        type=float,
        default=1e-4,
        help=(
            "Lower learning rate for the checkpoint-initialized species/ecotype "
            "branches; use 0 to keep those two branches fixed."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--head-batch-size", type=int, default=2048)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--preprocessing-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)
    parser.add_argument(
        "--reuse-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching cached embeddings (default: true).",
    )
    parser.add_argument(
        "--automatic-trigger-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance background/positive trigger loss from the training counts.",
    )
    parser.add_argument(
        "--trigger-class-weights",
        help="Override trigger weights, e.g. background=1,any_target_cetacean=2.",
    )
    parser.add_argument(
        "--species-class-weights",
        help="Optional species label=value weights.",
    )
    parser.add_argument(
        "--ecotype-class-weights",
        help="Optional ecotype label=value weights.",
    )
    parser.add_argument("--trigger-loss-weight", type=float, default=1.0)
    parser.add_argument("--species-loss-weight", type=float, default=1.0)
    parser.add_argument("--ecotype-loss-weight", type=float, default=1.0)

    parser.add_argument(
        "--mean-subtract",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint mean subtraction; otherwise use saved setting.",
    )
    parser.add_argument(
        "--high-pass-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint high-pass setting; otherwise use saved setting.",
    )
    parser.add_argument("--high-pass-cutoff-hz", type=float)
    parser.add_argument("--high-pass-order", type=int)
    parser.add_argument("--max-duration", type=float, default=3.0)
    parser.add_argument(
        "--save-backbone-state",
        action="store_true",
        help="Also save the large original multitask backbone state with the head bundle.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.embedding_batch_size < 1 or args.head_batch_size < 1:
        parser.error("Epoch and batch-size values must be positive.")
    if args.learning_rate <= 0 or args.existing_head_learning_rate < 0:
        parser.error("Trigger LR must be positive and existing-head LR non-negative.")
    if args.weight_decay < 0:
        parser.error("Weight decay must be non-negative.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative.")
    for name in ("trigger_loss_weight", "species_loss_weight", "ecotype_loss_weight"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_preprocessing(
    model_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = load_multitask_checkpoint_files(model_name)
    metadata = checkpoint[0] if checkpoint is not None else {}
    augmentation = metadata.get("augmentation", {})
    preprocessing = {
        "mean_subtract": bool(augmentation.get("mean_subtract", False)),
        "high_pass_filter": bool(augmentation.get("high_pass_filter", False)),
        "high_pass_cutoff_hz": float(augmentation.get("high_pass_cutoff_hz", 50.0)),
        "high_pass_order": int(augmentation.get("high_pass_order", 4)),
    }
    identity = (
        file_signature(str(checkpoint[1]))
        if checkpoint is not None
        else {"model_name": model_name, "checkpoint_weights": None}
    )
    return preprocessing, identity


def resolved_preprocessing(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings, checkpoint_identity = checkpoint_preprocessing(args.model_name)
    if args.mean_subtract is not None:
        settings["mean_subtract"] = args.mean_subtract
    if args.high_pass_filter is not None:
        settings["high_pass_filter"] = args.high_pass_filter
    if args.high_pass_cutoff_hz is not None:
        settings["high_pass_cutoff_hz"] = args.high_pass_cutoff_hz
    if args.high_pass_order is not None:
        settings["high_pass_order"] = args.high_pass_order
    settings["max_duration"] = args.max_duration
    return settings, checkpoint_identity


def selected_subset(dataset: Any, maximum: Optional[int], seed: int) -> Any:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("Maximum file counts must be positive.")
    indices = np.random.default_rng(seed).choice(len(dataset), size=maximum, replace=False)
    return Subset(dataset, sorted(indices.tolist()))


def file_signature(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_signature(
    split: str,
    manifest: str,
    dataset_root: Optional[str],
    checkpoint_identity: dict[str, Any],
    preprocessing: dict[str, Any],
    max_files: Optional[int],
    seed: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "manifest": file_signature(manifest),
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root else None,
        "checkpoint_identity": checkpoint_identity,
        "preprocessing": preprocessing,
        "max_files": max_files,
        "seed": seed,
    }


def cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{split}_embeddings.npy",
        cache_dir / f"{split}_labels.npz",
        cache_dir / f"{split}_cache.json",
    )


def load_matching_cache(
    cache_dir: Path,
    split: str,
    signature: dict[str, Any],
) -> Optional[tuple[np.ndarray, dict[str, np.ndarray]]]:
    embedding_path, label_path, metadata_path = cache_paths(cache_dir, split)
    if not (embedding_path.exists() and label_path.exists() and metadata_path.exists()):
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("signature") != signature:
        return None
    embeddings = np.load(embedding_path, mmap_mode="r")
    label_file = np.load(label_path, allow_pickle=False)
    labels = {name: label_file[name] for name in label_file.files}
    if len(embeddings) != len(labels["species"]):
        return None
    print(f"Reusing {split} embedding cache: {len(embeddings):,} clips")
    return embeddings, labels


def extract_embeddings(
    split: str,
    dataset: Any,
    model: Any,
    collator: DCLDEAudioCollator,
    device: torch.device,
    cache_dir: Path,
    signature: dict[str, Any],
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    embedding_path, label_path, metadata_path = cache_paths(cache_dir, split)
    hidden_size = int(model.config.hidden_size)
    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), hidden_size),
    )
    species = np.empty(len(dataset), dtype=np.int64)
    ecotype = np.empty(len(dataset), dtype=np.int64)
    paths: list[str] = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model.eval()
    offset = 0
    print(f"Extracting {split} embeddings from {len(dataset):,} clips...")
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            inputs = batch["input_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.ast(input_values=inputs, return_dict=True)
                pooled = model._pool_ast_output(output.last_hidden_state)
            count = len(inputs)
            embeddings[offset : offset + count] = pooled.float().cpu().numpy()
            species[offset : offset + count] = batch["species_labels"].numpy()
            ecotype[offset : offset + count] = batch["ecotype_labels"].numpy()
            offset += count
            if batch_index % 250 == 0 or offset == len(dataset):
                print(f"  {split}: {offset:,}/{len(dataset):,}")

    # Keep paths in exactly the same subset order as DataLoader.
    if isinstance(dataset, Subset):
        paths = [dataset.dataset.clip_paths[index] for index in dataset.indices]
    else:
        paths = list(dataset.clip_paths)
    embeddings.flush()
    np.savez_compressed(
        label_path,
        species=species,
        ecotype=ecotype,
        clip_path=np.asarray(paths, dtype=str),
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"signature": signature, "rows": len(dataset), "hidden_size": hidden_size},
            file,
            indent=2,
        )
    return np.load(embedding_path, mmap_mode="r"), {
        "species": species,
        "ecotype": ecotype,
        "clip_path": np.asarray(paths, dtype=str),
    }


def trigger_labels(species: np.ndarray, positive_species: list[str]) -> np.ndarray:
    positive_ids = [SPECIES_LABELS[name] for name in positive_species]
    return np.isin(species, positive_ids).astype(np.int64)


def f1_score_value(
    true: np.ndarray,
    predicted: np.ndarray,
    average: str,
    labels: Optional[list[int]] = None,
    pos_label: int = 1,
) -> float:
    return float(
        precision_recall_fscore_support(
            true,
            predicted,
            average=average,
            labels=labels,
            pos_label=pos_label,
            zero_division=0,
        )[2]
    )


def class_f1(true: np.ndarray, predicted: np.ndarray, class_id: int) -> float:
    values = precision_recall_fscore_support(
        true,
        predicted,
        labels=[class_id],
        average=None,
        zero_division=0,
    )[2]
    return float(values[0]) if len(values) else 0.0


def calculate_metrics(
    trigger_true: np.ndarray,
    trigger_pred: np.ndarray,
    species_true: np.ndarray,
    species_pred: np.ndarray,
    ecotype_true: np.ndarray,
    ecotype_pred: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "trigger_accuracy": float(accuracy_score(trigger_true, trigger_pred)),
        "trigger_f1": f1_score_value(trigger_true, trigger_pred, "binary"),
        "species_accuracy": float(accuracy_score(species_true, species_pred)),
        "species_macro_f1": f1_score_value(species_true, species_pred, "macro"),
    }
    for class_id, class_name in SPECIES_ID2LABEL.items():
        metrics[f"species_f1_{class_name}"] = class_f1(
            species_true, species_pred, class_id
        )
    mask = ecotype_true != IGNORE_INDEX
    if np.any(mask):
        eco_true = ecotype_true[mask]
        eco_pred = ecotype_pred[mask]
        metrics["ecotype_accuracy"] = float(accuracy_score(eco_true, eco_pred))
        metrics["ecotype_macro_f1"] = f1_score_value(eco_true, eco_pred, "macro")
        for class_id, class_name in ECOTYPE_ID2LABEL.items():
            metrics[f"ecotype_f1_{class_name}"] = class_f1(
                eco_true, eco_pred, class_id
            )
        metrics["ecotype_srkw_tkw_f1"] = f1_score_value(
            eco_true,
            eco_pred,
            "macro",
            labels=[ECOTYPE_LABELS["SRKW"], ECOTYPE_LABELS["TKW"]],
        )
    else:
        metrics.update(
            ecotype_accuracy=0.0,
            ecotype_macro_f1=0.0,
            ecotype_srkw_tkw_f1=0.0,
        )
    metrics["combined_score"] = (
        0.4 * metrics["trigger_f1"]
        + 0.3 * metrics["species_macro_f1"]
        + 0.3 * metrics["ecotype_srkw_tkw_f1"]
    )
    return metrics


def predict_head(
    head: nn.Module,
    embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    positive_species: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(embeddings))),
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts: list[list[np.ndarray]] = [[], [], []]
    head.eval()
    with torch.inference_mode():
        for (features,) in loader:
            outputs = head(features.to(device, non_blocking=True))
            for index, output in enumerate(outputs):
                logits_parts[index].append(output.cpu().numpy())
    trigger_logits, species_logits, ecotype_logits = [
        np.concatenate(parts) for parts in logits_parts
    ]
    trigger_true = trigger_labels(labels["species"], positive_species)
    trigger_pred = trigger_logits.argmax(axis=1)
    species_pred = species_logits.argmax(axis=1)
    ecotype_pred = ecotype_logits.argmax(axis=1)
    metrics = calculate_metrics(
        trigger_true,
        trigger_pred,
        labels["species"],
        species_pred,
        labels["ecotype"],
        ecotype_pred,
    )
    return metrics, {
        "trigger_probability": torch.softmax(torch.from_numpy(trigger_logits), dim=1)[
            :, 1
        ].numpy(),
        "trigger_prediction": trigger_pred,
        "species_prediction": species_pred,
        "ecotype_prediction": ecotype_pred,
        "species_logits": species_logits,
        "ecotype_logits": ecotype_logits,
    }


def threshold_report(
    true: np.ndarray, probabilities: np.ndarray
) -> tuple[pd.DataFrame, float]:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 199):
        predicted = (probabilities >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predicted, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(
        ["f1", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return frame, float(best["threshold"])


def confusion_frame(
    true: np.ndarray,
    predicted: np.ndarray,
    id2label: dict[int, str],
) -> pd.DataFrame:
    class_ids = sorted(id2label)
    names = [id2label[class_id] for class_id in class_ids]
    return pd.DataFrame(
        confusion_matrix(true, predicted, labels=class_ids),
        index=[f"actual_{name}" for name in names],
        columns=[f"predicted_{name}" for name in names],
    )


def weight_tensor(values: Optional[list[float]], device: torch.device) -> Optional[torch.Tensor]:
    return torch.tensor(values, dtype=torch.float32, device=device) if values else None


def train_head(
    args: argparse.Namespace,
    head: FrozenBackboneMultitaskHead,
    train_embeddings: np.ndarray,
    train_labels: dict[str, np.ndarray],
    val_embeddings: np.ndarray,
    val_labels: dict[str, np.ndarray],
    device: torch.device,
    class_weights: dict[str, Optional[list[float]]],
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    train_trigger = trigger_labels(train_labels["species"], args.trigger_species)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_embeddings)),
        torch.from_numpy(train_trigger),
        torch.from_numpy(train_labels["species"]),
        torch.from_numpy(train_labels["ecotype"]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    head.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": head.trigger_classifier.parameters(),
                "lr": args.learning_rate,
            },
            {
                "params": list(head.species_classifier.parameters())
                + list(head.ecotype_classifier.parameters()),
                "lr": args.existing_head_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    trigger_weights = weight_tensor(class_weights["trigger"], device)
    species_weights = weight_tensor(class_weights["species"], device)
    ecotype_weights = weight_tensor(class_weights["ecotype"], device)

    history = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] = {}
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        samples = 0
        for features, trigger, species, ecotype in loader:
            features = features.to(device, non_blocking=True)
            trigger = trigger.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            ecotype = ecotype.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            trigger_logits, species_logits, ecotype_logits = head(features)
            losses = [
                args.trigger_loss_weight
                * nn.functional.cross_entropy(
                    trigger_logits, trigger, weight=trigger_weights
                ),
                args.species_loss_weight
                * nn.functional.cross_entropy(
                    species_logits, species, weight=species_weights
                ),
            ]
            mask = ecotype != IGNORE_INDEX
            if torch.any(mask):
                losses.append(
                    args.ecotype_loss_weight
                    * nn.functional.cross_entropy(
                        ecotype_logits[mask], ecotype[mask], weight=ecotype_weights
                    )
                )
            loss = torch.stack(losses).sum()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            samples += len(features)

        metrics, _ = predict_head(
            head,
            val_embeddings,
            val_labels,
            args.trigger_species,
            args.head_batch_size,
            device,
        )
        row = {"epoch": epoch, "train_loss": total_loss / max(samples, 1), **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:02d}: loss={row['train_loss']:.5f}, "
            f"trigger_f1={metrics['trigger_f1']:.4f}, "
            f"species_macro_f1={metrics['species_macro_f1']:.4f}, "
            f"SRKW/TKW_f1={metrics['ecotype_srkw_tkw_f1']:.4f}, "
            f"combined={metrics['combined_score']:.4f}"
        )
        if metrics["combined_score"] > best_score + 1e-8:
            best_score = metrics["combined_score"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
                print(f"Early stopping after {stale_epochs} non-improving epochs.")
                break
    if not best_state:
        raise RuntimeError("No best head state was selected.")
    return best_state, pd.DataFrame(history)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessing, checkpoint_identity = resolved_preprocessing(args)

    print("\nFrozen-backbone multitask head settings")
    print(f"Base checkpoint:       {args.model_name}")
    print(f"Device:                {device}")
    print(f"Trigger-positive:      {', '.join(args.trigger_species)}")
    print(f"New-trigger LR:        {args.learning_rate:.3e}")
    print(f"Existing-head LR:      {args.existing_head_learning_rate:.3e}")
    print(f"Mean subtraction:      {preprocessing['mean_subtract']}")
    print(f"High-pass filter:      {preprocessing['high_pass_filter']}")
    if preprocessing["high_pass_filter"]:
        print(
            f"High-pass settings:    {preprocessing['high_pass_cutoff_hz']:g} Hz, "
            f"order {preprocessing['high_pass_order']}"
        )

    train_base = load_manifest(
        args.train_manifest,
        drop_unknown_labels=args.drop_unknown_labels,
        clip_path_dataset_root=args.train_dataset_root,
    )
    val_base = load_manifest(
        args.val_manifest,
        drop_unknown_labels=args.drop_unknown_labels,
        clip_path_dataset_root=args.val_dataset_root,
    )
    train_dataset = selected_subset(train_base, args.max_train_files, args.seed)
    val_dataset = selected_subset(val_base, args.max_val_files, args.seed + 1)
    print(f"Training clips:        {len(train_dataset):,}")
    print(f"Validation clips:      {len(val_dataset):,}")

    train_signature = cache_signature(
        "train",
        args.train_manifest,
        args.train_dataset_root,
        checkpoint_identity,
        preprocessing,
        args.max_train_files,
        args.seed,
    )
    val_signature = cache_signature(
        "validation",
        args.val_manifest,
        args.val_dataset_root,
        checkpoint_identity,
        preprocessing,
        args.max_val_files,
        args.seed + 1,
    )
    train_cache = (
        load_matching_cache(cache_dir, "train", train_signature)
        if args.reuse_embedding_cache
        else None
    )
    val_cache = (
        load_matching_cache(cache_dir, "validation", val_signature)
        if args.reuse_embedding_cache
        else None
    )

    # This invokes the same strict checkpoint compatibility remapper/validator
    # used by multispecies training and evaluation. Partial loads are refused.
    model = load_training_model(
        model_name=args.model_name,
        dropout=args.dropout,
        kw_loss_weight=1.0,
        species_loss_weight=1.0,
        ecotype_loss_weight=1.0,
        freeze_backbone=True,
    )
    hidden_size = int(model.config.hidden_size)
    head = FrozenBackboneMultitaskHead(hidden_size, args.dropout)
    head.species_classifier.load_state_dict(model.species_classifier.state_dict())
    head.ecotype_classifier.load_state_dict(model.ecotype_classifier.state_dict())

    if train_cache is None or val_cache is None:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
        collator = DCLDEAudioCollator(
            feature_extractor=feature_extractor,
            max_duration=args.max_duration,
            mean_subtract=preprocessing["mean_subtract"],
            high_pass_cutoff_hz=(
                preprocessing["high_pass_cutoff_hz"]
                if preprocessing["high_pass_filter"]
                else None
            ),
            high_pass_order=preprocessing["high_pass_order"],
        )
        model.to(device)
        if train_cache is None:
            train_cache = extract_embeddings(
                "train",
                train_dataset,
                model,
                collator,
                device,
                cache_dir,
                train_signature,
                args.embedding_batch_size,
                args.preprocessing_workers,
            )
        if val_cache is None:
            val_cache = extract_embeddings(
                "validation",
                val_dataset,
                model,
                collator,
                device,
                cache_dir,
                val_signature,
                args.embedding_batch_size,
                args.preprocessing_workers,
            )

    train_embeddings, train_label_data = train_cache
    val_embeddings, val_label_data = val_cache
    train_trigger = trigger_labels(train_label_data["species"], args.trigger_species)

    trigger_weights = parse_class_weights(
        args.trigger_class_weights, HEAD_LABELS, "--trigger-class-weights"
    )
    if trigger_weights is None and args.automatic_trigger_class_weights:
        counts = np.bincount(train_trigger, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Both trigger classes are required; counts={counts.tolist()}")
        trigger_weights = [1.0, float(counts[0] / counts[1])]
    species_weights = parse_class_weights(
        args.species_class_weights, SPECIES_LABELS, "--species-class-weights"
    )
    ecotype_weights = parse_class_weights(
        args.ecotype_class_weights, ECOTYPE_LABELS, "--ecotype-class-weights"
    )
    class_weights = {
        "trigger": trigger_weights,
        "species": species_weights,
        "ecotype": ecotype_weights,
    }
    print(f"Trigger counts:        {np.bincount(train_trigger, minlength=2).tolist()}")
    print(f"Trigger class weights: {trigger_weights}")

    backbone_state = copy.deepcopy(model.state_dict()) if args.save_backbone_state else None
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_state, history = train_head(
        args,
        head,
        train_embeddings,
        train_label_data,
        val_embeddings,
        val_label_data,
        device,
        class_weights,
    )
    head.load_state_dict(best_state)
    head.to(device)
    best_metrics, predictions = predict_head(
        head,
        val_embeddings,
        val_label_data,
        args.trigger_species,
        args.head_batch_size,
        device,
    )
    validation_trigger = trigger_labels(
        val_label_data["species"], args.trigger_species
    )
    thresholds, best_threshold = threshold_report(
        validation_trigger, predictions["trigger_probability"]
    )

    head_path = output_dir / "any_whale_multitask_head.pt"
    torch.save(best_state, head_path)
    history.to_csv(output_dir / "training_history.csv", index=False)
    thresholds.to_csv(output_dir / "trigger_threshold_metrics.csv", index=False)
    confusion_frame(
        validation_trigger,
        predictions["trigger_prediction"],
        HEAD_ID2LABEL,
    ).to_csv(output_dir / "trigger_confusion_matrix.csv")
    confusion_frame(
        val_label_data["species"],
        predictions["species_prediction"],
        SPECIES_ID2LABEL,
    ).to_csv(output_dir / "species_confusion_matrix.csv")
    ecotype_mask = val_label_data["ecotype"] != IGNORE_INDEX
    confusion_frame(
        val_label_data["ecotype"][ecotype_mask],
        predictions["ecotype_prediction"][ecotype_mask],
        ECOTYPE_ID2LABEL,
    ).to_csv(output_dir / "ecotype_confusion_matrix.csv")

    species_probabilities = torch.softmax(
        torch.from_numpy(predictions["species_logits"]), dim=1
    ).numpy()
    ecotype_probabilities = torch.softmax(
        torch.from_numpy(predictions["ecotype_logits"]), dim=1
    ).numpy()
    report = pd.DataFrame(
        {
            "clip_path": val_label_data["clip_path"],
            "trigger_true": validation_trigger,
            "trigger_probability": predictions["trigger_probability"],
            "trigger_pred_0_5": predictions["trigger_prediction"],
            "species_true": val_label_data["species"],
            "species_pred": predictions["species_prediction"],
            "ecotype_true": val_label_data["ecotype"],
            "ecotype_pred": predictions["ecotype_prediction"],
        }
    )
    for class_id, class_name in SPECIES_ID2LABEL.items():
        report[f"species_probability_{class_name}"] = species_probabilities[:, class_id]
    for class_id, class_name in ECOTYPE_ID2LABEL.items():
        report[f"ecotype_probability_{class_name}"] = ecotype_probabilities[:, class_id]
    report.to_csv(output_dir / "validation_predictions.csv", index=False)

    configuration = {
        "format": "dclde_frozen_backbone_multitask_head_v1",
        "base_model": args.model_name,
        "base_checkpoint_identity": checkpoint_identity,
        "hidden_size": hidden_size,
        "dropout": args.dropout,
        "trigger_labels": HEAD_LABELS,
        "trigger_positive_species": args.trigger_species,
        "species_labels": SPECIES_LABELS,
        "ecotype_labels": ECOTYPE_LABELS,
        "preprocessing": preprocessing,
        "class_weights": class_weights,
        "loss_weights": {
            "trigger": args.trigger_loss_weight,
            "species": args.species_loss_weight,
            "ecotype": args.ecotype_loss_weight,
        },
        "learning_rates": {
            "new_trigger": args.learning_rate,
            "existing_species_ecotype": args.existing_head_learning_rate,
        },
        "best_validation_trigger_threshold": best_threshold,
        "best_validation_metrics_at_0_5": best_metrics,
        "seed": args.seed,
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "val_manifest": str(Path(args.val_manifest).resolve()),
    }
    with (output_dir / "any_whale_multitask_head_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configuration, file, indent=2)
    if backbone_state is not None:
        torch.save(
            {
                "base_model_state_dict": backbone_state,
                "head_state_dict": best_state,
                "configuration": configuration,
            },
            output_dir / "backbone_and_multitask_head_bundle.pt",
        )

    print("\nBest validation metrics (classification threshold 0.5)")
    for name, value in best_metrics.items():
        print(f"{name:30s}: {value:.6f}")
    print(f"Best validation trigger threshold: {best_threshold:.4f}")
    print(f"Saved head: {head_path}")
    print(
        "The clip-level trigger threshold is diagnostic. Retune the operating "
        "threshold on long-recording validation data before deployment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
