#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Fine-tune a shared AST backbone with DCLDE multi-task classification heads.

Tasks:
  - KW detection: KW vs not-KW
  - Species: background, KW, HW, AB
  - KW ecotype: NRKW, SRKW, OKW, SAR, TKW

The species labels are read from ClassSpecies:
  KW     -> killer whale
  HW     -> humpback whale
  AB     -> Pacific white-sided dolphin
  UndBio -> background
  BKG    -> background

The ecotype loss is only computed for rows whose ClassSpecies is KW and whose
Ecotype value is one of NRKW, SRKW, OKW, SAR, or TKW.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

import datasets.config

datasets.config.AUDIO_BACKENDS_USE_TORCH = False
datasets.config.AUDIOCODEC_DEFAULT_DECODER = "soundfile"

from datasets import Audio, Dataset, DatasetDict  # noqa: E402
from transformers import (  # noqa: E402
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
DEFAULT_MAX_DURATION = 3.0
CHECKPOINT_SAVE_LIMIT = 6
DEFAULT_MAX_PREPROCESSING_WORKERS = 8

KW_LABELS = {"not_kw": 0, "kw": 1}
KW_ID2LABEL = {v: k for k, v in KW_LABELS.items()}
SPECIES_LABELS = {"background": 0, "KW": 1, "HW": 2, "AB": 3}
SPECIES_ID2LABEL = {v: k for k, v in SPECIES_LABELS.items()}
ECOTYPE_LABELS = {"NRKW": 0, "SRKW": 1, "OKW": 2, "SAR": 3, "TKW": 4}
ECOTYPE_ID2LABEL = {v: k for k, v in ECOTYPE_LABELS.items()}
BACKGROUND_SPECIES = {"UndBio", "BKG"}
KNOWN_SPECIES = set(SPECIES_LABELS) | BACKGROUND_SPECIES
IGNORE_INDEX = -100


def resolve_path(path: str) -> Path:
    """Resolve absolute paths as-is and repo-relative paths under REPO_ROOT."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def normalize_label(value: Any) -> str:
    """Normalize CSV label values to stable strings."""
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    if normalized.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return normalized


def map_species_label(class_species: str) -> int:
    """Map ClassSpecies to the species-head target."""
    if class_species in BACKGROUND_SPECIES:
        return SPECIES_LABELS["background"]
    if class_species in SPECIES_LABELS:
        return SPECIES_LABELS[class_species]
    raise ValueError(f"Unknown ClassSpecies label: {class_species!r}")


def map_kw_label(class_species: str) -> int:
    """Map ClassSpecies to the KW-vs-not-KW target."""
    return KW_LABELS["kw"] if class_species == "KW" else KW_LABELS["not_kw"]


def map_ecotype_label(class_species: str, ecotype: str) -> int:
    """Map Ecotype to target ID, or IGNORE_INDEX when not applicable."""
    if class_species != "KW":
        return IGNORE_INDEX
    if not ecotype:
        return IGNORE_INDEX
    if ecotype not in ECOTYPE_LABELS:
        raise ValueError(f"Unknown KW Ecotype label: {ecotype!r}")
    return ECOTYPE_LABELS[ecotype]


def load_manifest(manifest_path: str, drop_unknown_labels: bool = False) -> Dataset:
    """Load a DCLDE manifest into a Hugging Face Dataset."""
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

    df["kw_labels"] = df["ClassSpecies"].map(map_kw_label)
    df["species_labels"] = df["ClassSpecies"].map(map_species_label)
    df["ecotype_labels"] = [
        map_ecotype_label(class_species, ecotype)
        for class_species, ecotype in zip(df["ClassSpecies"], df["Ecotype"])
    ]

    dataset = Dataset.from_dict(
        {
            "audio": df["clip_path"].tolist(),
            "kw_labels": df["kw_labels"].astype(int).tolist(),
            "species_labels": df["species_labels"].astype(int).tolist(),
            "ecotype_labels": df["ecotype_labels"].astype(int).tolist(),
        }
    )
    return dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


class WaveformAugmenter:
    """Simple waveform augmentations applied during preprocessing."""

    def __init__(
        self,
        sample_rate: int,
        random_gain: bool = False,
        gain_db: float = 6.0,
        time_shift: bool = False,
        max_shift_ms: float = 250.0,
        gaussian_noise: bool = False,
        noise_std: float = 0.002,
    ) -> None:
        self.sample_rate = sample_rate
        self.random_gain = random_gain
        self.gain_db = gain_db
        self.time_shift = time_shift
        self.max_shift = int(sample_rate * max_shift_ms / 1000)
        self.gaussian_noise = gaussian_noise
        self.noise_std = noise_std

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32, copy=True)
        if self.random_gain:
            gain = random.uniform(-self.gain_db, self.gain_db)
            audio *= 10 ** (gain / 20)
        if self.time_shift and self.max_shift > 0:
            audio = np.roll(audio, random.randint(-self.max_shift, self.max_shift))
        if self.gaussian_noise:
            audio = audio + np.random.normal(0, self.noise_std, size=audio.shape)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)


def preprocess_function(
    examples: dict[str, Any],
    feature_extractor: Any,
    max_duration: float,
    augmenter: Optional[WaveformAugmenter] = None,
) -> dict[str, Any]:
    """Pad/truncate audio, optionally augment it, and run the AST feature extractor."""
    target_length = int(max_duration * SAMPLE_RATE)
    processed_audio = []
    for item in examples["audio"]:
        audio = item["array"]
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
        if augmenter is not None:
            audio = augmenter(audio)
        processed_audio.append(audio)

    inputs = feature_extractor(processed_audio, sampling_rate=SAMPLE_RATE, padding=True)
    inputs["kw_labels"] = examples["kw_labels"]
    inputs["species_labels"] = examples["species_labels"]
    inputs["ecotype_labels"] = examples["ecotype_labels"]
    return inputs


class MultiTaskASTForDCLDE(nn.Module):
    """AST backbone with KW, species, and ecotype classifier heads."""

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        kw_loss_weight: float = 1.0,
        species_loss_weight: float = 1.0,
        ecotype_loss_weight: float = 1.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        base_model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.config = base_model.config
        self.ast = base_model.audio_spectrogram_transformer
        hidden_size = int(getattr(self.config, "hidden_size"))
        self.dropout = nn.Dropout(dropout)
        self.kw_classifier = nn.Linear(hidden_size, len(KW_LABELS))
        self.species_classifier = nn.Linear(hidden_size, len(SPECIES_LABELS))
        self.ecotype_classifier = nn.Linear(hidden_size, len(ECOTYPE_LABELS))
        self.kw_loss_weight = kw_loss_weight
        self.species_loss_weight = species_loss_weight
        self.ecotype_loss_weight = ecotype_loss_weight

        if freeze_backbone:
            print("Freezing AST backbone and training classification heads only.")
            for param in self.ast.parameters():
                param.requires_grad = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None) -> None:
        """Allow Trainer gradient checkpointing to reach the AST backbone when available."""
        if hasattr(self.ast, "gradient_checkpointing_enable"):
            self.ast.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the AST backbone when available."""
        if hasattr(self.ast, "gradient_checkpointing_disable"):
            self.ast.gradient_checkpointing_disable()

    def _pool_ast_output(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """Use AST's CLS/distillation-token pooled embedding."""
        if sequence_output.shape[1] >= 2:
            return (sequence_output[:, 0] + sequence_output[:, 1]) / 2.0
        return sequence_output[:, 0]

    def forward(
        self,
        input_values: torch.Tensor,
        kw_labels: Optional[torch.Tensor] = None,
        species_labels: Optional[torch.Tensor] = None,
        ecotype_labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, Any]:
        outputs = self.ast(input_values=input_values, return_dict=True)
        pooled = self.dropout(self._pool_ast_output(outputs.last_hidden_state))

        kw_logits = self.kw_classifier(pooled)
        species_logits = self.species_classifier(pooled)
        ecotype_logits = self.ecotype_classifier(pooled)

        loss = None
        losses = []
        if kw_labels is not None:
            losses.append(
                self.kw_loss_weight
                * nn.functional.cross_entropy(kw_logits, kw_labels.long())
            )
        if species_labels is not None:
            losses.append(
                self.species_loss_weight
                * nn.functional.cross_entropy(species_logits, species_labels.long())
            )
        if ecotype_labels is not None:
            ecotype_mask = ecotype_labels != IGNORE_INDEX
            if torch.any(ecotype_mask):
                losses.append(
                    self.ecotype_loss_weight
                    * nn.functional.cross_entropy(
                        ecotype_logits[ecotype_mask],
                        ecotype_labels[ecotype_mask].long(),
                    )
                )
        if losses:
            loss = torch.stack(losses).sum()

        return {
            "loss": loss,
            "logits": (kw_logits, species_logits, ecotype_logits),
        }


def _f1(labels: np.ndarray, predictions: np.ndarray, **kwargs: Any) -> float:
    """Return an F1 score with sklearn's zero-division behavior fixed."""
    return float(
        precision_recall_fscore_support(
            labels,
            predictions,
            zero_division=0,
            **kwargs,
        )[2]
    )


def _class_f1(labels: np.ndarray, predictions: np.ndarray, class_id: int) -> float:
    """Return one-vs-rest F1 for a single class in a multiclass target."""
    scores = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[class_id],
        average=None,
        zero_division=0,
    )[2]
    return float(scores[0]) if len(scores) else 0.0


def _unpack_three(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Handle Trainer outputs as either (a, b, c) or ((a, b, c),)."""
    if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        raise ValueError(f"Expected three prediction/label arrays, got {type(value).__name__}")
    return value[0], value[1], value[2]


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute metrics for all heads and a combined model-selection score."""
    kw_logits, species_logits, ecotype_logits = _unpack_three(eval_pred.predictions)
    kw_labels, species_labels, ecotype_labels = _unpack_three(eval_pred.label_ids)

    kw_predictions = np.argmax(kw_logits, axis=1)
    species_predictions = np.argmax(species_logits, axis=1)
    ecotype_predictions = np.argmax(ecotype_logits, axis=1)

    metrics: dict[str, float] = {
        "kw_accuracy": float(accuracy_score(kw_labels, kw_predictions)),
        "kw_f1": _f1(kw_labels, kw_predictions, average="binary", pos_label=KW_LABELS["kw"]),
        "species_accuracy": float(accuracy_score(species_labels, species_predictions)),
        "species_macro_f1": _f1(species_labels, species_predictions, average="macro"),
    }

    for class_id, class_name in SPECIES_ID2LABEL.items():
        metrics[f"species_f1_{class_name}"] = _class_f1(
            species_labels,
            species_predictions,
            class_id,
        )

    ecotype_mask = ecotype_labels != IGNORE_INDEX
    if np.any(ecotype_mask):
        ecotype_true = ecotype_labels[ecotype_mask]
        ecotype_pred = ecotype_predictions[ecotype_mask]
        metrics["ecotype_accuracy"] = float(accuracy_score(ecotype_true, ecotype_pred))
        metrics["ecotype_macro_f1"] = _f1(ecotype_true, ecotype_pred, average="macro")
        for class_id, class_name in ECOTYPE_ID2LABEL.items():
            metrics[f"ecotype_f1_{class_name}"] = _class_f1(
                ecotype_true,
                ecotype_pred,
                class_id,
            )
        srkw_tkw_labels = [ECOTYPE_LABELS["SRKW"], ECOTYPE_LABELS["TKW"]]
        metrics["ecotype_srkw_tkw_f1"] = _f1(
            ecotype_true,
            ecotype_pred,
            average="macro",
            labels=srkw_tkw_labels,
        )
    else:
        metrics["ecotype_accuracy"] = 0.0
        metrics["ecotype_macro_f1"] = 0.0
        metrics["ecotype_srkw_tkw_f1"] = 0.0

    metrics["combined_score"] = (
        0.4 * metrics["kw_f1"]
        + 0.3 * metrics["species_macro_f1"]
        + 0.3 * metrics["ecotype_srkw_tkw_f1"]
    )
    return metrics


def get_preprocessing_workers(dataset: DatasetDict, requested_workers: int) -> int:
    """Choose a safe number of dataset map workers."""
    if requested_workers < 1:
        raise ValueError(f"preprocessing_workers must be at least 1, got {requested_workers}")
    split_sizes = [len(split_dataset) for split_dataset in dataset.values()]
    return max(1, min(requested_workers, min(split_sizes)))


def analyze_dataset(dataset: DatasetDict) -> None:
    """Print target distributions for each split."""
    print("\n" + "=" * 72)
    print("DATASET ANALYSIS")
    print("=" * 72)
    for split_name, split_dataset in dataset.items():
        print(f"\n{split_name.upper()} split: {len(split_dataset)} samples")
        for label_column, id2label in (
            ("kw_labels", KW_ID2LABEL),
            ("species_labels", SPECIES_ID2LABEL),
            ("ecotype_labels", ECOTYPE_ID2LABEL),
        ):
            labels = [label for label in split_dataset[label_column] if label != IGNORE_INDEX]
            counts = Counter(labels)
            print(f"  {label_column}:")
            if not labels:
                print("    no labeled rows")
                continue
            for label_id in sorted(counts):
                print(f"    {id2label[label_id]:12s}: {counts[label_id]:7d}")
    print("=" * 72 + "\n")


def save_training_curves(trainer: Trainer, output_dir: Path) -> None:
    """Save training and validation curves from Trainer logs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training curve plots.")
        return

    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    combined_steps: list[int] = []
    combined_scores: list[float] = []

    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(int(entry["step"]))
            train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_steps.append(int(entry["step"]))
            eval_losses.append(float(entry["eval_loss"]))
        if "eval_combined_score" in entry:
            combined_steps.append(int(entry["step"]))
            combined_scores.append(float(entry["eval_combined_score"]))

    if train_losses or eval_losses:
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_steps, train_losses, label="Training loss")
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="Validation loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        loss_path = output_dir / "loss_curve.png"
        plt.savefig(loss_path, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curve to {loss_path}")

    if combined_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(combined_steps, combined_scores, label="Combined validation score")
        plt.xlabel("Training step")
        plt.ylabel("Score")
        plt.title("Validation Score")
        plt.legend()
        plt.grid(True)
        score_path = output_dir / "validation_score_curve.png"
        plt.savefig(score_path, bbox_inches="tight")
        plt.close()
        print(f"Saved validation score curve to {score_path}")


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    """Write label maps and training configuration next to model artifacts."""
    metadata = {
        "model_type": "podsai_dclde_multitask_ast",
        "base_model": args.model_name,
        "kw_label2id": KW_LABELS,
        "kw_id2label": KW_ID2LABEL,
        "species_label2id": SPECIES_LABELS,
        "species_id2label": SPECIES_ID2LABEL,
        "ecotype_label2id": ECOTYPE_LABELS,
        "ecotype_id2label": ECOTYPE_ID2LABEL,
        "ignore_index": IGNORE_INDEX,
        "loss_weights": {
            "kw": args.kw_loss_weight,
            "species": args.species_loss_weight,
            "ecotype": args.ecotype_loss_weight,
        },
    }
    with (output_dir / "multitask_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def main() -> int:
    """Train the DCLDE multi-task AST model."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a shared AST model for DCLDE KW/species/ecotype tasks."
    )
    parser.add_argument("--train-manifest", required=True, help="Training manifest CSV.")
    parser.add_argument("--val-manifest", required=True, help="Validation manifest CSV.")
    parser.add_argument(
        "--output-dir",
        default="model/dclde_multitask",
        help="Directory to save checkpoints and final artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="davethaler/whale-call-detector",
        help="Existing AST model or Hugging Face model ID to fine-tune.",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20000,
        help="Validate every N optimizer steps (default: 20000).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoints every N optimizer steps (default: same as --eval-steps).",
    )
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--save-total-limit", type=int, default=CHECKPOINT_SAVE_LIMIT)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument(
        "--preprocessing-workers",
        type=int,
        default=max(1, min(DEFAULT_MAX_PREPROCESSING_WORKERS, os.cpu_count() or 1)),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kw-loss-weight", type=float, default=1.0)
    parser.add_argument("--species-loss-weight", type=float, default=1.0)
    parser.add_argument("--ecotype-loss-weight", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--drop-unknown-labels", action="store_true")
    parser.add_argument("--random-gain", action="store_true")
    parser.add_argument("--gain-db", type=float, default=6.0)
    parser.add_argument("--time-shift", action="store_true")
    parser.add_argument("--max-shift-ms", type=float, default=250.0)
    parser.add_argument("--gaussian-noise", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    args = parser.parse_args()

    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifests...")
    train_dataset = load_manifest(args.train_manifest, drop_unknown_labels=args.drop_unknown_labels)
    validation_dataset = load_manifest(args.val_manifest, drop_unknown_labels=args.drop_unknown_labels)
    dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    analyze_dataset(dataset)

    print(f"Loading feature extractor: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    augmenter = None
    if args.random_gain or args.time_shift or args.gaussian_noise:
        augmenter = WaveformAugmenter(
            sample_rate=SAMPLE_RATE,
            random_gain=args.random_gain,
            gain_db=args.gain_db,
            time_shift=args.time_shift,
            max_shift_ms=args.max_shift_ms,
            gaussian_noise=args.gaussian_noise,
            noise_std=args.noise_std,
        )

    preprocessing_workers = get_preprocessing_workers(dataset, args.preprocessing_workers)
    print(f"Preprocessing dataset with {preprocessing_workers} worker(s)...")
    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": ["audio"],
    }
    if preprocessing_workers > 1:
        map_kwargs["num_proc"] = preprocessing_workers

    dataset["train"] = dataset["train"].map(
        partial(
            preprocess_function,
            feature_extractor=feature_extractor,
            max_duration=args.max_duration,
            augmenter=augmenter,
        ),
        **map_kwargs,
    )
    dataset["validation"] = dataset["validation"].map(
        partial(
            preprocess_function,
            feature_extractor=feature_extractor,
            max_duration=args.max_duration,
            augmenter=None,
        ),
        **map_kwargs,
    )

    print(f"Loading multi-task AST model from {args.model_name}")
    model = MultiTaskASTForDCLDE(
        model_name=args.model_name,
        dropout=args.dropout,
        kw_loss_weight=args.kw_loss_weight,
        species_loss_weight=args.species_loss_weight,
        ecotype_loss_weight=args.ecotype_loss_weight,
        freeze_backbone=args.freeze_backbone,
    )

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        label_names=["kw_labels", "species_labels", "ecotype_labels"],
        push_to_hub=args.push_to_hub,
        hub_strategy="all_checkpoints" if args.push_to_hub else "end",
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Evaluating best/final model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    print(f"Saving final model artifacts to {output_dir}")
    trainer.save_model(str(output_dir))
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    feature_extractor.save_pretrained(str(output_dir))
    save_metadata(output_dir, args)
    save_training_curves(trainer, output_dir)

    if args.push_to_hub:
        print(f"Pushing feature extractor to Hugging Face Hub: {args.hub_model_id}")
        feature_extractor.push_to_hub(args.hub_model_id)

    print("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
