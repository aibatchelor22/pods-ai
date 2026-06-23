#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Train a PODS-AI audio classification model for orca call detection.

This script fine-tunes a HuggingFace audio classification model on orca call
audio. The default base model uses spectrogram features rather than raw-audio
Wav2Vec2 embeddings. The trained model can be pushed to HuggingFace Hub or
saved locally.

Usage:
    # Binary classification (other vs any call)
    python train_podsai_model.py --num_classes 2 --output_dir ./model/binary

    # Multi-class classification (water, resident, transient, humpback, vessel, jingle, human)
    python train_podsai_model.py --num_classes 7 --output_dir ./model/multiclass
"""

import argparse
from collections import Counter
from functools import partial
import os
from pathlib import Path
from typing import Protocol
import matplotlib.pyplot as plt

import numpy as np
import torch

# Configure datasets to use soundfile for audio decoding BEFORE importing datasets components.
import datasets.config
datasets.config.AUDIO_BACKENDS_USE_TORCH = False
datasets.config.AUDIOCODEC_DEFAULT_DECODER = "soundfile"

from datasets import Dataset, Audio, DatasetDict, ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import evaluate

# Verify audio decoding dependencies are available.
try:
    import librosa
    import soundfile
except ImportError as e:
    print("Error: Missing required audio decoding libraries.")
    print("Please install the required dependencies:")
    print("  pip install -r requirements.txt")
    print(f"\nSpecific error: {e}")
    raise

# Get repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Label mappings (will be set based on num_classes).
LABEL2ID = {}
ID2LABEL = {}


class _SklearnMetricFallback:
    """Fallback metric implementation used when evaluate metrics are unavailable."""

    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def compute(
        self,
        predictions: list[int] | np.ndarray,
        references: list[int] | np.ndarray,
        average: str | None = None,
        labels: list[int] | None = None,
    ) -> dict:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

        if self.metric_name == "accuracy":
            return {"accuracy": float(accuracy_score(references, predictions))}

        precision, recall, f1, _ = precision_recall_fscore_support(
            references,
            predictions,
            average=average,
            labels=labels,
            zero_division=0,
        )
        metric_value = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }[self.metric_name]
        return {self.metric_name: metric_value}


def _load_metric(metric_name: str):
    """Load an evaluate metric with sklearn fallback if unavailable."""
    try:
        return evaluate.load(metric_name)
    except FileNotFoundError:
        print(
            f"Warning: evaluate metric '{metric_name}' unavailable; using sklearn fallback.",
        )
        return _SklearnMetricFallback(metric_name)


# Load metrics once at module scope.
ACCURACY_METRIC = _load_metric("accuracy")
PRECISION_METRIC = _load_metric("precision")
RECALL_METRIC = _load_metric("recall")
F1_METRIC = _load_metric("f1")

# Whale classes for optimization in multi-class mode.
WHALE_CLASS_NAMES = {"humpback", "resident", "transient"}
CHECKPOINT_SAVE_LIMIT = 6
DEFAULT_MAX_PREPROCESSING_WORKERS = 8


class FeatureExtractorProtocol(Protocol):
    """Protocol for audio feature extractors used during PODS-AI training."""

    def __call__(self, processed_audio: list[np.ndarray], *, sampling_rate: int, padding: bool) -> dict:
        """Convert audio arrays into model inputs.

        Args:
            processed_audio: Fixed-length audio clips for the current batch.
            sampling_rate: Sampling rate of the provided clips.
            padding: Whether the extractor should apply its batch padding logic.

        Returns:
            Dictionary of model-ready features such as input_values or attention_mask.
        """


def setup_label_mappings(num_classes: int) -> None:
    """
    Set up label mappings based on number of classes.

    Args:
        num_classes: 2 for binary (other vs call), 7 for multi-class
    """
    global LABEL2ID, ID2LABEL

    if num_classes == 2:
        # Binary: other (0) vs any whale (1).
        LABEL2ID = {"other": 0, "whale": 1}
        ID2LABEL = {0: "other", 1: "whale"}
        print("Using BINARY classification: other vs whale")
    elif num_classes == 7:
        # Multi-class: water, resident, transient, humpback, vessel, jingle, human.
        LABEL2ID = {"water": 0, "resident": 1, "transient": 2, "humpback": 3, "vessel": 4, "jingle": 5, "human": 6}
        ID2LABEL = {0: "water", 1: "resident", 2: "transient", 3: "humpback", 4: "vessel", 5: "jingle", 6: "human"}
        print("Using MULTI-CLASS classification: water, resident, transient, humpback, vessel, jingle, human")
    else:
        raise ValueError(f"num_classes must be 2 or 7, got {num_classes}")


def load_manifest(
    manifest_path: str,
    num_classes: int,
) -> Dataset:
    import pandas as pd

    df = pd.read_csv(manifest_path)

    required_columns = {"clip_path", "Category"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{manifest_path} missing columns: {missing}"
        )

    if num_classes == 2:
        whale_categories = {
            "resident",
            "transient",
            "humpback",
        }

        df["label"] = df["Category"].apply(
            lambda x: LABEL2ID["whale"]
            if x in whale_categories
            else LABEL2ID["other"]
        )

    else:
        df = df[df["Category"].isin(LABEL2ID.keys())].copy()
        df["label"] = df["Category"].map(LABEL2ID)

    dataset = Dataset.from_dict(
        {
            "audio": df["clip_path"].tolist(),
            "label": df["label"].tolist(),
        }
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000),
    )

    dataset = dataset.cast_column(
        "label",
        ClassLabel(names=list(ID2LABEL.values()))
    )

    return dataset

def preprocess_function(examples: dict, feature_extractor: FeatureExtractorProtocol, max_duration: float = 3.0) -> dict:
    """
    Preprocess audio files for the model.

    Args:
        examples: Batch of examples with audio data
        feature_extractor: HuggingFace audio feature extractor instance
        max_duration: Maximum audio duration in seconds

    Returns:
        Processed inputs for the model (as NumPy arrays for serialization)
    """
    audio_arrays = [x["array"] for x in examples["audio"]]

    # Pad or truncate to max_duration so every training example has the same clip length.
    target_length = int(max_duration * 16000)  # 16kHz sample rate.
    processed_audio = []
    for audio in audio_arrays:
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        processed_audio.append(audio)

    # The audio is already padded/truncated above, so we only need the extractor's
    # feature conversion here.
    inputs = feature_extractor(
        processed_audio,
        sampling_rate=16000,
        padding=True,
    )

    # Ensure input_values is a NumPy array (feature_extractor returns this by default).
    # Convert to list of arrays for proper serialization in datasets library.
    inputs["labels"] = examples["label"]

    return inputs


def get_preprocessing_workers(dataset: DatasetDict, requested_workers: int) -> int:
    """Determine a safe number of dataset preprocessing workers.

    Args:
        dataset: Dataset splits that will be preprocessed.
        requested_workers: User-requested worker count.

    Returns:
        Effective worker count capped by the smallest dataset split.
    """
    if requested_workers < 1:
        raise ValueError(f"preprocessing_workers must be at least 1, got {requested_workers}")

    split_sizes = [len(split_dataset) for split_dataset in dataset.values()]
    if not split_sizes:
        return 1

    return max(1, min(requested_workers, min(split_sizes)))


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Compute evaluation metrics with per-class breakdown.

    Args:
        eval_pred: Predictions and labels

    Returns:
        Dictionary of metrics
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # Overall metrics.
    accuracy = ACCURACY_METRIC.compute(predictions=predictions, references=labels)
    precision = PRECISION_METRIC.compute(predictions=predictions, references=labels, average="weighted")
    recall = RECALL_METRIC.compute(predictions=predictions, references=labels, average="weighted")
    f1 = F1_METRIC.compute(predictions=predictions, references=labels, average="weighted")

    # Per-class metrics with explicit labels to ensure alignment with ID2LABEL ordering.
    all_labels = list(ID2LABEL.keys())
    precision_per_class = PRECISION_METRIC.compute(predictions=predictions, references=labels, average=None, labels=all_labels)
    recall_per_class = RECALL_METRIC.compute(predictions=predictions, references=labels, average=None, labels=all_labels)
    f1_per_class = F1_METRIC.compute(predictions=predictions, references=labels, average=None, labels=all_labels)

    # Confusion matrix analysis.
    print("\n" + "="*60)
    print("DETAILED EVALUATION METRICS")
    print("="*60)
    print("Dataset: user-supplied validation manifest.")

    # Class distribution in predictions vs ground truth.
    print("\nClass Distribution:")
    for class_id, class_name in ID2LABEL.items():
        true_count = np.sum(labels == class_id)
        pred_count = np.sum(predictions == class_id)
        print(f"  {class_name:12s} - True: {true_count:3d}, Predicted: {pred_count:3d}")

    # Per-class performance.
    print("\nPer-Class Performance:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    for class_id, class_name in ID2LABEL.items():
        prec = precision_per_class['precision'][class_id] if len(precision_per_class['precision']) > class_id else 0
        rec = recall_per_class['recall'][class_id] if len(recall_per_class['recall']) > class_id else 0
        f1_score = f1_per_class['f1'][class_id] if len(f1_per_class['f1']) > class_id else 0
        print(f"{class_name:<12} {prec:<12.3f} {rec:<12.3f} {f1_score:<12.3f}")

    # Confusion matrix.
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>12}", end="")
    for class_name in ID2LABEL.values():
        print(f"{class_name[:8]:>10}", end="")
    print()

    for true_class_id, true_class_name in ID2LABEL.items():
        print(f"{true_class_name:>12}", end="")
        for pred_class_id in range(len(ID2LABEL)):
            count = np.sum((labels == true_class_id) & (predictions == pred_class_id))
            print(f"{count:>10}", end="")
        print()

    print("="*60 + "\n")

    # Optimize model selection for whale classes by using macro F1 over
    # humpback, resident, and transient when those labels are present.
    whale_class_ids = sorted(class_id for class_id, class_name in ID2LABEL.items() if class_name in WHALE_CLASS_NAMES)
    if whale_class_ids:
        f1_whale = F1_METRIC.compute(
            predictions=predictions,
            references=labels,
            average="macro",
            labels=whale_class_ids,
        )
        f1_for_training = f1_whale["f1"]
    else:
        f1_for_training = f1["f1"]

    # Return metrics for training logs.
    metrics = {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1_for_training,
    }

    # Add per-class F1 to tracking.
    for class_id, class_name in ID2LABEL.items():
        if len(f1_per_class['f1']) > class_id:
            metrics[f"f1_{class_name}"] = f1_per_class['f1'][class_id]

    return metrics


def analyze_dataset(dataset: DatasetDict) -> None:
    """
    Analyze dataset statistics and distribution.

    Args:
        dataset: DatasetDict with train and test splits
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)

    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        labels = split_data["label"]

        print(f"\n{split_name.upper()} Split ({len(labels)} samples):")

        # Count per class.
        label_counts = Counter(labels)
        for class_id in sorted(label_counts.keys()):
            class_name = ID2LABEL[class_id]
            count = label_counts[class_id]
            percentage = 100 * count / len(labels)
            print(f"  {class_name:12s}: {count:4d} samples ({percentage:5.1f}%)")

        # Check for severe imbalance.
        if len(label_counts) > 0:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
            if imbalance_ratio > 10:
                print("  WARNING: Severe class imbalance detected!")

    print("="*60 + "\n")

def save_loss_plot(trainer, output_dir):
    """
    Save training and validation loss curves from Trainer logs.
    """

    train_steps = []
    train_losses = []

    eval_steps = []
    eval_losses = []

    for entry in trainer.state.log_history:

        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        print("No training loss data found.")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_steps,
        train_losses,
        label="Training Loss",
    )

    if eval_losses:
        plt.plot(
            eval_steps,
            eval_losses,
            label="Validation Loss",
        )

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    output_path = Path(output_dir) / "loss_curve.png"

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved loss plot to {output_path}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train PODS-AI audio classification model for orca calls"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        choices=[2, 7],
        default=7,
        help="Number of classes: 2 for binary (other vs whale), 7 for multi-class (default: 7)",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="Training manifest CSV",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        required=True,
        help="Validation manifest CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/podsai",
        help="Directory to save the trained model (default: model/podsai)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MIT/ast-finetuned-audioset-10-10-0.4593",
        help=(
            "Base model to fine-tune "
            "(default: MIT/ast-finetuned-audioset-10-10-0.4593)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--preprocessing_workers",
        type=int,
        default=max(1, min(DEFAULT_MAX_PREPROCESSING_WORKERS, os.cpu_count() or 1)),
        help="Number of parallel workers for AST feature preprocessing (default: min of 8 or available CPU cores)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="orca-call-detector",
        help="HuggingFace Hub model ID (default: orca-call-detector)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a specific checkpoint to resume training from",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze AST backbone and train classifier head only",
    )

    args = parser.parse_args()

    # Set up label mappings based on num_classes.
    setup_label_mappings(args.num_classes)

    # Set up paths.
    # data_dir = REPO_ROOT / args.data_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # print(f"Loading dataset from {data_dir}...")
    # dataset = load_audio_dataset(data_dir, args.num_classes)
    # print(f"Dataset: {dataset}")

    print(f"Loading manifests...")
    
    train_dataset = load_manifest(
        args.train_manifest,
        args.num_classes,
    )

    train_df = pd.read_csv(args.train_manifest)
    print("\nTraining manifest:")
    print(train_df["Category"].value_counts())

    
    val_dataset = load_manifest(
        args.val_manifest,
        args.num_classes,
    )

    val_df = pd.read_csv(args.val_manifest)
    print("\nValidation manifest:")
    print(val_df["Category"].value_counts())

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "test": val_dataset,
        }
    )

    # Load feature extractor and model.
    print(f"Loading feature extractor and model: {args.model_name}")

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    except Exception as e:
        error_msg = f"Error loading feature extractor from {args.model_name}: {type(e).__name__}: {e}"
        print(error_msg)
        print("Please ensure the model name is correct and you have internet connectivity.")
        raise RuntimeError(error_msg) from e

    try:
        model = AutoModelForAudioClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABEL2ID),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
        if args.freeze_backbone:

            print("Freezing backbone and training classifier head only...")

            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True

            trainable = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            total = sum(
                p.numel()
                for p in model.parameters()
            )

            print(
                f"Trainable parameters: "
                f"{trainable:,} / {total:,}"
            )
    except Exception as e:
        error_msg = f"Error loading model from {args.model_name}: {type(e).__name__}: {e}"
        print(error_msg)
        print("Please ensure the model name is correct and you have internet connectivity.")
        raise RuntimeError(error_msg) from e

    # Preprocess dataset.
    preprocessing_workers = get_preprocessing_workers(dataset, args.preprocessing_workers)
    print(f"Preprocessing dataset with {preprocessing_workers} worker(s)...")
    map_kwargs = {
        "batched": True,
        "remove_columns": ["audio"],
    }
    if preprocessing_workers > 1:
        map_kwargs["num_proc"] = preprocessing_workers
    dataset = dataset.map(
        partial(preprocess_function, feature_extractor=feature_extractor),
        **map_kwargs,
    )

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=CHECKPOINT_SAVE_LIMIT,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=args.push_to_hub,
        hub_strategy="all_checkpoints" if args.push_to_hub else "end",
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    # Create trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Resume from checkpoint if provided.
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        # Train.
        print("Starting training...")
        trainer.train()

    # Evaluate.
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    # Save model and feature extractor.
    print(f"Saving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))

    print("Training complete!")

    if args.push_to_hub:
        # The Trainer already pushed the model weights; also push the feature extractor
        # so that inference code can call AutoFeatureExtractor.from_pretrained()
        # on the Hub model ID.
        print(f"Pushing feature extractor to HuggingFace Hub: {args.hub_model_id}...")
        feature_extractor.push_to_hub(args.hub_model_id)
        print(f"Model pushed to HuggingFace Hub: {args.hub_model_id}")

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    save_loss_plot(
        trainer,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
