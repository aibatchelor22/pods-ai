#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Fine-tune a PODS-AI detector head after transplanting AST backbone weights.

This script creates a detector model from davethaler/whale-call-detector (or a
user-specified detector model), loads only the AST backbone weights from another
model, and then trains the detector classifier head on the existing PODS-AI WAV
directory dataset.

The source model's classification/detection head is intentionally ignored. This
is useful when the source model has a different head, such as a DCLDE multi-task
model, but its AST backbone has useful acoustic representations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

import train_podsai_model as base_train

try:
    from multispecies_train_model import load_training_model as load_multitask_training_model
except Exception:
    load_multitask_training_model = None


DEFAULT_DETECTOR_MODEL = "davethaler/whale-call-detector"


def resolve_path(path: str) -> Path:
    """Resolve absolute paths as-is and repo-relative paths under REPO_ROOT."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return base_train.REPO_ROOT / path_obj


def get_ast_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the AST backbone module from known PODS-AI model wrappers."""
    if hasattr(model, "audio_spectrogram_transformer"):
        return model.audio_spectrogram_transformer
    if hasattr(model, "ast"):
        return model.ast
    raise ValueError(
        f"Could not find an AST backbone on {type(model).__name__}. "
        "Expected .audio_spectrogram_transformer or .ast."
    )


def load_source_ast_model(model_name: str) -> torch.nn.Module:
    """Load a source model whose AST backbone will be copied."""
    try:
        return AutoModelForAudioClassification.from_pretrained(model_name)
    except Exception as auto_error:
        if load_multitask_training_model is None:
            raise auto_error
        print(
            "AutoModelForAudioClassification could not load the AST source; "
            "trying multispecies_train_model.py checkpoint loading."
        )
        try:
            return load_multitask_training_model(
                model_name=model_name,
                dropout=0.0,
                kw_loss_weight=1.0,
                species_loss_weight=1.0,
                ecotype_loss_weight=1.0,
                freeze_backbone=False,
            )
        except Exception as multitask_error:
            raise RuntimeError(
                f"Could not load source AST model {model_name!r} as either a standard "
                "audio-classification model or a DCLDE multi-task checkpoint."
            ) from multitask_error


def transplant_ast_backbone(target_model: torch.nn.Module, source_model: torch.nn.Module) -> None:
    """Copy source AST backbone weights into target AST backbone."""
    target_ast = get_ast_module(target_model)
    source_ast = get_ast_module(source_model)

    source_state = source_ast.state_dict()
    missing_keys, unexpected_keys = target_ast.load_state_dict(source_state, strict=False)
    if missing_keys:
        print(f"Warning: missing AST keys while transplanting backbone: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: unexpected AST keys while transplanting backbone: {unexpected_keys}")
    print(
        "Transplanted AST backbone weights: "
        f"{sum(param.numel() for param in target_ast.parameters()):,} parameters"
    )


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze only the AST backbone."""
    ast = get_ast_module(model)
    for param in ast.parameters():
        param.requires_grad = trainable
    state = "trainable" if trainable else "frozen"
    print(f"AST backbone is {state}.")


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print trainable/total parameter counts."""
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PODS-AI detector head after copying AST backbone weights "
            "from another Hugging Face model."
        )
    )
    parser.add_argument(
        "--ast-weights-model",
        "--ast_weights_model",
        required=True,
        help="Model path or Hugging Face ID whose AST backbone weights should be copied.",
    )
    parser.add_argument(
        "--detector-model",
        "--detector_model",
        default=DEFAULT_DETECTOR_MODEL,
        help=f"Detector architecture/head source model (default: {DEFAULT_DETECTOR_MODEL}).",
    )
    parser.add_argument(
        "--num-classes",
        "--num_classes",
        type=int,
        choices=[2, 7],
        default=7,
        help="Number of detector classes: 2 or 7 (default: 7).",
    )
    parser.add_argument("--data-dir", "--data_dir", default="output/wav")
    parser.add_argument("--output-dir", "--output_dir", default="model/podsai_ast_transplant")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--preprocessing-workers",
        "--preprocessing_workers",
        type=int,
        default=base_train.DEFAULT_MAX_PREPROCESSING_WORKERS,
    )
    parser.add_argument(
        "--unfreeze-backbone",
        "--unfreeze_backbone",
        action="store_true",
        help="Train the transplanted AST backbone too. By default only the detector head trains.",
    )
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", default=None)
    parser.add_argument("--push-to-hub", "--push_to_hub", action="store_true")
    parser.add_argument("--hub-model-id", "--hub_model_id", default="orca-call-detector-ast-transplant")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.preprocessing_workers < 1:
        raise ValueError("--preprocessing-workers must be at least 1.")

    base_train.setup_label_mappings(args.num_classes)

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_dir}...")
    dataset = base_train.load_audio_dataset(data_dir, args.num_classes)
    print(f"Dataset: {dataset}")
    base_train.analyze_dataset(dataset)

    print(f"Loading feature extractor from detector model: {args.detector_model}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.detector_model)

    print(f"Loading detector model/head from: {args.detector_model}")
    target_model = AutoModelForAudioClassification.from_pretrained(
        args.detector_model,
        num_labels=len(base_train.LABEL2ID),
        label2id=base_train.LABEL2ID,
        id2label=base_train.ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    print(f"Loading source AST backbone from: {args.ast_weights_model}")
    source_model = load_source_ast_model(args.ast_weights_model)
    transplant_ast_backbone(target_model, source_model)
    del source_model

    set_backbone_trainable(target_model, trainable=args.unfreeze_backbone)
    print_trainable_parameters(target_model)

    preprocessing_workers = base_train.get_preprocessing_workers(
        dataset,
        args.preprocessing_workers,
    )
    print(f"Preprocessing dataset with {preprocessing_workers} worker(s)...")
    map_kwargs = {
        "batched": True,
        "remove_columns": ["audio"],
    }
    if preprocessing_workers > 1:
        map_kwargs["num_proc"] = preprocessing_workers
    dataset = dataset.map(
        base_train.partial(
            base_train.preprocess_function,
            feature_extractor=feature_extractor,
        ),
        **map_kwargs,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=base_train.CHECKPOINT_SAVE_LIMIT,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=args.push_to_hub,
        hub_strategy="all_checkpoints" if args.push_to_hub else "end",
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    trainer = Trainer(
        model=target_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=base_train.compute_metrics,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    print(f"Saving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))

    if args.push_to_hub:
        print(f"Pushing feature extractor to Hugging Face Hub: {args.hub_model_id}...")
        feature_extractor.push_to_hub(args.hub_model_id)
        print(f"Model pushed to Hugging Face Hub: {args.hub_model_id}")

    print("Training complete!")


if __name__ == "__main__":
    main()
