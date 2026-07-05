#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Generate AST embeddings from a DCLDE multi-task model for every row in a manifest.

The output CSV preserves all original manifest columns and appends:
  - KW/species/ecotype predictions, confidences, and per-class probabilities
  - one embedding column per AST hidden dimension

Embeddings are the same pooled AST representation used by the multi-task heads:
the average of AST token 0 and token 1 when both are present, otherwise token 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from multispecies_train_model import (
    DEFAULT_MAX_DURATION,
    ECOTYPE_ID2LABEL,
    KW_ID2LABEL,
    REPO_ROOT,
    SAMPLE_RATE,
    SPECIES_ID2LABEL,
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


def iter_batches(total_rows: int, batch_size: int):
    """Yield batch start offsets with tqdm when available."""
    starts = range(0, total_rows, batch_size)
    if tqdm is None:
        return starts
    return tqdm(starts, desc="Embedding batches")


def load_audio_batch(
    paths: list[str],
    sample_rate: int = SAMPLE_RATE,
    duration: float = DEFAULT_MAX_DURATION,
) -> list[np.ndarray]:
    """Load, mono-convert, pad, and truncate a batch of audio clips."""
    target_length = int(sample_rate * duration)
    audio_batch = []

    for path in paths:
        try:
            audio, _ = librosa.load(path, sr=sample_rate, mono=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio clip: {path}") from exc

        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
        audio_batch.append(audio.astype(np.float32, copy=False))

    return audio_batch


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move feature extractor outputs to the inference device."""
    return {key: value.to(device) for key, value in batch.items()}


def probability_columns(prefix: str, probabilities: np.ndarray, id2label: dict[int, str]) -> dict[str, float]:
    """Return per-class probability columns for one prediction row."""
    return {
        f"{prefix}_prob_{id2label[class_id]}": float(probabilities[class_id])
        for class_id in sorted(id2label)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate pooled AST embeddings from a DCLDE multi-task model manifest."
    )
    parser.add_argument("--model-name", "--model_name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", "--output_csv", required=True)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--sample-rate", "--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument(
        "--clip-path-column",
        default="clip_path",
        help="Manifest column containing audio clip paths.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda, cpu. Defaults to cuda when available.",
    )
    args = parser.parse_args()

    model_source = resolve_model_source(args.model_name)
    manifest_path = resolve_path(args.manifest)
    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {device}")
    print(f"Loading manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if args.clip_path_column not in manifest.columns:
        raise ValueError(
            f"{manifest_path} missing clip path column {args.clip_path_column!r}. "
            f"Available columns: {list(manifest.columns)}"
        )

    print(f"Loading feature extractor from: {model_source}")
    feature_extractor = load_feature_extractor(model_source)

    print(f"Loading multi-task model from: {model_source}")
    model = load_training_model(
        model_name=model_source,
        dropout=0.0,
        kw_loss_weight=1.0,
        species_loss_weight=1.0,
        ecotype_loss_weight=1.0,
        freeze_backbone=False,
    )
    model.to(device)
    model.eval()

    rows_written = 0
    wrote_header = False
    preview: Optional[pd.DataFrame] = None
    total_rows = len(manifest)
    for start in iter_batches(total_rows, args.batch_size):
        batch = manifest.iloc[start : start + args.batch_size]
        clip_paths = batch[args.clip_path_column].astype(str).tolist()
        audio = load_audio_batch(
            clip_paths,
            sample_rate=args.sample_rate,
            duration=args.duration,
        )

        inputs = feature_extractor(
            audio,
            sampling_rate=args.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = to_device(inputs, device)

        with torch.inference_mode():
            ast_outputs = model.ast(input_values=inputs["input_values"], return_dict=True)
            embeddings_tensor = model._pool_ast_output(ast_outputs.last_hidden_state)
            kw_logits = model.kw_classifier(embeddings_tensor)
            species_logits = model.species_classifier(embeddings_tensor)
            ecotype_logits = model.ecotype_classifier(embeddings_tensor)

            embeddings = embeddings_tensor.cpu().float().numpy()
            kw_probs = torch.softmax(kw_logits, dim=-1).cpu().numpy()
            species_probs = torch.softmax(species_logits, dim=-1).cpu().numpy()
            ecotype_probs = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()

        kw_pred = np.argmax(kw_probs, axis=1)
        species_pred = np.argmax(species_probs, axis=1)
        ecotype_pred = np.argmax(ecotype_probs, axis=1)

        batch_rows: list[dict[str, Any]] = []
        for row_index, (_, manifest_row) in enumerate(batch.iterrows()):
            result = manifest_row.to_dict()

            kw_id = int(kw_pred[row_index])
            species_id = int(species_pred[row_index])
            ecotype_id = int(ecotype_pred[row_index])
            result.update(
                {
                    "kw_prediction": kw_id,
                    "kw_prediction_label": KW_ID2LABEL[kw_id],
                    "kw_confidence": float(kw_probs[row_index, kw_id]),
                    "species_prediction": species_id,
                    "species_prediction_label": SPECIES_ID2LABEL[species_id],
                    "species_confidence": float(species_probs[row_index, species_id]),
                    "ecotype_prediction": ecotype_id,
                    "ecotype_prediction_label": ECOTYPE_ID2LABEL[ecotype_id],
                    "ecotype_confidence": float(ecotype_probs[row_index, ecotype_id]),
                }
            )
            result.update(probability_columns("kw", kw_probs[row_index], KW_ID2LABEL))
            result.update(
                probability_columns("species", species_probs[row_index], SPECIES_ID2LABEL)
            )
            result.update(
                probability_columns("ecotype", ecotype_probs[row_index], ECOTYPE_ID2LABEL)
            )

            for embedding_index, value in enumerate(embeddings[row_index]):
                result[f"embedding_{embedding_index}"] = float(value)

            batch_rows.append(result)

        output_batch = pd.DataFrame(batch_rows)
        if preview is None:
            preview = output_batch.head()
        output_batch.to_csv(
            output_csv,
            index=False,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
        )
        wrote_header = True
        rows_written += len(output_batch)

    if preview is not None:
        print(preview)
    print()
    print(f"Saved {rows_written:,} embeddings to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
