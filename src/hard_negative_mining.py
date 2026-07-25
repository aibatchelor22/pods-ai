#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Mine hard negative 3-second clips from source audio referenced by a training manifest.

The script only considers source files present in the provided training manifest.
For each source audio file, it downloads one GCS object, samples candidate windows
that do not overlap existing manifest clip intervals, runs the multi-species AST
model, saves high-confidence non-background false positives as BKG examples, then
deletes the downloaded source file.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor

try:
    import gcsfs
except ImportError:
    gcsfs = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from multispecies_train_model import (
    DEFAULT_MAX_DURATION,
    ECOTYPE_ID2LABEL,
    ECOTYPE_LABELS,
    REPO_ROOT,
    SAMPLE_RATE,
    SPECIES_ID2LABEL,
    SPECIES_LABELS,
    load_multitask_checkpoint_files,
    load_training_model,
)


BACKGROUND_LABEL = "BKG"
BACKGROUND_SOURCE_LABELS = {"BKG", "UndBio"}
MODEL_SPECIES_LABELS = ("background", "AB", "HW", "KW")
MODEL_ECOTYPE_LABELS = ("SRKW", "TKW", "NRKW", "OKW", "SAR")
EXTRA_COLUMNS = [
    "p_BKG",
    "p_AB",
    "p_HW",
    "p_KW",
    "p_SRKW",
    "p_TKW",
    "p_NRKW",
    "p_OKW",
    "p_SAR",
    "predicted_species",
    "predicted_ecotype",
    "hard_negative_score",
]


def resolve_path(path: str) -> Path:
    """Resolve absolute paths as-is and repo-relative paths under REPO_ROOT."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def progress(iterable, **kwargs):
    """Wrap an iterable in tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def safe_name(value: str) -> str:
    """Make a filesystem-safe stem component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def normalize_gcs_path(value: Any) -> str:
    """Normalize manifest GCS paths to the form expected by gcsfs."""
    path = str(value).strip()
    if path.startswith("gs://"):
        return path[5:]
    return path


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


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the training manifest."""
    df = pd.read_csv(path, low_memory=False)
    required = {"gcs_path", "ClassSpecies", "ClipStartSec", "ClipEndSec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["gcs_path"] = df["gcs_path"].map(normalize_gcs_path)
    df = df[df["gcs_path"].astype(str).str.len() > 0].reset_index(drop=True)
    return df


def interval_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Return the best available start/end columns for labeled intervals."""
    if {"ClipStartSec", "ClipEndSec"}.issubset(frame.columns):
        return "ClipStartSec", "ClipEndSec"
    return "FileBeginSec", "FileEndSec"


def labeled_intervals(frame: pd.DataFrame, buffer_sec: float) -> list[tuple[float, float]]:
    """Return sorted labeled intervals for a source file, expanded by buffer_sec."""
    start_col, end_col = interval_columns(frame)
    starts = pd.to_numeric(frame[start_col], errors="coerce")
    ends = pd.to_numeric(frame[end_col], errors="coerce")

    intervals = []
    for start, end in zip(starts, ends):
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        if end <= start:
            continue
        intervals.append((max(0.0, float(start) - buffer_sec), float(end) + buffer_sec))
    return sorted(intervals)


def overlaps_any(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    """Return True when [start, end) overlaps any labeled interval."""
    for interval_start, interval_end in intervals:
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        return True
    return False


def sample_candidate_starts(
    duration_sec: float,
    clip_duration: float,
    intervals: list[tuple[float, float]],
    max_candidates: int,
    max_attempts: int,
    rng: np.random.Generator,
) -> list[float]:
    """Sample non-overlapping candidate starts outside existing labeled intervals."""
    latest_start = duration_sec - clip_duration
    if latest_start < 0:
        return []

    starts = []
    seen = set()
    attempts = 0
    while len(starts) < max_candidates and attempts < max_attempts:
        attempts += 1
        start = float(rng.uniform(0.0, latest_start)) if latest_start > 0 else 0.0
        start = round(start, 3)
        if start in seen:
            continue
        seen.add(start)
        end = start + clip_duration
        if overlaps_any(start, end, intervals):
            continue
        starts.append(start)
    return sorted(starts)


def read_clip(
    source_path: Path,
    start_sec: float,
    clip_duration: float,
    target_sample_rate: int,
) -> np.ndarray:
    """Read one clip from a local WAV/FLAC file and return mono target-rate audio."""
    with sf.SoundFile(source_path) as sound_file:
        source_sample_rate = sound_file.samplerate
        frame_start = max(0, int(round(start_sec * source_sample_rate)))
        frame_count = int(math.ceil(clip_duration * source_sample_rate))
        sound_file.seek(min(frame_start, len(sound_file)))
        audio = sound_file.read(frames=frame_count, dtype="float32", always_2d=True)

    if audio.size == 0:
        audio = np.zeros((0,), dtype=np.float32)
    else:
        audio = audio.mean(axis=1).astype(np.float32, copy=False)

    if source_sample_rate != target_sample_rate and len(audio) > 0:
        audio = librosa.resample(audio, orig_sr=source_sample_rate, target_sr=target_sample_rate)

    target_length = int(round(clip_duration * target_sample_rate))
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    return audio.astype(np.float32, copy=False)


def run_batch_inference(
    model: torch.nn.Module,
    feature_extractor: Any,
    audio_batch: list[np.ndarray],
    sample_rate: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return species and ecotype probabilities for a batch of clips."""
    inputs = feature_extractor(
        audio_batch,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        _, species_logits, ecotype_logits = outputs["logits"]
        species_probs = torch.softmax(species_logits, dim=-1).cpu().numpy()
        ecotype_probs = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()
    return species_probs, ecotype_probs


def prediction_metadata(species_probs: np.ndarray, ecotype_probs: np.ndarray) -> dict[str, Any]:
    """Build hard-negative score and probability metadata for one prediction."""
    species_id = int(np.argmax(species_probs))
    ecotype_id = int(np.argmax(ecotype_probs))
    predicted_species = SPECIES_ID2LABEL[species_id]
    predicted_ecotype = ECOTYPE_ID2LABEL[ecotype_id]
    whale_score = max(
        float(species_probs[SPECIES_LABELS["AB"]]),
        float(species_probs[SPECIES_LABELS["HW"]]),
        float(species_probs[SPECIES_LABELS["KW"]]),
    )

    return {
        "p_BKG": float(species_probs[SPECIES_LABELS["background"]]),
        "p_AB": float(species_probs[SPECIES_LABELS["AB"]]),
        "p_HW": float(species_probs[SPECIES_LABELS["HW"]]),
        "p_KW": float(species_probs[SPECIES_LABELS["KW"]]),
        "p_SRKW": float(ecotype_probs[ECOTYPE_LABELS["SRKW"]]),
        "p_TKW": float(ecotype_probs[ECOTYPE_LABELS["TKW"]]),
        "p_NRKW": float(ecotype_probs[ECOTYPE_LABELS["NRKW"]]),
        "p_OKW": float(ecotype_probs[ECOTYPE_LABELS["OKW"]]),
        "p_SAR": float(ecotype_probs[ECOTYPE_LABELS["SAR"]]),
        "predicted_species": predicted_species,
        "predicted_ecotype": predicted_ecotype,
        "hard_negative_score": whale_score,
    }


def build_manifest_row(
    template: pd.Series,
    manifest_columns: list[str],
    clip_path: Path,
    clip_filename: str,
    start_sec: float,
    clip_duration: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a training-compatible BKG manifest row plus mining metadata."""
    row = {column: template.get(column, "") for column in manifest_columns}
    end_sec = start_sec + clip_duration
    center_sec = start_sec + clip_duration / 2.0

    row.update(
        {
            "ClassSpecies": BACKGROUND_LABEL,
            "KW": 0,
            "KW_certain": "",
            "Ecotype": "",
            "AnnotationLevel": "HardNegative",
            "FileBeginSec": start_sec,
            "FileEndSec": end_sec,
            "CenterSec": center_sec,
            "ClipStartSec": start_sec,
            "ClipEndSec": end_sec,
            "clip_filename": clip_filename,
            "clip_path": str(clip_path),
            "Generated": True,
            "BackgroundBufferSec": "",
        }
    )
    row.update(metadata)
    return row


def append_rows(rows: list[dict[str, Any]], output_manifest: Path, fieldnames: list[str]) -> None:
    """Append mined rows to the output manifest."""
    if not rows:
        return
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_manifest.exists()
    pd.DataFrame(rows, columns=fieldnames).to_csv(
        output_manifest,
        mode="a",
        header=write_header,
        index=False,
    )


def download_gcs_file(fs: Any, gcs_path: str, destination: Path) -> None:
    """Download one GCS object to destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fs.get(gcs_path, str(destination))


def already_processed_sources(output_manifest: Path) -> set[str]:
    """Return source GCS paths already represented in an existing output manifest."""
    if not output_manifest.exists():
        return set()
    try:
        output_df = pd.read_csv(output_manifest, usecols=["gcs_path"], low_memory=False)
    except ValueError:
        return set()
    return set(output_df["gcs_path"].dropna().map(normalize_gcs_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine high-confidence non-background false positives as hard negatives."
    )
    parser.add_argument("--model-name", required=True, help="Local path or Hugging Face ID for the checkpoint.")
    parser.add_argument("--training-manifest", required=True, help="Training manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory where hard-negative clips are saved.")
    parser.add_argument("--output-manifest", required=True, help="CSV manifest for mined hard negatives.")
    parser.add_argument("--temp-dir", default=None, help="Temporary download directory.")
    parser.add_argument("--score-threshold", type=float, default=0.80)
    parser.add_argument("--max-candidate-clips-per-file", type=int, default=100)
    parser.add_argument("--max-hard-negatives-per-file", type=int, default=20)
    parser.add_argument("--clip-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--min-gap-from-label-sec", type=float, default=0.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--gcs-token", default="anon", help="gcsfs token setting, default: anon.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip sources already in output manifest.")
    parser.add_argument("--shuffle-files", action="store_true", help="Shuffle source-file order before mining.")
    args = parser.parse_args()

    if gcsfs is None:
        raise ImportError("hard_negative_mining.py requires gcsfs. Install it with: pip install gcsfs")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if args.max_candidate_clips_per_file <= 0:
        raise ValueError("--max-candidate-clips-per-file must be positive.")
    if args.max_hard_negatives_per_file <= 0:
        raise ValueError("--max-hard-negatives-per-file must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.clip_duration <= 0:
        raise ValueError("--clip-duration must be positive.")

    manifest_path = resolve_path(args.training_manifest)
    output_dir = resolve_path(args.output_dir)
    output_manifest = resolve_path(args.output_manifest)
    temp_root = Path(args.temp_dir) if args.temp_dir else Path(tempfile.gettempdir()) / "podsai_hard_negative_mining"
    temp_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    manifest_columns = list(manifest.columns)
    output_columns = manifest_columns + [column for column in EXTRA_COLUMNS if column not in manifest_columns]

    grouped = list(manifest.groupby("gcs_path", sort=False))
    if args.shuffle_files:
        rng_for_order = np.random.default_rng(args.seed)
        rng_for_order.shuffle(grouped)
    if args.max_files is not None:
        grouped = grouped[: args.max_files]

    if not args.no_resume:
        done_sources = already_processed_sources(output_manifest)
        if done_sources:
            grouped = [(gcs_path, group) for gcs_path, group in grouped if gcs_path not in done_sources]
            print(f"Resume enabled: skipping {len(done_sources):,} source files already in output manifest.")

    model_source = resolve_model_source(args.model_name)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
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

    fs = gcsfs.GCSFileSystem(token=args.gcs_token)
    rng = np.random.default_rng(args.seed)
    total_saved = 0
    total_candidates = 0

    for source_index, (gcs_path, group) in enumerate(
        progress(grouped, desc="Source files", unit="file"),
        start=1,
    ):
        soundfile_name = str(group.iloc[0].get("Soundfile", Path(gcs_path).name))
        local_source = temp_root / safe_name(soundfile_name)
        saved_for_file = 0

        try:
            download_gcs_file(fs, gcs_path, local_source)
            with sf.SoundFile(local_source) as source_audio:
                duration_sec = len(source_audio) / float(source_audio.samplerate)

            intervals = labeled_intervals(group, buffer_sec=args.min_gap_from_label_sec)
            candidate_starts = sample_candidate_starts(
                duration_sec=duration_sec,
                clip_duration=args.clip_duration,
                intervals=intervals,
                max_candidates=args.max_candidate_clips_per_file,
                max_attempts=args.max_candidate_clips_per_file * 50,
                rng=rng,
            )
            total_candidates += len(candidate_starts)

            pending_rows = []
            for batch_start in range(0, len(candidate_starts), args.batch_size):
                if saved_for_file >= args.max_hard_negatives_per_file:
                    break

                starts = candidate_starts[batch_start : batch_start + args.batch_size]
                audio_batch = [
                    read_clip(local_source, start, args.clip_duration, args.sample_rate)
                    for start in starts
                ]
                species_probs, ecotype_probs = run_batch_inference(
                    model=model,
                    feature_extractor=feature_extractor,
                    audio_batch=audio_batch,
                    sample_rate=args.sample_rate,
                    device=device,
                )

                for row_index, start_sec in enumerate(starts):
                    if saved_for_file >= args.max_hard_negatives_per_file:
                        break

                    metadata = prediction_metadata(species_probs[row_index], ecotype_probs[row_index])
                    if metadata["predicted_species"] == "background":
                        continue
                    if metadata["hard_negative_score"] < args.score_threshold:
                        continue

                    score_milli = int(round(metadata["hard_negative_score"] * 1000))
                    start_milli = int(round(start_sec * 1000))
                    clip_filename = (
                        f"hardneg_{safe_name(Path(soundfile_name).stem)}_"
                        f"{start_milli:010d}ms_s{score_milli:03d}.wav"
                    )
                    clip_path = output_dir / clip_filename
                    sf.write(clip_path, audio_batch[row_index], args.sample_rate)

                    pending_rows.append(
                        build_manifest_row(
                            template=group.iloc[0],
                            manifest_columns=manifest_columns,
                            clip_path=clip_path,
                            clip_filename=clip_filename,
                            start_sec=start_sec,
                            clip_duration=args.clip_duration,
                            metadata=metadata,
                        )
                    )
                    saved_for_file += 1
                    total_saved += 1

            append_rows(pending_rows, output_manifest, output_columns)
            print(
                f"[{source_index}/{len(grouped)}] {soundfile_name}: "
                f"candidates={len(candidate_starts)}, saved={saved_for_file}"
            )

        except Exception as exc:
            print(f"Error processing {gcs_path}: {type(exc).__name__}: {exc}")
        finally:
            try:
                local_source.unlink(missing_ok=True)
            except OSError:
                shutil.rmtree(local_source, ignore_errors=True)

    print()
    print(f"Done. Candidate clips evaluated: {total_candidates:,}")
    print(f"Hard negatives saved: {total_saved:,}")
    print(f"Clips directory: {output_dir}")
    print(f"Manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
