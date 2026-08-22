#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Evaluate a DCLDE multi-task AST detector on NOAA long recordings.

Annotations.csv supplies both ground truth and the unique Soundfile inventory.
Audio objects are discovered anonymously in the public NOAA GCS bucket and are
downloaded, evaluated, and deleted one recording at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from io import TextIOWrapper
from typing import Any, Iterable, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from compare_new_models_experimantal_2 import load_multispecies_feature_extractor
from multispecies_train_model import (
    ECOTYPE_ID2LABEL,
    ECOTYPE_LABELS,
    KW_LABELS,
    SAMPLE_RATE,
    SPECIES_ID2LABEL,
    SPECIES_LABELS,
    load_multitask_checkpoint_files,
    load_training_model,
)


LOGGER = logging.getLogger("dclde_long_evaluation")
ECOTYPE_NAMES = ("SRKW", "NRKW", "TKW", "OKW", "SAR")
KW_SPECIES_NAMES = {"kw", "killer whale", "killer_whale", "killerwhale", "orca"}
REMOTE_SCHEMES = ("gs://", "http://", "https://")
DEFAULT_ANNOTATIONS = (
    "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/"
    "dclde_2027_killer_whales/Annotations.csv"
)
DEFAULT_GCS_BASE = (
    "noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales"
)
DEFAULT_GCS_PROVIDERS = (
    "dfo_crp", "dfo_wdlp", "onc", "orcasound", "scripps", "simres", "smru", "uaf", "vfpa"
)


@dataclass(frozen=True)
class Recording:
    soundfile: str
    audio_source: str
    provider: str
    dataset: str
    duration_sec: float


@dataclass(frozen=True)
class GroundTruthEvent:
    event_id: str
    soundfile: str
    start_sec: float
    end_sec: float
    ecotype: str
    provider: str
    dataset: str


@dataclass(frozen=True)
class LabeledAnnotation:
    soundfile: str
    start_sec: float
    end_sec: float
    species: str
    ecotype: str


@dataclass(frozen=True)
class WindowPrediction:
    soundfile: str
    start_sec: float
    end_sec: float
    kw_probability: float
    kw_prediction: bool
    species_prediction: str
    species_probabilities: tuple[float, ...]
    ecotype_prediction: str
    ecotype_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PredictedEvent:
    event_id: str
    soundfile: str
    start_sec: float
    end_sec: float
    max_kw_probability: float
    mean_kw_probability: float
    number_of_windows: int
    predicted_ecotype: str
    ecotype_confidence: float
    ecotype_probabilities: tuple[float, ...]
    provider: str
    dataset: str


@dataclass(frozen=True)
class EventMatch:
    soundfile: str
    status: str
    predicted_event_id: str
    true_event_id: str
    predicted_start_sec: Optional[float]
    predicted_end_sec: Optional[float]
    true_start_sec: Optional[float]
    true_end_sec: Optional[float]
    temporal_iou: float
    overlap_sec: float
    predicted_ecotype: str
    true_ecotype: str
    ambiguous_ecotype_overlap: bool
    overlapping_true_ecotypes: str
    provider: str
    dataset: str


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def find_column(fieldnames: Iterable[str], aliases: Iterable[str], required: bool = True) -> Optional[str]:
    lookup = {name.casefold(): name for name in fieldnames}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    if required:
        raise ValueError(f"Missing required column; expected one of {list(aliases)}")
    return None


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = str(path)
    if source.startswith(("http://", "https://")):
        response = urllib.request.urlopen(source, timeout=120)
        file_context = TextIOWrapper(response, encoding="utf-8-sig", newline="")
    else:
        file_context = Path(path).open(newline="", encoding="utf-8-sig")
    with file_context as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"CSV has no header: {source}")
        return fields, list(reader)


def build_noaa_gcs_index(
    base_gcs: str,
    providers: tuple[str, ...],
    required_names: set[str],
    max_file_gb: float,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Index public NOAA audio objects, retaining only requested basenames."""
    try:
        import gcsfs
    except ImportError as error:
        raise ImportError(
            "GCS discovery requires gcsfs. On Kaggle run: !pip install -q gcsfs"
        ) from error

    fs = gcsfs.GCSFileSystem(token="anon")
    matches: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    oversized = []
    for provider in providers:
        prefix = f"{base_gcs.rstrip('/')}/{provider}/audio"
        LOGGER.info("Indexing gs://%s", prefix)
        try:
            objects = fs.find(prefix, detail=True)
        except Exception as error:
            LOGGER.warning("Could not index %s: %s", prefix, error)
            continue
        items = objects.items() if isinstance(objects, dict) else ((item, {}) for item in objects)
        for object_path, metadata in items:
            scanned += 1
            basename = Path(object_path).name
            if basename not in required_names:
                continue
            size_bytes = int((metadata or {}).get("size", 0) or 0)
            size_gb = size_bytes / 1024**3
            if size_bytes and size_gb > max_file_gb:
                oversized.append((basename, size_gb))
                continue
            matches[basename].append(f"gs://{object_path}")
    return matches, {
        "gcs_objects_scanned": scanned,
        "gcs_soundfiles_matched": len(matches),
        "oversized_audio_count": len(oversized),
        "oversized_audio_examples": oversized[:25],
    }


def remote_download_url(source: str) -> str:
    if source.startswith("gs://"):
        bucket_and_object = source[5:]
        if "/" not in bucket_and_object:
            raise ValueError(f"Google Storage URI has no object name: {source}")
        bucket, object_name = bucket_and_object.split("/", 1)
        return (
            f"https://storage.googleapis.com/{urllib.parse.quote(bucket)}/"
            f"{urllib.parse.quote(object_name, safe='/')}"
        )
    return source


@contextmanager
def materialize_audio(source: str, soundfile: str, temp_dir: Optional[str]):
    """Yield a local audio path, downloading one remote object at a time."""
    if not source.startswith(REMOTE_SCHEMES):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        yield path
        return

    parent = Path(temp_dir) if temp_dir else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dclde_audio_", dir=parent) as directory:
        suffix = Path(soundfile).suffix or Path(urllib.parse.urlparse(source).path).suffix or ".audio"
        destination = Path(directory) / f"recording{suffix}"
        request = urllib.request.Request(
            remote_download_url(source),
            headers={"User-Agent": "pods-ai-dclde-evaluator/1.0"},
        )
        LOGGER.info("Downloading %s", source)
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as file:
            shutil.copyfileobj(response, file, length=8 * 1024 * 1024)
        yield destination


def audio_metadata(path: Path) -> tuple[float, int, int]:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Invalid audio metadata: {path}")
    return info.frames / info.samplerate, int(info.samplerate), int(info.channels)


def load_recordings(
    args: argparse.Namespace,
    annotation_table: tuple[list[str], list[dict[str, str]]],
) -> tuple[list[Recording], dict[str, Any]]:
    fields, rows = annotation_table
    sound_col = find_column(fields, ("Soundfile", "soundfile", "filename", "file"))
    provider_col = find_column(fields, ("Provider", "provider"), required=False)
    dataset_col = find_column(fields, ("Dataset", "dataset"), required=False)
    metadata_by_name: dict[str, tuple[str, str]] = {}
    metadata_conflicts = []
    for row in rows:
        soundfile = normalize_text(row.get(sound_col))
        if not soundfile:
            continue
        metadata = (
            normalize_text(row.get(provider_col)) if provider_col else "",
            normalize_text(row.get(dataset_col)) if dataset_col else "",
        )
        if soundfile in metadata_by_name and metadata_by_name[soundfile] != metadata:
            metadata_conflicts.append((soundfile, metadata_by_name[soundfile], metadata))
        else:
            metadata_by_name[soundfile] = metadata

    requested_names = list(metadata_by_name)
    if args.max_files is not None:
        requested_names = requested_names[: args.max_files]
    required_names = set(requested_names)
    gcs_index, gcs_sanity = build_noaa_gcs_index(
        args.gcs_base,
        tuple(item.strip() for item in args.gcs_providers.split(",") if item.strip()),
        required_names,
        args.max_file_gb,
    )
    recordings = []
    missing = []
    ambiguous = []
    for soundfile in requested_names:
        candidates = gcs_index.get(soundfile, [])
        if not candidates:
            missing.append(soundfile)
            continue
        if len(candidates) > 1:
            ambiguous.append((soundfile, candidates))
        provider, dataset = metadata_by_name[soundfile]
        recordings.append(
            Recording(
                soundfile=soundfile,
                audio_source=sorted(candidates)[0],
                provider=provider,
                dataset=dataset,
                duration_sec=0.0,
            )
        )
    sanity = {
        "annotation_rows_used_as_inventory": len(rows),
        "unique_annotated_soundfiles": len(metadata_by_name),
        "requested_soundfiles": len(requested_names),
        "matched_recordings": len(recordings),
        "missing_gcs_audio_count": len(missing),
        "missing_gcs_audio_examples": missing[:25],
        "ambiguous_gcs_basename_count": len(ambiguous),
        "ambiguous_gcs_basename_examples": ambiguous[:10],
        "metadata_conflict_count": len(metadata_conflicts),
        "metadata_conflict_examples": metadata_conflicts[:25],
        **gcs_sanity,
    }
    return recordings, sanity


def is_kw_species(value: Any) -> bool:
    return normalize_text(value).casefold() in KW_SPECIES_NAMES


def load_annotations(
    path: str | Path,
    recordings: list[Recording],
    annotation_table: Optional[tuple[list[str], list[dict[str, str]]]] = None,
) -> tuple[dict[str, list[GroundTruthEvent]], dict[str, Any]]:
    fields, rows = annotation_table if annotation_table is not None else read_csv(path)
    sound_col = find_column(fields, ("Soundfile", "soundfile", "filename", "file"))
    start_col = find_column(
        fields, ("Start", "start", "start_sec", "start_time_sec", "FileBeginSec")
    )
    end_col = find_column(
        fields, ("End", "end", "end_sec", "end_time_sec", "FileEndSec")
    )
    species_col = find_column(fields, ("ClassSpecies", "class_species", "species"))
    ecotype_col = find_column(fields, ("Ecotype", "ecotype"), required=False)
    provider_col = find_column(fields, ("Provider", "provider"), required=False)
    dataset_col = find_column(fields, ("Dataset", "dataset"), required=False)
    recording_by_name = {recording.soundfile: recording for recording in recordings}
    by_file: dict[str, list[GroundTruthEvent]] = defaultdict(list)
    outside = []
    malformed = []
    duplicate_keys = Counter()
    kw_rows = 0

    for row_index, row in enumerate(rows):
        if not is_kw_species(row.get(species_col)):
            continue
        kw_rows += 1
        soundfile = normalize_text(row.get(sound_col))
        recording = recording_by_name.get(soundfile)
        if recording is None:
            continue
        try:
            start = float(row[start_col])
            end = float(row[end_col])
        except (TypeError, ValueError):
            malformed.append(f"row {row_index}: invalid Start/End")
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            malformed.append(f"row {row_index}: invalid interval {start}..{end}")
            continue
        if start < 0 or (recording.duration_sec > 0 and end > recording.duration_sec):
            outside.append(
                f"row {row_index} {soundfile}: {start:.3f}..{end:.3f} outside 0..{recording.duration_sec:.3f}"
            )
        start = max(0.0, start)
        if recording.duration_sec > 0:
            end = min(recording.duration_sec, end)
        if end <= start:
            continue
        ecotype = normalize_text(row.get(ecotype_col)).upper() if ecotype_col else ""
        if ecotype not in ECOTYPE_LABELS:
            ecotype = ""
        provider = normalize_text(row.get(provider_col)) if provider_col else recording.provider
        dataset = normalize_text(row.get(dataset_col)) if dataset_col else recording.dataset
        duplicate_key = (soundfile, round(start, 6), round(end, 6), ecotype)
        duplicate_keys[duplicate_key] += 1
        by_file[soundfile].append(
            GroundTruthEvent(
                event_id=f"true_{row_index}",
                soundfile=soundfile,
                start_sec=start,
                end_sec=end,
                ecotype=ecotype,
                provider=provider,
                dataset=dataset,
            )
        )
    for events in by_file.values():
        events.sort(key=lambda event: (event.start_sec, event.end_sec))
    duplicates = [key for key, count in duplicate_keys.items() if count > 1]
    sanity = {
        "annotation_rows": len(rows),
        "kw_annotation_rows": kw_rows,
        "valid_kw_events": sum(len(events) for events in by_file.values()),
        "ecotype_counts": dict(
            Counter(event.ecotype or "unknown" for events in by_file.values() for event in events)
        ),
        "outside_duration_count": len(outside),
        "outside_duration_examples": outside[:25],
        "malformed_annotation_count": len(malformed),
        "malformed_annotation_examples": malformed[:25],
        "duplicate_annotation_count": len(duplicates),
        "duplicate_annotation_examples": [str(key) for key in duplicates[:25]],
    }
    return by_file, sanity


def load_all_labeled_annotations(
    annotation_table: tuple[list[str], list[dict[str, str]]],
    recordings: list[Recording],
) -> dict[str, list[LabeledAnnotation]]:
    fields, rows = annotation_table
    sound_col = find_column(fields, ("Soundfile", "soundfile", "filename", "file"))
    start_col = find_column(fields, ("FileBeginSec", "Start", "start", "start_sec"))
    end_col = find_column(fields, ("FileEndSec", "End", "end", "end_sec"))
    species_col = find_column(fields, ("ClassSpecies", "class_species", "species"))
    ecotype_col = find_column(fields, ("Ecotype", "ecotype"), required=False)
    selected = {recording.soundfile for recording in recordings}
    by_file: dict[str, list[LabeledAnnotation]] = defaultdict(list)
    for row in rows:
        soundfile = normalize_text(row.get(sound_col))
        if soundfile not in selected:
            continue
        raw_species = normalize_text(row.get(species_col))
        species = "background" if raw_species in {"UndBio", "BKG"} else raw_species
        if species not in SPECIES_ID2LABEL.values():
            continue
        try:
            start = float(row[start_col])
            end = float(row[end_col])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        ecotype = normalize_text(row.get(ecotype_col)).upper() if ecotype_col else ""
        if ecotype not in ECOTYPE_LABELS:
            ecotype = ""
        by_file[soundfile].append(
            LabeledAnnotation(soundfile, max(0.0, start), end, species, ecotype)
        )
    for events in by_file.values():
        events.sort(key=lambda event: (event.start_sec, event.end_sec))
    return by_file


def overlapping_ground_truth(
    start: float,
    end: float,
    annotations: list[LabeledAnnotation],
) -> tuple[str, str, bool]:
    overlapping = [
        event
        for event in annotations
        if interval_overlap(start, end, event.start_sec, event.end_sec) > 0
    ]
    if not overlapping:
        return "background", "", False
    species = {event.species for event in overlapping}
    ecotypes = {event.ecotype for event in overlapping if event.species == "KW" and event.ecotype}
    best = max(
        overlapping,
        key=lambda event: interval_overlap(start, end, event.start_sec, event.end_sec),
    )
    return best.species, best.ecotype if best.species == "KW" else "", (
        len(species) > 1 or len(ecotypes) > 1
    )


def thresholded_label(
    scores: dict[str, float], thresholds: dict[str, float]
) -> tuple[str, float]:
    surviving = {
        label: float(score)
        for label, score in scores.items()
        if float(score) >= thresholds[label]
    }
    if not surviving:
        return "unclassified", 0.0
    return max(surviving.items(), key=lambda item: item[1])


def clip_truths_to_duration(
    truths: list[GroundTruthEvent], duration_sec: float
) -> list[GroundTruthEvent]:
    clipped = []
    for truth in truths:
        start = max(0.0, truth.start_sec)
        end = min(duration_sec, truth.end_sec)
        if end > start:
            clipped.append(
                GroundTruthEvent(
                    truth.event_id,
                    truth.soundfile,
                    start,
                    end,
                    truth.ecotype,
                    truth.provider,
                    truth.dataset,
                )
            )
    return clipped


def resolve_preprocessing(
    model_path: str,
    mean_subtract_override: Optional[bool],
    high_pass_override: Optional[bool],
    cutoff_override: Optional[float],
    order_override: Optional[int],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    checkpoint = load_multitask_checkpoint_files(model_path)
    if checkpoint is not None:
        metadata = checkpoint[0]
    augmentation = metadata.get("augmentation", {})
    return {
        "mean_subtract": (
            bool(augmentation.get("mean_subtract", False))
            if mean_subtract_override is None
            else mean_subtract_override
        ),
        "high_pass_filter": (
            bool(augmentation.get("high_pass_filter", False))
            if high_pass_override is None
            else high_pass_override
        ),
        "high_pass_cutoff_hz": (
            float(augmentation.get("high_pass_cutoff_hz", 50.0))
            if cutoff_override is None
            else cutoff_override
        ),
        "high_pass_order": (
            int(augmentation.get("high_pass_order", 4))
            if order_override is None
            else order_override
        ),
    }


def build_high_pass_sos(settings: dict[str, Any], sample_rate: int) -> Optional[np.ndarray]:
    if not settings["high_pass_filter"]:
        return None
    cutoff = settings["high_pass_cutoff_hz"]
    order = settings["high_pass_order"]
    if not 0 < cutoff < sample_rate / 2:
        raise ValueError(f"High-pass cutoff must be between 0 and {sample_rate / 2}")
    if order < 1:
        raise ValueError("High-pass order must be positive")
    return butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")


def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    settings: dict[str, Any],
    high_pass_sos: Optional[np.ndarray],
) -> np.ndarray:
    """Isolated model waveform preprocessing, matching the training collator."""
    waveform = waveform.astype(np.float32, copy=False)
    if settings["mean_subtract"]:
        waveform = waveform - float(waveform.mean())
    if high_pass_sos is not None:
        try:
            waveform = sosfiltfilt(high_pass_sos, waveform)
        except ValueError:
            waveform = sosfilt(high_pass_sos, waveform)
    return waveform.astype(np.float32, copy=False)


def read_window_from_open_file(
    audio_file: sf.SoundFile,
    start_sec: float,
    window_sec: float,
    target_sample_rate: int,
    settings: dict[str, Any],
    high_pass_sos: Optional[np.ndarray],
) -> np.ndarray:
    source_rate = int(audio_file.samplerate)
    start_frame = max(0, round(start_sec * source_rate))
    frame_count = max(1, round(window_sec * source_rate))
    audio_file.seek(min(start_frame, len(audio_file)))
    data = audio_file.read(frame_count, dtype="float32", always_2d=True)
    waveform = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if source_rate != target_sample_rate and len(waveform):
        waveform = librosa.resample(
            waveform,
            orig_sr=source_rate,
            target_sr=target_sample_rate,
        )
    target_length = round(window_sec * target_sample_rate)
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]
    return preprocess_waveform(waveform, target_sample_rate, settings, high_pass_sos)


def read_single_window(
    path: Path,
    start_sec: float,
    window_sec: float,
    target_sample_rate: int,
    settings: dict[str, Any],
    high_pass_sos: Optional[np.ndarray],
) -> np.ndarray:
    with sf.SoundFile(str(path)) as audio_file:
        return read_window_from_open_file(
            audio_file,
            start_sec,
            window_sec,
            target_sample_rate,
            settings,
            high_pass_sos,
        )


def window_starts(duration_sec: float, window_sec: float, hop_sec: float) -> list[float]:
    if duration_sec <= 0:
        return []
    count = max(1, int(math.floor((duration_sec - window_sec) / hop_sec)) + 1)
    return [index * hop_sec for index in range(count)]


def run_model_batch(
    model: Any,
    feature_extractor: Any,
    waveforms: list[np.ndarray],
    sample_rate: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-specific batch adapter, isolated for easy replacement."""
    inputs = feature_extractor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        kw_logits, species_logits, ecotype_logits = outputs["logits"]
        kw_probs = torch.softmax(kw_logits, dim=-1).cpu().numpy()
        species_probs = torch.softmax(species_logits, dim=-1).cpu().numpy()
        ecotype_probs = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()
    return kw_probs, species_probs, ecotype_probs


def load_model(model_path: str, device_name: Optional[str]) -> tuple[Any, Any, torch.device]:
    """Load the existing DCLDE model and feature extractor with compatibility safeguards."""
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_extractor = load_multispecies_feature_extractor(model_path)
    model = load_training_model(
        model_name=model_path,
        dropout=0.0,
        kw_loss_weight=1.0,
        species_loss_weight=1.0,
        ecotype_loss_weight=1.0,
        freeze_backbone=False,
    )
    model.to(device)
    model.eval()
    return model, feature_extractor, device


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = interval_overlap(start_a, end_a, start_b, end_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    return overlap / union if union > 0 else 0.0


def window_has_kw(start: float, end: float, truth_events: list[GroundTruthEvent]) -> bool:
    return any(interval_overlap(start, end, event.start_sec, event.end_sec) > 0 for event in truth_events)


def update_binary_counts(counts: BinaryCounts, actual: bool, predicted: bool) -> None:
    if actual and predicted:
        counts.tp += 1
    elif not actual and predicted:
        counts.fp += 1
    elif actual and not predicted:
        counts.fn += 1
    else:
        counts.tn += 1


def binary_metrics(counts: BinaryCounts) -> dict[str, Any]:
    precision = safe_divide(counts.tp, counts.tp + counts.fp)
    recall = safe_divide(counts.tp, counts.tp + counts.fn)
    accuracy = safe_divide(counts.tp + counts.tn, counts.tp + counts.fp + counts.fn + counts.tn)
    return {
        **asdict(counts),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
    }


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def harmonic_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def generate_events(
    positive_windows: list[WindowPrediction],
    recording: Recording,
    merge_gap_sec: float,
    min_duration_sec: float,
    ecotype_thresholds: Optional[dict[str, float]] = None,
) -> list[PredictedEvent]:
    if not positive_windows:
        return []
    groups: list[list[WindowPrediction]] = []
    current = [positive_windows[0]]
    current_end = positive_windows[0].end_sec
    for window in positive_windows[1:]:
        if window.start_sec - current_end <= merge_gap_sec:
            current.append(window)
            current_end = max(current_end, window.end_sec)
        else:
            groups.append(current)
            current = [window]
            current_end = window.end_sec
    groups.append(current)

    events = []
    for group in groups:
        start = group[0].start_sec
        end = max(window.end_sec for window in group)
        if end - start < min_duration_sec:
            continue
        ecotype_mean = np.mean(
            np.asarray([window.ecotype_probabilities for window in group]), axis=0
        )
        species_kw_mean = float(
            np.mean(
                [
                    window.species_probabilities[SPECIES_LABELS["KW"]]
                    for window in group
                ]
            )
        )
        ecotype_scores = {
            label: species_kw_mean * float(ecotype_mean[index])
            for label, index in ECOTYPE_LABELS.items()
        }
        if ecotype_thresholds:
            predicted_ecotype, ecotype_confidence = thresholded_label(
                ecotype_scores, ecotype_thresholds
            )
        else:
            ecotype_id = int(np.argmax(ecotype_mean))
            predicted_ecotype = ECOTYPE_ID2LABEL[ecotype_id]
            ecotype_confidence = float(ecotype_mean[ecotype_id])
        kw_values = [window.kw_probability for window in group]
        events.append(
            PredictedEvent(
                event_id=f"pred_{recording.soundfile}_{len(events)}",
                soundfile=recording.soundfile,
                start_sec=start,
                end_sec=end,
                max_kw_probability=max(kw_values),
                mean_kw_probability=float(np.mean(kw_values)),
                number_of_windows=len(group),
                predicted_ecotype=predicted_ecotype,
                ecotype_confidence=ecotype_confidence,
                ecotype_probabilities=tuple(float(value) for value in ecotype_mean),
                provider=recording.provider,
                dataset=recording.dataset,
            )
        )
    return events


def match_events(
    predictions: list[PredictedEvent],
    truths: list[GroundTruthEvent],
    iou_threshold: float,
) -> list[EventMatch]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            overlap = interval_overlap(
                prediction.start_sec, prediction.end_sec, truth.start_sec, truth.end_sec
            )
            iou = temporal_iou(
                prediction.start_sec, prediction.end_sec, truth.start_sec, truth.end_sec
            )
            if overlap > 0 and iou >= iou_threshold:
                candidates.append((iou, overlap, pred_index, truth_index))
    candidates.sort(reverse=True)
    matched_predictions = set()
    matched_truths = set()
    matches = []

    for iou, overlap, pred_index, truth_index in candidates:
        if pred_index in matched_predictions or truth_index in matched_truths:
            continue
        prediction = predictions[pred_index]
        truth = truths[truth_index]
        overlapping_ecotypes = sorted(
            {
                event.ecotype
                for event in truths
                if event.ecotype
                and interval_overlap(
                    prediction.start_sec,
                    prediction.end_sec,
                    event.start_sec,
                    event.end_sec,
                )
                > 0
            }
        )
        matches.append(
            EventMatch(
                soundfile=prediction.soundfile,
                status="TP",
                predicted_event_id=prediction.event_id,
                true_event_id=truth.event_id,
                predicted_start_sec=prediction.start_sec,
                predicted_end_sec=prediction.end_sec,
                true_start_sec=truth.start_sec,
                true_end_sec=truth.end_sec,
                temporal_iou=iou,
                overlap_sec=overlap,
                predicted_ecotype=prediction.predicted_ecotype,
                true_ecotype=truth.ecotype,
                ambiguous_ecotype_overlap=len(overlapping_ecotypes) > 1,
                overlapping_true_ecotypes=";".join(overlapping_ecotypes),
                provider=prediction.provider,
                dataset=prediction.dataset,
            )
        )
        matched_predictions.add(pred_index)
        matched_truths.add(truth_index)

    for index, prediction in enumerate(predictions):
        if index not in matched_predictions:
            matches.append(
                EventMatch(
                    prediction.soundfile,
                    "FP",
                    prediction.event_id,
                    "",
                    prediction.start_sec,
                    prediction.end_sec,
                    None,
                    None,
                    0.0,
                    0.0,
                    prediction.predicted_ecotype,
                    "",
                    False,
                    "",
                    prediction.provider,
                    prediction.dataset,
                )
            )
    for index, truth in enumerate(truths):
        if index not in matched_truths:
            matches.append(
                EventMatch(
                    truth.soundfile,
                    "FN",
                    "",
                    truth.event_id,
                    None,
                    None,
                    truth.start_sec,
                    truth.end_sec,
                    0.0,
                    0.0,
                    "",
                    truth.ecotype,
                    False,
                    truth.ecotype,
                    truth.provider,
                    truth.dataset,
                )
            )
    return matches


def multiclass_metrics(
    pairs: list[tuple[str, str]], labels: tuple[str, ...]
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]], dict[str, Any]]:
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in pairs:
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in labels)
    rows = []
    for label in labels:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        rows.append(
            {
                "label": label,
                "support": tp + fn,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "precision": precision,
                "recall": recall,
                "f1": harmonic_f1(precision, recall),
            }
        )
    f1_values = [row["f1"] for row in rows if row["support"] > 0 and row["f1"] is not None]
    overall = {
        "evaluated": total,
        "correct": correct,
        "accuracy": safe_divide(correct, total),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
    }
    return matrix, rows, overall


def focused_srkw_tkw_metrics(
    pairs: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]], dict[str, Any]]:
    """Score SRKW/TKW truths while retaining other ecotype predictions as errors."""
    labels = ("SRKW", "TKW", "other")
    normalized = [
        (actual, predicted if predicted in {"SRKW", "TKW"} else "other")
        for actual, predicted in pairs
        if actual in {"SRKW", "TKW"}
    ]
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in ("SRKW", "TKW")}
    for actual, predicted in normalized:
        matrix[actual][predicted] += 1
    rows = []
    for label in ("SRKW", "TKW"):
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[actual][label] for actual in ("SRKW", "TKW") if actual != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        rows.append(
            {
                "label": label,
                "support": tp + fn,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "precision": precision,
                "recall": recall,
                "f1": harmonic_f1(precision, recall),
            }
        )
    f1_values = [row["f1"] for row in rows if row["support"] and row["f1"] is not None]
    overall = {
        "evaluated": len(normalized),
        "correct": sum(matrix[label][label] for label in ("SRKW", "TKW")),
        "accuracy": safe_divide(
            sum(matrix[label][label] for label in ("SRKW", "TKW")), len(normalized)
        ),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
    }
    return matrix, rows, overall


def event_detection_metrics(matches: list[EventMatch], audio_hours: float) -> dict[str, Any]:
    counts = Counter(match.status for match in matches)
    precision = safe_divide(counts["TP"], counts["TP"] + counts["FP"])
    recall = safe_divide(counts["TP"], counts["TP"] + counts["FN"])
    return {
        "true_positives": counts["TP"],
        "false_positives": counts["FP"],
        "false_negatives": counts["FN"],
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
        "false_positives_per_hour": safe_divide(counts["FP"], audio_hours),
        "misses_per_hour": safe_divide(counts["FN"], audio_hours),
    }


def grouped_metrics(
    recordings: list[Recording],
    matches: list[EventMatch],
    field: str,
) -> list[dict[str, Any]]:
    durations: dict[str, float] = defaultdict(float)
    for recording in recordings:
        durations[getattr(recording, field) or "unknown"] += recording.duration_sec / 3600
    grouped_matches: dict[str, list[EventMatch]] = defaultdict(list)
    for match in matches:
        grouped_matches[getattr(match, field) or "unknown"].append(match)
    rows = []
    for group in sorted(set(durations) | set(grouped_matches)):
        metrics = event_detection_metrics(grouped_matches[group], durations[group])
        rows.append(
            {
                field: group,
                "audio_hours": durations[group],
                "ground_truth_kw_events": metrics["true_positives"] + metrics["false_negatives"],
                "predicted_kw_events": metrics["true_positives"] + metrics["false_positives"],
                **metrics,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: dict[str, dict[str, int]], labels: tuple[str, ...]) -> None:
    write_rows(
        path,
        [{"actual_label": actual, **matrix[actual]} for actual in labels],
        ["actual_label", *labels],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--model-path", "--model_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--gcs-base", default=DEFAULT_GCS_BASE)
    parser.add_argument(
        "--gcs-providers",
        default=",".join(DEFAULT_GCS_PROVIDERS),
        help="Comma-separated NOAA bucket provider directories to index.",
    )
    parser.add_argument("--max-file-gb", type=float, default=10.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for one-at-a-time remote audio downloads (deleted after each file).",
    )
    parser.add_argument("--sample-rate", "--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--window-sec", "--window_sec", type=float, default=3.0)
    parser.add_argument("--hop-sec", "--hop_sec", type=float, default=1.0)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument(
        "--multispecies-threshold", type=float, default=0.25,
        help="Base threshold for KW, AB, background, NRKW, OKW, and SAR.",
    )
    parser.add_argument("--kw-threshold", "--kw_threshold", type=float, default=0.25)
    parser.add_argument("--multispecies-humpback-threshold", type=float, default=0.475)
    parser.add_argument("--multispecies-resident-threshold", type=float, default=0.05)
    parser.add_argument("--multispecies-transient-threshold", type=float, default=0.20)
    parser.add_argument(
        "--event-merge-gap-sec", "--event_merge_gap_sec", type=float, default=1.0
    )
    parser.add_argument(
        "--min-event-duration-sec", "--min_event_duration_sec", type=float, default=0.0
    )
    parser.add_argument(
        "--event-iou-threshold", "--event_iou_threshold", type=float, default=0.1
    )
    parser.add_argument("--flush-every-windows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mean-subtract",
        "--mean_subtract",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--high-pass-filter",
        "--high_pass_filter",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--high-pass-cutoff-hz", type=float, default=None)
    parser.add_argument("--high-pass-order", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("sample_rate", "batch_size", "flush_every_windows"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("window_sec", "hop_sec"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_file_gb <= 0 or (args.max_files is not None and args.max_files < 1):
        raise ValueError("--max-file-gb and --max-files (when set) must be positive")
    threshold_values = (
        args.kw_threshold,
        args.multispecies_threshold,
        args.multispecies_humpback_threshold,
        args.multispecies_resident_threshold,
        args.multispecies_transient_threshold,
        args.event_iou_threshold,
    )
    if any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("Probability and IoU thresholds must be between 0 and 1")
    if args.event_merge_gap_sec < 0 or args.min_event_duration_sec < 0:
        raise ValueError("Event gap and minimum duration cannot be negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)

    annotation_table = read_csv(args.annotations)
    recordings, recording_inventory_sanity = load_recordings(args, annotation_table)
    annotations_by_file, annotation_sanity = load_annotations(
        args.annotations, recordings, annotation_table
    )
    labeled_annotations_by_file = load_all_labeled_annotations(
        annotation_table, recordings
    )
    if not recordings:
        raise ValueError("No readable recordings were found")
    known_audio_hours = sum(recording.duration_sec for recording in recordings) / 3600
    LOGGER.info("Recordings: %s", len(recordings))
    if known_audio_hours:
        LOGGER.info("Locally known audio hours: %.3f", known_audio_hours)
    LOGGER.info("Ground-truth KW events: %s", annotation_sanity["valid_kw_events"])
    LOGGER.info("Ecotype counts: %s", annotation_sanity["ecotype_counts"])
    if recording_inventory_sanity["missing_gcs_audio_count"]:
        LOGGER.warning(
            "Annotated soundfiles not found in indexed GCS providers: %s",
            recording_inventory_sanity["missing_gcs_audio_count"],
        )
    if annotation_sanity["outside_duration_count"]:
        LOGGER.warning(
            "Annotations outside duration (clipped for evaluation): %s",
            annotation_sanity["outside_duration_count"],
        )
    if annotation_sanity["duplicate_annotation_count"]:
        LOGGER.warning("Duplicate annotations: %s", annotation_sanity["duplicate_annotation_count"])

    preprocessing = resolve_preprocessing(
        args.model_path,
        args.mean_subtract,
        args.high_pass_filter,
        args.high_pass_cutoff_hz,
        args.high_pass_order,
    )
    LOGGER.info("Waveform preprocessing: %s", preprocessing)
    high_pass_sos = build_high_pass_sos(preprocessing, args.sample_rate)
    model, feature_extractor, device = load_model(args.model_path, args.device)
    LOGGER.info("Inference device: %s", device)
    species_thresholds = {
        "background": args.multispecies_threshold,
        "KW": args.multispecies_threshold,
        "HW": args.multispecies_humpback_threshold,
        "AB": args.multispecies_threshold,
    }
    ecotype_thresholds = {
        "SRKW": args.multispecies_resident_threshold,
        "TKW": args.multispecies_transient_threshold,
        "NRKW": args.multispecies_threshold,
        "OKW": args.multispecies_threshold,
        "SAR": args.multispecies_threshold,
    }
    LOGGER.info("Species thresholds: %s", species_thresholds)
    LOGGER.info("Ecotype score thresholds: %s", ecotype_thresholds)

    window_fields = [
        "Soundfile", "window_start_sec", "window_end_sec", "Provider", "Dataset",
        "kw_probability", "kw_prediction", "kw_ground_truth",
        "species_ground_truth", "ecotype_ground_truth", "ground_truth_ambiguous",
        "species_argmax_prediction", "species_prediction", "species_confidence",
        *[f"species_probability_{SPECIES_ID2LABEL[index]}" for index in sorted(SPECIES_ID2LABEL)],
        "ecotype_argmax_prediction", "ecotype_prediction", "ecotype_confidence",
        "ecotype_meaningful",
        *[f"ecotype_probability_{ECOTYPE_ID2LABEL[index]}" for index in sorted(ECOTYPE_ID2LABEL)],
        *[f"ecotype_score_{ECOTYPE_ID2LABEL[index]}" for index in sorted(ECOTYPE_ID2LABEL)],
        "comparison_prediction", "comparison_confidence", "comparison_ground_truth",
    ]
    window_path = output_dir / "window_predictions.csv"
    all_predictions: list[PredictedEvent] = []
    all_truths: list[GroundTruthEvent] = []
    all_matches: list[EventMatch] = []
    evaluated_recordings: list[Recording] = []
    species_window_pairs: list[tuple[str, str]] = []
    ecotype_window_pairs: list[tuple[str, str]] = []
    comparison_window_pairs: list[tuple[str, str]] = []
    window_counts = BinaryCounts()
    failed_files = []
    window_rows_since_flush = 0
    file_iterable = tqdm(recordings, desc="Long recordings") if tqdm else recordings

    with window_path.open("w", newline="", encoding="utf-8") as window_file:
        window_writer = csv.DictWriter(window_file, fieldnames=window_fields)
        window_writer.writeheader()
        for recording in file_iterable:
            try:
                with materialize_audio(
                    recording.audio_source, recording.soundfile, args.temp_dir
                ) as local_audio_path:
                    duration_sec, _, _ = audio_metadata(local_audio_path)
                    evaluated_recording = Recording(
                        recording.soundfile,
                        recording.audio_source,
                        recording.provider,
                        recording.dataset,
                        duration_sec,
                    )
                    truths = clip_truths_to_duration(
                        annotations_by_file.get(recording.soundfile, []), duration_sec
                    )
                    labeled_truths = labeled_annotations_by_file.get(recording.soundfile, [])
                    positive_windows: list[WindowPrediction] = []
                    recording_window_counts = BinaryCounts()
                    recording_species_pairs: list[tuple[str, str]] = []
                    recording_ecotype_pairs: list[tuple[str, str]] = []
                    recording_comparison_pairs: list[tuple[str, str]] = []
                    starts = window_starts(duration_sec, args.window_sec, args.hop_sec)
                    open_audio = sf.SoundFile(str(local_audio_path)) if args.num_workers == 0 else None
                    executor = (
                        ThreadPoolExecutor(max_workers=args.num_workers)
                        if args.num_workers > 0 else None
                    )
                    try:
                        for batch_start in range(0, len(starts), args.batch_size):
                            batch_starts = starts[batch_start : batch_start + args.batch_size]
                            if executor is None:
                                waveforms = [
                                    read_window_from_open_file(
                                        open_audio, start, args.window_sec, args.sample_rate,
                                        preprocessing, high_pass_sos,
                                    )
                                    for start in batch_starts
                                ]
                            else:
                                futures = [
                                    executor.submit(
                                        read_single_window,
                                        local_audio_path,
                                        start,
                                        args.window_sec,
                                        args.sample_rate,
                                        preprocessing,
                                        high_pass_sos,
                                    )
                                    for start in batch_starts
                                ]
                                waveforms = [future.result() for future in futures]
                            kw_probs, species_probs, ecotype_probs = run_model_batch(
                                model, feature_extractor, waveforms, args.sample_rate, device
                            )
                            for item_index, start in enumerate(batch_starts):
                                end = min(start + args.window_sec, duration_sec)
                                kw_probability = float(kw_probs[item_index, KW_LABELS["kw"]])
                                kw_prediction = kw_probability >= args.kw_threshold
                                species_id = int(np.argmax(species_probs[item_index]))
                                ecotype_id = int(np.argmax(ecotype_probs[item_index]))
                                actual_kw = window_has_kw(start, end, truths)
                                actual_species, actual_ecotype, truth_ambiguous = (
                                    overlapping_ground_truth(start, end, labeled_truths)
                                )
                                species_scores = {
                                    label: float(species_probs[item_index, index])
                                    for label, index in SPECIES_LABELS.items()
                                }
                                predicted_species, species_confidence = thresholded_label(
                                    species_scores, species_thresholds
                                )
                                species_kw_probability = species_scores["KW"]
                                ecotype_scores = {
                                    label: species_kw_probability
                                    * float(ecotype_probs[item_index, index])
                                    for label, index in ECOTYPE_LABELS.items()
                                }
                                predicted_ecotype, ecotype_confidence = thresholded_label(
                                    ecotype_scores, ecotype_thresholds
                                )
                                comparison_scores = {
                                    "HW": species_scores["HW"],
                                    "AB": species_scores["AB"],
                                    **ecotype_scores,
                                }
                                comparison_thresholds = {
                                    "HW": species_thresholds["HW"],
                                    "AB": species_thresholds["AB"],
                                    **ecotype_thresholds,
                                }
                                comparison_prediction, comparison_confidence = thresholded_label(
                                    comparison_scores, comparison_thresholds
                                )
                                if comparison_prediction == "unclassified":
                                    comparison_prediction = "background"
                                comparison_truth = (
                                    actual_ecotype
                                    if actual_species == "KW" and actual_ecotype
                                    else actual_species
                                )
                                update_binary_counts(
                                    recording_window_counts, actual_kw, kw_prediction
                                )
                                if not truth_ambiguous:
                                    recording_species_pairs.append(
                                        (actual_species, predicted_species)
                                    )
                                    if actual_species == "KW" and actual_ecotype:
                                        recording_ecotype_pairs.append(
                                            (actual_ecotype, predicted_ecotype)
                                        )
                                    if comparison_truth != "KW":
                                        recording_comparison_pairs.append(
                                            (comparison_truth, comparison_prediction)
                                        )
                                prediction = WindowPrediction(
                                    recording.soundfile, start, end, kw_probability, kw_prediction,
                                    predicted_species,
                                    tuple(float(value) for value in species_probs[item_index]),
                                    predicted_ecotype,
                                    tuple(float(value) for value in ecotype_probs[item_index]),
                                )
                                if kw_prediction:
                                    positive_windows.append(prediction)
                                row = {
                                    "Soundfile": recording.soundfile,
                                    "window_start_sec": start,
                                    "window_end_sec": end,
                                    "Provider": recording.provider,
                                    "Dataset": recording.dataset,
                                    "kw_probability": kw_probability,
                                    "kw_prediction": kw_prediction,
                                    "kw_ground_truth": actual_kw,
                                    "species_ground_truth": actual_species,
                                    "ecotype_ground_truth": actual_ecotype,
                                    "ground_truth_ambiguous": truth_ambiguous,
                                    "species_argmax_prediction": SPECIES_ID2LABEL[species_id],
                                    "species_prediction": prediction.species_prediction,
                                    "species_confidence": species_confidence,
                                    "ecotype_argmax_prediction": ECOTYPE_ID2LABEL[ecotype_id],
                                    "ecotype_prediction": prediction.ecotype_prediction,
                                    "ecotype_confidence": ecotype_confidence,
                                    "ecotype_meaningful": (
                                        kw_prediction or prediction.species_prediction == "KW"
                                    ),
                                    "comparison_prediction": comparison_prediction,
                                    "comparison_confidence": comparison_confidence,
                                    "comparison_ground_truth": comparison_truth,
                                }
                                row.update({
                                    f"species_probability_{SPECIES_ID2LABEL[index]}": prediction.species_probabilities[index]
                                    for index in sorted(SPECIES_ID2LABEL)
                                })
                                row.update({
                                    f"ecotype_probability_{ECOTYPE_ID2LABEL[index]}": prediction.ecotype_probabilities[index]
                                    for index in sorted(ECOTYPE_ID2LABEL)
                                })
                                row.update({
                                    f"ecotype_score_{label}": score
                                    for label, score in ecotype_scores.items()
                                })
                                window_writer.writerow(row)
                                window_rows_since_flush += 1
                                if window_rows_since_flush >= args.flush_every_windows:
                                    window_file.flush()
                                    window_rows_since_flush = 0
                    finally:
                        if open_audio is not None:
                            open_audio.close()
                        if executor is not None:
                            executor.shutdown(wait=True)
                predictions = generate_events(
                    positive_windows,
                    evaluated_recording,
                    args.event_merge_gap_sec,
                    args.min_event_duration_sec,
                    ecotype_thresholds,
                )
                matches = match_events(predictions, truths, args.event_iou_threshold)
                for field in ("tp", "fp", "fn", "tn"):
                    setattr(
                        window_counts,
                        field,
                        getattr(window_counts, field) + getattr(recording_window_counts, field),
                    )
                evaluated_recordings.append(evaluated_recording)
                species_window_pairs.extend(recording_species_pairs)
                ecotype_window_pairs.extend(recording_ecotype_pairs)
                comparison_window_pairs.extend(recording_comparison_pairs)
                all_truths.extend(truths)
                all_predictions.extend(predictions)
                all_matches.extend(matches)
            except Exception as error:
                failed_files.append(
                    {
                        "Soundfile": recording.soundfile,
                        "audio_path": recording.audio_source,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                LOGGER.exception("Failed recording %s; continuing", recording.soundfile)
        window_file.flush()

    audio_hours = sum(recording.duration_sec for recording in evaluated_recordings) / 3600

    event_rows = []
    for event in all_predictions:
        row = asdict(event)
        probabilities = row.pop("ecotype_probabilities")
        row.update({
            f"ecotype_probability_{ECOTYPE_ID2LABEL[index]}": probabilities[index]
            for index in sorted(ECOTYPE_ID2LABEL)
        })
        event_rows.append(row)
    predicted_event_fields = [
        "event_id", "soundfile", "start_sec", "end_sec", "max_kw_probability",
        "mean_kw_probability", "number_of_windows", "predicted_ecotype",
        "ecotype_confidence", "provider", "dataset",
        *[f"ecotype_probability_{ECOTYPE_ID2LABEL[index]}" for index in sorted(ECOTYPE_ID2LABEL)],
    ]
    write_rows(output_dir / "predicted_events.csv", event_rows, predicted_event_fields)
    write_rows(
        output_dir / "event_matches.csv",
        [asdict(match) for match in all_matches],
        list(EventMatch.__dataclass_fields__),
    )
    write_rows(
        output_dir / "failed_files.csv",
        failed_files,
        ["Soundfile", "audio_path", "error"],
    )

    event_metrics = event_detection_metrics(all_matches, audio_hours)
    window_metrics = binary_metrics(window_counts)
    ecotype_pairs = [
        (match.true_ecotype, match.predicted_ecotype)
        for match in all_matches
        if match.status == "TP"
        and not match.ambiguous_ecotype_overlap
        and match.true_ecotype in ECOTYPE_LABELS
    ]
    ecotype_output_labels = (*ECOTYPE_NAMES, "unclassified")
    ecotype_matrix, ecotype_rows, ecotype_overall = multiclass_metrics(
        ecotype_pairs, ecotype_output_labels
    )
    srkw_tkw_pairs = [pair for pair in ecotype_pairs if pair[0] in {"SRKW", "TKW"}]
    focused_matrix, focused_rows, focused_overall = focused_srkw_tkw_metrics(
        srkw_tkw_pairs
    )
    provider_rows = grouped_metrics(evaluated_recordings, all_matches, "provider")
    dataset_rows = grouped_metrics(evaluated_recordings, all_matches, "dataset")
    grouped_fields = [
        "provider", "audio_hours", "ground_truth_kw_events", "predicted_kw_events",
        "true_positives", "false_positives", "false_negatives", "precision", "recall",
        "f1", "false_positives_per_hour", "misses_per_hour",
    ]
    write_rows(output_dir / "provider_metrics.csv", provider_rows, grouped_fields)
    write_rows(
        output_dir / "dataset_metrics.csv",
        dataset_rows,
        ["dataset", *grouped_fields[1:]],
    )
    ecotype_fields = [
        "label", "support", "true_positive_count", "false_positive_count",
        "false_negative_count", "precision", "recall", "f1",
    ]
    write_rows(output_dir / "ecotype_metrics.csv", ecotype_rows, ecotype_fields)
    write_rows(output_dir / "srkw_tkw_metrics.csv", focused_rows, ecotype_fields)
    write_matrix(
        output_dir / "ecotype_confusion_matrix.csv",
        ecotype_matrix,
        ecotype_output_labels,
    )
    write_rows(
        output_dir / "srkw_tkw_confusion_matrix.csv",
        [{"actual_label": actual, **focused_matrix[actual]} for actual in ("SRKW", "TKW")],
        ["actual_label", "SRKW", "TKW", "other"],
    )
    write_rows(
        output_dir / "window_confusion_matrix.csv",
        [
            {"actual": "not_kw", "predicted_not_kw": window_counts.tn, "predicted_kw": window_counts.fp},
            {"actual": "kw", "predicted_not_kw": window_counts.fn, "predicted_kw": window_counts.tp},
        ],
    )
    species_output_labels = ("background", "KW", "HW", "AB", "unclassified")
    species_window_matrix, species_window_rows, species_window_overall = multiclass_metrics(
        species_window_pairs, species_output_labels
    )
    ecotype_window_matrix, ecotype_window_rows, ecotype_window_overall = multiclass_metrics(
        ecotype_window_pairs, ecotype_output_labels
    )
    comparison_labels = ("background", "HW", "AB", *ECOTYPE_NAMES)
    comparison_window_matrix, comparison_window_rows, comparison_window_overall = (
        multiclass_metrics(comparison_window_pairs, comparison_labels)
    )
    write_rows(
        output_dir / "window_species_metrics.csv",
        species_window_rows,
        ecotype_fields,
    )
    write_matrix(
        output_dir / "window_species_confusion_matrix.csv",
        species_window_matrix,
        species_output_labels,
    )
    write_rows(
        output_dir / "window_ecotype_metrics.csv",
        ecotype_window_rows,
        ecotype_fields,
    )
    write_matrix(
        output_dir / "window_ecotype_confusion_matrix.csv",
        ecotype_window_matrix,
        ecotype_output_labels,
    )
    write_rows(
        output_dir / "window_all_classes_metrics.csv",
        comparison_window_rows,
        ecotype_fields,
    )
    write_matrix(
        output_dir / "window_all_classes_confusion_matrix.csv",
        comparison_window_matrix,
        comparison_labels,
    )
    overall = {
        "audio_hours": audio_hours,
        "evaluated_recordings": len(evaluated_recordings),
        "ground_truth_kw_events": len(all_truths),
        "predicted_kw_events": len(all_predictions),
        "failed_recordings": len(failed_files),
        "window_metrics": window_metrics,
        "window_species_metrics": species_window_overall,
        "window_ecotype_metrics": ecotype_window_overall,
        "window_all_classes_metrics": comparison_window_overall,
        "event_metrics": event_metrics,
        "ecotype_metrics": ecotype_overall,
        "srkw_vs_tkw_metrics": focused_overall,
        "recording_inventory_sanity": recording_inventory_sanity,
        "annotation_sanity": annotation_sanity,
        "preprocessing": preprocessing,
        "thresholds": {
            "binary_kw": args.kw_threshold,
            "species": species_thresholds,
            "ecotype_composite_scores": ecotype_thresholds,
        },
    }
    with (output_dir / "overall_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(overall, file, indent=2)

    print("\nLong-recording DCLDE evaluation")
    print("-" * 40)
    print(f"Audio hours:            {audio_hours:.3f}")
    print(f"Ground-truth KW events: {len(all_truths)}")
    print(f"Predicted KW events:    {len(all_predictions)}")
    print(f"TP:                     {event_metrics['true_positives']}")
    print(f"FP:                     {event_metrics['false_positives']}")
    print(f"FN:                     {event_metrics['false_negatives']}")
    print(f"Precision:              {event_metrics['precision']}")
    print(f"Recall:                 {event_metrics['recall']}")
    print(f"F1:                     {event_metrics['f1']}")
    print(f"FP/hour:                {event_metrics['false_positives_per_hour']}")
    print(f"Window species accuracy:{species_window_overall['accuracy']}")
    print(f"All-class window F1:    {comparison_window_overall['macro_f1']}")
    print(f"Ecotype accuracy:       {ecotype_overall['accuracy']}")
    print(f"Ecotype macro F1:       {ecotype_overall['macro_f1']}")
    ecotype_by_name = {row['label']: row for row in ecotype_rows}
    print(f"SRKW F1:                {ecotype_by_name['SRKW']['f1']}")
    print(f"TKW F1:                 {ecotype_by_name['TKW']['f1']}")
    print("\nProvider-level results:")
    for row in provider_rows:
        print(
            f"  {row['provider']}: hours={row['audio_hours']:.3f}, "
            f"TP={row['true_positives']}, FP={row['false_positives']}, "
            f"FN={row['false_negatives']}, F1={row['f1']}, "
            f"FP/hour={row['false_positives_per_hour']}"
        )
    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
