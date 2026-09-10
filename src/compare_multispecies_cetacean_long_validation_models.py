#!/usr/bin/env python3
"""Compare V2 multispecies models on local-mode validation recordings.

The source-recording plan defines the recording-disjoint validation set. By
default this script excludes ``remote_seek`` recordings, downloads each
remaining source recording once, extracts each AST feature batch once, and
passes that shared batch through every candidate model. Window probabilities
are checkpointed after every recording so an interrupted Kaggle run can resume.

The CPU evaluation stage forms stable events for KW, HW, and UndBio, performs a
small per-model threshold search, and ranks models by class-aware macro F1.
Ecotype metrics are calculated only for correctly detected KW events.
Validation-selected settings must be frozen before evaluating a test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt
from transformers import AutoFeatureExtractor

from train_multispecies_cetacean_model import (
    ECOTYPE_ID2LABEL,
    ECOTYPE_LABELS,
    SAMPLE_RATE,
    SOURCE_LABELS,
    TRIGGER_LABELS,
    checkpoint_files,
    load_model,
)


DEFAULT_SOURCE_PLAN = (
    "/kaggle/input/datasets/leonisviridis/orca-detector-misc/"
    "multispecies_cetacean_plan_v4/multispecies_cetacean_source_recording_plan.csv"
)
DEFAULT_ANNOTATIONS = (
    "/kaggle/input/datasets/leonisviridis/orca-detector-misc/"
    "multispecies_cetacean_plan_v4/multispecies_cetacean_master_cleaned_annotations.csv"
)
DEFAULT_OUTPUT = "/kaggle/working/multispecies_cetacean_long_validation_comparison"
TARGET_CLASSES = ("KW", "HW", "UndBio")
REMOTE_SCHEMES = ("gs://", "http://", "https://")


@dataclass(frozen=True)
class Recording:
    recording_id: str
    provider: str
    dataset: str
    soundfile: str
    audio_source: str
    extraction_mode: str
    source_size_bytes: int


@dataclass(frozen=True)
class TruthEvent:
    event_id: str
    recording_id: str
    species: str
    start_sec: float
    end_sec: float
    ecotype: str
    provider: str
    dataset: str

    @property
    def center_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


@dataclass
class ModelBundle:
    name: str
    slug: str
    model: Any
    device: torch.device
    cache_path: Path
    metadata_path: Path
    completed_recordings: set[str]
    preprocessing: dict[str, Any]
    feature_signature: dict[str, Any]


@dataclass(frozen=True)
class StableConfig:
    moving_average_windows: int
    threshold: float
    support_ratio: float
    continuation_ratio: float
    minimum_support_windows: int
    support_radius_windows: int
    maximum_gap_windows: int
    peak_suppression_windows: int
    event_top_k: int


@dataclass(frozen=True)
class PredictedEvent:
    event_id: str
    recording_id: str
    species: str
    peak_window_index: int
    peak_time_sec: float
    start_sec: float
    end_sec: float
    peak_score: float
    event_score: float
    normalized_score: float
    supporting_windows: int
    predicted_ecotype: str
    ecotype_confidence: float
    provider: str
    dataset: str


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "n/a"} else text


def parse_csv_list(value: str, cast: type, name: str) -> list[Any]:
    try:
        values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def safe_slug(model_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-._") or "model"
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{stem[-80:]}-{digest}"


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preprocessing_for_model(model_name: str) -> dict[str, Any]:
    checkpoint = checkpoint_files(model_name)
    metadata = checkpoint[0] if checkpoint is not None else {}
    values = metadata.get("preprocessing", metadata.get("augmentation", {}))
    return {
        "mean_subtract": bool(values.get("mean_subtract", False)),
        "high_pass_filter": bool(values.get("high_pass_filter", False)),
        "high_pass_cutoff_hz": float(values.get("high_pass_cutoff_hz", 50.0)),
        "high_pass_order": int(values.get("high_pass_order", 4)),
    }


def feature_signature(extractor: Any) -> dict[str, Any]:
    data = extractor.to_dict()
    keys = (
        "feature_size",
        "sampling_rate",
        "num_mel_bins",
        "max_length",
        "mean",
        "std",
        "do_normalize",
    )
    return {key: data.get(key) for key in keys}


def apply_preprocessing_overrides(settings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(settings)
    if args.mean_subtract is not None:
        result["mean_subtract"] = args.mean_subtract
    if args.high_pass_filter is not None:
        result["high_pass_filter"] = args.high_pass_filter
    if args.high_pass_cutoff_hz is not None:
        result["high_pass_cutoff_hz"] = args.high_pass_cutoff_hz
    if args.high_pass_order is not None:
        result["high_pass_order"] = args.high_pass_order
    return result


def load_recordings(args: argparse.Namespace) -> list[Recording]:
    rows = read_rows(Path(args.source_plan))
    modes = {item.casefold() for item in parse_csv_list(args.extraction_modes, str, "extraction modes")}
    selected: list[Recording] = []
    for row in rows:
        if clean(row.get("split")).casefold() != args.split.casefold():
            continue
        mode = clean(row.get("extraction_mode"))
        if mode.casefold() not in modes:
            continue
        source = clean(row.get("gcs_path"))
        if not source:
            continue
        if source.startswith("/") is False and not source.startswith(REMOTE_SCHEMES):
            source = "gs://" + source.lstrip("/")
        recording_id = clean(row.get("source_recording_id"))
        if not recording_id:
            recording_id = "|".join(
                (clean(row.get("Provider")), clean(row.get("Dataset")), clean(row.get("Soundfile")))
            )
        selected.append(
            Recording(
                recording_id=recording_id,
                provider=clean(row.get("Provider")),
                dataset=clean(row.get("Dataset")),
                soundfile=clean(row.get("Soundfile")),
                audio_source=source,
                extraction_mode=mode,
                source_size_bytes=int(float(clean(row.get("source_size_bytes")) or 0)),
            )
        )
    selected.sort(key=lambda item: (item.provider, item.dataset, item.soundfile))
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("--max-files must be positive")
        rng = random.Random(args.seed)
        selected = rng.sample(selected, min(args.max_files, len(selected)))
        selected.sort(key=lambda item: (item.provider, item.dataset, item.soundfile))
    if not selected:
        raise ValueError(
            f"No recordings matched split={args.split!r}, extraction_modes={sorted(modes)}"
        )
    return selected


def load_truth(path: Path, recordings: list[Recording]) -> dict[str, list[TruthEvent]]:
    selected = {item.recording_id: item for item in recordings}
    selected_by_triplet = {
        (item.provider, item.dataset, item.soundfile): item for item in recordings
    }
    result: dict[str, list[TruthEvent]] = defaultdict(list)
    for row_number, row in enumerate(read_rows(path), start=2):
        recording_id = clean(row.get("source_recording_id"))
        recording = selected.get(recording_id)
        if recording is None:
            recording = selected_by_triplet.get(
                (clean(row.get("Provider")), clean(row.get("Dataset")), clean(row.get("Soundfile")))
            )
        if recording is None:
            continue
        species = clean(row.get("model_source_label") or row.get("clean_class_species") or row.get("ClassSpecies"))
        aliases = {"kw": "KW", "hw": "HW", "undbio": "UndBio", "ab": "Abiotic", "abiotic": "Abiotic"}
        species = aliases.get(species.casefold(), species)
        if species not in TARGET_CLASSES:
            continue
        try:
            start = float(row.get("FileBeginSec", ""))
            end = float(row.get("FileEndSec", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        ecotype = clean(row.get("clean_ecotype") or row.get("Ecotype")).upper()
        if species != "KW" or ecotype not in ECOTYPE_LABELS:
            ecotype = ""
        result[recording.recording_id].append(
            TruthEvent(
                event_id=clean(row.get("annotation_id")) or f"truth_{row_number}",
                recording_id=recording.recording_id,
                species=species,
                start_sec=max(0.0, start),
                end_sec=end,
                ecotype=ecotype,
                provider=recording.provider,
                dataset=recording.dataset,
            )
        )
    for events in result.values():
        events.sort(key=lambda item: (item.start_sec, item.end_sec, item.species))
    return result


def remote_download_url(source: str) -> str:
    if not source.startswith("gs://"):
        return source
    bucket_object = source[5:]
    bucket, object_name = bucket_object.split("/", 1)
    return (
        f"https://storage.googleapis.com/{urllib.parse.quote(bucket)}/"
        f"{urllib.parse.quote(object_name, safe='/')}"
    )


@contextmanager
def materialize_audio(recording: Recording, temp_root: Path | None) -> Iterator[Path]:
    source = recording.audio_source
    if not source.startswith(REMOTE_SCHEMES):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path
        return
    parent = str(temp_root) if temp_root is not None else None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="multispecies_long_", dir=parent) as directory:
        suffix = Path(recording.soundfile).suffix or ".audio"
        destination = Path(directory) / f"recording{suffix}"
        request = urllib.request.Request(
            remote_download_url(source), headers={"User-Agent": "pods-ai-v2-long-evaluator/1.0"}
        )
        with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        yield destination


def window_starts(duration_sec: float, window_sec: float, hop_sec: float) -> np.ndarray:
    if duration_sec <= 0:
        return np.empty(0, dtype=np.float64)
    count = max(1, int(math.floor((duration_sec - window_sec) / hop_sec)) + 1)
    return np.arange(count, dtype=np.float64) * hop_sec


def high_pass_filter(settings: dict[str, Any]) -> np.ndarray | None:
    if not settings["high_pass_filter"]:
        return None
    cutoff = float(settings["high_pass_cutoff_hz"])
    if not 0 < cutoff < SAMPLE_RATE / 2:
        raise ValueError(f"Invalid high-pass cutoff: {cutoff}")
    return butter(
        int(settings["high_pass_order"]), cutoff, btype="highpass", fs=SAMPLE_RATE, output="sos"
    )


def preprocess_window(audio: np.ndarray, settings: dict[str, Any], sos: np.ndarray | None) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32)
    if settings["mean_subtract"]:
        result = result - float(result.mean())
    if sos is not None:
        try:
            result = sosfiltfilt(sos, result)
        except ValueError:
            result = sosfilt(sos, result)
    return np.asarray(result, dtype=np.float32)


def read_contiguous_batch(
    handle: sf.SoundFile,
    starts: np.ndarray,
    window_sec: float,
    settings: dict[str, Any],
    sos: np.ndarray | None,
) -> list[np.ndarray]:
    source_rate = int(handle.samplerate)
    target_length = round(window_sec * SAMPLE_RATE)
    first_source = round(float(starts[0]) * source_rate)
    final_source = round((float(starts[-1]) + window_sec) * source_rate)
    handle.seek(min(first_source, len(handle)))
    block = handle.read(max(0, final_source - first_source), dtype="float32", always_2d=True)
    # Match V2 training: use channel zero rather than averaging channels.
    mono = block[:, 0] if len(block) else np.empty(0, dtype=np.float32)
    if source_rate != SAMPLE_RATE and len(mono):
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, source_rate // divisor)
    windows: list[np.ndarray] = []
    origin = float(starts[0])
    for start in starts:
        relative = round((float(start) - origin) * SAMPLE_RATE)
        waveform = np.asarray(mono[relative : relative + target_length], dtype=np.float32)
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        windows.append(preprocess_window(waveform, settings, sos))
    return windows


def completed_recordings(cache_path: Path) -> set[str]:
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(cache_path, usecols=["recording_id"])["recording_id"].astype(str).unique())


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    exists = path.is_file() and path.stat().st_size > 0
    frame.to_csv(path, mode="a" if exists else "w", header=not exists, index=False)


def commit_partial_csv(partial: Path, destination: Path) -> None:
    """Append one completed recording while omitting a duplicate CSV header."""
    destination_exists = destination.is_file() and destination.stat().st_size > 0
    with partial.open("r", encoding="utf-8", newline="") as source, destination.open(
        "a" if destination_exists else "w", encoding="utf-8", newline=""
    ) as target:
        if destination_exists:
            next(source, None)
        shutil.copyfileobj(source, target, length=1024 * 1024)
    partial.unlink()


def load_bundles(args: argparse.Namespace, output_dir: Path) -> tuple[list[ModelBundle], Any]:
    devices = [item.strip() for item in args.model_devices.split(",") if item.strip()]
    if not devices:
        devices = [args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")]
    bundles: list[ModelBundle] = []
    shared_signature: dict[str, Any] | None = None
    shared_preprocessing: dict[str, Any] | None = None
    shared_extractor: Any = None
    for index, name in enumerate(args.model_names):
        device = torch.device(devices[index % len(devices)])
        print(f"Loading model {index + 1}/{len(args.model_names)} on {device}: {name}")
        model, identity, feature_source = load_model(name, dropout=0.0, freeze_backbone=False)
        extractor = AutoFeatureExtractor.from_pretrained(feature_source)
        signature = feature_signature(extractor)
        preprocessing = apply_preprocessing_overrides(preprocessing_for_model(name), args)
        if shared_signature is None:
            shared_signature = signature
            shared_preprocessing = preprocessing
            shared_extractor = extractor
        elif signature != shared_signature:
            raise ValueError(
                f"Models do not share one AST feature-extractor configuration: {name}: {signature} "
                f"!= {shared_signature}"
            )
        elif preprocessing != shared_preprocessing:
            raise ValueError(
                f"Models have different deterministic preprocessing, which prevents shared inference: "
                f"{name}: {preprocessing} != {shared_preprocessing}. Supply explicit preprocessing overrides."
            )
        model.to(device).eval()
        slug = safe_slug(name)
        model_dir = output_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_path = model_dir / "window_predictions.csv"
        metadata_path = model_dir / "window_predictions.json"
        if not args.reuse_window_cache:
            cache_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        metadata = {
            "model_name": name,
            "model_identity": identity,
            "source_plan": file_identity(Path(args.source_plan)),
            "annotations": file_identity(Path(args.annotations)),
            "split": args.split,
            "extraction_modes": args.extraction_modes,
            "max_files": args.max_files,
            "seed": args.seed,
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "preprocessing": preprocessing,
            "feature_signature": signature,
        }
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous != metadata:
                raise RuntimeError(
                    f"Stale/incompatible cache metadata for {name}. Use a new --output-dir or delete {model_dir}."
                )
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        bundles.append(
            ModelBundle(
                name=name,
                slug=slug,
                model=model,
                device=device,
                cache_path=cache_path,
                metadata_path=metadata_path,
                completed_recordings=completed_recordings(cache_path),
                preprocessing=preprocessing,
                feature_signature=signature,
            )
        )
    return bundles, shared_extractor


def infer_recordings(
    args: argparse.Namespace,
    recordings: list[Recording],
    bundles: list[ModelBundle],
    extractor: Any,
) -> dict[str, float]:
    timings = Counter()
    settings = bundles[0].preprocessing
    sos = high_pass_filter(settings)
    temp_root = Path(args.temp_dir) if args.temp_dir else None
    for file_index, recording in enumerate(recordings, start=1):
        active = [bundle for bundle in bundles if recording.recording_id not in bundle.completed_recordings]
        if not active:
            print(f"[{file_index}/{len(recordings)}] cached: {recording.soundfile}")
            continue
        size_gb = recording.source_size_bytes / 1024**3
        print(
            f"[{file_index}/{len(recordings)}] {recording.provider}/{recording.soundfile} "
            f"({size_gb:.2f} GB; {len(active)} model(s))"
        )
        download_started = time.perf_counter()
        try:
            with materialize_audio(recording, temp_root) as audio_path:
                timings["download_seconds"] += time.perf_counter() - download_started
                with sf.SoundFile(str(audio_path)) as handle:
                    duration = len(handle) / float(handle.samplerate)
                    starts = window_starts(duration, args.window_sec, args.hop_sec)
                    partial_paths = {
                        bundle.slug: bundle.cache_path.with_name(
                            f".{bundle.cache_path.stem}.{hashlib.sha1(recording.recording_id.encode()).hexdigest()[:12]}.partial.csv"
                        )
                        for bundle in active
                    }
                    for partial in partial_paths.values():
                        partial.unlink(missing_ok=True)
                    for offset in range(0, len(starts), args.batch_size):
                        batch_starts = starts[offset : offset + args.batch_size]
                        io_started = time.perf_counter()
                        waveforms = read_contiguous_batch(
                            handle, batch_starts, args.window_sec, settings, sos
                        )
                        features = extractor(
                            waveforms,
                            sampling_rate=SAMPLE_RATE,
                            padding=True,
                            return_tensors="pt",
                        )["input_values"]
                        timings["frontend_seconds"] += time.perf_counter() - io_started
                        by_device: dict[str, list[ModelBundle]] = defaultdict(list)
                        for bundle in active:
                            by_device[str(bundle.device)].append(bundle)

                        def evaluate_device(device_bundles: list[ModelBundle]) -> list[tuple[ModelBundle, np.ndarray, np.ndarray, np.ndarray, float]]:
                            device_values = features.to(device_bundles[0].device, non_blocking=True)
                            results = []
                            for bundle in device_bundles:
                                model_started = time.perf_counter()
                                with torch.inference_mode():
                                    output = bundle.model(input_values=device_values)
                                    trigger = torch.softmax(output[0], dim=-1).cpu().numpy()
                                    source = torch.softmax(output[1], dim=-1).cpu().numpy()
                                    ecotype = torch.softmax(output[2], dim=-1).cpu().numpy()
                                results.append(
                                    (bundle, trigger, source, ecotype, time.perf_counter() - model_started)
                                )
                            return results

                        device_groups = list(by_device.values())
                        if args.parallel_model_devices and len(device_groups) > 1:
                            with ThreadPoolExecutor(max_workers=len(device_groups)) as executor:
                                grouped_results = list(executor.map(evaluate_device, device_groups))
                        else:
                            grouped_results = [evaluate_device(group) for group in device_groups]
                        for device_results in grouped_results:
                            for bundle, trigger, source, ecotype, elapsed in device_results:
                                timings[f"model_seconds::{bundle.slug}"] += elapsed
                                rows: list[dict[str, Any]] = []
                                for local_index, start_sec in enumerate(batch_starts):
                                    row: dict[str, Any] = {
                                        "recording_id": recording.recording_id,
                                        "provider": recording.provider,
                                        "dataset": recording.dataset,
                                        "soundfile": recording.soundfile,
                                        "duration_sec": duration,
                                        "window_index": offset + local_index,
                                        "window_start_sec": float(start_sec),
                                        "window_end_sec": float(start_sec + args.window_sec),
                                    }
                                    for label, class_index in TRIGGER_LABELS.items():
                                        row[f"trigger_{label}"] = float(trigger[local_index, class_index])
                                    for label, class_index in SOURCE_LABELS.items():
                                        row[f"source_{label}"] = float(source[local_index, class_index])
                                    for label, class_index in ECOTYPE_LABELS.items():
                                        row[f"ecotype_{label}"] = float(ecotype[local_index, class_index])
                                    rows.append(row)
                                append_rows(partial_paths[bundle.slug], rows)
                    for bundle in active:
                        commit_partial_csv(partial_paths[bundle.slug], bundle.cache_path)
                        bundle.completed_recordings.add(recording.recording_id)
                    print(f"  {len(starts):,} windows cached")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
            with (Path(args.output_dir) / "failed_recordings.csv").open(
                "a", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                if handle.tell() == 0:
                    writer.writerow(["recording_id", "soundfile", "error"])
                writer.writerow([recording.recording_id, recording.soundfile, repr(exc)])
    return {key: float(value) for key, value in timings.items()}


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) < width:
        return values.copy()
    kernel = np.full(width, 1.0 / width)
    return np.convolve(values, kernel, mode="same")


def local_peaks(scores: np.ndarray, threshold: float) -> list[int]:
    candidates = []
    for index, value in enumerate(scores):
        if value < threshold:
            continue
        left = scores[index - 1] if index else -math.inf
        right = scores[index + 1] if index + 1 < len(scores) else -math.inf
        if value >= left and value >= right:
            candidates.append(index)
    return candidates


def suppress_peaks(scores: np.ndarray, peaks: list[int], radius: int) -> list[int]:
    selected: list[int] = []
    for index in sorted(peaks, key=lambda item: (-scores[item], item)):
        if all(abs(index - kept) > radius for kept in selected):
            selected.append(index)
    return sorted(selected)


def extend_event(scores: np.ndarray, peak: int, threshold: float, maximum_gap: int) -> tuple[int, int]:
    left = peak
    gap = 0
    cursor = peak - 1
    while cursor >= 0:
        if scores[cursor] >= threshold:
            left, gap = cursor, 0
        else:
            gap += 1
            if gap > maximum_gap:
                break
        cursor -= 1
    right = peak
    gap = 0
    cursor = peak + 1
    while cursor < len(scores):
        if scores[cursor] >= threshold:
            right, gap = cursor, 0
        else:
            gap += 1
            if gap > maximum_gap:
                break
        cursor += 1
    return left, right


def score_vector(group: pd.DataFrame, species: str, trigger_mode: str, gate: float) -> np.ndarray:
    source = group[f"source_{species}"].to_numpy(dtype=np.float64)
    if species == "UndBio":
        return source
    trigger = group["trigger_known_whale"].to_numpy(dtype=np.float64)
    if trigger_mode == "product":
        return trigger * source
    if trigger_mode == "minimum":
        return np.minimum(trigger, source)
    if trigger_mode == "gate":
        return source * (trigger >= gate)
    raise ValueError(trigger_mode)


def form_events(
    frame: pd.DataFrame,
    species: str,
    config: StableConfig,
    trigger_mode: str,
    trigger_gate: float,
) -> list[PredictedEvent]:
    events: list[PredictedEvent] = []
    for recording_id, raw_group in frame.groupby("recording_id", sort=False):
        group = raw_group.sort_values("window_index").reset_index(drop=True)
        scores = moving_average(score_vector(group, species, trigger_mode, trigger_gate), config.moving_average_windows)
        support_threshold = config.threshold * config.support_ratio
        continuation_threshold = config.threshold * config.continuation_ratio
        candidates = []
        for peak in local_peaks(scores, config.threshold):
            lo = max(0, peak - config.support_radius_windows)
            hi = min(len(scores), peak + config.support_radius_windows + 1)
            if int(np.sum(scores[lo:hi] >= support_threshold)) >= config.minimum_support_windows:
                candidates.append(peak)
        peaks = suppress_peaks(scores, candidates, config.peak_suppression_windows)
        first = group.iloc[0]
        recording_events: list[PredictedEvent] = []
        for event_number, peak in enumerate(peaks):
            left, right = extend_event(
                scores, peak, continuation_threshold, config.maximum_gap_windows
            )
            envelope = np.arange(left, right + 1)
            supporting = envelope[scores[envelope] >= support_threshold]
            if len(supporting) < config.minimum_support_windows:
                continue
            top = supporting[np.argsort(scores[supporting])[-config.event_top_k :]]
            event_score = float(np.mean(scores[top]))
            threshold = config.threshold
            normalized = (event_score - threshold) / max(1e-9, 1.0 - threshold)
            predicted_ecotype = ""
            ecotype_confidence = math.nan
            if species == "KW":
                probabilities = np.asarray(
                    [
                        [float(group.iloc[index][f"ecotype_{label}"]) for label in ECOTYPE_LABELS]
                        for index in top
                    ]
                ).mean(axis=0)
                ecotype_index = int(np.argmax(probabilities))
                predicted_ecotype = ECOTYPE_ID2LABEL[ecotype_index]
                ecotype_confidence = float(probabilities[ecotype_index])
            peak_start = float(group.iloc[peak]["window_start_sec"])
            peak_end = float(group.iloc[peak]["window_end_sec"])
            recording_events.append(
                PredictedEvent(
                    event_id=f"pred_{hashlib.sha1(str(recording_id).encode()).hexdigest()[:10]}_{species}_{event_number}",
                    recording_id=str(recording_id),
                    species=species,
                    peak_window_index=int(group.iloc[peak]["window_index"]),
                    peak_time_sec=(peak_start + peak_end) / 2.0,
                    start_sec=float(group.iloc[left]["window_start_sec"]),
                    end_sec=float(group.iloc[right]["window_end_sec"]),
                    peak_score=float(scores[peak]),
                    event_score=event_score,
                    normalized_score=normalized,
                    supporting_windows=len(supporting),
                    predicted_ecotype=predicted_ecotype,
                    ecotype_confidence=ecotype_confidence,
                    provider=clean(first["provider"]),
                    dataset=clean(first["dataset"]),
                )
            )
        # A flat or multi-peaked score envelope can yield several local peaks.
        # Treat overlapping hysteresis envelopes as one event, retaining the
        # strongest candidate instead of counting duplicate detections.
        kept: list[PredictedEvent] = []
        for event in sorted(
            recording_events, key=lambda item: (-item.event_score, -item.peak_score)
        ):
            if all(
                event.end_sec < other.start_sec or event.start_sec > other.end_sec
                for other in kept
            ):
                kept.append(event)
        events.extend(sorted(kept, key=lambda item: item.peak_time_sec))
    return events


def match_same_class(
    predictions: list[PredictedEvent], truths: list[TruthEvent], collar: float
) -> tuple[int, int, int, list[tuple[int, int, float]]]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if prediction.recording_id != truth.recording_id or prediction.species != truth.species:
                continue
            if truth.start_sec - collar <= prediction.peak_time_sec <= truth.end_sec + collar:
                candidates.append((abs(prediction.peak_time_sec - truth.center_sec), pred_index, truth_index))
    candidates.sort()
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    pairs = []
    for distance, pred_index, truth_index in candidates:
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
        pairs.append((pred_index, truth_index, distance))
    return len(pairs), len(predictions) - len(pairs), len(truths) - len(pairs), pairs


def detection_metrics(tp: int, fp: int, fn: int, audio_hours: float) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fp_per_hour": fp / audio_hours if audio_hours else 0.0,
    }


def select_class_config(
    frame: pd.DataFrame,
    truths: list[TruthEvent],
    species: str,
    args: argparse.Namespace,
    audio_hours: float,
) -> tuple[StableConfig, list[PredictedEvent], list[dict[str, Any]]]:
    thresholds = parse_csv_list(args.thresholds, float, "thresholds")
    moving_averages = parse_csv_list(args.moving_average_windows, int, "moving averages")
    species_truths = [truth for truth in truths if truth.species == species]
    grid: list[dict[str, Any]] = []
    event_cache: dict[tuple[int, float], list[PredictedEvent]] = {}
    for width in moving_averages:
        for threshold in thresholds:
            config = StableConfig(
                moving_average_windows=width,
                threshold=threshold,
                support_ratio=args.support_ratio,
                continuation_ratio=args.continuation_ratio,
                minimum_support_windows=args.minimum_support_windows,
                support_radius_windows=args.support_radius_windows,
                maximum_gap_windows=args.maximum_gap_windows,
                peak_suppression_windows=args.peak_suppression_windows,
                event_top_k=args.event_top_k,
            )
            events = form_events(frame, species, config, args.trigger_combination, args.trigger_gate_threshold)
            event_cache[(width, threshold)] = events
            tp, fp, fn, _ = match_same_class(events, species_truths, args.collar_sec)
            metrics = detection_metrics(tp, fp, fn, audio_hours)
            grid.append({"species": species, **asdict(config), **metrics})
    eligible = grid
    if args.max_fp_per_hour is not None:
        constrained = [row for row in grid if float(row["fp_per_hour"]) <= args.max_fp_per_hour]
        if constrained:
            eligible = constrained
    best = max(
        eligible,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            -float(row["fp_per_hour"]),
            float(row["threshold"]),
        ),
    )
    config = StableConfig(
        **{name: best[name] for name in StableConfig.__dataclass_fields__}
    )
    return config, event_cache[(config.moving_average_windows, config.threshold)], grid


def suppress_cross_class(events: list[PredictedEvent], seconds: float) -> list[PredictedEvent]:
    by_recording: dict[str, list[PredictedEvent]] = defaultdict(list)
    for event in events:
        by_recording[event.recording_id].append(event)
    selected = []
    for group in by_recording.values():
        kept: list[PredictedEvent] = []
        for event in sorted(group, key=lambda item: (-item.normalized_score, -item.event_score)):
            if all(abs(event.peak_time_sec - other.peak_time_sec) > seconds for other in kept):
                kept.append(event)
        selected.extend(kept)
    return sorted(selected, key=lambda item: (item.recording_id, item.peak_time_sec))


def joint_evaluation(
    events: list[PredictedEvent], truths: list[TruthEvent], collar: float, audio_hours: float
) -> tuple[
    dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]
]:
    candidates = []
    for pred_index, prediction in enumerate(events):
        for truth_index, truth in enumerate(truths):
            if prediction.recording_id != truth.recording_id:
                continue
            if truth.start_sec - collar <= prediction.peak_time_sec <= truth.end_sec + collar:
                candidates.append(
                    (
                        prediction.species != truth.species,
                        abs(prediction.peak_time_sec - truth.center_sec),
                        pred_index,
                        truth_index,
                    )
                )
    candidates.sort()
    pred_to_truth: dict[int, tuple[int, float]] = {}
    used_truths: set[int] = set()
    for _, distance, pred_index, truth_index in candidates:
        if pred_index in pred_to_truth or truth_index in used_truths:
            continue
        pred_to_truth[pred_index] = (truth_index, distance)
        used_truths.add(truth_index)

    labels = list(TARGET_CLASSES)
    matrix = pd.DataFrame(
        0, index=labels + ["no_truth"], columns=labels + ["missed"], dtype=int
    )
    match_rows: list[dict[str, Any]] = []
    for pred_index, prediction in enumerate(events):
        matched = pred_to_truth.get(pred_index)
        if matched is None:
            matrix.loc["no_truth", prediction.species] += 1
            match_rows.append({**asdict(prediction), "status": "FP", "true_event_id": "", "true_species": "", "true_ecotype": "", "timing_error_sec": math.nan})
            continue
        truth_index, distance = matched
        truth = truths[truth_index]
        matrix.loc[truth.species, prediction.species] += 1
        status = "TP" if truth.species == prediction.species else "species_error"
        match_rows.append({**asdict(prediction), "status": status, "true_event_id": truth.event_id, "true_species": truth.species, "true_ecotype": truth.ecotype, "timing_error_sec": distance})
    for truth_index, truth in enumerate(truths):
        if truth_index not in used_truths:
            matrix.loc[truth.species, "missed"] += 1
            match_rows.append({"event_id": "", "recording_id": truth.recording_id, "species": "", "peak_window_index": -1, "peak_time_sec": math.nan, "start_sec": math.nan, "end_sec": math.nan, "peak_score": math.nan, "event_score": math.nan, "normalized_score": math.nan, "supporting_windows": 0, "predicted_ecotype": "", "ecotype_confidence": math.nan, "provider": truth.provider, "dataset": truth.dataset, "status": "FN", "true_event_id": truth.event_id, "true_species": truth.species, "true_ecotype": truth.ecotype, "timing_error_sec": math.nan})

    per_class = []
    for label in labels:
        tp = int(matrix.loc[label, label])
        fp = int(matrix[label].sum() - tp)
        fn = int(matrix.loc[label].sum() - tp)
        per_class.append({"species": label, **detection_metrics(tp, fp, fn, audio_hours)})
    detection_tp = len(pred_to_truth)
    detection_fp = len(events) - detection_tp
    detection_fn = len(truths) - detection_tp
    detection = detection_metrics(detection_tp, detection_fp, detection_fn, audio_hours)
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))
    total_tp = sum(int(row["tp"]) for row in per_class)
    total_fp = sum(int(row["fp"]) for row in per_class)
    total_fn = sum(int(row["fn"]) for row in per_class)
    micro = detection_metrics(total_tp, total_fp, total_fn, audio_hours)

    ecotype_labels = list(ECOTYPE_LABELS)
    ecotype_matrix = pd.DataFrame(0, index=ecotype_labels, columns=ecotype_labels, dtype=int)
    for row in match_rows:
        if row["status"] != "TP" or row["species"] != "KW":
            continue
        actual = clean(row["true_ecotype"])
        predicted = clean(row["predicted_ecotype"])
        if actual in ecotype_matrix.index and predicted in ecotype_matrix.columns:
            ecotype_matrix.loc[actual, predicted] += 1
    ecotype_total = int(ecotype_matrix.to_numpy().sum())
    ecotype_correct = int(np.trace(ecotype_matrix.to_numpy()))
    ecotype_f1s = []
    for label in ecotype_labels:
        tp = int(ecotype_matrix.loc[label, label])
        fp = int(ecotype_matrix[label].sum() - tp)
        fn = int(ecotype_matrix.loc[label].sum() - tp)
        if tp + fn:
            denominator = 2 * tp + fp + fn
            ecotype_f1s.append(2 * tp / denominator if denominator else 0.0)
    metrics = {
        "audio_hours": audio_hours,
        "truth_events": len(truths),
        "predicted_events": len(events),
        "detection": detection,
        "class_aware_micro": micro,
        "class_aware_macro_f1": macro_f1,
        "species_misclassifications": int(sum(matrix.loc[label, other] for label in labels for other in labels if other != label)),
        "ecotype_evaluated_events": ecotype_total,
        "ecotype_accuracy": ecotype_correct / ecotype_total if ecotype_total else None,
        "ecotype_macro_f1": float(np.mean(ecotype_f1s)) if ecotype_f1s else None,
    }
    return metrics, per_class, matrix, ecotype_matrix, match_rows


def grouped_event_metrics(
    match_rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    duration_hours = (
        frame.groupby("recording_id", sort=False)
        .first()
        .groupby(group_column)["duration_sec"]
        .sum()
        .div(3600.0)
        .to_dict()
    )
    groups = sorted({clean(row.get(group_column)) for row in match_rows})
    output = []
    for group in groups:
        rows = [row for row in match_rows if clean(row.get(group_column)) == group]
        hours = float(duration_hours.get(group, 0.0))
        for species in TARGET_CLASSES:
            tp = sum(
                row["status"] == "TP" and row["species"] == species for row in rows
            )
            fp = sum(
                row["species"] == species and row["status"] in {"FP", "species_error"}
                for row in rows
            )
            fn = sum(
                row["true_species"] == species and row["status"] in {"FN", "species_error"}
                for row in rows
            )
            output.append(
                {
                    group_column: group,
                    "species": species,
                    "audio_hours": hours,
                    **detection_metrics(tp, fp, fn, hours),
                }
            )
    return pd.DataFrame(output)


def evaluate_model(
    bundle: ModelBundle,
    truth_by_recording: dict[str, list[TruthEvent]],
    common_recordings: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame = pd.read_csv(bundle.cache_path, low_memory=False)
    frame = frame.loc[frame["recording_id"].astype(str).isin(common_recordings)].copy()
    truths = [
        event
        for recording_id, events in truth_by_recording.items()
        if recording_id in common_recordings
        for event in events
    ]
    durations = frame.groupby("recording_id")["duration_sec"].first()
    audio_hours = float(durations.sum()) / 3600.0
    all_events: list[PredictedEvent] = []
    grid_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, Any] = {}
    for species in TARGET_CLASSES:
        config, events, grid = select_class_config(frame, truths, species, args, audio_hours)
        selected_configs[species] = asdict(config)
        all_events.extend(events)
        grid_rows.extend(grid)
    events = suppress_cross_class(all_events, args.cross_class_suppression_sec)
    metrics, species_rows, species_matrix, ecotype_matrix, matches = joint_evaluation(
        events, truths, args.collar_sec, audio_hours
    )
    model_dir = bundle.cache_path.parent
    pd.DataFrame(grid_rows).to_csv(model_dir / "threshold_grid_results.csv", index=False)
    pd.DataFrame([asdict(event) for event in events]).to_csv(model_dir / "selected_events.csv", index=False)
    pd.DataFrame(matches).to_csv(model_dir / "event_matches.csv", index=False)
    pd.DataFrame(species_rows).to_csv(model_dir / "species_event_metrics.csv", index=False)
    species_matrix.to_csv(model_dir / "species_event_confusion_matrix.csv")
    ecotype_matrix.to_csv(model_dir / "ecotype_confusion_matrix.csv")
    grouped_event_metrics(matches, frame, "provider").to_csv(
        model_dir / "provider_species_event_metrics.csv", index=False
    )
    grouped_event_metrics(matches, frame, "dataset").to_csv(
        model_dir / "dataset_species_event_metrics.csv", index=False
    )
    report = {
        "model_name": bundle.name,
        "evaluated_recordings": len(common_recordings),
        "selected_configs": selected_configs,
        "metrics": metrics,
    }
    (model_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_comparison(reports: list[dict[str, Any]]) -> None:
    print("\n# Ranked long-validation model comparison\n")
    print(
        f"{'Rank':>4}  {'Model':48s} {'Macro F1':>9} {'Micro F1':>9} "
        f"{'Detect F1':>9} {'FP/h':>9} {'Eco F1':>9}"
    )
    for rank, report in enumerate(reports, start=1):
        metrics = report["metrics"]
        eco = metrics["ecotype_macro_f1"]
        eco_text = "None" if eco is None else f"{eco:.4f}"
        print(
            f"{rank:4d}  {report['model_name'][:48]:48s} "
            f"{metrics['class_aware_macro_f1']:9.4f} "
            f"{metrics['class_aware_micro']['f1']:9.4f} "
            f"{metrics['detection']['f1']:9.4f} "
            f"{metrics['class_aware_micro']['fp_per_hour']:9.3f} {eco_text:>9s}"
        )
    best = reports[0]
    print(f"\n# Top model: {best['model_name']}")
    print("\nSelected settings")
    for species, config in best["selected_configs"].items():
        print(
            f"{species:7s}: MA={config['moving_average_windows']}, "
            f"threshold={config['threshold']}, support={config['support_ratio']}, "
            f"continuation={config['continuation_ratio']}"
        )
    matrix_path = Path(best["model_dir"]) / "species_event_confusion_matrix.csv"
    print("\nClass-aware confusion matrix")
    print(pd.read_csv(matrix_path, index_col=0).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--source-plan", default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--extraction-modes",
        default="download_then_extract",
        help="Comma-separated source-plan extraction modes; remote_seek is excluded by default.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--model-devices",
        default="",
        help="Comma-separated devices assigned round-robin, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--parallel-model-devices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run distinct model-device groups concurrently (default: enabled).",
    )
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reuse-window-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mean-subtract", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-pass-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-pass-cutoff-hz", type=float, default=None)
    parser.add_argument("--high-pass-order", type=int, default=None)
    parser.add_argument("--trigger-combination", choices=("product", "minimum", "gate"), default="product")
    parser.add_argument("--trigger-gate-threshold", type=float, default=0.50)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    parser.add_argument("--moving-average-windows", default="1,2")
    parser.add_argument("--support-ratio", type=float, default=0.70)
    parser.add_argument("--continuation-ratio", type=float, default=0.40)
    parser.add_argument("--minimum-support-windows", type=int, default=2)
    parser.add_argument("--support-radius-windows", type=int, default=2)
    parser.add_argument("--maximum-gap-windows", type=int, default=0)
    parser.add_argument("--peak-suppression-windows", type=int, default=1)
    parser.add_argument("--event-top-k", type=int, default=2)
    parser.add_argument("--collar-sec", type=float, default=1.5)
    parser.add_argument("--cross-class-suppression-sec", type=float, default=1.0)
    parser.add_argument("--max-fp-per-hour", type=float, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.model_names)) != len(args.model_names):
        raise ValueError("--model-names contains duplicates")
    for path in (Path(args.source_plan), Path(args.annotations)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1 or args.window_sec <= 0 or args.hop_sec <= 0:
        raise ValueError("Batch size, window seconds, and hop seconds must be positive")
    if args.minimum_support_windows < 1 or args.event_top_k < 1:
        raise ValueError("Support windows and event top-k must be positive")
    if args.support_radius_windows < 0 or args.maximum_gap_windows < 0 or args.peak_suppression_windows < 0:
        raise ValueError("Window-radius settings cannot be negative")
    for name in ("support_ratio", "continuation_ratio", "trigger_gate_threshold"):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    thresholds = parse_csv_list(args.thresholds, float, "thresholds")
    if any(not 0 < value < 1 for value in thresholds):
        raise ValueError("All thresholds must lie strictly between zero and one")


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings(args)
    truth_by_recording = load_truth(Path(args.annotations), recordings)
    truth_counts = Counter(event.species for events in truth_by_recording.values() for event in events)
    print(f"Selected local validation recordings: {len(recordings):,}")
    print(f"Ground-truth events: {dict(truth_counts)}")
    bundles, extractor = load_bundles(args, output_dir)
    timings = infer_recordings(args, recordings, bundles, extractor)
    (output_dir / "inference_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    reports = []
    completed_sets = [
        completed_recordings(bundle.cache_path)
        for bundle in bundles
        if bundle.cache_path.is_file()
    ]
    common_recordings = set.intersection(*completed_sets) if completed_sets else set()
    if not common_recordings:
        raise RuntimeError("No recording has a complete window cache for every model")
    print(f"Common completed recordings used for comparison: {len(common_recordings):,}")
    for bundle in bundles:
        if not bundle.cache_path.is_file():
            print(f"Skipping evaluation with no completed cache: {bundle.name}")
            continue
        print(f"Evaluating cached events: {bundle.name}")
        report = evaluate_model(bundle, truth_by_recording, common_recordings, args)
        report["model_dir"] = str(bundle.cache_path.parent)
        reports.append(report)
    if not reports:
        raise RuntimeError("No model produced a usable window cache")
    reports.sort(
        key=lambda item: (
            -item["metrics"]["class_aware_macro_f1"],
            -item["metrics"]["class_aware_micro"]["f1"],
            item["metrics"]["class_aware_micro"]["fp_per_hour"],
        )
    )
    rows = []
    for rank, report in enumerate(reports, start=1):
        metrics = report["metrics"]
        rows.append(
            {
                "rank": rank,
                "model_name": report["model_name"],
                "class_aware_macro_f1": metrics["class_aware_macro_f1"],
                "class_aware_micro_f1": metrics["class_aware_micro"]["f1"],
                "detection_f1_ignoring_species": metrics["detection"]["f1"],
                "class_aware_fp_per_hour": metrics["class_aware_micro"]["fp_per_hour"],
                "ecotype_accuracy": metrics["ecotype_accuracy"],
                "ecotype_macro_f1": metrics["ecotype_macro_f1"],
                "audio_hours": metrics["audio_hours"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "ranked_model_comparison.csv", index=False)
    print_comparison(reports)
    print(f"\nSaved comparison and per-model reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
