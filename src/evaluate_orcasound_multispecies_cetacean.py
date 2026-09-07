#!/usr/bin/env python3
"""Evaluate the V2 multispecies cetacean model on Orcasound 60-second clips.

Each recording is divided into overlapping three-second windows. The model's
known-whale trigger, four-class source head, and five-class ecotype head are
cached for threshold tuning. Aggregation uses per-class gates followed by the
mean of each class's strongest qualifying windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt
from transformers import AutoFeatureExtractor

from train_multispecies_cetacean_model import (
    ECOTYPE_LABELS,
    SAMPLE_RATE,
    SOURCE_LABELS,
    TRIGGER_LABELS,
    checkpoint_files,
    load_model,
)


EVALUATION_LABELS = ("humpback", "resident", "transient", "other/background")
WHALE_LABELS = EVALUATION_LABELS[:3]
PROBABILITY_COLUMNS = (
    "trigger_not_whale",
    "trigger_known_whale",
    "source_Abiotic",
    "source_KW",
    "source_HW",
    "source_UndBio",
    "ecotype_NRKW",
    "ecotype_SRKW",
    "ecotype_OKW",
    "ecotype_SAR",
    "ecotype_TKW",
)


@dataclass(frozen=True)
class AggregationConfig:
    trigger_threshold: float = 0.50
    trigger_combination: str = "gate"
    ecotype_mode: str = "srkw_tkw_conditional"
    kw_source_threshold: float = 0.0
    hw_source_threshold: float = 0.0
    resident_ecotype_threshold: float = 0.0
    transient_ecotype_threshold: float = 0.0
    humpback_threshold: float = 0.40
    resident_threshold: float = 0.20
    transient_threshold: float = 0.20
    top_k: int = 2
    humpback_min_windows: int = 2
    resident_min_windows: int = 2
    transient_min_windows: int = 2
    smoothing: bool = False


@dataclass
class CachedSample:
    sample_id: int
    category: str
    actual_label: str
    wav_path: str
    window_index: np.ndarray
    window_start_sec: np.ndarray
    probabilities: dict[str, np.ndarray]


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "n/a"} else text


def normalize_label(category: str) -> str:
    value = clean(category).casefold().replace("_", " ").replace("-", " ")
    if value in {"resident", "srkw", "southern resident", "southern residents"}:
        return "resident"
    if value in {"transient", "transients", "tkw", "biggs", "bigg's"}:
        return "transient"
    if value in {"humpback", "hw", "humpback whale"}:
        return "humpback"
    return "other/background"


def load_test_manifest(path: Path, category: str | None, maximum: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "Category" not in frame:
        raise ValueError(f"{path} is missing Category; columns={list(frame.columns)}")
    if category is not None:
        frame = frame.loc[frame["Category"].fillna("").astype(str).eq(category)]
    if maximum is not None:
        if maximum < 1:
            raise ValueError("--max-samples must be positive")
        frame = frame.iloc[:maximum]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No matching test rows")
    return frame


def resolve_wav(row: pd.Series, wav_dir: Path) -> Path:
    for column in ("wav_path", "audio_path", "clip_path", "path"):
        value = clean(row.get(column))
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_file():
            return candidate.resolve()
        rooted = wav_dir / candidate
        if rooted.is_file():
            return rooted.resolve()
    category = clean(row.get("Category"))
    node = clean(row.get("NodeName")).replace("_", "-")
    timestamp = clean(row.get("Timestamp"))
    filename = f"{node}_{timestamp}.wav"
    candidates = (wav_dir / category / filename, wav_dir / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def checkpoint_preprocessing(model_name: str) -> dict[str, Any]:
    checkpoint = checkpoint_files(model_name)
    metadata = checkpoint[0] if checkpoint is not None else {}
    augmentation = metadata.get("preprocessing", metadata.get("augmentation", {}))
    return {
        "mean_subtract": bool(augmentation.get("mean_subtract", False)),
        "high_pass_filter": bool(augmentation.get("high_pass_filter", False)),
        "high_pass_cutoff_hz": float(augmentation.get("high_pass_cutoff_hz", 50.0)),
        "high_pass_order": int(augmentation.get("high_pass_order", 4)),
    }


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = audio[:, 0]
    if int(rate) != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        waveform = resample_poly(waveform, SAMPLE_RATE // divisor, int(rate) // divisor)
    waveform = np.asarray(waveform, dtype=np.float32)
    return waveform


def audio_windows(
    audio: np.ndarray,
    segment_seconds: float,
    hop_seconds: float,
    preprocessing: dict[str, Any],
) -> tuple[list[np.ndarray], np.ndarray]:
    segment_samples = round(segment_seconds * SAMPLE_RATE)
    hop_samples = round(hop_seconds * SAMPLE_RATE)
    positions = max(1, math.floor((len(audio) - segment_samples) / hop_samples) + 1)
    windows: list[np.ndarray] = []
    starts: list[float] = []
    high_pass_sos: np.ndarray | None = None
    if preprocessing["high_pass_filter"]:
        high_pass_sos = butter(
            preprocessing["high_pass_order"],
            preprocessing["high_pass_cutoff_hz"],
            btype="highpass",
            fs=SAMPLE_RATE,
            output="sos",
        )
    for index in range(positions):
        start = index * hop_samples
        segment = audio[start : start + segment_samples]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        segment = np.asarray(segment, dtype=np.float32)
        # Match ArchiveAudioCollator: deterministic preprocessing is applied
        # independently to every three-second example, not to the full file.
        if preprocessing["mean_subtract"]:
            segment = segment - float(segment.mean())
        if high_pass_sos is not None:
            try:
                segment = sosfiltfilt(high_pass_sos, segment).astype(np.float32)
            except ValueError:
                segment = sosfilt(high_pass_sos, segment).astype(np.float32)
        windows.append(np.asarray(segment, dtype=np.float32))
        starts.append(start / SAMPLE_RATE)
    return windows, np.asarray(starts, dtype=np.float64)


def infer_window_cache(
    manifest: pd.DataFrame,
    wav_dir: Path,
    model_name: str,
    batch_size: int,
    device: str,
    segment_seconds: float,
    hop_seconds: float,
    preprocessing: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    print(f"Loading V2 multispecies model: {model_name}")
    model, identity, feature_source = load_model(model_name, dropout=0.0, freeze_backbone=False)
    try:
        extractor = AutoFeatureExtractor.from_pretrained(feature_source)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    inference_seconds = 0.0
    print(f"Model identity: {identity}")
    print(f"Preprocessing: {preprocessing}")
    for sample_id, row in manifest.iterrows():
        wav_path = resolve_wav(row, wav_dir)
        audio = load_audio(wav_path)
        windows, starts = audio_windows(audio, segment_seconds, hop_seconds, preprocessing)
        trigger_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        ecotype_parts: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_start in range(0, len(windows), batch_size):
                batch = windows[batch_start : batch_start + batch_size]
                features = extractor(
                    batch,
                    sampling_rate=SAMPLE_RATE,
                    padding=True,
                    return_tensors="pt",
                )
                values = features["input_values"].to(device)
                trigger_logits, source_logits, ecotype_logits = model(input_values=values)
                trigger_parts.append(torch.softmax(trigger_logits, dim=-1).cpu().numpy())
                source_parts.append(torch.softmax(source_logits, dim=-1).cpu().numpy())
                ecotype_parts.append(torch.softmax(ecotype_logits, dim=-1).cpu().numpy())
        inference_seconds += time.perf_counter() - started
        trigger = np.concatenate(trigger_parts)
        source = np.concatenate(source_parts)
        ecotype = np.concatenate(ecotype_parts)
        category = clean(row.get("Category"))
        actual = normalize_label(category)
        for window_index, start_sec in enumerate(starts):
            output: dict[str, Any] = {
                "sample_id": int(sample_id),
                "category": category,
                "actual_label": actual,
                "wav_path": str(wav_path),
                "window_index": window_index,
                "window_start_sec": start_sec,
            }
            for label, index in TRIGGER_LABELS.items():
                output[f"trigger_{label}"] = float(trigger[window_index, index])
            for label, index in SOURCE_LABELS.items():
                output[f"source_{label}"] = float(source[window_index, index])
            for label, index in ECOTYPE_LABELS.items():
                output[f"ecotype_{label}"] = float(ecotype[window_index, index])
            rows.append(output)
        print(f"[{sample_id + 1}/{len(manifest)}] {wav_path.name}: {len(windows)} windows")
    model.to("cpu")
    return pd.DataFrame(rows), inference_seconds


def load_cached_samples(frame: pd.DataFrame) -> list[CachedSample]:
    required = {"sample_id", "category", "actual_label", "wav_path", "window_index", "window_start_sec", *PROBABILITY_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Window cache is missing columns: {sorted(missing)}")
    samples: list[CachedSample] = []
    for sample_id, group in frame.groupby("sample_id", sort=True):
        group = group.sort_values("window_index")
        first = group.iloc[0]
        samples.append(
            CachedSample(
                sample_id=int(sample_id),
                category=clean(first["category"]),
                actual_label=clean(first["actual_label"]),
                wav_path=clean(first["wav_path"]),
                window_index=group["window_index"].to_numpy(dtype=np.int64),
                window_start_sec=group["window_start_sec"].to_numpy(dtype=np.float64),
                probabilities={column: group[column].to_numpy(dtype=np.float64) for column in PROBABILITY_COLUMNS},
            )
        )
    return samples


def smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    result = values.copy()
    result[1:-1] = (values[:-2] + values[1:-1]) / 2.0
    return result


def class_window_scores(sample: CachedSample, config: AggregationConfig) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probability = sample.probabilities
    trigger = probability["trigger_known_whale"]
    kw = probability["source_KW"]
    hw = probability["source_HW"]
    srkw = probability["ecotype_SRKW"]
    tkw = probability["ecotype_TKW"]
    if config.ecotype_mode == "srkw_tkw_conditional":
        denominator = srkw + tkw + 1e-12
        srkw_component = srkw / denominator
        tkw_component = tkw / denominator
    elif config.ecotype_mode == "raw":
        srkw_component, tkw_component = srkw, tkw
    else:
        raise ValueError(f"Unknown ecotype mode: {config.ecotype_mode}")
    multiplier = trigger if config.trigger_combination == "product" else np.ones_like(trigger)
    if config.trigger_combination not in {"gate", "product"}:
        raise ValueError(f"Unknown trigger combination: {config.trigger_combination}")
    scores = {
        "humpback": multiplier * hw,
        "resident": multiplier * kw * srkw_component,
        "transient": multiplier * kw * tkw_component,
    }
    gates = {
        "humpback": (trigger >= config.trigger_threshold) & (hw >= config.hw_source_threshold),
        "resident": (
            (trigger >= config.trigger_threshold)
            & (kw >= config.kw_source_threshold)
            & (srkw_component >= config.resident_ecotype_threshold)
        ),
        "transient": (
            (trigger >= config.trigger_threshold)
            & (kw >= config.kw_source_threshold)
            & (tkw_component >= config.transient_ecotype_threshold)
        ),
    }
    if config.smoothing:
        scores = {label: smooth(values) for label, values in scores.items()}
        # Gates remain unsmoothed so a neighboring trigger cannot manufacture a call.
    return scores, gates


def aggregate_sample(sample: CachedSample, config: AggregationConfig) -> dict[str, Any]:
    if config.top_k < 1 or min(
        config.humpback_min_windows, config.resident_min_windows, config.transient_min_windows
    ) < 1:
        raise ValueError("top-k and minimum-window values must be positive")
    thresholds = {
        "humpback": config.humpback_threshold,
        "resident": config.resident_threshold,
        "transient": config.transient_threshold,
    }
    minimums = {
        "humpback": config.humpback_min_windows,
        "resident": config.resident_min_windows,
        "transient": config.transient_min_windows,
    }
    scores, gates = class_window_scores(sample, config)
    candidates: dict[str, tuple[float, int, int]] = {}
    output: dict[str, Any] = {}
    for label in WHALE_LABELS:
        qualifying = gates[label] & (scores[label] >= thresholds[label])
        indices = np.flatnonzero(qualifying)
        output[f"{label}_positive_windows"] = int(len(indices))
        output[f"{label}_peak_score"] = float(np.max(scores[label])) if len(scores[label]) else 0.0
        if len(indices) < minimums[label]:
            output[f"{label}_aggregate_score"] = 0.0
            continue
        ordered = indices[np.argsort(scores[label][indices])[::-1]]
        selected = ordered[: config.top_k]
        aggregate = float(np.mean(scores[label][selected]))
        peak_index = int(ordered[0])
        output[f"{label}_aggregate_score"] = aggregate
        candidates[label] = (aggregate, len(indices), peak_index)
    if candidates:
        predicted = max(candidates, key=lambda label: (candidates[label][0], candidates[label][1]))
        confidence, _, peak_index = candidates[predicted]
        output.update(
            predicted_label=predicted,
            confidence=confidence,
            top_window_index=int(sample.window_index[peak_index]),
            top_window_start_sec=float(sample.window_start_sec[peak_index]),
        )
    else:
        output.update(predicted_label="other/background", confidence=0.0, top_window_index=-1, top_window_start_sec=math.nan)
    output.update(
        sample_id=sample.sample_id,
        category=sample.category,
        actual_label=sample.actual_label,
        wav_path=sample.wav_path,
    )
    return output


def confusion_and_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.DataFrame(0, index=EVALUATION_LABELS, columns=EVALUATION_LABELS, dtype=int)
    for _, row in predictions.iterrows():
        matrix.loc[row["actual_label"], row["predicted_label"]] += 1
    total = int(matrix.to_numpy().sum())
    correct = int(np.trace(matrix.to_numpy()))
    per_class: dict[str, dict[str, float | int]] = {}
    for label in EVALUATION_LABELS:
        true_positive = int(matrix.loc[label, label])
        false_positive = int(matrix[label].sum() - true_positive)
        false_negative = int(matrix.loc[label].sum() - true_positive)
        truth = true_positive + false_negative
        predicted = true_positive + false_positive
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / truth if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        negative = total - truth
        per_class[label] = {
            "truth": truth,
            "predicted": predicted,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive / negative if negative else 0.0,
            "false_negative_rate": false_negative / truth if truth else 0.0,
        }
    whale_f1 = float(np.mean([per_class[label]["f1"] for label in WHALE_LABELS]))
    all_f1 = float(np.mean([per_class[label]["f1"] for label in EVALUATION_LABELS]))
    return matrix, {
        "evaluated": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "whale_macro_f1": whale_f1,
        "all_class_macro_f1": all_f1,
        "per_class": per_class,
    }


def evaluate_cached_samples(samples: list[CachedSample], config: AggregationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions = pd.DataFrame(aggregate_sample(sample, config) for sample in samples)
    matrix, metrics = confusion_and_metrics(predictions)
    return predictions, matrix, metrics


def print_report(name: str, matrix: pd.DataFrame, metrics: dict[str, Any], average_time: float | None = None) -> None:
    resident = metrics["per_class"]["resident"]
    transient = metrics["per_class"]["transient"]
    humpback = metrics["per_class"]["humpback"]
    time_text = "—" if average_time is None else f"{average_time:.3f}s"
    print("\n## Model           Evaluated   Correct  Accuracy      F1    RFP%    RFN%    TFP%    TFN%    HFP%    HFN%   Avg Time")
    print(
        f"{name:18s} {metrics['evaluated']:9,d} {metrics['correct']:9,d} "
        f"{100 * metrics['accuracy']:8.1f}% {metrics['whale_macro_f1']:7.3f} "
        f"{100 * resident['false_positive_rate']:7.1f}% {100 * resident['false_negative_rate']:7.1f}% "
        f"{100 * transient['false_positive_rate']:7.1f}% {100 * transient['false_negative_rate']:7.1f}% "
        f"{100 * humpback['false_positive_rate']:7.1f}% {100 * humpback['false_negative_rate']:7.1f}% {time_text:>10s}"
    )
    shown = matrix.copy()
    shown["total"] = shown.sum(axis=1)
    print("\n" + shown.to_string())
    print("\nPer-class metrics")
    for label in EVALUATION_LABELS:
        values = metrics["per_class"][label]
        print(
            f"{label:18s} P={values['precision']:.4f} R={values['recall']:.4f} "
            f"F1={values['f1']:.4f} FP={values['fp']:,} FN={values['fn']:,}"
        )


def add_aggregation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trigger-threshold", type=float, default=0.50)
    parser.add_argument("--trigger-combination", choices=("gate", "product"), default="gate")
    parser.add_argument("--ecotype-mode", choices=("srkw_tkw_conditional", "raw"), default="srkw_tkw_conditional")
    parser.add_argument("--kw-source-threshold", type=float, default=0.0)
    parser.add_argument("--hw-source-threshold", type=float, default=0.0)
    parser.add_argument("--resident-ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--transient-ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--humpback-threshold", type=float, default=0.40)
    parser.add_argument("--resident-threshold", type=float, default=0.20)
    parser.add_argument("--transient-threshold", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--humpback-min-windows", type=int, default=2)
    parser.add_argument("--resident-min-windows", type=int, default=2)
    parser.add_argument("--transient-min-windows", type=int, default=2)
    parser.add_argument("--smoothing", action=argparse.BooleanOptionalAction, default=False)


def config_from_args(args: argparse.Namespace) -> AggregationConfig:
    return AggregationConfig(**{
        field: getattr(args, field)
        for field in AggregationConfig.__dataclass_fields__
    })


def validate_config(config: AggregationConfig) -> None:
    for name, value in asdict(config).items():
        if name.endswith("threshold") and not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.top_k < 1 or min(config.humpback_min_windows, config.resident_min_windows, config.transient_min_windows) < 1:
        raise ValueError("top-k and minimum-window values must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=False)
    parser.add_argument("--testing-csv", default="output/csv/testing_60s_samples.csv")
    parser.add_argument("--wav-dir", default="output/testing-wav")
    parser.add_argument("--window-cache", default="orcasound_multispecies_cetacean_evaluation/window_predictions.csv")
    parser.add_argument("--reuse-window-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="orcasound_multispecies_cetacean_evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--mean-subtract", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-pass-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-pass-cutoff-hz", type=float, default=None)
    parser.add_argument("--high-pass-order", type=int, default=None)
    add_aggregation_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    validate_config(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.window_cache)
    inference_seconds: float | None = None
    if args.reuse_window_cache and cache_path.is_file():
        print(f"Reusing window cache: {cache_path}")
        cache = pd.read_csv(cache_path, low_memory=False)
    else:
        if not args.model_name:
            raise ValueError("--model-name is required when creating the window cache")
        testing_csv = Path(args.testing_csv)
        wav_dir = Path(args.wav_dir)
        if not testing_csv.is_file():
            raise FileNotFoundError(testing_csv)
        if not wav_dir.is_dir():
            raise FileNotFoundError(wav_dir)
        manifest = load_test_manifest(testing_csv, args.category, args.max_samples)
        preprocessing = checkpoint_preprocessing(args.model_name)
        if args.mean_subtract is not None:
            preprocessing["mean_subtract"] = args.mean_subtract
        if args.high_pass_filter is not None:
            preprocessing["high_pass_filter"] = args.high_pass_filter
        if args.high_pass_cutoff_hz is not None:
            preprocessing["high_pass_cutoff_hz"] = args.high_pass_cutoff_hz
        if args.high_pass_order is not None:
            preprocessing["high_pass_order"] = args.high_pass_order
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        cache, inference_seconds = infer_window_cache(
            manifest, wav_dir, args.model_name, args.batch_size, device,
            args.segment_seconds, args.hop_seconds, preprocessing,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)
        metadata = {
            "model_name": args.model_name,
            "testing_csv": str(testing_csv.resolve()),
            "wav_dir": str(wav_dir.resolve()),
            "preprocessing": preprocessing,
            "segment_seconds": args.segment_seconds,
            "hop_seconds": args.hop_seconds,
            "samples": int(cache["sample_id"].nunique()),
            "windows": len(cache),
        }
        cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved window cache: {cache_path}")

    samples = load_cached_samples(cache)
    predictions, matrix, metrics = evaluate_cached_samples(samples, config)
    average_time = inference_seconds / len(samples) if inference_seconds is not None else None
    predictions.to_csv(output_dir / "clip_predictions.csv", index=False)
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    report = {"aggregation": asdict(config), "metrics": metrics, "average_model_time_seconds": average_time}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nAggregation configuration")
    for name, value in asdict(config).items():
        print(f"{name:31s}: {value}")
    print_report("Multispecies-V2", matrix, metrics, average_time)
    print(f"\nSaved report files to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
