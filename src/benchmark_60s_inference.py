#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Benchmark deployment-style inference on local 60-second Orcasound WAVs.

The timed boundary is ``predict(wav_path)`` for both implementations. This is
the same boundary reported as Avg Time by compare_models.py. Model creation,
Hugging Face downloads, and acquisition of the 60-second WAV are deliberately
outside the timer. Models are benchmarked sequentially so each receives the
same device without competing for GPU memory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


DEFAULT_PODSAI_MODEL = "davethaler/whale-call-detector"
DEFAULT_PODSAI_REVISION = "db51f75da131de0e53e8080a1f2c5f4b534810aa"


@dataclass(frozen=True)
class AudioSample:
    row_number: int
    category: str
    node_name: str
    timestamp: str
    wav_path: Path


@dataclass(frozen=True)
class TimingRow:
    model: str
    model_path: str
    sample_number: int
    manifest_row: int
    repeat: int
    category: str
    wav_path: str
    predict_seconds: float
    predicted_label: str
    global_confidence: float | None


def parse_csv_values(value: str) -> list[str]:
    result = [item.strip().casefold() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one model must be selected")
    unknown = sorted(set(result) - {"v2", "v2_optimized", "podsai"})
    if unknown:
        raise ValueError(f"Unknown --models values: {unknown}")
    return result


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def first_value(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = clean(row.get(name))
        if value:
            return value
    return ""


def resolve_wav(row: dict[str, str], wav_root: Path) -> Path:
    explicit = first_value(row, ("wav_path", "file_path", "clip_path", "audio_path"))
    candidates: list[Path] = []
    if explicit:
        path = Path(explicit)
        candidates.extend((path, wav_root / path))

    category = first_value(row, ("Category", "category", "label"))
    node = first_value(row, ("NodeName", "node_name")).replace("_", "-")
    timestamp = first_value(row, ("Timestamp", "timestamp"))
    if node and timestamp:
        filename = f"{node}_{timestamp}.wav"
        candidates.extend((wav_root / category / filename, wav_root / filename))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(item) for item in candidates) or "no usable path fields"
    raise FileNotFoundError(rendered)


def load_samples(manifest: Path, wav_root: Path) -> tuple[list[AudioSample], list[str]]:
    samples: list[AudioSample] = []
    missing: list[str] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            try:
                wav_path = resolve_wav(row, wav_root)
            except FileNotFoundError as error:
                missing.append(f"row {row_number}: {error}")
                continue
            samples.append(
                AudioSample(
                    row_number=row_number,
                    category=first_value(row, ("Category", "category", "label")),
                    node_name=first_value(row, ("NodeName", "node_name")),
                    timestamp=first_value(row, ("Timestamp", "timestamp")),
                    wav_path=wav_path,
                )
            )
    return samples, missing


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_predict(
    predictor: Any,
    wav_path: Path,
    device: str,
    segment_seconds: int,
    hop_seconds: int,
) -> tuple[dict[str, Any], float]:
    synchronize(device)
    started = time.perf_counter()
    result = predictor.predict(
        str(wav_path),
        segment_duration=segment_seconds,
        hop_duration=hop_seconds,
    )
    synchronize(device)
    return result, time.perf_counter() - started


def benchmark_model(
    label: str,
    model_path: str,
    predictor: Any,
    samples: list[AudioSample],
    device: str,
    warmup_runs: int,
    repeats: int,
    segment_seconds: int,
    hop_seconds: int,
) -> list[TimingRow]:
    print(f"\nWarming up {label}: {warmup_runs} run(s)")
    for index in range(warmup_runs):
        sample = samples[index % len(samples)]
        predictor.predict(
            str(sample.wav_path),
            segment_duration=segment_seconds,
            hop_duration=hop_seconds,
        )
    synchronize(device)

    rows: list[TimingRow] = []
    total = len(samples) * repeats
    completed = 0
    for repeat in range(1, repeats + 1):
        for sample_number, sample in enumerate(samples, start=1):
            result, elapsed = timed_predict(
                predictor,
                sample.wav_path,
                device,
                segment_seconds,
                hop_seconds,
            )
            completed += 1
            confidence = result.get("global_confidence")
            rows.append(
                TimingRow(
                    model=label,
                    model_path=model_path,
                    sample_number=sample_number,
                    manifest_row=sample.row_number,
                    repeat=repeat,
                    category=sample.category,
                    wav_path=str(sample.wav_path),
                    predict_seconds=elapsed,
                    predicted_label=clean(result.get("global_prediction_label")),
                    global_confidence=(float(confidence) if confidence is not None else None),
                )
            )
            print(
                f"[{label} {completed}/{total}] {sample.wav_path.name}: "
                f"{elapsed:.3f}s"
            )
    return rows


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def summarize(rows: list[TimingRow], clip_seconds: float) -> dict[str, Any]:
    values = [row.predict_seconds for row in rows]
    mean = statistics.fmean(values)
    return {
        "model": rows[0].model,
        "model_path": rows[0].model_path,
        "timed_predictions": len(values),
        "mean_predict_seconds": mean,
        "median_predict_seconds": statistics.median(values),
        "p90_predict_seconds": percentile(values, 90),
        "p95_predict_seconds": percentile(values, 95),
        "minimum_predict_seconds": min(values),
        "maximum_predict_seconds": max(values),
        "real_time_factor": mean / clip_seconds,
        "realtime_speed_multiple": clip_seconds / mean,
        "estimated_clips_per_hour": 3600.0 / mean,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_v2_predictions(rows: list[TimingRow]) -> list[dict[str, Any]]:
    reference = {
        (row.manifest_row, row.repeat): row
        for row in rows
        if row.model == "multispecies_v2"
    }
    optimized = {
        (row.manifest_row, row.repeat): row
        for row in rows
        if row.model.startswith("multispecies_v2_full_spectrogram_")
    }
    comparisons: list[dict[str, Any]] = []
    for key in sorted(reference.keys() & optimized.keys()):
        original = reference[key]
        faster = optimized[key]
        confidence_difference = None
        if (
            original.global_confidence is not None
            and faster.global_confidence is not None
        ):
            confidence_difference = abs(
                original.global_confidence - faster.global_confidence
            )
        comparisons.append(
            {
                "manifest_row": original.manifest_row,
                "repeat": original.repeat,
                "category": original.category,
                "wav_path": original.wav_path,
                "reference_prediction": original.predicted_label,
                "optimized_prediction": faster.predicted_label,
                "predictions_agree": (
                    original.predicted_label == faster.predicted_label
                ),
                "reference_confidence": original.global_confidence,
                "optimized_confidence": faster.global_confidence,
                "absolute_confidence_difference": confidence_difference,
            }
        )
    return comparisons


def release_model(predictor: Any) -> None:
    model = getattr(predictor, "model", None)
    if model is not None and hasattr(model, "to"):
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testing-csv", default="output/csv/testing_60s_samples.csv")
    parser.add_argument("--wav-dir", default="output/testing-wav")
    parser.add_argument(
        "--models",
        default="v2,v2_optimized,podsai",
        help="Comma-separated selection: v2, v2_optimized, and/or podsai.",
    )
    parser.add_argument("--v2-model-path")
    parser.add_argument("--v2-model-revision")
    parser.add_argument(
        "--optimized-preserve-max-length",
        action="store_true",
        help=(
            "Compute one full spectrogram but preserve the checkpoint's padded AST "
            "frame length. By default v2_optimized uses compact PODS-AI-style frames."
        ),
    )
    parser.add_argument("--aggregation-json", type=Path)
    parser.add_argument("--podsai-model-path", default=DEFAULT_PODSAI_MODEL)
    parser.add_argument("--podsai-model-revision", default=DEFAULT_PODSAI_REVISION)
    parser.add_argument(
        "--pods-ai-src",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing model_inference.py and podsai_inference.py.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=int, default=3)
    parser.add_argument("--hop-seconds", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", default="output/benchmark_60s_inference")
    args = parser.parse_args()
    args.models = parse_csv_values(args.models)
    if {"v2", "v2_optimized"}.intersection(args.models) and not args.v2_model_path:
        parser.error("--v2-model-path is required when benchmarking v2")
    for name in (
        "inference_batch_size",
        "segment_seconds",
        "hop_seconds",
        "max_files",
        "repeats",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    manifest = Path(args.testing_csv)
    wav_root = Path(args.wav_dir)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    samples, missing = load_samples(manifest, wav_root)
    if not samples:
        raise RuntimeError("No usable WAV files were found from the testing manifest")
    rng = random.Random(args.seed)
    samples = rng.sample(samples, min(args.max_files, len(samples)))
    samples.sort(key=lambda item: (item.category, item.node_name, item.timestamp))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected WAV files:       {len(samples):,}")
    print(f"Missing manifest WAVs:    {len(missing):,}")
    print(f"Device:                   {device}")
    print(f"Window/hop seconds:       {args.segment_seconds}/{args.hop_seconds}")
    print(f"Inference batch size:     {args.inference_batch_size}")
    print("Timed scope: predict(wav); excludes model loading and WAV download")

    timing_rows: list[TimingRow] = []
    summaries: list[dict[str, Any]] = []
    aggregation: dict[str, Any] = {}
    if args.aggregation_json is not None:
        loaded_aggregation = json.loads(
            args.aggregation_json.read_text(encoding="utf-8")
        )
        aggregation = loaded_aggregation.get("aggregation", loaded_aggregation)
        if not isinstance(aggregation, dict):
            raise ValueError(
                "--aggregation-json must contain an object or an 'aggregation' object"
            )

    source_dir = args.pods_ai_src.resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    if "v2" in args.models:
        from multispecies_cetacean_inference import MultispeciesCetaceanInference

        predictor = MultispeciesCetaceanInference(
            args.v2_model_path,
            device=device,
            model_revision=args.v2_model_revision,
            inference_batch_size=args.inference_batch_size,
            aggregation_config=aggregation,
        )
        rows = benchmark_model(
            "multispecies_v2",
            args.v2_model_path,
            predictor,
            samples,
            device,
            args.warmup_runs,
            args.repeats,
            args.segment_seconds,
            args.hop_seconds,
        )
        timing_rows.extend(rows)
        summaries.append(summarize(rows, 60.0))
        release_model(predictor)
        del predictor

    if "v2_optimized" in args.models:
        from multispecies_cetacean_inference_full_spectrogram import (
            FullSpectrogramMultispeciesCetaceanInference,
        )

        predictor = FullSpectrogramMultispeciesCetaceanInference(
            args.v2_model_path,
            device=device,
            model_revision=args.v2_model_revision,
            inference_batch_size=args.inference_batch_size,
            aggregation_config=aggregation,
            compact_ast_frames=not args.optimized_preserve_max_length,
        )
        rows = benchmark_model(
            (
                "multispecies_v2_full_spectrogram_padded"
                if args.optimized_preserve_max_length
                else "multispecies_v2_full_spectrogram_compact"
            ),
            args.v2_model_path,
            predictor,
            samples,
            device,
            args.warmup_runs,
            args.repeats,
            args.segment_seconds,
            args.hop_seconds,
        )
        timing_rows.extend(rows)
        summaries.append(summarize(rows, 60.0))
        release_model(predictor)
        del predictor

    if "podsai" in args.models:
        if not (source_dir / "model_inference.py").is_file():
            raise FileNotFoundError(f"model_inference.py not found under {source_dir}")
        from model_inference import get_model_inference

        predictor = get_model_inference(
            model_type="podsai",
            model_path=args.podsai_model_path,
            model_revision=args.podsai_model_revision,
            device=device,
            inference_batch_size=args.inference_batch_size,
        )
        rows = benchmark_model(
            "podsai",
            args.podsai_model_path,
            predictor,
            samples,
            device,
            args.warmup_runs,
            args.repeats,
            args.segment_seconds,
            args.hop_seconds,
        )
        timing_rows.extend(rows)
        summaries.append(summarize(rows, 60.0))
        release_model(predictor)
        del predictor

    timing_dicts = [asdict(row) for row in timing_rows]
    write_csv(
        output_dir / "per_clip_inference_times.csv",
        timing_dicts,
        list(TimingRow.__dataclass_fields__),
    )
    write_csv(
        output_dir / "inference_benchmark_summary.csv",
        summaries,
        list(summaries[0]),
    )
    prediction_comparisons = compare_v2_predictions(timing_rows)
    if prediction_comparisons:
        write_csv(
            output_dir / "v2_prediction_agreement.csv",
            prediction_comparisons,
            list(prediction_comparisons[0]),
        )
    report = {
        "settings": {
            "testing_csv": str(manifest),
            "wav_dir": str(wav_root),
            "selected_files": len(samples),
            "missing_files": len(missing),
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "device": device,
            "inference_batch_size": args.inference_batch_size,
            "segment_seconds": args.segment_seconds,
            "hop_seconds": args.hop_seconds,
            "timed_scope": "predict(wav); excludes model loading and WAV download",
        },
        "models": summaries,
    }
    if prediction_comparisons:
        agreement = statistics.fmean(
            float(row["predictions_agree"])
            for row in prediction_comparisons
        )
        report["v2_prediction_agreement"] = agreement
    (output_dir / "inference_benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if missing:
        (output_dir / "missing_wavs.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )

    print("\n# Comparable 60-second predict() timing")
    print(
        f"{'Model':20s} {'Mean':>9s} {'Median':>9s} {'P90':>9s} "
        f"{'P95':>9s} {'RT factor':>10s} {'x realtime':>11s}"
    )
    for summary in summaries:
        print(
            f"{summary['model']:20s} "
            f"{summary['mean_predict_seconds']:8.3f}s "
            f"{summary['median_predict_seconds']:8.3f}s "
            f"{summary['p90_predict_seconds']:8.3f}s "
            f"{summary['p95_predict_seconds']:8.3f}s "
            f"{summary['real_time_factor']:10.4f} "
            f"{summary['realtime_speed_multiple']:10.2f}x"
        )
    if prediction_comparisons:
        print(
            "Reference/optimized global-label agreement: "
            f"{100.0 * report['v2_prediction_agreement']:.2f}%"
        )
    print(f"\nSaved benchmark outputs to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
