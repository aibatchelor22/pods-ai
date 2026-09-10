#!/usr/bin/env python3
"""Rank multiple V2 Hugging Face models on Orcasound 60-second clips.

Audio is prepared once for every distinct preprocessing configuration. Each
model is then run once to cache its three-second window probabilities, and a
shared reproducible subset of the aggregation grid is evaluated on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor

import evaluate_orcasound_multispecies_cetacean as evaluation
import grid_search_orcasound_multispecies_cetacean as grid_search
from train_multispecies_cetacean_model import ECOTYPE_LABELS, SAMPLE_RATE, SOURCE_LABELS, TRIGGER_LABELS, load_model


def model_slug(model_name: str) -> str:
    readable = "".join(character if character.isalnum() else "_" for character in model_name).strip("_")
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:10]
    return f"{readable[-80:]}_{digest}"


def models_from_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--models-file", type=Path)
    parser.add_argument("--testing-csv", default="output/csv/testing_60s_samples.csv")
    parser.add_argument("--wav-dir", default="output/testing-wav")
    parser.add_argument("--output-dir", default="orcasound_multispecies_cetacean_model_comparison")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--category")
    parser.add_argument("--mean-subtract", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-pass-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-pass-cutoff-hz", type=float)
    parser.add_argument("--high-pass-order", type=int)
    parser.add_argument(
        "--reuse-window-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse compatible per-model probability caches for resumable comparisons.",
    )
    parser.add_argument(
        "--ranking-metric",
        choices=["whale_macro_f1", "all_class_macro_f1"],
        default="whale_macro_f1",
    )

    # Compact defaults: the same randomized configuration set is used for all
    # models, making their independently tuned rankings directly comparable.
    parser.add_argument("--trigger-thresholds", type=grid_search.floats, default=grid_search.floats("0.25,0.5,0.75,0.9,0.97"))
    parser.add_argument("--trigger-combinations", type=grid_search.strings({"gate", "product"}), default=["gate", "product"])
    parser.add_argument("--ecotype-modes", type=grid_search.strings({"srkw_tkw_conditional", "raw"}), default=["srkw_tkw_conditional", "raw"])
    parser.add_argument("--kw-source-thresholds", type=grid_search.floats, default=grid_search.floats("0,0.25,0.5,0.75"))
    parser.add_argument("--hw-source-thresholds", type=grid_search.floats, default=grid_search.floats("0,0.25,0.5,0.75"))
    parser.add_argument("--resident-ecotype-thresholds", type=grid_search.floats, default=grid_search.floats("0,0.5,0.7"))
    parser.add_argument("--transient-ecotype-thresholds", type=grid_search.floats, default=grid_search.floats("0,0.5,0.7"))
    parser.add_argument("--humpback-thresholds", type=grid_search.floats, default=grid_search.floats("0.3,0.4,0.5,0.6"))
    parser.add_argument("--resident-thresholds", type=grid_search.floats, default=grid_search.floats("0.02,0.05,0.1,0.2,0.4"))
    parser.add_argument("--transient-thresholds", type=grid_search.floats, default=grid_search.floats("0.05,0.1,0.2,0.3,0.5"))
    parser.add_argument("--top-ks", type=grid_search.integers, default=grid_search.integers("1,2,3"))
    parser.add_argument("--humpback-min-windows-values", type=grid_search.integers, default=grid_search.integers("1,2"))
    parser.add_argument("--resident-min-windows-values", type=grid_search.integers, default=grid_search.integers("1,2,3"))
    parser.add_argument("--transient-min-windows-values", type=grid_search.integers, default=grid_search.integers("1,2,3"))
    parser.add_argument("--smoothing-values", type=grid_search.booleans, default=[False])
    parser.add_argument(
        "--max-grid-runs",
        type=int,
        default=2000,
        help="Shared randomized subset of the Cartesian grid; 0 searches all combinations.",
    )
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    # Retained because grid_search.configurations expects the standard parser namespace.
    parser.add_argument("--resident-max-fp-rate", type=float, default=1.0)
    args = parser.parse_args()
    supplied = [*args.models, *models_from_file(args.models_file)]
    args.models = list(dict.fromkeys(model.strip() for model in supplied if model.strip()))
    if not args.models:
        parser.error("Provide at least one model with --models or --models-file")
    if args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("--batch-size and --checkpoint-every must be positive")
    return args


def input_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_preprocessing(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    settings = evaluation.checkpoint_preprocessing(model_name)
    if args.mean_subtract is not None:
        settings["mean_subtract"] = args.mean_subtract
    if args.high_pass_filter is not None:
        settings["high_pass_filter"] = args.high_pass_filter
    if args.high_pass_cutoff_hz is not None:
        settings["high_pass_cutoff_hz"] = args.high_pass_cutoff_hz
    if args.high_pass_order is not None:
        settings["high_pass_order"] = args.high_pass_order
    return settings


def prepare_windows(
    manifest: pd.DataFrame,
    wav_dir: Path,
    segment_seconds: float,
    hop_seconds: float,
    preprocessing: dict[str, Any],
) -> tuple[list[np.ndarray], pd.DataFrame]:
    windows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    print(f"Preparing audio once for preprocessing profile: {preprocessing}")
    for sample_id, row in manifest.iterrows():
        wav_path = evaluation.resolve_wav(row, wav_dir)
        audio = evaluation.load_audio(wav_path)
        sample_windows, starts = evaluation.audio_windows(
            audio, segment_seconds, hop_seconds, preprocessing
        )
        category = evaluation.clean(row.get("Category"))
        actual_label = evaluation.normalize_label(category)
        for window_index, (window, start_sec) in enumerate(zip(sample_windows, starts)):
            windows.append(window)
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "category": category,
                    "actual_label": actual_label,
                    "wav_path": str(wav_path),
                    "window_index": window_index,
                    "window_start_sec": float(start_sec),
                }
            )
        if (sample_id + 1) % 25 == 0 or sample_id + 1 == len(manifest):
            print(f"  prepared {sample_id + 1:,}/{len(manifest):,} recordings; {len(windows):,} windows")
    return windows, pd.DataFrame(rows)


def infer_prepared_windows(
    model_name: str,
    windows: list[np.ndarray],
    metadata: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    print(f"\nLoading model: {model_name}")
    model, identity, feature_source = load_model(model_name, dropout=0.0, freeze_backbone=False)
    try:
        extractor = AutoFeatureExtractor.from_pretrained(feature_source)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    trigger = np.empty((len(windows), len(TRIGGER_LABELS)), dtype=np.float32)
    source = np.empty((len(windows), len(SOURCE_LABELS)), dtype=np.float32)
    ecotype = np.empty((len(windows), len(ECOTYPE_LABELS)), dtype=np.float32)
    started = time.perf_counter()
    use_amp = amp and device.type == "cuda"
    with torch.inference_mode():
        for batch_start in range(0, len(windows), batch_size):
            batch_end = min(batch_start + batch_size, len(windows))
            features = extractor(
                windows[batch_start:batch_end],
                sampling_rate=SAMPLE_RATE,
                padding=True,
                return_tensors="pt",
            )
            values = features["input_values"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(input_values=values)
            trigger[batch_start:batch_end] = torch.softmax(outputs[0].float(), dim=-1).cpu().numpy()
            source[batch_start:batch_end] = torch.softmax(outputs[1].float(), dim=-1).cpu().numpy()
            ecotype[batch_start:batch_end] = torch.softmax(outputs[2].float(), dim=-1).cpu().numpy()
            if batch_end % (batch_size * 20) == 0 or batch_end == len(windows):
                print(f"  inferred {batch_end:,}/{len(windows):,} windows")
    elapsed = time.perf_counter() - started
    frame = metadata.copy()
    for label, index in TRIGGER_LABELS.items():
        frame[f"trigger_{label}"] = trigger[:, index]
    for label, index in SOURCE_LABELS.items():
        frame[f"source_{label}"] = source[:, index]
    for label, index in ECOTYPE_LABELS.items():
        frame[f"ecotype_{label}"] = ecotype[:, index]
    model.to("cpu")
    del model, extractor, trigger, source, ecotype
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frame, elapsed, identity


def cache_is_compatible(cache_path: Path, expected: dict[str, Any]) -> bool:
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("cache_inputs") == expected


def evaluate_grid(
    model_name: str,
    cache: pd.DataFrame,
    configurations: list[evaluation.AggregationConfig],
    complete_grid_size: int,
    output_dir: Path,
    checkpoint_every: int,
    selection_metric: str,
) -> tuple[pd.Series, Path]:
    samples = evaluation.load_cached_samples(cache)
    rows: list[dict[str, Any]] = []
    model_dir = output_dir / "models" / model_slug(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / "grid_search_results.csv"
    for run, config in enumerate(configurations, start=1):
        _, _, metrics = evaluation.evaluate_cached_samples(samples, config)
        row = grid_search.result_row(run, config, metrics)
        row["model_name"] = model_name
        rows.append(row)
        if run % checkpoint_every == 0 or run == len(configurations):
            pd.DataFrame(rows).to_csv(result_path, index=False)
            best = max(rows, key=lambda item: (item[selection_metric], item["accuracy"]))
            print(
                f"  grid {run:,}/{len(configurations):,}; "
                f"best {selection_metric}={best[selection_metric]:.4f}"
            )
    results = pd.DataFrame(rows)
    results["complete_grid_size"] = complete_grid_size
    tie_metric = (
        "all_class_macro_f1"
        if selection_metric == "whale_macro_f1"
        else "whale_macro_f1"
    )
    results.sort_values(
        [selection_metric, tie_metric, "accuracy"], ascending=False
    ).to_csv(result_path, index=False)
    best = results.sort_values(
        [selection_metric, tie_metric, "accuracy"], ascending=False
    ).iloc[0]
    return best, result_path


def main() -> int:
    args = parse_args()
    testing_csv = Path(args.testing_csv).expanduser().resolve()
    wav_dir = Path(args.wav_dir).expanduser().resolve()
    if not testing_csv.is_file():
        raise FileNotFoundError(testing_csv)
    if not wav_dir.is_dir():
        raise NotADirectoryError(wav_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / "window_caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = evaluation.load_test_manifest(testing_csv, args.category, args.max_samples)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config_iterator, complete_grid_size, selected_runs = grid_search.configurations(args)
    configurations = list(config_iterator)
    print(f"Models:                 {len(args.models):,}")
    print(f"Test recordings:        {len(manifest):,}")
    print(f"Grid configurations:    {selected_runs:,} of {complete_grid_size:,}")
    print(f"Device / AMP:           {device} / {args.amp and device.type == 'cuda'}")

    profiles: dict[str, dict[str, Any]] = {}
    model_records: dict[str, dict[str, Any]] = {}
    for model_name in args.models:
        preprocessing = model_preprocessing(model_name, args)
        profile_key = json.dumps(preprocessing, sort_keys=True)
        profiles.setdefault(profile_key, {"preprocessing": preprocessing, "models": []})["models"].append(model_name)
        cache_path = cache_dir / f"{model_slug(model_name)}.csv"
        cache_inputs = {
            "model_name": model_name,
            "testing_csv": input_identity(testing_csv),
            "wav_dir": str(wav_dir),
            "preprocessing": preprocessing,
            "segment_seconds": args.segment_seconds,
            "hop_seconds": args.hop_seconds,
            "max_samples": args.max_samples,
            "category": args.category,
            "amp": args.amp and device.type == "cuda",
        }
        model_records[model_name] = {
            "cache_path": cache_path,
            "cache_inputs": cache_inputs,
            "preprocessing": preprocessing,
        }

    for profile in profiles.values():
        pending = [
            model_name
            for model_name in profile["models"]
            if not (
                args.reuse_window_caches
                and cache_is_compatible(
                    model_records[model_name]["cache_path"],
                    model_records[model_name]["cache_inputs"],
                )
            )
        ]
        for model_name in profile["models"]:
            if model_name not in pending:
                print(f"Reusing model window cache: {model_records[model_name]['cache_path']}")
        if not pending:
            continue
        windows, window_metadata = prepare_windows(
            manifest,
            wav_dir,
            args.segment_seconds,
            args.hop_seconds,
            profile["preprocessing"],
        )
        for model_name in pending:
            cache, inference_seconds, identity = infer_prepared_windows(
                model_name,
                windows,
                window_metadata,
                args.batch_size,
                device,
                args.amp,
            )
            record = model_records[model_name]
            cache.to_csv(record["cache_path"], index=False)
            record["cache_path"].with_suffix(".json").write_text(
                json.dumps(
                    {
                        "cache_inputs": record["cache_inputs"],
                        "model_identity": identity,
                        "inference_seconds": inference_seconds,
                        "recordings": int(cache["sample_id"].nunique()),
                        "windows": len(cache),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            print(f"Saved model window cache: {record['cache_path']}")
        del windows, window_metadata
        gc.collect()

    ranking_rows: list[dict[str, Any]] = []
    cache_paths: dict[str, Path] = {}
    for model_index, model_name in enumerate(args.models, start=1):
        print(f"\n[{model_index}/{len(args.models)}] Grid search: {model_name}")
        record = model_records[model_name]
        cache_path = record["cache_path"]
        cache = pd.read_csv(cache_path, low_memory=False)
        best, _ = evaluate_grid(
            model_name,
            cache,
            configurations,
            complete_grid_size,
            output_dir,
            args.checkpoint_every,
            args.ranking_metric,
        )
        cache_metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        ranking = best.to_dict()
        ranking["inference_seconds"] = cache_metadata.get("inference_seconds")
        ranking_rows.append(ranking)
        cache_paths[model_name] = cache_path
        del cache

    tie_metric = "all_class_macro_f1" if args.ranking_metric == "whale_macro_f1" else "whale_macro_f1"
    rankings = pd.DataFrame(ranking_rows).sort_values(
        [args.ranking_metric, tie_metric, "accuracy"], ascending=False
    ).reset_index(drop=True)
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    rankings.to_csv(output_dir / "model_rankings.csv", index=False)

    top = rankings.iloc[0]
    top_model = str(top["model_name"])
    top_cache = pd.read_csv(cache_paths[top_model], low_memory=False)
    top_samples = evaluation.load_cached_samples(top_cache)
    top_config = grid_search.config_from_result(top)
    predictions, matrix, metrics = evaluation.evaluate_cached_samples(top_samples, top_config)
    predictions.to_csv(output_dir / "top_model_clip_predictions.csv", index=False)
    matrix.to_csv(output_dir / "top_model_confusion_matrix.csv")
    (output_dir / "top_model_report.json").write_text(
        json.dumps(
            {
                "model_name": top_model,
                "ranking_metric": args.ranking_metric,
                "aggregation": asdict(top_config),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n# Ranked models")
    shown_columns = [
        "rank", "model_name", "whale_macro_f1", "all_class_macro_f1", "accuracy",
        "humpback_f1", "resident_f1", "transient_f1", "inference_seconds",
    ]
    print(rankings[shown_columns].to_string(index=False))
    print("\n# Top model aggregation configuration")
    for name, value in asdict(top_config).items():
        print(f"{name:31s}: {value}")
    evaluation.print_report("top_model", matrix, metrics)
    print(f"\nSaved comparison outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
