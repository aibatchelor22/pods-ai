#!/usr/bin/env python3
"""Compare V2 multispecies checkpoints on the complete extracted validation split.

This evaluates the native three-second task used during training.  It is meant
for model selection, so the default split is ``validation``; use ``--split
test`` only after the model and decision rule have been locked.

Every checkpoint is evaluated with the preprocessing recorded in its own
``multispecies_cetacean_config.json``.  Results are reported overall, for local
and remote shards separately, and by provider and provider/dataset domain.
Completed per-model predictions are cached so an interrupted Kaggle run can be
resumed without recomputing finished models.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from train_multispecies_cetacean_model import (
    ECOTYPE_ID2LABEL,
    ECOTYPE_LABELS,
    IGNORE_INDEX,
    REMOTE_MARKER,
    SAMPLE_RATE,
    SOURCE_ID2LABEL,
    SOURCE_LABELS,
    TRIGGER_ID2LABEL,
    TRIGGER_LABELS,
    ActiveRmsNormalizer,
    ArchiveAudioCollator,
    ArchiveManifestDataset,
    clean,
    data_provenance,
    discover_rows,
    file_identity,
    load_model,
    load_overlap_audit,
    metrics_for_predictions,
    checkpoint_files,
)


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
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Root searched recursively for shard manifests; repeatable (default: /kaggle/input).",
    )
    parser.add_argument("--manifest-name", default="multispecies_cetacean_manifest.csv")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/multispecies_cetacean_validation_comparison",
    )
    parser.add_argument(
        "--overlap-audit-csv",
        type=Path,
        help="Overlap sidecar used during training; recommended for matching supervision masks.",
    )
    parser.add_argument(
        "--overlap-conflict-policy",
        choices=("mask", "exclude", "include"),
        default="mask",
    )
    parser.add_argument(
        "--require-overlap-audit-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--undbio-trigger-policy",
        choices=("ignore", "positive", "negative"),
        default="ignore",
    )
    parser.add_argument(
        "--domain-columns",
        nargs="+",
        default=["Provider", "Dataset"],
        help="Manifest columns defining provider/dataset domains.",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--preprocessing-workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--maximum-clips",
        type=int,
        help="Optional deterministic random subset for a quick code test.",
    )
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument(
        "--require-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if no remote-shard rows are present (default: true).",
    )
    parser.add_argument(
        "--minimum-group-clips",
        type=int,
        default=50,
        help="Minimum rows for a provider/domain to enter macro-domain summaries.",
    )
    parser.add_argument(
        "--ranking-metric",
        choices=(
            "balanced_local_remote_score",
            "overall_combined_score",
            "remote_combined_score",
            "provider_macro_combined_score",
        ),
        default="balanced_local_remote_score",
    )
    parser.add_argument(
        "--reuse-model-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    supplied = [*args.models, *models_from_file(args.models_file)]
    args.models = list(dict.fromkeys(value.strip() for value in supplied if value.strip()))
    if not args.models:
        parser.error("Provide at least one checkpoint with --models or --models-file")
    if args.data_root is None:
        args.data_root = ["/kaggle/input"]
    if args.batch_size < 1 or args.preprocessing_workers < 0:
        parser.error("Batch size must be positive and preprocessing workers nonnegative")
    if args.maximum_clips is not None and args.maximum_clips < 1:
        parser.error("--maximum-clips must be positive")
    if args.minimum_group_clips < 1:
        parser.error("--minimum-group-clips must be positive")
    return args


def safe_slug(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[-72:]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def checkpoint_preprocessing(model_name: str) -> dict[str, Any]:
    checkpoint = checkpoint_files(model_name)
    metadata = checkpoint[0] if checkpoint is not None else {}
    values = metadata.get("preprocessing", metadata.get("augmentation", {}))
    return {
        "mean_subtract": bool(values.get("mean_subtract", False)),
        "high_pass_filter": bool(values.get("high_pass_filter", False)),
        "high_pass_cutoff_hz": float(values.get("high_pass_cutoff_hz", 50.0)),
        "high_pass_order": int(values.get("high_pass_order", 4)),
        "level_normalization": bool(values.get("level_normalization", False)),
        "level_normalization_mode": str(values.get("level_normalization_mode", "active_rms")),
        "target_active_rms_dbfs": float(values.get("target_active_rms_dbfs", -45.0)),
        "level_normalization_max_gain_db": float(
            values.get("level_normalization_max_gain_db", 12.0)
        ),
        "level_normalization_max_attenuation_db": float(
            values.get("level_normalization_max_attenuation_db", 12.0)
        ),
        "level_normalization_floor_dbfs": float(
            values.get("level_normalization_floor_dbfs", -70.0)
        ),
        "level_normalization_active_percentile": float(
            values.get("level_normalization_active_percentile", 80.0)
        ),
    }


def row_is_remote(row: dict[str, Any]) -> bool:
    fields = (
        row.get("manifest_path", ""),
        row.get("input_dataset_id", ""),
        row.get("input_dataset_root", ""),
    )
    return any(REMOTE_MARKER.search(str(value)) for value in fields)


def dataset_hash(rows: list[dict[str, Any]], split: str) -> str:
    digest = hashlib.sha256(split.encode("utf-8"))
    for row in rows:
        values = (
            row["clip_id"],
            row["trigger_label"],
            row["source_label"],
            row["source_original_label"],
            row["ecotype_label"],
            row["manifest_path"],
            row["archive_member_path"],
        )
        digest.update(json.dumps(values, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_identity(model_name: str) -> dict[str, Any]:
    checkpoint = checkpoint_files(model_name)
    if checkpoint is None:
        return {"model_name": model_name, "checkpoint": None}
    metadata, weights, kind = checkpoint
    return {
        "model_name": model_name,
        "weights": file_identity(weights),
        "kind": kind,
        "base_model": clean(metadata.get("base_model")),
    }


def prediction_frame(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "clip_id": [row["clip_id"] for row in rows],
            "provider": [row["domain_key"][0] for row in rows],
            "dataset": [
                row["domain_key"][1] if len(row["domain_key"]) > 1 else "<missing>"
                for row in rows
            ],
            "is_remote": [row_is_remote(row) for row in rows],
            "manifest_path": [row["manifest_path"] for row in rows],
            "trigger_true": [row["trigger_label"] for row in rows],
            "source_true": [row["source_label"] for row in rows],
            "source_original_true": [row["source_original_label"] for row in rows],
            "ecotype_true": [row["ecotype_label"] for row in rows],
            "trigger_pred": predictions["trigger"],
            "source_pred": predictions["source"],
            "ecotype_pred": predictions["ecotype"],
        }
    )
    for head, labels in (
        ("trigger", TRIGGER_LABELS),
        ("source", SOURCE_LABELS),
        ("ecotype", ECOTYPE_LABELS),
    ):
        probabilities = predictions[f"{head}_probabilities"]
        for label, class_id in labels.items():
            frame[f"{head}_p_{label}"] = probabilities[:, class_id]
    return frame


def infer_model(
    model_name: str,
    rows: list[dict[str, Any]],
    preprocessing: dict[str, Any],
    batch_size: int,
    workers: int,
    device: torch.device,
    amp: bool,
    multi_gpu: bool,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    print(f"\nLoading model: {model_name}")
    print(f"Preprocessing: {preprocessing}")
    model, identity, feature_source = load_model(model_name, dropout=0.0, freeze_backbone=True)
    model.to(device)
    if multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
    model.eval()
    try:
        extractor = AutoFeatureExtractor.from_pretrained(feature_source)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
    normalizer = None
    if preprocessing["level_normalization"]:
        normalizer = ActiveRmsNormalizer(
            sample_rate=SAMPLE_RATE,
            target_dbfs=preprocessing["target_active_rms_dbfs"],
            max_gain_db=preprocessing["level_normalization_max_gain_db"],
            max_attenuation_db=preprocessing["level_normalization_max_attenuation_db"],
            floor_dbfs=preprocessing["level_normalization_floor_dbfs"],
            active_percentile=preprocessing["level_normalization_active_percentile"],
        )
    collator = ArchiveAudioCollator(
        extractor,
        clip_seconds=3.0,
        mean_subtract=preprocessing["mean_subtract"],
        high_pass_cutoff_hz=(
            preprocessing["high_pass_cutoff_hz"]
            if preprocessing["high_pass_filter"]
            else None
        ),
        high_pass_order=preprocessing["high_pass_order"],
        level_normalizer=normalizer,
    )
    loader = DataLoader(
        ArchiveManifestDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    parts: dict[str, list[np.ndarray]] = {"trigger": [], "source": [], "ecotype": []}
    started = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            values = batch["input_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(values)
            for name, output in zip(("trigger", "source", "ecotype"), outputs):
                parts[name].append(torch.softmax(output.float(), dim=1).cpu().numpy())
            completed += len(values)
            if completed % max(batch_size * 100, 1) < len(values):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(rows) - completed) / max(rate, 1e-9)
                print(
                    f"  {completed:,}/{len(rows):,} clips; "
                    f"{rate:.1f} clips/s; ETA {remaining / 60:.1f} min"
                )
    seconds = time.perf_counter() - started
    probabilities = {name: np.concatenate(value) for name, value in parts.items()}
    predictions: dict[str, np.ndarray] = {
        name: values.argmax(axis=1) for name, values in probabilities.items()
    }
    predictions.update(
        {f"{name}_probabilities": values for name, values in probabilities.items()}
    )
    frame = prediction_frame(rows, predictions)
    del loader, collator, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frame, seconds, identity


def arrays_from_frame(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = {
        "trigger": frame["trigger_true"].to_numpy(dtype=np.int64),
        "source": frame["source_true"].to_numpy(dtype=np.int64),
        "source_original": frame["source_original_true"].to_numpy(dtype=np.int64),
        "ecotype": frame["ecotype_true"].to_numpy(dtype=np.int64),
    }
    predictions = {
        "trigger": frame["trigger_pred"].to_numpy(dtype=np.int64),
        "source": frame["source_pred"].to_numpy(dtype=np.int64),
        "ecotype": frame["ecotype_pred"].to_numpy(dtype=np.int64),
    }
    return labels, predictions


def subset_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    labels, predictions = arrays_from_frame(frame)
    result = metrics_for_predictions(labels, predictions)
    result["clips"] = float(len(frame))
    result["trigger_evaluated"] = float(np.sum(labels["trigger"] != IGNORE_INDEX))
    result["source_evaluated"] = float(np.sum(labels["source"] != IGNORE_INDEX))
    result["ecotype_evaluated"] = float(np.sum(labels["ecotype"] != IGNORE_INDEX))
    return result


def prefixed(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def group_metrics(
    frame: pd.DataFrame,
    columns: list[str],
    model_name: str,
    minimum_clips: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for keys, group in frame.groupby(grouper, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = subset_metrics(group)
        record: dict[str, Any] = {"model_name": model_name}
        record.update(dict(zip(columns, keys)))
        record.update(metrics)
        record["eligible_for_macro_summary"] = len(group) >= minimum_clips
        records.append(record)
    return pd.DataFrame(records)


def save_confusions(frame: pd.DataFrame, directory: Path, subset_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("trigger", TRIGGER_LABELS, TRIGGER_ID2LABEL),
        ("source", SOURCE_LABELS, SOURCE_ID2LABEL),
        ("ecotype", ECOTYPE_LABELS, ECOTYPE_ID2LABEL),
    )
    for head, label_map, id_to_label in definitions:
        true = frame[f"{head}_true"].to_numpy(dtype=np.int64)
        pred = frame[f"{head}_pred"].to_numpy(dtype=np.int64)
        mask = true != IGNORE_INDEX
        label_ids = list(range(len(label_map)))
        matrix = confusion_matrix(true[mask], pred[mask], labels=label_ids)
        names = [id_to_label[index] for index in label_ids]
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            directory / f"{subset_name}_{head}_confusion_matrix.csv"
        )


def mean_eligible(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty or metric not in frame:
        return float("nan")
    eligible = frame.loc[frame["eligible_for_macro_summary"].astype(bool), metric]
    return float(eligible.mean()) if len(eligible) else float("nan")


def main() -> int:
    args = parse_args()
    roots = [Path(value).expanduser().resolve() for value in args.data_root]
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / "model_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = (
        args.overlap_audit_csv.expanduser().resolve()
        if args.overlap_audit_csv is not None
        else None
    )
    overlap = load_overlap_audit(overlap_path)
    audit_usage: Counter[str] = Counter()
    rows, manifests, _ = discover_rows(
        roots,
        args.split,
        args.manifest_name,
        args.undbio_trigger_policy,
        args.domain_columns,
        overlap,
        args.overlap_conflict_policy,
        args.require_overlap_audit_coverage,
        audit_usage,
    )
    if args.maximum_clips is not None and args.maximum_clips < len(rows):
        indices = np.random.default_rng(args.seed).choice(
            len(rows), size=args.maximum_clips, replace=False
        )
        rows = [rows[index] for index in sorted(indices.tolist())]
    remote_count = sum(row_is_remote(row) for row in rows)
    local_count = len(rows) - remote_count
    if args.require_remote and remote_count == 0:
        raise RuntimeError(
            "No remote validation rows were discovered. Attach every validation_remote "
            "Kaggle shard, or use --no-require-remote only for a code test."
        )
    signature = dataset_hash(rows, args.split)
    provenance = data_provenance(rows, args.split)
    pd.DataFrame(provenance).to_csv(output_dir / "input_data_provenance.csv", index=False)
    inventory = pd.DataFrame(
        {
            "clip_id": [row["clip_id"] for row in rows],
            "provider": [row["domain_key"][0] for row in rows],
            "dataset": [
                row["domain_key"][1] if len(row["domain_key"]) > 1 else "<missing>"
                for row in rows
            ],
            "is_remote": [row_is_remote(row) for row in rows],
            "manifest_path": [row["manifest_path"] for row in rows],
        }
    )
    inventory.to_csv(output_dir / "evaluation_inventory.csv", index=False)
    print("\nFull multispecies validation comparison")
    print(f"Split:                  {args.split}")
    print(f"Models:                 {len(args.models):,}")
    print(f"Clips:                  {len(rows):,}")
    print(f"Local / remote clips:   {local_count:,} / {remote_count:,}")
    print(f"Providers / datasets:   {inventory['provider'].nunique():,} / {inventory[['provider', 'dataset']].drop_duplicates().shape[0]:,}")
    print(f"Manifests used:         {len(manifests):,}")
    print(f"Overlap audit actions:  {dict(sorted(audit_usage.items()))}")
    print("\nRows by local/remote and source label")
    source_names = inventory.copy()
    source_names["source"] = [SOURCE_ID2LABEL[row["source_original_label"]] for row in rows]
    print(pd.crosstab(source_names["is_remote"], source_names["source"]).to_string())

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ranking_rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(args.models, start=1):
        print(f"\n=== Model {model_index}/{len(args.models)}: {model_name} ===")
        preprocessing = checkpoint_preprocessing(model_name)
        identity = checkpoint_identity(model_name)
        cache_key = safe_slug(model_name)
        prediction_path = cache_dir / f"{cache_key}.csv"
        metadata_path = cache_dir / f"{cache_key}.json"
        expected_cache = {
            "model_name": model_name,
            "checkpoint_identity": identity,
            "dataset_hash": signature,
            "split": args.split,
            "preprocessing": preprocessing,
        }
        cached = False
        if args.reuse_model_caches and prediction_path.is_file() and metadata_path.is_file():
            observed = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = observed.get("cache_inputs") == expected_cache
        if cached:
            print(f"Reusing completed predictions: {prediction_path}")
            predictions = pd.read_csv(prediction_path)
            inference_seconds = float(
                json.loads(metadata_path.read_text(encoding="utf-8")).get(
                    "inference_seconds", float("nan")
                )
            )
        else:
            predictions, inference_seconds, loaded_identity = infer_model(
                model_name,
                rows,
                preprocessing,
                args.batch_size,
                args.preprocessing_workers,
                device,
                args.amp,
                args.multi_gpu,
            )
            if loaded_identity != identity:
                # Both identities describe the same checkpoint; retain the identity
                # produced by the strict model loader in the saved diagnostic data.
                expected_cache["loaded_checkpoint_identity"] = loaded_identity
            predictions.to_csv(prediction_path, index=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "cache_inputs": {
                            key: value
                            for key, value in expected_cache.items()
                            if key != "loaded_checkpoint_identity"
                        },
                        "loaded_checkpoint_identity": loaded_identity,
                        "inference_seconds": inference_seconds,
                        "clips": len(predictions),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            print(f"Saved completed predictions: {prediction_path}")

        overall = subset_metrics(predictions)
        local = subset_metrics(predictions.loc[~predictions["is_remote"].astype(bool)])
        remote = subset_metrics(predictions.loc[predictions["is_remote"].astype(bool)])
        provider = group_metrics(
            predictions, ["provider"], model_name, args.minimum_group_clips
        )
        domain = group_metrics(
            predictions,
            ["provider", "dataset"],
            model_name,
            args.minimum_group_clips,
        )
        model_dir = output_dir / "model_reports" / cache_key
        model_dir.mkdir(parents=True, exist_ok=True)
        provider.to_csv(model_dir / "provider_metrics.csv", index=False)
        domain.to_csv(model_dir / "provider_dataset_metrics.csv", index=False)
        save_confusions(predictions, model_dir, "overall")
        save_confusions(
            predictions.loc[~predictions["is_remote"].astype(bool)], model_dir, "local"
        )
        save_confusions(
            predictions.loc[predictions["is_remote"].astype(bool)], model_dir, "remote"
        )
        balanced = float(
            np.mean([local.get("combined_score", np.nan), remote.get("combined_score", np.nan)])
        )
        record: dict[str, Any] = {
            "model_name": model_name,
            "inference_seconds": inference_seconds,
            "clips_per_second": len(predictions) / max(inference_seconds, 1e-9),
            "balanced_local_remote_score": balanced,
            "provider_macro_combined_score": mean_eligible(provider, "combined_score"),
            "domain_macro_combined_score": mean_eligible(domain, "combined_score"),
        }
        record.update(prefixed(overall, "overall"))
        record.update(prefixed(local, "local"))
        record.update(prefixed(remote, "remote"))
        ranking_rows.append(record)
        (model_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "preprocessing": preprocessing,
                    "inference_seconds": inference_seconds,
                    "overall": overall,
                    "local": local,
                    "remote": remote,
                    "balanced_local_remote_score": balanced,
                    "provider_macro_combined_score": record[
                        "provider_macro_combined_score"
                    ],
                    "domain_macro_combined_score": record[
                        "domain_macro_combined_score"
                    ],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            f"Overall combined={overall.get('combined_score', float('nan')):.4f}; "
            f"local={local.get('combined_score', float('nan')):.4f}; "
            f"remote={remote.get('combined_score', float('nan')):.4f}; "
            f"balanced={balanced:.4f}"
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        [args.ranking_metric, "overall_combined_score"],
        ascending=False,
        na_position="last",
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_dir / "model_rankings.csv", index=False)
    display_columns = [
        "rank",
        "model_name",
        "balanced_local_remote_score",
        "overall_combined_score",
        "local_combined_score",
        "remote_combined_score",
        "overall_source_macro_f1",
        "remote_source_macro_f1",
        "overall_ecotype_macro_f1",
        "remote_ecotype_macro_f1",
        "provider_macro_combined_score",
        "inference_seconds",
    ]
    display_columns = [column for column in display_columns if column in ranking]
    print(f"\n# Ranked models by {args.ranking_metric}\n")
    print(ranking[display_columns].to_string(index=False))
    print(f"\nSaved reports to: {output_dir}")
    if args.split == "validation":
        print(
            "Validation is appropriate for model selection. Keep the test split "
            "untouched until the model and evaluation rule are locked."
        )
    else:
        print(
            "WARNING: this evaluated the test split. Do not use these results for "
            "additional model or hyperparameter selection."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
