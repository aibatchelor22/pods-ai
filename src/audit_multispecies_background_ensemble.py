#!/usr/bin/env python3
"""Conservative, independent-detector audit of multispecies background clips.

This script is intended for Kaggle.  It discovers the background clips in the
multispecies-cetacean shard manifests, materializes them as ordinary WAV files,
runs one or more independent whale detectors, and creates a manual-review
queue.  It never deletes clips or rewrites a training manifest.

The four supported detector backends are deliberately independent of the
multispecies model being audited:

* cook_inlet: Microsoft Cook Inlet Belugas binary whale detector
* google_whale: Google multi-species whale model, via BACPIPE
* orcahello: OrcaHello SRKW detector
* google_humpback: Google/NOAA humpback-song model from TF Hub

Every backend writes its own resumable ``scores_<name>.csv``.  It is therefore
safe (and often easier) to run the detector backends in separate notebook
sessions, then invoke ``--report-only`` to combine their votes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError:  # permit --help and --report-only in lightweight environments
    sf = None  # type: ignore[assignment]

try:
    from scipy.signal import resample_poly
except ImportError:  # permit --help and score-report merging without SciPy
    resample_poly = None  # type: ignore[assignment]


MANIFEST_NAME = "multispecies_cetacean_manifest.csv"
SUPPORTED_DETECTORS = ("cook_inlet", "google_whale", "orcahello", "google_humpback")
DEFAULT_THRESHOLDS = {
    "cook_inlet": 0.30,
    "google_whale": 0.30,
    "orcahello": 0.30,
    "google_humpback": 0.30,
}
SCORE_COLUMNS = {
    "cook_inlet": "cook_inlet_whale_score",
    "google_whale": "google_whale_any_score",
    "orcahello": "orcahello_srkw_score",
    "google_humpback": "google_humpback_score",
}


def require_audio_dependencies() -> None:
    if sf is None or resample_poly is None:
        raise RuntimeError(
            "Audio processing requires soundfile and scipy. Install them with "
            "`pip install soundfile scipy` before preparing or scoring clips."
        )


def clean(value: object) -> str:
    if value is None:
        return ""
    result = str(value).strip()
    return "" if result.casefold() in {"", "nan", "none", "null", "na"} else result


def number(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def is_remote_row(row: dict[str, Any], manifest: Path) -> bool:
    fields = (
        str(manifest), row.get("storage_key", ""), row.get("shard_id", ""),
        row.get("kaggle_dataset_id", ""), row.get("extraction_mode", ""),
    )
    return "remote" in " ".join(map(str, fields)).casefold()


def discover_background_rows(
    data_root: Path, splits: set[str], scope: str
) -> list[dict[str, Any]]:
    manifests = sorted(data_root.rglob(MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"No {MANIFEST_NAME} files found below {data_root}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        for raw in read_csv_rows(manifest):
            if clean(raw.get("split")).casefold() not in splits:
                continue
            if clean(raw.get("clip_kind")).casefold() != "background":
                continue
            clip_id = clean(raw.get("clip_id"))
            if not clip_id or clip_id in seen:
                continue
            audit_group = "remote" if is_remote_row(raw, manifest) else "local"
            if scope != "all" and audit_group != scope:
                continue
            seen.add(clip_id)
            row: dict[str, Any] = dict(raw)
            row.update(manifest_path=str(manifest), audit_group=audit_group)
            rows.append(row)
    if not rows:
        raise ValueError("No matching background rows were found")
    return rows


def stratified_sample(rows: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        return rows
    rng = random.Random(seed)
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            clean(row.get("split")), clean(row.get("Provider")),
            clean(row.get("Dataset")), clean(row.get("audit_group")),
        )
        strata[key].append(row)
    keys = sorted(strata)
    rng.shuffle(keys)
    for values in strata.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < maximum:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
    return selected


def safe_name(clip_id: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id)[:120]
    digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}.wav"


def source_reference(row: dict[str, Any]) -> tuple[Path | None, Path | None, str]:
    manifest = Path(str(row["manifest_path"]))
    archive_text = clean(row.get("archive_path"))
    member = clean(row.get("archive_member_path")).replace("\\", "/")
    archive = (manifest.parent / archive_text).resolve() if archive_text else None
    direct_text = clean(row.get("audio_file")) or clean(row.get("clip_path"))
    direct = (manifest.parent / direct_text).resolve() if direct_text else None
    return archive, direct, member


def read_source_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    require_audio_dependencies()
    archive, direct, member = source_reference(row)
    candidates: list[Path] = []
    if direct is not None:
        candidates.append(direct)
    if archive is not None and member:
        candidates.extend([archive / Path(member), archive.with_suffix("") / Path(member)])
    for candidate in candidates:
        if candidate.is_file():
            audio, sample_rate = sf.read(candidate, dtype="float32", always_2d=True)
            return audio[:, 0], int(sample_rate)
    if archive is not None and archive.is_file() and member:
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(member) as handle:
                audio, sample_rate = sf.read(handle, dtype="float32", always_2d=True)
                return audio[:, 0], int(sample_rate)
    raise FileNotFoundError(f"Could not resolve audio for {row.get('clip_id')}: {archive}::{member}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def materialize_inventory(rows: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    require_audio_dependencies()
    stage = output_dir / "staged_wavs"
    stage.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "background_inventory.csv"
    existing: dict[str, dict[str, Any]] = {}
    if inventory_path.is_file():
        for item in pd.read_csv(inventory_path).fillna("").to_dict("records"):
            existing[str(item["clip_id"])] = item
    output: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        clip_id = clean(row.get("clip_id"))
        staged = stage / safe_name(clip_id)
        old = existing.get(clip_id, {})
        if not staged.is_file():
            try:
                audio, sample_rate = read_source_audio(row)
                sf.write(staged, audio, sample_rate, subtype="PCM_16")
            except Exception as exc:  # retain all failures for inspection
                failures.append({"clip_id": clip_id, "error": repr(exc)})
                continue
        try:
            info = sf.info(staged)
        except Exception as exc:
            failures.append({"clip_id": clip_id, "error": repr(exc)})
            continue
        result = dict(row)
        result.update(
            staged_wav=str(staged.resolve()), sample_rate=int(info.samplerate),
            duration_seconds=float(info.duration), channels=int(info.channels),
        )
        for key, value in old.items():
            if key not in result:
                result[key] = value
        output.append(result)
        if index % 250 == 0 or index == len(rows):
            print(f"Staged {index:,}/{len(rows):,} background clips")
            atomic_csv(pd.DataFrame(output), inventory_path)
    frame = pd.DataFrame(output)
    atomic_csv(frame, inventory_path)
    atomic_csv(pd.DataFrame(failures, columns=["clip_id", "error"]), output_dir / "resolution_failures.csv")
    return frame


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    require_audio_dependencies()
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    ratio = Fraction(target_rate, source_rate).limit_denominator()
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, usecols=["clip_id"])
    return set(frame["clip_id"].astype(str))


def append_checkpoint(rows: list[dict[str, Any]], path: Path, every: int, force: bool = False) -> None:
    if force or (rows and len(rows) % every == 0):
        old = pd.read_csv(path).to_dict("records") if path.is_file() else []
        merged = pd.DataFrame(old + rows).drop_duplicates("clip_id", keep="last")
        atomic_csv(merged, path)
        rows.clear()


def run_orcahello(inventory: pd.DataFrame, args: argparse.Namespace) -> None:
    score_path = args.output_dir / "scores_orcahello.csv"
    done = completed_ids(score_path)
    repo = args.orcahello_repo.resolve()
    module_root = repo / "InferenceSystem" / "src"
    if not module_root.is_dir():
        raise FileNotFoundError(
            f"OrcaHello source not found at {module_root}. Clone "
            "https://github.com/orcasound/aifororcas-livesystem first."
        )
    sys.path.insert(0, str(module_root))
    from model import OrcaHelloSRKWDetectorV1  # type: ignore

    print("Loading OrcaHello SRKW detector")
    detector = OrcaHelloSRKWDetectorV1.from_pretrained(args.orcahello_model)
    pending: list[dict[str, Any]] = []
    todo = inventory[~inventory["clip_id"].astype(str).isin(done)]
    for count, row in enumerate(todo.itertuples(index=False), 1):
        result = detector.detect_srkw_from_file(str(row.staged_wav))
        local = np.asarray(getattr(result, "local_confidences", []), dtype=float)
        pending.append({
            "clip_id": str(row.clip_id),
            "orcahello_srkw_score": float(np.max(local)) if local.size else 0.0,
            "orcahello_global_score": float(getattr(result, "global_confidence", math.nan)),
        })
        append_checkpoint(pending, score_path, args.checkpoint_every)
        if count % 100 == 0:
            print(f"OrcaHello: {count:,}/{len(todo):,}")
    append_checkpoint(pending, score_path, args.checkpoint_every, force=True)


def run_google_humpback(inventory: pd.DataFrame, args: argparse.Namespace) -> None:
    score_path = args.output_dir / "scores_google_humpback.csv"
    done = completed_ids(score_path)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(args.model_cache_dir / "tfhub"))
    import tensorflow as tf  # type: ignore
    import tensorflow_hub as hub  # type: ignore

    print(f"Loading TF Hub humpback model: {args.google_humpback_url}")
    model = hub.load(args.google_humpback_url)
    score_fn = model.signatures["score"]
    pending: list[dict[str, Any]] = []
    todo = inventory[~inventory["clip_id"].astype(str).isin(done)]
    for count, row in enumerate(todo.itertuples(index=False), 1):
        audio, sample_rate = sf.read(row.staged_wav, dtype="float32", always_2d=True)
        mono = resample(audio[:, 0], int(sample_rate), 10_000)
        waveform = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(mono), 1), 0)
        values = score_fn(waveform=waveform, context_step_samples=tf.cast(10_000, tf.int64))
        scores = np.asarray(values["scores"].numpy()).reshape(-1)
        pending.append({
            "clip_id": str(row.clip_id),
            "google_humpback_score": float(np.max(scores)) if scores.size else 0.0,
            "google_humpback_mean_score": float(np.mean(scores)) if scores.size else 0.0,
        })
        append_checkpoint(pending, score_path, args.checkpoint_every)
        if count % 100 == 0:
            print(f"Google humpback: {count:,}/{len(todo):,}")
    append_checkpoint(pending, score_path, args.checkpoint_every, force=True)


def probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0:
        return values
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def run_google_whale(inventory: pd.DataFrame, args: argparse.Namespace) -> None:
    score_path = args.output_dir / "scores_google_whale.csv"
    done = completed_ids(score_path)
    if args.bacpipe_repo:
        sys.path.insert(0, str(args.bacpipe_repo.resolve()))
    import torch  # type: ignore
    from bacpipe.model_pipelines.feature_extractors.google_whale import Model  # type: ignore
    from bacpipe.core.workflows import ensure_models_exist  # type: ignore

    cache = args.model_cache_dir / "bacpipe"
    cache.mkdir(parents=True, exist_ok=True)
    ensure_models_exist(str(cache), model_names=["google_whale"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        model_name="google_whale", device=device, model_base_path=str(cache),
        global_batch_size=args.batch_size, run_pretrained_classifier=True,
    )
    model.prepare_inference()
    labels = [str(value) for value in model.classes]
    orca_indices = [i for i, label in enumerate(labels) if "orca" in label.casefold()]
    hump_indices = [i for i, label in enumerate(labels) if "humpback" in label.casefold()]
    if not orca_indices or not hump_indices:
        raise RuntimeError(f"Unexpected Google Whale class list: {labels}")
    todo = inventory[~inventory["clip_id"].astype(str).isin(done)]
    pending: list[dict[str, Any]] = []
    records = list(todo.itertuples(index=False))
    for offset in range(0, len(records), args.batch_size):
        current = records[offset : offset + args.batch_size]
        frames: list[np.ndarray] = []
        frame_owners: list[int] = []
        for row_index, row in enumerate(current):
            audio, sample_rate = sf.read(row.staged_wav, dtype="float32", always_2d=True)
            mono = resample(audio[:, 0], int(sample_rate), 24_000)
            target = 50_000
            if len(mono) >= target:
                starts = sorted({0, len(mono) - target})
                clips = [mono[start : start + target] for start in starts]
            else:
                clips = [np.pad(mono, (0, target - len(mono)))]
            for clip in clips:
                frames.append(clip)
                frame_owners.append(row_index)
        prepared = model.preprocess(torch.as_tensor(np.stack(frames), dtype=torch.float32))
        embeddings = model(prepared)
        values = probabilities(np.asarray(model.classifier_predictions(embeddings)))
        if values.ndim == 1:
            values = values[None, :]
        for row_index, row in enumerate(current):
            scores = values[np.asarray(frame_owners) == row_index].max(axis=0)
            pending.append({
                "clip_id": str(row.clip_id),
                "google_whale_any_score": float(np.max(scores)),
                "google_whale_orca_score": float(np.max(scores[orca_indices])),
                "google_whale_humpback_score": float(np.max(scores[hump_indices])),
                "google_whale_top_class": labels[int(np.argmax(scores))],
            })
        append_checkpoint(pending, score_path, args.checkpoint_every)
        print(f"Google Whale: {min(offset + len(current), len(records)):,}/{len(records):,}")
    append_checkpoint(pending, score_path, args.checkpoint_every, force=True)


def run_cook_inlet(inventory: pd.DataFrame, args: argparse.Namespace) -> None:
    score_path = args.output_dir / "scores_cook_inlet.csv"
    if len(completed_ids(score_path)) == len(inventory):
        print("Cook Inlet scores are already complete")
        return
    repo = args.cook_inlet_repo.resolve()
    script = repo / "inference.py"
    config = repo / "data" / "data_config.yaml"
    if not script.is_file():
        raise FileNotFoundError(
            f"Cook Inlet detector not found at {repo}. Clone "
            "https://github.com/microsoft/CookInlet_Belugas first."
        )
    if not args.cook_inlet_checkpoint.is_file():
        raise FileNotFoundError(
            f"Cook Inlet checkpoint not found: {args.cook_inlet_checkpoint}. "
            "Download the official binary best.ckpt checkpoint."
        )
    dataset = "multispecies_background_audit"
    command = [
        sys.executable, str(script), "--config", str(config),
        "--checkpoint", str(args.cook_inlet_checkpoint.resolve()),
        "--audios_source", str((args.output_dir / "staged_wavs").resolve()),
        "--dataset", dataset, "--temperature", str(args.cook_inlet_temperature),
        "--normalize", "--batch_size", str(args.batch_size),
        "--num_workers", str(args.workers),
    ]
    if args.device:
        command.extend(["--device", args.device])
    print("Running official Cook Inlet inference")
    subprocess.run(command, cwd=repo, check=True)
    result_path = repo / "inference" / dataset / "binary_inference_results.csv"
    if not result_path.is_file():
        matches = list((repo / "inference" / dataset).rglob("*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Could not identify Cook Inlet output below {result_path.parent}")
        result_path = matches[0]
    raw = pd.read_csv(result_path)
    audio_col = next((c for c in raw if c.casefold() in {"audio", "file", "filename", "path"}), None)
    score_col = next((c for c in raw if c.casefold() in {"probability", "score", "confidence"}), None)
    if audio_col is None or score_col is None:
        raise ValueError(f"Unexpected Cook Inlet output columns: {list(raw.columns)}")
    file_to_clip = {
        Path(str(row.staged_wav)).name: str(row.clip_id)
        for row in inventory.itertuples(index=False)
    }
    raw["_name"] = raw[audio_col].astype(str).map(lambda value: Path(value).name)
    raw["clip_id"] = raw["_name"].map(file_to_clip)
    raw = raw.dropna(subset=["clip_id"])
    result = (
        raw.groupby("clip_id", as_index=False)[score_col].max()
        .rename(columns={score_col: "cook_inlet_whale_score"})
    )
    atomic_csv(result, score_path)


def parse_thresholds(values: Sequence[str]) -> dict[str, float]:
    result = dict(DEFAULT_THRESHOLDS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Threshold must have detector=value form: {value}")
        name, raw = value.split("=", 1)
        name = name.strip()
        if name not in SUPPORTED_DETECTORS:
            raise ValueError(f"Unknown detector in threshold: {name}")
        result[name] = float(raw)
    return result


def annotate_geometry(inventory: pd.DataFrame, annotation_path: Path | None, collar: float) -> pd.DataFrame:
    result = inventory.copy()
    result["annotation_exact_overlap"] = False
    result["annotation_within_collar"] = False
    result["nearest_annotation_seconds"] = math.nan
    if annotation_path is None:
        return result
    annotations = pd.read_csv(annotation_path, low_memory=False)
    required = {"source_recording_id", "FileBeginSec", "FileEndSec"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Annotation table lacks columns: {sorted(required - set(annotations.columns))}")
    groups: dict[str, np.ndarray] = {}
    for recording, group in annotations.groupby("source_recording_id", dropna=False):
        starts = pd.to_numeric(group["FileBeginSec"], errors="coerce").to_numpy(float)
        ends = pd.to_numeric(group["FileEndSec"], errors="coerce").to_numpy(float)
        valid = np.isfinite(starts) & np.isfinite(ends)
        groups[str(recording)] = np.stack([starts[valid], ends[valid]], axis=1)
    exact: list[bool] = []
    nearby: list[bool] = []
    distances: list[float] = []
    for row in result.itertuples(index=False):
        intervals = groups.get(str(getattr(row, "source_recording_id", "")), np.empty((0, 2)))
        start = number(getattr(row, "actual_clip_start_sec", math.nan))
        duration = number(getattr(row, "duration_seconds", 3.0), 3.0)
        end = start + duration
        if not len(intervals) or not math.isfinite(start):
            exact.append(False); nearby.append(False); distances.append(math.nan)
            continue
        overlap = np.any((intervals[:, 0] < end) & (intervals[:, 1] > start))
        distance = np.minimum(np.abs(intervals[:, 0] - end), np.abs(start - intervals[:, 1]))
        distance[(intervals[:, 0] < end) & (intervals[:, 1] > start)] = 0.0
        nearest = float(np.min(distance))
        exact.append(bool(overlap)); nearby.append(nearest <= collar); distances.append(nearest)
    result["annotation_exact_overlap"] = exact
    result["annotation_within_collar"] = nearby
    result["nearest_annotation_seconds"] = distances
    return result


def risk_report(args: argparse.Namespace, thresholds: dict[str, float]) -> pd.DataFrame:
    inventory_path = args.output_dir / "background_inventory.csv"
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")
    combined = annotate_geometry(pd.read_csv(inventory_path, low_memory=False), args.master_annotations_csv, args.annotation_collar_seconds)
    available: list[str] = []
    for detector in SUPPORTED_DETECTORS:
        score_path = args.output_dir / f"scores_{detector}.csv"
        if not score_path.is_file():
            continue
        scores = pd.read_csv(score_path)
        combined = combined.merge(scores, on="clip_id", how="left", validate="one_to_one")
        score_column = SCORE_COLUMNS[detector]
        if score_column in combined:
            values = pd.to_numeric(combined[score_column], errors="coerce")
            combined[f"{detector}_scored"] = values.notna()
            combined[f"{detector}_flag"] = values >= thresholds[detector]
            available.append(detector)
    if not available:
        raise RuntimeError("No detector score files are available; run at least one detector first")
    flag_columns = [f"{name}_flag" for name in available]
    combined["detector_flag_count"] = combined[flag_columns].fillna(False).sum(axis=1).astype(int)
    scored_columns = [f"{name}_scored" for name in available]
    combined["detector_score_count"] = combined[scored_columns].fillna(False).sum(axis=1).astype(int)
    combined["detectors_flagged"] = combined.apply(
        lambda row: ";".join(name for name in available if bool(row[f"{name}_flag"])), axis=1
    )
    combined["detectors_missing"] = combined.apply(
        lambda row: ";".join(name for name in available if not bool(row[f"{name}_scored"])), axis=1
    )
    combined["risk_tier"] = np.select(
        [
            combined["annotation_exact_overlap"].astype(bool) | (combined["detector_flag_count"] >= 2),
            combined["detector_flag_count"] == 1,
            combined["detector_score_count"] < len(available),
            combined["annotation_within_collar"].astype(bool),
        ],
        ["red", "orange", "incomplete", "yellow"],
        default="provisional_clean",
    )
    combined["risk_reason"] = combined.apply(
        lambda row: (
            "annotation_overlap" if row["annotation_exact_overlap"] else
            (f"{int(row['detector_flag_count'])}_detector_consensus" if row["detector_flag_count"] >= 2 else
             (f"single_detector:{row['detectors_flagged']}" if row["detector_flag_count"] == 1 else
              (f"missing_scores:{row['detectors_missing']}" if row["detector_score_count"] < len(available) else
               ("annotation_collar" if row["annotation_within_collar"] else "no_flag"))))
        ), axis=1,
    )
    combined["manual_label"] = ""
    combined["review_notes"] = ""
    atomic_csv(combined, args.output_dir / "ensemble_background_audit.csv")
    return combined


def summary_tables(frame: pd.DataFrame, output_dir: Path) -> None:
    tier_order = ["red", "orange", "incomplete", "yellow", "provisional_clean"]
    atomic_csv(
        frame["risk_tier"].value_counts().reindex(tier_order, fill_value=0).rename_axis("risk_tier").reset_index(name="clips"),
        output_dir / "risk_tier_summary.csv",
    )
    for column, name in (("Provider", "provider_audit_summary.csv"), ("Dataset", "dataset_audit_summary.csv")):
        table = pd.crosstab(frame[column].fillna(""), frame["risk_tier"]).reindex(columns=tier_order, fill_value=0)
        table["total"] = table.sum(axis=1)
        table["red_or_orange_percent"] = 100.0 * (table["red"] + table["orange"]) / table["total"].clip(lower=1)
        atomic_csv(table.reset_index(), output_dir / name)


def balanced_review_sample(frame: pd.DataFrame, tier: str, limit: int, seed: int) -> pd.DataFrame:
    subset = frame[frame["risk_tier"] == tier].copy()
    if limit <= 0 or len(subset) <= limit:
        return subset
    rows = subset.to_dict("records")
    return pd.DataFrame(stratified_sample(rows, limit, seed))


def make_review_bundle(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    selected = pd.concat([
        balanced_review_sample(frame, "red", args.review_red, args.seed),
        balanced_review_sample(frame, "orange", args.review_orange, args.seed + 1),
        balanced_review_sample(frame, "incomplete", args.review_incomplete, args.seed + 2),
        balanced_review_sample(frame, "yellow", args.review_yellow, args.seed + 3),
        balanced_review_sample(frame, "provisional_clean", args.review_clean, args.seed + 4),
    ], ignore_index=True).drop_duplicates("clip_id")
    review_dir = args.output_dir / "manual_review_audio"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_paths: list[str] = []
    for row in selected.itertuples(index=False):
        destination = review_dir / str(row.risk_tier) / Path(str(row.staged_wav)).name
        source = Path(str(row.staged_wav))
        if not source.is_file():
            review_paths.append("")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copy2(source, destination)
        review_paths.append(str(destination.relative_to(args.output_dir)))
    selected["review_audio_path"] = review_paths
    selected["manual_label"] = ""
    selected["review_notes"] = ""
    atomic_csv(selected, args.output_dir / "manual_review_queue.csv")
    archive = args.output_dir / "manual_review_audio.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in review_dir.rglob("*.wav"):
            bundle.write(path, path.relative_to(review_dir))


def write_report(frame: pd.DataFrame, args: argparse.Namespace, thresholds: dict[str, float]) -> None:
    counts = frame["risk_tier"].value_counts()
    lines = [
        "# Multispecies background ensemble audit", "",
        f"Background clips audited: {len(frame):,}",
        f"Providers: {frame['Provider'].nunique():,}",
        f"Datasets: {frame['Dataset'].nunique():,}", "",
        "Risk tiers:",
    ]
    for tier in ("red", "orange", "incomplete", "yellow", "provisional_clean"):
        lines.append(f"  {tier:18s} {int(counts.get(tier, 0)):,}")
    lines.extend(["", "Detector thresholds (provisional until calibrated):"])
    for detector in SUPPORTED_DETECTORS:
        if f"{detector}_flag" in frame:
            lines.append(f"  {detector:18s} {thresholds[detector]:.4f}")
    lines.extend([
        "", "Interpretation:",
        "  red: exact annotation overlap or at least two independent detector flags",
        "  orange: exactly one independent detector flag",
        "  incomplete: one or more available detector score files lack this clip",
        "  yellow: within the annotation collar, with no detector flag",
        "  provisional_clean: none of the checks flagged the clip",
        "", "No clips were deleted and no training manifest was rewritten.",
        "Manual review is required before excluding or relabeling clips.", "",
    ])
    (args.output_dir / "ensemble_background_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/multispecies_background_ensemble_audit"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--scope", choices=["all", "local", "remote"], default="all")
    parser.add_argument("--max-clips", type=int, default=0, help="0 audits every discovered background clip")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--detectors", nargs="+", choices=SUPPORTED_DETECTORS, default=list(SUPPORTED_DETECTORS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-inventory", action="store_true",
        help="Score the existing background_inventory.csv without rediscovery/restaging",
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-missing-detectors", action="store_true")
    parser.add_argument("--threshold", action="append", default=[], metavar="DETECTOR=VALUE")
    parser.add_argument("--master-annotations-csv", type=Path)
    parser.add_argument("--annotation-collar-seconds", type=float, default=60.0)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("/kaggle/working/background_audit_models"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--cook-inlet-repo", type=Path, default=Path("/kaggle/working/CookInlet_Belugas"))
    parser.add_argument("--cook-inlet-checkpoint", type=Path, default=Path("/kaggle/input/cook-inlet-belugas-checkpoints/binary/best.ckpt"))
    parser.add_argument("--cook-inlet-temperature", type=float, default=3.0)
    parser.add_argument("--bacpipe-repo", type=Path, default=Path("/kaggle/working/bacpipe"))
    parser.add_argument("--orcahello-repo", type=Path, default=Path("/kaggle/working/aifororcas-livesystem"))
    parser.add_argument("--orcahello-model", default="orcasound/orcahello-srkw-detector-v1")
    parser.add_argument("--google-humpback-url", default="https://tfhub.dev/google/humpback_whale/1")
    parser.add_argument("--review-red", type=int, default=0, help="0 includes every red clip")
    parser.add_argument("--review-orange", type=int, default=500)
    parser.add_argument("--review-incomplete", type=int, default=100)
    parser.add_argument("--review-yellow", type=int, default=100)
    parser.add_argument("--review-clean", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.threshold)
    inventory_path = args.output_dir / "background_inventory.csv"
    if args.report_only or args.reuse_inventory:
        if not inventory_path.is_file():
            raise FileNotFoundError(f"Prepared inventory not found: {inventory_path}")
        inventory = pd.read_csv(inventory_path, low_memory=False)
    else:
        splits = {value.casefold() for value in args.splits}
        rows = discover_background_rows(args.data_root.resolve(), splits, args.scope)
        rows = stratified_sample(rows, args.max_clips, args.seed)
        print(f"Discovered {len(rows):,} background clips")
        inventory = materialize_inventory(rows, args.output_dir)
        print(f"Prepared {len(inventory):,} readable clips in {args.output_dir / 'staged_wavs'}")
    settings = vars(args).copy()
    settings = {key: (str(value) if isinstance(value, Path) else value) for key, value in settings.items()}
    settings["thresholds"] = thresholds
    (args.output_dir / "audit_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if args.prepare_only:
        return 0
    if not args.report_only:
        args.model_cache_dir.mkdir(parents=True, exist_ok=True)
        runners = {
            "cook_inlet": run_cook_inlet,
            "google_whale": run_google_whale,
            "orcahello": run_orcahello,
            "google_humpback": run_google_humpback,
        }
        failures: list[dict[str, str]] = []
        for detector in args.detectors:
            started = time.time()
            try:
                runners[detector](inventory, args)
                print(f"Completed {detector} in {(time.time() - started) / 60:.1f} minutes")
            except Exception as exc:
                failures.append({"detector": detector, "error": repr(exc)})
                print(f"ERROR: {detector} failed: {exc}", file=sys.stderr)
                if not args.allow_missing_detectors:
                    atomic_csv(pd.DataFrame(failures), args.output_dir / "detector_failures.csv")
                    raise
        atomic_csv(pd.DataFrame(failures, columns=["detector", "error"]), args.output_dir / "detector_failures.csv")
    audit = risk_report(args, thresholds)
    summary_tables(audit, args.output_dir)
    make_review_bundle(audit, args)
    write_report(audit, args, thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
