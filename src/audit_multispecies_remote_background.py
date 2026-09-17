#!/usr/bin/env python3
"""Audit remote training-background clips for possible unlabeled whale calls.

The audit compares balanced samples of remote and local training-background
clips using a checkpoint trained before remote data were introduced. It writes
per-clip probabilities, a remote-vs-local score plot, and a ZIP containing the
highest-scoring remote clips for manual review.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import spectrogram
from transformers import AutoFeatureExtractor

from train_multispecies_cetacean_model import (
    ArchiveAudioCollator,
    ECOTYPE_LABELS,
    SAMPLE_RATE,
    SOURCE_LABELS,
    TRIGGER_LABELS,
    checkpoint_metadata,
    load_model,
)


DEFAULT_MODEL = "aibatchelor22/multi_species_v2_epoch_7"
MANIFEST_NAME = "multispecies_cetacean_manifest.csv"


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def read_manifest(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def is_remote_row(row: dict[str, str], manifest: Path) -> bool:
    values = [
        str(manifest),
        row.get("storage_key", ""),
        row.get("shard_id", ""),
        row.get("kaggle_dataset_id", ""),
        row.get("extraction_mode", ""),
    ]
    return "remote" in " ".join(values).casefold()


def discover_background_rows(root: Path, split: str) -> list[dict[str, Any]]:
    manifests = sorted(root.rglob(MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"No {MANIFEST_NAME} files found below {root}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        for raw in read_manifest(manifest):
            if clean(raw.get("split")).casefold() != split.casefold():
                continue
            if clean(raw.get("clip_kind")).casefold() != "background":
                continue
            clip_id = clean(raw.get("clip_id"))
            if not clip_id or clip_id in seen:
                continue
            seen.add(clip_id)
            row: dict[str, Any] = dict(raw)
            row["manifest_path"] = str(manifest)
            row["audit_group"] = "remote" if is_remote_row(raw, manifest) else "local"
            rows.append(row)
    if not rows:
        raise ValueError(f"No {split!r} background rows found below {root}")
    return rows


def balanced_sample(rows: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("Sample counts must be positive")
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(clean(row.get("Provider")), clean(row.get("Dataset")))].append(row)
    keys = list(strata)
    rng.shuffle(keys)
    for values in strata.values():
        rng.shuffle(values)
    result: list[dict[str, Any]] = []
    while len(result) < min(maximum, len(rows)):
        added = False
        for key in keys:
            if strata[key]:
                result.append(strata[key].pop())
                added = True
                if len(result) >= maximum:
                    break
        if not added:
            break
    return result


def resolve_audio(row: dict[str, Any]) -> dict[str, Any]:
    manifest = Path(row["manifest_path"])
    archive = (manifest.parent / clean(row.get("archive_path"))).resolve()
    member = clean(row.get("archive_member_path")).replace("\\", "/")
    if not member:
        raise ValueError(f"Missing archive_member_path for {row.get('clip_id')}")
    resolved = dict(row)
    resolved.update(
        {
            "archive_file": "",
            "audio_file": "",
            "archive_member_path": member,
            "trigger_label": 0,
            "source_label": 0,
            "source_original_label": 0,
            "ecotype_label": -100,
            "event_group_size": 1,
        }
    )
    if archive.is_file():
        resolved["archive_file"] = str(archive)
        return resolved
    for directory in (archive, archive.with_suffix("")):
        candidate = directory.joinpath(*Path(member).parts)
        if candidate.is_file():
            resolved["audio_file"] = str(candidate.resolve())
            return resolved
    raise FileNotFoundError(f"Could not resolve {archive}::{member}")


def preprocessing_for_model(model_name: str) -> dict[str, Any]:
    checkpoint = checkpoint_metadata(model_name)
    metadata = checkpoint[0] if checkpoint is not None else {}
    values = metadata.get("preprocessing", metadata.get("augmentation", {}))
    return {
        "mean_subtract": bool(values.get("mean_subtract", False)),
        "high_pass_filter": bool(values.get("high_pass_filter", False)),
        "high_pass_cutoff_hz": float(values.get("high_pass_cutoff_hz", 50.0)),
        "high_pass_order": int(values.get("high_pass_order", 4)),
    }


def score_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    device_name: str | None,
) -> pd.DataFrame:
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Loading audit checkpoint on {device}: {model_name}")
    model, identity, feature_source = load_model(model_name, dropout=0.0, freeze_backbone=True)
    model.to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(feature_source)
    preprocessing = preprocessing_for_model(model_name)
    print(f"Checkpoint identity: {identity}")
    print(f"Deterministic preprocessing: {preprocessing}")
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
    )

    output_rows: list[dict[str, Any]] = []
    for offset in range(0, len(rows), batch_size):
        current = rows[offset : offset + batch_size]
        batch = collator(current)
        with torch.inference_mode():
            trigger_logits, source_logits, ecotype_logits = model(
                input_values=batch["input_values"].to(device)
            )
            trigger = torch.softmax(trigger_logits, dim=-1).cpu().numpy()
            source = torch.softmax(source_logits, dim=-1).cpu().numpy()
            ecotype = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()
        for index, row in enumerate(current):
            result = {
                "clip_id": clean(row.get("clip_id")),
                "audit_group": row["audit_group"],
                "Provider": clean(row.get("Provider")),
                "Dataset": clean(row.get("Dataset")),
                "Soundfile": clean(row.get("Soundfile")),
                "source_recording_id": clean(row.get("source_recording_id")),
                "actual_clip_start_sec": clean(row.get("actual_clip_start_sec")),
                "manifest_path": row["manifest_path"],
                "archive_file": row["archive_file"],
                "audio_file": row["audio_file"],
                "archive_member_path": row["archive_member_path"],
            }
            for label, class_index in TRIGGER_LABELS.items():
                result[f"trigger_{label}"] = float(trigger[index, class_index])
            for label, class_index in SOURCE_LABELS.items():
                result[f"source_{label}"] = float(source[index, class_index])
            for label, class_index in ECOTYPE_LABELS.items():
                result[f"ecotype_{label}"] = float(ecotype[index, class_index])
            known_source = max(result["source_KW"], result["source_HW"])
            result["known_whale_source_score"] = known_source
            result["joint_known_whale_score"] = result["trigger_known_whale"] * known_source
            result["predicted_source"] = max(
                SOURCE_LABELS, key=lambda label: result[f"source_{label}"]
            )
            output_rows.append(result)
        print(f"Scored {min(offset + len(current), len(rows)):,}/{len(rows):,} clips")
    return pd.DataFrame(output_rows)


def copy_review_clip(row: pd.Series, destination: Path) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(row["clip_id"]))
    target = destination / f"{float(row['joint_known_whale_score']):.4f}_{safe_id}.wav"
    if clean(row.get("audio_file")):
        shutil.copy2(Path(str(row["audio_file"])), target)
    else:
        with zipfile.ZipFile(str(row["archive_file"])) as archive:
            with archive.open(str(row["archive_member_path"])) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return target


def plot_score_distributions(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for group, color in (("local", "#4C78A8"), ("remote", "#E45756")):
        values = frame.loc[frame["audit_group"] == group]
        axes[0].hist(
            values["trigger_known_whale"], bins=np.linspace(0, 1, 31),
            alpha=0.55, density=True, label=group, color=color,
        )
        axes[1].hist(
            values["joint_known_whale_score"], bins=np.linspace(0, 1, 31),
            alpha=0.55, density=True, label=group, color=color,
        )
    axes[0].set_title("Any-whale trigger")
    axes[1].set_title("Trigger × max(KW, HW)")
    for axis in axes:
        axis.set_xlabel("Score")
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_review_spectrograms(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    paths = paths[:24]
    columns = 4
    rows = math.ceil(len(paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.2 * rows), squeeze=False)
    for axis, path in zip(axes.flat, paths):
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio[:, 0]
        frequencies, times, power = spectrogram(
            audio, fs=rate, nperseg=min(1024, len(audio)), noverlap=min(768, max(0, len(audio) - 1))
        )
        mask = frequencies <= min(8000, rate / 2)
        axis.pcolormesh(times, frequencies[mask], 10 * np.log10(power[mask] + 1e-12), shading="auto")
        axis.set_title(path.stem[:58], fontsize=8)
        axis.set_xlabel("Seconds")
        axis.set_ylabel("Hz")
    for axis in axes.flat[len(paths) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="/kaggle/working/remote_background_audit")
    parser.add_argument("--split", default="train")
    parser.add_argument("--remote-samples", type=int, default=500)
    parser.add_argument("--local-control-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--trigger-threshold", type=float, default=0.80)
    parser.add_argument("--source-threshold", type=float, default=0.50)
    parser.add_argument("--review-clips", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = discover_background_rows(Path(args.data_root), args.split)
    remote = [row for row in all_rows if row["audit_group"] == "remote"]
    local = [row for row in all_rows if row["audit_group"] == "local"]
    if not remote:
        raise ValueError("No remote background rows were found in the attached datasets")
    if not local:
        raise ValueError("No local background rows were found for the control sample")
    print(f"Available backgrounds: remote={len(remote):,}, local={len(local):,}")
    selected = balanced_sample(remote, args.remote_samples, args.seed)
    selected += balanced_sample(local, args.local_control_samples, args.seed + 1)
    print("Selected by group/provider:")
    print(dict(sorted(Counter((row["audit_group"], row.get("Provider", "")) for row in selected).items())))

    resolved: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in selected:
        try:
            resolved.append(resolve_audio(row))
        except Exception as error:
            failures.append({"clip_id": clean(row.get("clip_id")), "error": repr(error)})
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "resolution_failures.csv", index=False)
        print(f"WARNING: {len(failures)} sampled clips could not be resolved")
    if not resolved:
        raise RuntimeError("None of the sampled clips could be resolved")

    scores = score_rows(resolved, args.model_name, args.batch_size, args.device)
    scores["suspicious"] = (
        (scores["trigger_known_whale"] >= args.trigger_threshold)
        & (scores["known_whale_source_score"] >= args.source_threshold)
    )
    scores = scores.sort_values("joint_known_whale_score", ascending=False)
    scores.to_csv(output_dir / "background_audit_scores.csv", index=False)
    summary_rows = []
    for group, values in scores.groupby("audit_group"):
        summary_rows.append(
            {
                "audit_group": group,
                "clips": len(values),
                "suspicious_clips": int(values["suspicious"].sum()),
                "suspicious_percent": 100.0 * float(values["suspicious"].mean()),
                "median_trigger": float(values["trigger_known_whale"].median()),
                "p95_trigger": float(values["trigger_known_whale"].quantile(0.95)),
                "median_joint_score": float(values["joint_known_whale_score"].median()),
                "p95_joint_score": float(values["joint_known_whale_score"].quantile(0.95)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("audit_group")
    summary.to_csv(output_dir / "background_audit_summary.csv", index=False)
    print("\nRemote versus local background audit")
    print(summary.to_string(index=False))
    plot_score_distributions(scores, output_dir / "remote_vs_local_score_distributions.png")

    review = scores.loc[scores["audit_group"] == "remote"].head(args.review_clips).copy()
    review.insert(0, "manual_label", "")
    review.insert(1, "review_notes", "")
    review.to_csv(output_dir / "remote_background_manual_review.csv", index=False)
    review_dir = output_dir / "suspicious_remote_wavs"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_paths = [copy_review_clip(row, review_dir) for _, row in review.iterrows()]
    plot_review_spectrograms(review_paths, output_dir / "suspicious_remote_spectrograms.png")
    archive = shutil.make_archive(
        str(output_dir / "suspicious_remote_background_clips"), "zip", review_dir
    )
    print(f"\nScores:       {output_dir / 'background_audit_scores.csv'}")
    print(f"Review table: {output_dir / 'remote_background_manual_review.csv'}")
    print(f"Review WAVs:  {archive}")
    print(f"Spectrograms: {output_dir / 'suspicious_remote_spectrograms.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
