#!/usr/bin/env python3
"""Compare remote and local Multispecies Cetacean training-clip acoustics.

The audit samples extracted 3-second clips equally within remote/local source
labels, measures level and spectral characteristics, and repeats the analysis
after the deterministic training preprocessing (mean subtraction and optional
high-pass filtering). It is CPU-only and does not run model inference.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from scipy.stats import ks_2samp


MANIFEST_NAME = "multispecies_cetacean_manifest.csv"
REMOTE_MARKER = re.compile(r"(^|[\\/_-])remote([\\/_-]|$)", re.IGNORECASE)
EPSILON = 1e-12


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def read_manifest(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def is_remote(row: dict[str, str], manifest: Path) -> bool:
    if clean(row.get("extraction_mode")).casefold() == "remote_seek":
        return True
    values = (
        str(manifest),
        clean(row.get("storage_key")),
        clean(row.get("shard_id")),
        clean(row.get("kaggle_dataset_id")),
    )
    return bool(REMOTE_MARKER.search("/".join(values)))


def discover_rows(
    root: Path, split: str, clip_kind: str, include_synthetic: bool
) -> list[dict[str, Any]]:
    manifests = sorted(root.rglob(MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"No {MANIFEST_NAME} files found below {root}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        for raw in read_manifest(manifest):
            if clean(raw.get("split")).casefold() != split.casefold():
                continue
            kind = clean(raw.get("clip_kind")).casefold()
            if not include_synthetic and kind not in {"annotated", "background"}:
                continue
            if clip_kind != "all" and kind != clip_kind:
                continue
            clip_id = clean(raw.get("clip_id"))
            if not clip_id or clip_id in seen:
                continue
            seen.add(clip_id)
            row: dict[str, Any] = dict(raw)
            row["manifest_path"] = str(manifest)
            row["audit_group"] = "remote" if is_remote(raw, manifest) else "local"
            row["source_class"] = clean(raw.get("model_source_label")) or "<missing>"
            rows.append(row)
    if not rows:
        raise ValueError(f"No eligible {split!r} rows found below {root}")
    return rows


def stratified_sample(
    rows: list[dict[str, Any]], samples_per_group_label: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[
            (
                row["audit_group"],
                row["source_class"],
                clean(row.get("clip_kind")).casefold() or "<missing>",
            )
        ].append(row)
    selected: list[dict[str, Any]] = []
    label_kinds = sorted({(key[1], key[2]) for key in strata})
    for label, kind in label_kinds:
        local = strata.get(("local", label, kind), [])
        remote = strata.get(("remote", label, kind), [])
        if not local or not remote:
            print(
                f"Skipped unmatched stratum {label}/{kind}: "
                f"local={len(local):,}, remote={len(remote):,}"
            )
            continue
        rng.shuffle(local)
        rng.shuffle(remote)
        count = min(samples_per_group_label, len(local), len(remote))
        selected.extend(local[:count])
        selected.extend(remote[:count])
        print(
            f"Selected {count:,} local + {count:,} remote clips for "
            f"{label}/{kind} (available local={len(local):,}, remote={len(remote):,})"
        )
    if not selected:
        raise ValueError("No source-class/clip-kind strata occur in both local and remote data")
    rng.shuffle(selected)
    return selected


def resolve_audio(row: dict[str, Any]) -> tuple[Path, str]:
    manifest = Path(row["manifest_path"])
    archive = (manifest.parent / clean(row.get("archive_path"))).resolve()
    member = clean(row.get("archive_member_path")).replace("\\", "/")
    if not member:
        raise ValueError(f"Missing archive_member_path for {row.get('clip_id')}")
    if archive.is_file():
        return archive, member
    for directory in (archive, archive.with_suffix("")):
        candidate = directory.joinpath(*Path(member).parts)
        if candidate.is_file():
            return candidate.resolve(), ""
    raise FileNotFoundError(f"Could not resolve {archive}::{member}")


def read_audio(row: dict[str, Any]) -> tuple[np.ndarray, int, int]:
    source, member = resolve_audio(row)
    if member:
        with zipfile.ZipFile(source) as archive:
            payload = archive.read(member)
        values, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    else:
        values, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    channels = int(values.shape[1])
    # Match the dataset/training convention: use the first channel.
    return np.asarray(values[:, 0], dtype=np.float64), int(sample_rate), channels


def dbfs(value: float, floor: float = -160.0) -> float:
    if not math.isfinite(value) or value <= 0:
        return floor
    return max(20.0 * math.log10(value), floor)


def high_pass(samples: np.ndarray, sample_rate: int, cutoff_hz: float, order: int) -> np.ndarray:
    if cutoff_hz <= 0:
        return samples
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"High-pass cutoff {cutoff_hz} is invalid for {sample_rate} Hz audio")
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, samples).astype(np.float64, copy=False)


def spectral_quantile(freqs: np.ndarray, power: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(power)
    if not len(cumulative) or cumulative[-1] <= 0:
        return 0.0
    index = int(np.searchsorted(cumulative, cumulative[-1] * quantile, side="left"))
    return float(freqs[min(index, len(freqs) - 1)])


def measure(samples: np.ndarray, sample_rate: int, prefix: str) -> dict[str, float]:
    samples = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    frame_length = max(1, round(0.1 * sample_rate))
    usable = len(samples) - len(samples) % frame_length
    if usable:
        frames = samples[:usable].reshape(-1, frame_length)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        active_rms = float(np.percentile(frame_rms, 90))
        activity_range = float(np.percentile(frame_rms, 90) - np.percentile(frame_rms, 10))
        silent_percent = float(np.mean(frame_rms <= 10 ** (-75.0 / 20.0)) * 100.0)
    else:
        active_rms, activity_range = rms, 0.0
        silent_percent = float(rms <= 10 ** (-75.0 / 20.0)) * 100.0

    window = np.hanning(len(samples)) if len(samples) > 1 else np.ones(len(samples))
    spectrum = np.fft.rfft(samples * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    total_power = float(power.sum()) + EPSILON
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    positive_power = power[1:] if len(power) > 1 else power
    flatness = float(
        np.exp(np.mean(np.log(positive_power + EPSILON)))
        / (np.mean(positive_power + EPSILON))
    )

    result = {
        f"{prefix}rms_dbfs": dbfs(rms),
        f"{prefix}active_rms_dbfs": dbfs(active_rms),
        f"{prefix}peak_dbfs": dbfs(peak),
        f"{prefix}crest_factor_db": dbfs(peak / max(rms, EPSILON), floor=-160.0),
        f"{prefix}dc_offset": float(np.mean(samples)) if len(samples) else 0.0,
        f"{prefix}clipped_percent": float(np.mean(np.abs(samples) >= 0.999) * 100.0),
        f"{prefix}near_silent_frame_percent": silent_percent,
        f"{prefix}activity_range_linear": activity_range,
        f"{prefix}spectral_centroid_hz": centroid,
        f"{prefix}spectral_bandwidth_hz": bandwidth,
        f"{prefix}spectral_rolloff_85_hz": spectral_quantile(freqs, power, 0.85),
        f"{prefix}spectral_rolloff_95_hz": spectral_quantile(freqs, power, 0.95),
        f"{prefix}spectral_flatness": flatness,
    }
    bands = ((0, 100), (100, 500), (500, 2000), (2000, 4000), (4000, 8000))
    for low, high in bands:
        mask = (freqs >= low) & (freqs < min(high, sample_rate / 2.0 + 1e-9))
        fraction = float(power[mask].sum() / total_power) if np.any(mask) else 0.0
        result[f"{prefix}energy_{low}_{high}_fraction"] = fraction
    return result


def robust_effect(remote: np.ndarray, local: np.ndarray) -> float:
    combined = np.concatenate([remote, local])
    median = float(np.median(combined))
    mad = float(np.median(np.abs(combined - median))) * 1.4826
    return (float(np.median(remote)) - float(np.median(local))) / max(mad, EPSILON)


def summarize(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["audit_group", "source_class", "Provider", "Dataset"]
    for keys, group in frame.groupby(group_columns, dropna=False):
        record: dict[str, Any] = dict(zip(group_columns, keys))
        record["clips"] = len(group)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{metric}_p10"] = float(values.quantile(0.10))
            record[f"{metric}_median"] = float(values.median())
            record[f"{metric}_p90"] = float(values.quantile(0.90))
        records.append(record)
    return pd.DataFrame(records)


def compare_groups(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    labels = sorted(frame["source_class"].unique())
    for label in ["<all>", *labels]:
        subset = frame if label == "<all>" else frame[frame["source_class"] == label]
        for metric in metric_columns:
            remote = pd.to_numeric(
                subset.loc[subset["audit_group"] == "remote", metric], errors="coerce"
            ).dropna().to_numpy()
            local = pd.to_numeric(
                subset.loc[subset["audit_group"] == "local", metric], errors="coerce"
            ).dropna().to_numpy()
            if not len(remote) or not len(local):
                continue
            test = ks_2samp(remote, local, alternative="two-sided", method="auto")
            records.append(
                {
                    "source_class": label,
                    "metric": metric,
                    "remote_n": len(remote),
                    "local_n": len(local),
                    "remote_median": float(np.median(remote)),
                    "local_median": float(np.median(local)),
                    "remote_minus_local_median": float(np.median(remote) - np.median(local)),
                    "robust_effect_size": robust_effect(remote, local),
                    "ks_statistic": float(test.statistic),
                    "ks_pvalue": float(test.pvalue),
                }
            )
    result = pd.DataFrame(records)
    if not result.empty:
        result["absolute_robust_effect_size"] = result["robust_effect_size"].abs()
        result = result.sort_values(
            ["source_class", "absolute_robust_effect_size"], ascending=[True, False]
        )
    return result


def plot_metrics(frame: pd.DataFrame, output: Path) -> None:
    metrics = [
        "processed_rms_dbfs",
        "processed_active_rms_dbfs",
        "processed_spectral_centroid_hz",
        "processed_spectral_rolloff_85_hz",
        "processed_spectral_flatness",
        "processed_energy_0_100_fraction",
        "processed_energy_100_500_fraction",
        "processed_energy_2000_4000_fraction",
    ]
    labels = sorted(frame["source_class"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, metric in zip(axes.flat, metrics):
        positions: list[float] = []
        values: list[np.ndarray] = []
        tick_positions: list[float] = []
        for label_index, label in enumerate(labels):
            center = label_index * 3.0
            tick_positions.append(center + 0.5)
            for offset, group in enumerate(("local", "remote")):
                current = pd.to_numeric(
                    frame.loc[
                        (frame["source_class"] == label) & (frame["audit_group"] == group),
                        metric,
                    ],
                    errors="coerce",
                ).dropna().to_numpy()
                if len(current):
                    positions.append(center + offset)
                    values.append(current)
        if values:
            boxes = axis.boxplot(values, positions=positions, widths=0.7, showfliers=False, patch_artist=True)
            for index, box in enumerate(boxes["boxes"]):
                box.set_facecolor("#4C78A8" if index % 2 == 0 else "#E45756")
                box.set_alpha(0.7)
        axis.set_xticks(tick_positions, labels, rotation=35, ha="right")
        axis.set_title(metric.replace("processed_", "").replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Local (blue) versus remote (red) training-clip acoustics")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/remote_acoustic_audit"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-kind", choices=["all", "annotated", "background"], default="all")
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic mixture manifests (excluded by default).",
    )
    parser.add_argument("--samples-per-group-label", type=int, default=500)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--mean-subtract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high-pass-cutoff-hz", type=float, default=50.0)
    parser.add_argument("--high-pass-order", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_group_label < 1:
        parser.error("--samples-per-group-label must be positive")
    if args.high_pass_order < 1:
        parser.error("--high-pass-order must be positive")
    return args


def main() -> int:
    args = parse_args()
    rows = discover_rows(
        args.data_root, args.split, args.clip_kind, args.include_synthetic
    )
    selected = stratified_sample(rows, args.samples_per_group_label, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        try:
            samples, sample_rate, channels = read_audio(row)
            processed = samples.copy()
            if args.mean_subtract:
                processed -= float(np.mean(processed))
            if args.high_pass_cutoff_hz > 0:
                processed = high_pass(
                    processed, sample_rate, args.high_pass_cutoff_hz, args.high_pass_order
                )
            result: dict[str, Any] = {
                "clip_id": clean(row.get("clip_id")),
                "audit_group": row["audit_group"],
                "source_class": row["source_class"],
                "clip_kind": clean(row.get("clip_kind")),
                "Provider": clean(row.get("Provider")),
                "Dataset": clean(row.get("Dataset")),
                "source_recording_id": clean(row.get("source_recording_id")),
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_sec": len(samples) / sample_rate,
                "manifest_path": row["manifest_path"],
            }
            result.update(measure(samples, sample_rate, "raw_"))
            result.update(measure(processed, sample_rate, "processed_"))
            results.append(result)
        except Exception as exc:
            failures.append({"clip_id": clean(row.get("clip_id")), "error": repr(exc)})
        if index % 250 == 0 or index == len(selected):
            print(f"Measured {index:,}/{len(selected):,}; failures={len(failures):,}")

    if not results:
        raise RuntimeError("No clips were measured successfully")
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_dir / "remote_acoustic_clip_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "remote_acoustic_failures.csv", index=False)
    metric_columns = [
        column for column in frame.columns if column.startswith(("raw_", "processed_"))
    ]
    summary = summarize(frame, metric_columns)
    summary.to_csv(args.output_dir / "remote_acoustic_group_summary.csv", index=False)
    comparison = compare_groups(frame, metric_columns)
    comparison.to_csv(args.output_dir / "remote_acoustic_comparison.csv", index=False)
    plot_metrics(frame, args.output_dir / "remote_acoustic_comparison.png")

    all_comparison = comparison[comparison["source_class"] == "<all>"].head(15)
    report_lines = [
        "# Remote versus local acoustic audit",
        "",
        f"Eligible rows discovered: {len(rows):,}",
        f"Clips selected:          {len(selected):,}",
        f"Clips measured:          {len(frame):,}",
        f"Failures:                {len(failures):,}",
        f"Split / clip kind:       {args.split} / {args.clip_kind}",
        f"Mean subtraction:        {args.mean_subtract}",
        f"High-pass filter:        {args.high_pass_cutoff_hz:g} Hz, order {args.high_pass_order}",
        "",
        "Largest overall robust differences after matching sample counts by group and label:",
    ]
    for _, row in all_comparison.iterrows():
        report_lines.append(
            f"  {row['metric']}: remote median={row['remote_median']:.6g}, "
            f"local median={row['local_median']:.6g}, robust effect={row['robust_effect_size']:.3f}, "
            f"KS={row['ks_statistic']:.3f}"
        )
    report_lines.extend(
        [
            "",
            "Interpretation guide:",
            "  |robust effect| around 0.5 is noticeable; >=1 is a large distribution shift.",
            "  Inspect class-specific rows before attributing a difference to storage mode.",
            "  A difference remaining after processed_ metrics may warrant preprocessing or sampling changes.",
        ]
    )
    (args.output_dir / "remote_acoustic_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines))
    print(f"\nOutputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
