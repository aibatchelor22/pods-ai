#!/usr/bin/env python3
"""Generate Orcasound distractor/background mixtures for DCLDE training.

Bird, vessel, jingle, and human clips are mixed with ordinary underwater
backgrounds at controlled broadband SNRs. Every generated clip remains a
background example (ClassSpecies=BKG); this script never inserts whale or
dolphin calls and performs no model inference. Outputs include a mixture-only
manifest, the original training manifest plus mixtures without deduplication,
an audit table, provenance, and Kaggle dataset metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


DEFAULT_OUTPUT_DIR = "/kaggle/working/orcasound_distractor_background_mixtures"
DEFAULT_KAGGLE_DATASET_ID = "leonisviridis/orcasound-distractor-background-mixtures"
DEFAULT_CATEGORIES = ("bird", "vessel", "jingle", "human")
BACKGROUND_SPECIES = {"BKG"}
UNCERTAIN_OR_TARGET_SPECIES = {"KW", "HW", "AB", "UNDBIO"}
EPSILON = 1e-12


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "n/a"} else text


def parse_csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    rewrites = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid path rewrite {value!r}; expected OLD=NEW")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError("The OLD portion of a path rewrite cannot be empty")
        rewrites.append((old, new))
    return rewrites


def find_column(frame: pd.DataFrame, requested: Optional[str], candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested column {requested!r} not found; columns={list(frame.columns)}")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"None of the candidate columns {candidates} were found")


def candidate_paths(
    row: pd.Series,
    path_column: str,
    dataset_root: Optional[Path],
    rewrites: list[tuple[str, str]],
) -> list[Path]:
    raw = normalize_text(row.get(path_column))
    if not raw:
        return []
    for old, new in rewrites:
        if raw.startswith(old):
            raw = new + raw[len(old) :]
            break
    normalized = raw.replace("\\", "/")
    path = Path(normalized)
    candidates = [path]
    if dataset_root is not None:
        relative = normalize_text(row.get("relative_clip_path")).replace("\\", "/")
        filename = normalized.rsplit("/", 1)[-1]
        category = normalize_text(row.get("Category") or row.get("category")).casefold()
        node_name = normalize_text(row.get("NodeName") or row.get("node_name"))
        timestamp = normalize_text(row.get("Timestamp") or row.get("timestamp"))
        inferred_filename = (
            f"{node_name.replace('_', '-')}_{timestamp}.wav"
            if node_name and timestamp
            else ""
        )
        if relative:
            candidates.append(dataset_root / relative.lstrip("/"))
        if not path.is_absolute():
            candidates.append(dataset_root / path)
        candidates.extend(
            [
                dataset_root / "clips" / filename,
                dataset_root / filename,
            ]
        )
        if category:
            candidates.extend(
                [
                    dataset_root / category / filename,
                    dataset_root / "output" / "wav" / category / filename,
                    dataset_root / "src" / "output" / "wav" / category / filename,
                ]
            )
        if inferred_filename:
            candidates.extend(
                [
                    dataset_root / inferred_filename,
                    dataset_root / "clips" / inferred_filename,
                    dataset_root / category / inferred_filename,
                    dataset_root / "output" / "wav" / category / inferred_filename,
                    dataset_root / "src" / "output" / "wav" / category / inferred_filename,
                ]
            )
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def parse_orcasound_timestamp(value: Any) -> pd.Timestamp:
    """Parse the timestamp convention used by Orcasound manifests."""
    text = normalize_text(value)
    if not text:
        raise ValueError("missing Orcasound timestamp")
    return pd.to_datetime(
        text.removesuffix("_PST"),
        format="%Y_%m_%d_%H_%M_%S",
        errors="raise",
    )


def quarantine_same_node_day(
    distractors: pd.DataFrame,
    testing: pd.DataFrame,
    node_column: str = "NodeName",
    timestamp_column: str = "Timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove distractors from the same hydrophone/date as a held-out test row."""
    required = {node_column, timestamp_column}
    for name, frame in (("distractor", distractors), ("testing", testing)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest missing quarantine columns: {sorted(missing)}")

    held_out_keys = {
        (
            normalize_text(row[node_column]).casefold(),
            parse_orcasound_timestamp(row[timestamp_column]).date().isoformat(),
        )
        for _, row in testing.iterrows()
    }
    keys = distractors.apply(
        lambda row: (
            normalize_text(row[node_column]).casefold(),
            parse_orcasound_timestamp(row[timestamp_column]).date().isoformat(),
        ),
        axis=1,
    )
    excluded_mask = keys.isin(held_out_keys)
    excluded = distractors.loc[excluded_mask].copy()
    excluded["quarantine_reason"] = "same_hydrophone_and_calendar_date_as_test"
    return distractors.loc[~excluded_mask].copy(), excluded


def numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def basename(value: Any) -> str:
    return Path(normalize_text(value).replace("\\", "/")).name


def screen_backgrounds_with_annotations(
    backgrounds: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject BKG donor clips overlapping target or uncertain annotations."""
    required_annotations = {"Soundfile", "ClassSpecies", "FileBeginSec", "FileEndSec"}
    missing = required_annotations - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotations CSV missing columns: {sorted(missing)}")
    if "Soundfile" not in backgrounds.columns:
        return backgrounds.copy(), pd.DataFrame()

    unsafe = annotations.copy()
    unsafe["_species"] = unsafe["ClassSpecies"].map(normalize_text).str.upper()
    unsafe = unsafe.loc[unsafe["_species"].isin(UNCERTAIN_OR_TARGET_SPECIES)].copy()
    unsafe["_soundfile"] = unsafe["Soundfile"].map(basename)
    unsafe["_start"] = pd.to_numeric(unsafe["FileBeginSec"], errors="coerce")
    unsafe["_end"] = pd.to_numeric(unsafe["FileEndSec"], errors="coerce")
    unsafe = unsafe.loc[
        unsafe["_soundfile"].ne("")
        & unsafe["_start"].notna()
        & unsafe["_end"].notna()
        & unsafe["_end"].gt(unsafe["_start"])
    ]
    intervals = {
        soundfile: list(zip(group["_start"], group["_end"], group["_species"]))
        for soundfile, group in unsafe.groupby("_soundfile", sort=False)
    }

    excluded_indices = []
    reasons = {}
    for index, row in backgrounds.iterrows():
        soundfile = basename(row.get("Soundfile"))
        clip_start = numeric(row.get("ClipStartSec"))
        clip_end = numeric(row.get("ClipEndSec"))
        if not soundfile or not math.isfinite(clip_start) or not math.isfinite(clip_end):
            continue
        for annotation_start, annotation_end, species in intervals.get(soundfile, []):
            if clip_start < annotation_end and annotation_start < clip_end:
                excluded_indices.append(index)
                reasons[index] = f"overlaps_annotation_{species}"
                break

    excluded = backgrounds.loc[excluded_indices].copy()
    if not excluded.empty:
        excluded["background_exclusion_reason"] = [reasons[index] for index in excluded.index]
    return backgrounds.drop(index=excluded_indices).copy(), excluded


def resolve_audio_path(
    row: pd.Series,
    path_column: str,
    dataset_root: Optional[Path],
    rewrites: list[tuple[str, str]],
) -> Path:
    candidates = candidate_paths(row, path_column, dataset_root, rewrites)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if not candidates:
        raise FileNotFoundError(f"Empty {path_column}")
    raise FileNotFoundError(
        f"Audio file not found; tried: {', '.join(str(path) for path in candidates)}"
    )


class AudioCache:
    """Small LRU cache of decoded, mono, resampled waveforms."""

    def __init__(self, sample_rate: int, max_items: int) -> None:
        self.sample_rate = sample_rate
        self.max_items = max_items
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, path: Path) -> np.ndarray:
        key = str(path)
        cached = self.items.pop(key, None)
        if cached is not None:
            self.items[key] = cached
            return cached.copy()
        waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Invalid or empty waveform: {path}")
        if int(source_rate) != self.sample_rate:
            divisor = math.gcd(int(source_rate), self.sample_rate)
            waveform = resample_poly(
                waveform,
                self.sample_rate // divisor,
                int(source_rate) // divisor,
            ).astype(np.float32)
        if not np.all(np.isfinite(waveform)):
            raise ValueError(f"Non-finite waveform samples: {path}")
        self.items[key] = waveform
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return waveform.copy()


def fit_duration(waveform: np.ndarray, target_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(waveform) > target_samples:
        start = int(rng.integers(0, len(waveform) - target_samples + 1))
        return waveform[start : start + target_samples].copy()
    if len(waveform) < target_samples:
        left = int(rng.integers(0, target_samples - len(waveform) + 1))
        return np.pad(waveform, (left, target_samples - len(waveform) - left))
    return waveform.copy()


def zero_pad_shift(waveform: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return waveform.copy()
    output = np.zeros_like(waveform)
    if shift > 0:
        output[shift:] = waveform[:-shift]
    else:
        output[:shift] = waveform[-shift:]
    return output


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def active_rms(
    waveform: np.ndarray,
    sample_rate: int,
    frame_duration_ms: float = 100.0,
    active_quantile: float = 0.80,
) -> float:
    """RMS over frames at or above the waveform's frame-RMS quantile."""
    frame_samples = max(1, int(round(sample_rate * frame_duration_ms / 1000.0)))
    usable_samples = (len(waveform) // frame_samples) * frame_samples
    if usable_samples < frame_samples:
        return rms(waveform)
    frames = waveform[:usable_samples].reshape(-1, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    cutoff = float(np.quantile(frame_rms, active_quantile))
    active_frames = frames[frame_rms >= cutoff]
    return rms(active_frames) if active_frames.size else rms(waveform)


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPSILON))


def clipped_percent(waveform: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs(waveform) >= 0.999))


def validate_audio_quality(
    waveform: np.ndarray,
    role: str,
    minimum_rms_dbfs: float | None,
    maximum_clipped_percent: float | None,
) -> dict[str, float]:
    waveform_rms = rms(waveform)
    waveform_rms_dbfs = dbfs(waveform_rms)
    waveform_clipped_percent = clipped_percent(waveform)
    if minimum_rms_dbfs is not None and waveform_rms_dbfs < minimum_rms_dbfs:
        raise ValueError(
            f"{role}_rms_dbfs_below_minimum: "
            f"{waveform_rms_dbfs:.3f} < {minimum_rms_dbfs:.3f}"
        )
    if (
        maximum_clipped_percent is not None
        and waveform_clipped_percent > maximum_clipped_percent
    ):
        raise ValueError(
            f"{role}_clipped_percent_above_maximum: "
            f"{waveform_clipped_percent:.6f} > {maximum_clipped_percent:.6f}"
        )
    return {
        "rms": waveform_rms,
        "rms_dbfs": waveform_rms_dbfs,
        "clipped_percent": waveform_clipped_percent,
    }


def mix_distractor_and_background(
    distractor: np.ndarray,
    background: np.ndarray,
    target_snr_db: float,
    peak_limit: float,
    sample_rate: int = 16000,
    snr_reference: str = "rms",
) -> tuple[np.ndarray, dict[str, float]]:
    distractor_rms = rms(distractor)
    background_rms = rms(background)
    if distractor_rms <= EPSILON:
        raise ValueError("silent_distractor")
    if background_rms <= EPSILON:
        raise ValueError("silent_background")
    if snr_reference == "active_rms":
        distractor_reference_rms = active_rms(distractor, sample_rate)
        background_reference_rms = active_rms(background, sample_rate)
    elif snr_reference == "rms":
        distractor_reference_rms = distractor_rms
        background_reference_rms = background_rms
    else:
        raise ValueError(f"Unsupported SNR reference: {snr_reference}")
    if distractor_reference_rms <= EPSILON:
        raise ValueError("silent_distractor_reference")
    if background_reference_rms <= EPSILON:
        raise ValueError("silent_background_reference")
    target_distractor_rms = background_reference_rms * 10.0 ** (target_snr_db / 20.0)
    distractor_gain = target_distractor_rms / distractor_reference_rms
    scaled_distractor = distractor * distractor_gain
    scaled_distractor_reference_rms = (
        active_rms(scaled_distractor, sample_rate)
        if snr_reference == "active_rms"
        else rms(scaled_distractor)
    )
    measured_snr_db = 20.0 * math.log10(
        (scaled_distractor_reference_rms + EPSILON) / background_reference_rms
    )
    mixture = background + scaled_distractor
    peak_before_scale = float(np.max(np.abs(mixture)))
    peak_scale = min(1.0, peak_limit / peak_before_scale) if peak_before_scale > 0 else 1.0
    mixture = np.clip(mixture * peak_scale, -1.0, 1.0).astype(np.float32)
    return mixture, {
        "distractor_rms": distractor_rms,
        "background_rms": background_rms,
        "distractor_reference_rms": distractor_reference_rms,
        "background_reference_rms": background_reference_rms,
        "distractor_gain": distractor_gain,
        "measured_snr_db": measured_snr_db,
        "peak_before_scale": peak_before_scale,
        "peak_scale": peak_scale,
    }


def balanced_category_schedule(
    categories: list[str],
    total: int,
    rng: np.random.Generator,
) -> list[str]:
    if not categories or total < 1:
        raise ValueError("Categories and total mixture count must be non-empty")
    base, remainder = divmod(total, len(categories))
    schedule = [
        category
        for index, category in enumerate(categories)
        for _ in range(base + int(index < remainder))
    ]
    rng.shuffle(schedule)
    return schedule


def proportional_category_schedule(
    category_values: pd.Series,
    mixtures_per_source: int,
    rng: np.random.Generator,
) -> list[str]:
    schedule = [
        category
        for category in category_values.astype(str).tolist()
        for _ in range(mixtures_per_source)
    ]
    rng.shuffle(schedule)
    return schedule


def background_domains(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[int]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return {"all": frame.index.tolist()}
    groups: dict[str, list[int]] = {}
    for index, row in frame.iterrows():
        key = " | ".join(
            f"{column}={normalize_text(row.get(column)).casefold() or '<missing>'}"
            for column in available
        )
        groups.setdefault(key, []).append(index)
    return groups


def output_row(
    distractor: pd.Series,
    background: pd.Series,
    category: str,
    mixture_index: int,
    filename: str,
    local_path: Path,
    published_path: str,
    distractor_path: Path,
    background_path: Path,
    background_domain: str,
    target_snr_db: float,
    shift_samples: int,
    metrics: dict[str, float],
    seed: int,
    duration: float,
) -> dict[str, Any]:
    return {
        "Soundfile": filename,
        "Dataset": "orcasound_distractor_background_mixtures",
        "ClassSpecies": "BKG",
        "KW": 0,
        "KW_certain": 1,
        "Ecotype": "",
        "Provider": "OrcaSound",
        "AnnotationLevel": "synthetic_background",
        "FileBeginSec": 0.0,
        "FileEndSec": duration,
        "ClipStartSec": 0.0,
        "ClipEndSec": duration,
        "clip_filename": filename,
        "clip_path": published_path,
        "local_clip_path": str(local_path),
        "Generated": True,
        "sampling_source": "orcasound_distractor_background_mixture",
        "distractor_category": category,
        "mixture_index": mixture_index,
        "mixture_seed": seed,
        "mixture_method": "broadband_distractor_plus_underwater_background",
        "mixture_snr_reference": metrics.get("snr_reference", "rms"),
        "mixture_target_snr_db": target_snr_db,
        "mixture_measured_snr_db": metrics["measured_snr_db"],
        "mixture_distractor_gain": metrics["distractor_gain"],
        "mixture_peak_scale": metrics["peak_scale"],
        "mixture_shift_samples": shift_samples,
        "distractor_clip_path": str(distractor_path),
        "distractor_soundfile": normalize_text(distractor.get("Soundfile")),
        "distractor_dataset": normalize_text(distractor.get("Dataset")),
        "distractor_provider": normalize_text(distractor.get("Provider")),
        "background_clip_path": str(background_path),
        "background_soundfile": normalize_text(background.get("Soundfile")),
        "background_dataset": normalize_text(background.get("Dataset")),
        "background_provider": normalize_text(background.get("Provider")),
        "background_domain": background_domain,
        "distractor_rms_before_gain": metrics["distractor_rms"],
        "background_rms": metrics["background_rms"],
        "distractor_reference_rms_before_gain": metrics.get(
            "distractor_reference_rms", metrics["distractor_rms"]
        ),
        "background_reference_rms": metrics.get(
            "background_reference_rms", metrics["background_rms"]
        ),
        "distractor_clipped_percent": metrics.get("distractor_clipped_percent"),
        "background_clipped_percent": metrics.get("background_clipped_percent"),
    }


def atomic_write_audio(path: Path, waveform: np.ndarray, sample_rate: int, audio_format: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    file_format = "WAV" if audio_format == "wav" else "FLAC"
    sf.write(temporary, waveform, sample_rate, format=file_format)
    os.replace(temporary, path)


def write_outputs(
    mixture_manifest: Path,
    audit_path: Path,
    rows: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> None:
    pd.DataFrame(rows).to_csv(mixture_manifest, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distractor-manifest", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument(
        "--annotations-csv",
        required=True,
        help="Original Annotations.csv used to reject BKG donors overlapping target/UndBio annotations.",
    )
    parser.add_argument(
        "--testing-manifest",
        required=True,
        help="Held-out Orcasound 60s manifest used for same-hydrophone/day quarantine.",
    )
    parser.add_argument(
        "--ordinary-background-manifest",
        default=None,
        help="BKG/UndBio donor manifest; defaults to --training-manifest.",
    )
    parser.add_argument("--distractor-dataset-root", default=None)
    parser.add_argument("--background-dataset-root", default=None)
    parser.add_argument("--distractor-path-column", default=None)
    parser.add_argument("--background-path-column", default=None)
    parser.add_argument("--distractor-path-rewrite", action="append", default=[])
    parser.add_argument("--background-path-rewrite", action="append", default=[])
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--allow-undbio-backgrounds",
        action="store_true",
        help="Allow ClassSpecies=UndBio ordinary background donors (default: excluded).",
    )
    parser.add_argument("--mixtures-per-source", type=int, default=10)
    parser.add_argument("--num-mixtures", type=int, default=None)
    parser.add_argument(
        "--balance-categories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allocate equal mixture counts to available categories (default: true).",
    )
    parser.add_argument("--background-domain-columns", default="Provider,Dataset")
    parser.add_argument(
        "--uniform-background-domains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample recording domains uniformly before sampling a clip (default: true).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--snr-db-min", type=float, default=-12.0)
    parser.add_argument("--snr-db-max", type=float, default=12.0)
    parser.add_argument(
        "--snr-reference",
        choices=("rms", "active_rms"),
        default="rms",
        help="Level reference used to set SNR (default: whole-clip RMS).",
    )
    parser.add_argument(
        "--minimum-background-rms-dbfs",
        type=float,
        default=None,
        help="Reject fitted background donors quieter than this whole-clip RMS level.",
    )
    parser.add_argument(
        "--maximum-clipped-percent",
        type=float,
        default=None,
        help="Reject distractor or background clips with a larger percent of samples at |x| >= 0.999.",
    )
    parser.add_argument("--max-shift-ms", type=float, default=500.0)
    parser.add_argument("--peak-limit", type=float, default=0.99)
    parser.add_argument("--audio-format", choices=("wav", "flac"), default="wav")
    parser.add_argument("--audio-cache-items", type=int, default=256)
    parser.add_argument("--max-attempts-per-mixture", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--combined-manifest", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--kaggle-dataset-id", default=DEFAULT_KAGGLE_DATASET_ID)
    parser.add_argument("--kaggle-title", default="Orcasound Distractor Background Mixtures")
    parser.add_argument("--kaggle-license", default="CC0-1.0")
    parser.add_argument("--publish-action", choices=("none", "create", "version"), default="none")
    parser.add_argument("--version-message", default="Update Orcasound distractor mixtures")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mixtures_per_source < 1:
        raise ValueError("--mixtures-per-source must be positive")
    if args.num_mixtures is not None and args.num_mixtures < 1:
        raise ValueError("--num-mixtures must be positive")
    if args.sample_rate < 1000 or args.clip_duration <= 0:
        raise ValueError("--sample-rate and --clip-duration must be positive")
    if args.snr_db_min > args.snr_db_max:
        raise ValueError("--snr-db-min cannot exceed --snr-db-max")
    if args.max_shift_ms < 0:
        raise ValueError("--max-shift-ms cannot be negative")
    if not 0 < args.peak_limit <= 1:
        raise ValueError("--peak-limit must be in (0, 1]")
    if args.maximum_clipped_percent is not None and not (
        0 <= args.maximum_clipped_percent <= 100
    ):
        raise ValueError("--maximum-clipped-percent must be in [0, 100]")
    if args.audio_cache_items < 1 or args.max_attempts_per_mixture < 1:
        raise ValueError("Cache size and maximum attempts must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.kaggle_dataset_id.count("/") != 1:
        raise ValueError("--kaggle-dataset-id must have the form owner/dataset-slug")


def main() -> int:
    args = parse_args()
    validate_args(args)
    distractor_manifest = Path(args.distractor_manifest)
    training_manifest = Path(args.training_manifest)
    background_manifest = Path(args.ordinary_background_manifest or args.training_manifest)
    annotations_path = Path(args.annotations_csv)
    testing_manifest = Path(args.testing_manifest)
    for path in (
        distractor_manifest,
        training_manifest,
        background_manifest,
        annotations_path,
        testing_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else output_dir / "orcasound_distractor_mixture_manifest.csv"
    )
    combined_manifest = (
        Path(args.combined_manifest)
        if args.combined_manifest
        else output_dir / "training_manifest_with_orcasound_distractors.csv"
    )
    audit_path = output_dir / "generation_audit.csv"
    quarantine_path = output_dir / "quarantined_orcasound_distractors.csv"
    background_exclusions_path = output_dir / "background_annotation_exclusions.csv"

    distractors = pd.read_csv(distractor_manifest, low_memory=False)
    training_frame = pd.read_csv(training_manifest, low_memory=False)
    backgrounds = pd.read_csv(background_manifest, low_memory=False)
    annotations = pd.read_csv(annotations_path, low_memory=False)
    testing = pd.read_csv(testing_manifest, low_memory=False)
    distractor_path_column = find_column(
        distractors,
        args.distractor_path_column,
        ("clip_path", "wav_path", "audio_path", "local_clip_path", "FilePath", "URI"),
    )
    background_path_column = find_column(
        backgrounds,
        args.background_path_column,
        ("clip_path", "wav_path", "audio_path", "local_clip_path", "FilePath"),
    )
    category_column = find_column(
        distractors,
        args.category_column,
        ("Category", "category", "label", "original_label"),
    )
    if "ClassSpecies" not in backgrounds.columns:
        raise ValueError("Ordinary background manifest must contain ClassSpecies")

    allowed_categories = {value.casefold() for value in parse_csv_values(args.categories)}
    distractors = distractors.copy()
    distractors["_category"] = distractors[category_column].map(normalize_text).str.casefold()
    distractors = distractors.loc[distractors["_category"].isin(allowed_categories)].copy()
    if distractors.empty:
        raise ValueError(f"No distractor rows matched categories {sorted(allowed_categories)}")
    distractors_before_quarantine = len(distractors)
    distractors, quarantined_distractors = quarantine_same_node_day(distractors, testing)
    quarantined_distractors.to_csv(quarantine_path, index=False)
    if distractors.empty:
        raise ValueError("All distractor rows were removed by same-hydrophone/day quarantine")
    allowed_background_species = set(BACKGROUND_SPECIES)
    if args.allow_undbio_backgrounds:
        allowed_background_species.add("UndBio")
    backgrounds = backgrounds.loc[
        backgrounds["ClassSpecies"].map(normalize_text).isin(allowed_background_species)
    ].copy()
    if backgrounds.empty:
        raise ValueError("No eligible BKG rows were found in the ordinary background manifest")
    backgrounds_before_annotation_screen = len(backgrounds)
    backgrounds, background_exclusions = screen_backgrounds_with_annotations(
        backgrounds,
        annotations,
    )
    background_exclusions.to_csv(background_exclusions_path, index=False)
    if backgrounds.empty:
        raise ValueError("All ordinary backgrounds were rejected by annotation screening")

    categories = sorted(distractors["_category"].unique())
    total_mixtures = args.num_mixtures or len(distractors) * args.mixtures_per_source
    schedule_rng = np.random.default_rng(args.seed)
    if args.balance_categories:
        schedule = balanced_category_schedule(categories, total_mixtures, schedule_rng)
    else:
        if args.num_mixtures is not None:
            probabilities = distractors["_category"].value_counts(normalize=True)
            schedule = schedule_rng.choice(
                probabilities.index.to_numpy(),
                size=total_mixtures,
                p=probabilities.to_numpy(),
            ).tolist()
        else:
            schedule = proportional_category_schedule(
                distractors["_category"], args.mixtures_per_source, schedule_rng
            )

    domains = background_domains(
        backgrounds,
        parse_csv_values(args.background_domain_columns),
    )
    domain_names = sorted(domains)
    distractor_root = Path(args.distractor_dataset_root) if args.distractor_dataset_root else None
    background_root = Path(args.background_dataset_root) if args.background_dataset_root else None
    distractor_rewrites = parse_path_rewrites(args.distractor_path_rewrite)
    background_rewrites = parse_path_rewrites(args.background_path_rewrite)
    cache = AudioCache(args.sample_rate, args.audio_cache_items)
    target_samples = int(round(args.sample_rate * args.clip_duration))
    max_shift_samples = int(round(args.sample_rate * args.max_shift_ms / 1000.0))
    published_root = f"/kaggle/input/datasets/{args.kaggle_dataset_id}/clips"

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    if output_manifest.is_file() and not args.no_resume:
        existing = pd.read_csv(output_manifest, low_memory=False)
        rows = existing.to_dict("records")
        for row in rows:
            filename = normalize_text(row.get("clip_filename"))
            if filename:
                row["clip_path"] = f"{published_root}/{filename}"
        if "mixture_index" in existing.columns:
            completed_indices = set(existing["mixture_index"].dropna().astype(int))

    print("\nOrcasound distractor/background mixture generation")
    print("=" * 72)
    print(f"Distractor manifest:       {distractor_manifest}")
    print(f"Training manifest:         {training_manifest}")
    print(f"Annotations CSV:           {annotations_path}")
    print(f"Held-out testing manifest: {testing_manifest}")
    print(
        f"Distractor quarantine:     {distractors_before_quarantine - len(distractors):,} removed; "
        f"{len(distractors):,} retained"
    )
    print(
        f"Annotation donor screen:   {backgrounds_before_annotation_screen - len(backgrounds):,} "
        f"removed; {len(backgrounds):,} retained"
    )
    print(f"Ordinary backgrounds:      {len(backgrounds):,} across {len(domains):,} domains")
    print(f"Eligible distractors:      {len(distractors):,}")
    print(f"Distractor categories:     {dict(Counter(distractors['_category']))}")
    print(f"Requested mixtures:        {len(schedule):,} ({dict(Counter(schedule))})")
    print(f"Target SNR range:          {args.snr_db_min:g}..{args.snr_db_max:g} dB")
    print(f"SNR reference:             {args.snr_reference}")
    print(f"Minimum background RMS:    {args.minimum_background_rms_dbfs} dBFS")
    print(f"Maximum clipped samples:   {args.maximum_clipped_percent}%")
    print(f"Uniform background domains:{args.uniform_background_domains}")
    print(f"Already completed:         {len(completed_indices):,}")
    print("=" * 72 + "\n")

    failures = Counter()
    for mixture_index, category in enumerate(schedule):
        if mixture_index in completed_indices:
            continue
        example_rng = np.random.default_rng(np.random.SeedSequence([args.seed, mixture_index]))
        category_rows = distractors.loc[distractors["_category"] == category]
        success = False
        for attempt in range(1, args.max_attempts_per_mixture + 1):
            try:
                distractor = category_rows.iloc[int(example_rng.integers(0, len(category_rows)))]
                if args.uniform_background_domains:
                    domain = domain_names[int(example_rng.integers(0, len(domain_names)))]
                    indices = domains[domain]
                    background = backgrounds.loc[indices[int(example_rng.integers(0, len(indices)))]]
                else:
                    background = backgrounds.iloc[int(example_rng.integers(0, len(backgrounds)))]
                    domain = "proportional_all_rows"
                distractor_path = resolve_audio_path(
                    distractor, distractor_path_column, distractor_root, distractor_rewrites
                )
                background_path = resolve_audio_path(
                    background, background_path_column, background_root, background_rewrites
                )
                distractor_audio = fit_duration(
                    cache.load(distractor_path), target_samples, example_rng
                )
                background_audio = fit_duration(
                    cache.load(background_path), target_samples, example_rng
                )
                distractor_quality = validate_audio_quality(
                    distractor_audio,
                    "distractor",
                    minimum_rms_dbfs=None,
                    maximum_clipped_percent=args.maximum_clipped_percent,
                )
                background_quality = validate_audio_quality(
                    background_audio,
                    "background",
                    minimum_rms_dbfs=args.minimum_background_rms_dbfs,
                    maximum_clipped_percent=args.maximum_clipped_percent,
                )
                shift = (
                    int(example_rng.integers(-max_shift_samples, max_shift_samples + 1))
                    if max_shift_samples
                    else 0
                )
                distractor_audio = zero_pad_shift(distractor_audio, shift)
                target_snr_db = float(example_rng.uniform(args.snr_db_min, args.snr_db_max))
                mixture, metrics = mix_distractor_and_background(
                    distractor_audio,
                    background_audio,
                    target_snr_db,
                    args.peak_limit,
                    sample_rate=args.sample_rate,
                    snr_reference=args.snr_reference,
                )
                metrics["snr_reference"] = args.snr_reference
                metrics["distractor_clipped_percent"] = distractor_quality[
                    "clipped_percent"
                ]
                metrics["background_clipped_percent"] = background_quality[
                    "clipped_percent"
                ]
                extension = ".wav" if args.audio_format == "wav" else ".flac"
                filename = (
                    f"distractor_{mixture_index:06d}_{category}_"
                    f"snr{target_snr_db:+05.1f}db{extension}"
                )
                local_path = clips_dir / filename
                atomic_write_audio(local_path, mixture, args.sample_rate, args.audio_format)
                published_path = f"{published_root}/{filename}"
                rows.append(
                    output_row(
                        distractor,
                        background,
                        category,
                        mixture_index,
                        filename,
                        local_path,
                        published_path,
                        distractor_path,
                        background_path,
                        domain,
                        target_snr_db,
                        shift,
                        metrics,
                        args.seed,
                        args.clip_duration,
                    )
                )
                audit.append(
                    {
                        "mixture_index": mixture_index,
                        "category": category,
                        "attempt": attempt,
                        "status": "saved",
                        "filename": filename,
                    }
                )
                success = True
                break
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                failures[reason] += 1
                audit.append(
                    {
                        "mixture_index": mixture_index,
                        "category": category,
                        "attempt": attempt,
                        "status": "retry",
                        "reason": reason,
                    }
                )
        if not success:
            audit.append(
                {
                    "mixture_index": mixture_index,
                    "category": category,
                    "status": "failed",
                    "reason": "maximum_attempts_exhausted",
                }
            )
        if (mixture_index + 1) % args.checkpoint_every == 0:
            write_outputs(output_manifest, audit_path, rows, audit)
            print(f"Processed {mixture_index + 1:,}/{len(schedule):,}; saved {len(rows):,}")

    write_outputs(output_manifest, audit_path, rows, audit)
    combined = pd.concat([training_frame, pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined.to_csv(combined_manifest, index=False)
    metadata = {
        "title": args.kaggle_title,
        "id": args.kaggle_dataset_id,
        "licenses": [{"name": args.kaggle_license}],
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary = {
        "distractor_manifest": str(distractor_manifest),
        "training_manifest": str(training_manifest),
        "ordinary_background_manifest": str(background_manifest),
        "annotations_csv": str(annotations_path),
        "testing_manifest": str(testing_manifest),
        "distractors_before_quarantine": distractors_before_quarantine,
        "quarantined_distractors": len(quarantined_distractors),
        "eligible_distractors": len(distractors),
        "backgrounds_before_annotation_screen": backgrounds_before_annotation_screen,
        "background_annotation_exclusions": len(background_exclusions),
        "requested_mixtures": len(schedule),
        "saved_mixtures": len(rows),
        "saved_categories": dict(Counter(normalize_text(row.get("distractor_category")) for row in rows)),
        "combined_manifest": str(combined_manifest),
        "combined_manifest_rows": len(combined),
        "retry_reasons": dict(failures.most_common()),
        "arguments": vars(args),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\nGeneration complete")
    print("=" * 72)
    print(f"Saved mixtures:        {len(rows):,}")
    print(f"Mixture manifest:      {output_manifest}")
    print(f"Combined manifest:     {combined_manifest}")
    print(f"Combined rows:         {len(combined):,}")
    print(f"Audit:                 {audit_path}")
    print("Review and listen to examples from every category and SNR range before training.")

    if args.publish_action == "create":
        subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(output_dir), "--dir-mode", "zip"],
            check=True,
        )
    elif args.publish_action == "version":
        subprocess.run(
            [
                "kaggle", "datasets", "version", "-p", str(output_dir),
                "-m", args.version_message, "--dir-mode", "zip",
            ],
            check=True,
        )
    else:
        print("\nTo create the Kaggle dataset:")
        print(f"  kaggle datasets create -p {output_dir} --dir-mode zip")
        print("For a later version:")
        print(
            f'  kaggle datasets version -p {output_dir} -m "{args.version_message}" '
            "--dir-mode zip"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
