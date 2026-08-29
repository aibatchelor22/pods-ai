#!/usr/bin/env python3
"""Generate annotation-driven call/background mixtures for DCLDE training.

Positive donor calls come only from manifest annotations (KW, HW, or AB).
The script isolates each call inside its annotated time/frequency rectangle,
places it at a random position in a three-second background clip, and scales it
to a controlled band-limited SNR. It writes audio, a trainer-compatible
manifest, provenance, an audit table, and Kaggle dataset metadata.

This is an offline dataset generator. It never uses model predictions to decide
which clips are positive.
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
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, istft, resample_poly, sosfiltfilt, stft


DEFAULT_MANIFEST = (
    "/kaggle/input/datasets/leonisviridis/"
    "dclde-cetacean-detector-manifests-and-misc/"
    "dclde_orcasound_train_plus_multispecies_hard_negatives.csv"
)
DEFAULT_OUTPUT_DIR = "/kaggle/working/dclde_controlled_background_mixtures"
DEFAULT_KAGGLE_DATASET_ID = "leonisviridis/dclde-controlled-background-mixtures"
POSITIVE_SPECIES = ("KW", "HW", "AB")
BACKGROUND_SPECIES = {"BKG", "UndBio"}
MISSING_DOMAIN = "<missing>"
EPSILON = 1e-12


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "n/a"} else text


def numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def parse_csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --path-rewrite {value!r}; expected OLD=NEW")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError("The OLD portion of --path-rewrite cannot be empty")
        rewrites.append((old, new))
    return rewrites


def parse_species_counts(value: Optional[str], total: int) -> dict[str, int]:
    if value is None:
        base, remainder = divmod(total, len(POSITIVE_SPECIES))
        return {
            species: base + int(index < remainder)
            for index, species in enumerate(POSITIVE_SPECIES)
        }
    counts = {species: 0 for species in POSITIVE_SPECIES}
    for assignment in value.split(","):
        if "=" not in assignment:
            raise ValueError("--species-counts must look like KW=10000,HW=10000,AB=10000")
        label, raw_count = (part.strip() for part in assignment.split("=", 1))
        if label not in counts:
            raise ValueError(f"Unknown species in --species-counts: {label!r}")
        counts[label] = int(raw_count)
    if any(count < 0 for count in counts.values()) or sum(counts.values()) < 1:
        raise ValueError("--species-counts must contain non-negative counts with a positive sum")
    return counts


def canonical_domain(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for column in columns:
        text = normalize_text(row.get(column, "")).casefold() or MISSING_DOMAIN
        parts.append(f"{column.casefold()}={text}")
    return " | ".join(parts)


def canonical_domains(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Vectorized domain keys for a manifest frame."""
    result: Optional[pd.Series] = None
    for column in columns:
        values = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace(r"^(?i:nan|none|null|na|n/a)?$", MISSING_DOMAIN, regex=True)
            .str.casefold()
        )
        part = column.casefold() + "=" + values
        result = part if result is None else result + " | " + part
    if result is None:
        raise ValueError("At least one domain column is required")
    return result


def resolve_audio_path(
    raw_path: Any,
    dataset_root: Optional[Path],
    rewrites: list[tuple[str, str]],
) -> Path:
    value = normalize_text(raw_path)
    if not value:
        raise FileNotFoundError("Empty clip_path")
    for old, new in rewrites:
        if value.startswith(old):
            value = new + value[len(old) :]
            break
    path = Path(value)
    if path.is_file():
        return path
    if not path.is_absolute() and dataset_root is not None:
        rooted = dataset_root / path
        if rooted.is_file():
            return rooted
        return rooted
    return path


class AudioCache:
    """Small least-recently-used cache for decoded three-second clips."""

    def __init__(self, sample_rate: int, duration: float, max_items: int) -> None:
        self.sample_rate = sample_rate
        self.target_samples = int(round(sample_rate * duration))
        self.max_items = max_items
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, path: Path) -> np.ndarray:
        key = str(path)
        cached = self.items.pop(key, None)
        if cached is not None:
            self.items[key] = cached
            return cached.copy()
        if not path.is_file():
            raise FileNotFoundError(path)
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
        if len(waveform) < self.target_samples:
            waveform = np.pad(waveform, (0, self.target_samples - len(waveform)))
        else:
            waveform = waveform[: self.target_samples]
        if not np.all(np.isfinite(waveform)):
            raise ValueError(f"Non-finite waveform samples: {path}")
        self.items[key] = waveform
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return waveform.copy()


def annotation_geometry(
    row: pd.Series,
    clip_duration: float,
    sample_rate: int,
    frequency_margin_hz: float,
) -> tuple[float, float, float, float]:
    clip_start = numeric(row.get("ClipStartSec"))
    event_start = numeric(row.get("FileBeginSec"))
    event_end = numeric(row.get("FileEndSec"))
    low_hz = numeric(row.get("LowFreqHz"))
    high_hz = numeric(row.get("HighFreqHz"))
    if not all(math.isfinite(value) for value in (clip_start, event_start, event_end, low_hz, high_hz)):
        raise ValueError("missing_time_or_frequency_bounds")
    relative_start = event_start - clip_start
    relative_end = event_end - clip_start
    if relative_end <= relative_start:
        raise ValueError("invalid_event_time_bounds")
    if relative_end <= 0 or relative_start >= clip_duration:
        raise ValueError("event_outside_clip")
    relative_start = max(0.0, relative_start)
    relative_end = min(clip_duration, relative_end)
    nyquist = sample_rate / 2.0
    low_hz = max(20.0, low_hz - frequency_margin_hz)
    high_hz = min(nyquist - 20.0, high_hz + frequency_margin_hz)
    if high_hz <= low_hz:
        raise ValueError("invalid_event_frequency_bounds")
    return relative_start, relative_end, low_hz, high_hz


def isolate_annotated_call(
    waveform: np.ndarray,
    sample_rate: int,
    event_start: float,
    event_end: float,
    low_hz: float,
    high_hz: float,
    time_margin_sec: float,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, float, float]:
    """Return a smoothly time-frequency-masked call segment and event offsets."""
    clip_duration = len(waveform) / sample_rate
    segment_start = max(0.0, event_start - time_margin_sec)
    segment_end = min(clip_duration, event_end + time_margin_sec)
    frequencies, times, spectrum = stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )
    mask = (
        (frequencies[:, None] >= low_hz)
        & (frequencies[:, None] <= high_hz)
        & (times[None, :] >= segment_start)
        & (times[None, :] <= segment_end)
    ).astype(np.float32)
    if not np.any(mask):
        raise ValueError("empty_time_frequency_mask")
    frequency_bin_hz = sample_rate / n_fft
    sigma_frequency = max(0.5, 30.0 / frequency_bin_hz)
    sigma_time = max(0.5, 0.025 * sample_rate / hop_length)
    mask = gaussian_filter(mask, sigma=(sigma_frequency, sigma_time), mode="nearest")
    mask /= max(float(mask.max()), EPSILON)
    _, isolated = istft(
        spectrum * mask,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
        boundary=True,
    )
    if len(isolated) < len(waveform):
        isolated = np.pad(isolated, (0, len(waveform) - len(isolated)))
    isolated = isolated[: len(waveform)].astype(np.float32)
    start_sample = int(round(segment_start * sample_rate))
    end_sample = int(round(segment_end * sample_rate))
    segment = isolated[start_sample:end_sample].copy()
    if segment.size == 0:
        raise ValueError("empty_isolated_segment")
    event_offset_start = event_start - segment_start
    event_offset_end = event_end - segment_start
    return segment, event_offset_start, event_offset_end


def band_limited_rms(
    waveform: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> float:
    if waveform.size < 8:
        return 0.0
    nyquist = sample_rate / 2.0
    low = max(10.0, min(low_hz, nyquist - 30.0))
    high = max(low + 10.0, min(high_hz, nyquist - 10.0))
    sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    try:
        filtered = sosfiltfilt(sos, waveform)
    except ValueError:
        return 0.0
    return float(np.sqrt(np.mean(np.square(filtered, dtype=np.float64))))


def scale_for_target_snr(signal_rms: float, background_rms: float, snr_db: float) -> float:
    if signal_rms <= EPSILON:
        raise ValueError("isolated_signal_has_no_band_energy")
    if background_rms <= EPSILON:
        raise ValueError("background_has_no_band_energy")
    return background_rms * (10.0 ** (snr_db / 20.0)) / signal_rms


def mix_call_and_background(
    signal_segment: np.ndarray,
    event_offset_start: float,
    event_offset_end: float,
    background: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    target_snr_db: float,
    rng: np.random.Generator,
    peak_limit: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(signal_segment) > len(background):
        raise ValueError("isolated_signal_segment_longer_than_background")
    maximum_start = len(background) - len(signal_segment)
    placement_start_sample = int(rng.integers(0, maximum_start + 1))
    placement_end_sample = placement_start_sample + len(signal_segment)
    event_start_sample = placement_start_sample + int(round(event_offset_start * sample_rate))
    event_end_sample = placement_start_sample + int(round(event_offset_end * sample_rate))
    event_start_sample = max(placement_start_sample, min(event_start_sample, placement_end_sample - 1))
    event_end_sample = max(event_start_sample + 1, min(event_end_sample, placement_end_sample))

    event_signal_start = event_start_sample - placement_start_sample
    event_signal_end = event_end_sample - placement_start_sample
    signal_rms = band_limited_rms(
        signal_segment[event_signal_start:event_signal_end], sample_rate, low_hz, high_hz
    )
    background_rms = band_limited_rms(
        background[event_start_sample:event_end_sample], sample_rate, low_hz, high_hz
    )
    signal_gain = scale_for_target_snr(signal_rms, background_rms, target_snr_db)
    signal_track = np.zeros_like(background)
    signal_track[placement_start_sample:placement_end_sample] = signal_segment * signal_gain
    mixture = background.astype(np.float64) + signal_track.astype(np.float64)
    peak = float(np.max(np.abs(mixture)))
    mixture_scale = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    mixture = (mixture * mixture_scale).astype(np.float32)
    measured_snr_db = 20.0 * math.log10(
        max(signal_rms * signal_gain, EPSILON) / max(background_rms, EPSILON)
    )
    return mixture, {
        "target_snr_db": float(target_snr_db),
        "measured_snr_db": float(measured_snr_db),
        "signal_gain": float(signal_gain),
        "mixture_scale": float(mixture_scale),
        "placement_start_sec": placement_start_sample / sample_rate,
        "placement_end_sec": placement_end_sample / sample_rate,
        "mixed_event_start_sec": event_start_sample / sample_rate,
        "mixed_event_end_sec": event_end_sample / sample_rate,
        "signal_band_rms_before_gain": signal_rms,
        "background_band_rms": background_rms,
    }


def eligible_rows(
    frame: pd.DataFrame,
    domain_columns: list[str],
    clip_duration: float,
    sample_rate: int,
    frequency_margin_hz: float,
    annotation_levels: Optional[set[str]],
) -> tuple[dict[str, pd.DataFrame], dict[str, list[int]], list[dict[str, Any]]]:
    species = frame["ClassSpecies"].fillna("").astype(str).str.strip()
    domains = canonical_domains(frame, domain_columns)
    background_mask = species.isin(BACKGROUND_SPECIES)
    backgrounds_by_domain = {
        domain: list(indices)
        for domain, indices in frame.loc[background_mask].groupby(domains[background_mask]).groups.items()
    }

    positive_mask = species.isin(POSITIVE_SPECIES)
    reason = pd.Series("", index=frame.index, dtype=object)
    if annotation_levels is not None:
        levels = frame.get(
            "AnnotationLevel",
            pd.Series("", index=frame.index),
        ).fillna("").astype(str).str.strip().str.casefold()
        reason.loc[positive_mask & ~levels.isin(annotation_levels)] = "annotation_level"

    numeric_columns: dict[str, pd.Series] = {}
    for column in ("ClipStartSec", "FileBeginSec", "FileEndSec", "LowFreqHz", "HighFreqHz"):
        if column in frame:
            numeric_columns[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            numeric_columns[column] = pd.Series(np.nan, index=frame.index)
    missing_bounds = pd.concat(numeric_columns.values(), axis=1).isna().any(axis=1)
    available = positive_mask & reason.eq("")
    reason.loc[available & missing_bounds] = "missing_time_or_frequency_bounds"

    clip_start = numeric_columns["ClipStartSec"]
    event_start = numeric_columns["FileBeginSec"]
    event_end = numeric_columns["FileEndSec"]
    relative_start = event_start - clip_start
    relative_end = event_end - clip_start
    available = positive_mask & reason.eq("")
    reason.loc[available & (event_end <= event_start)] = "invalid_event_time_bounds"
    available = positive_mask & reason.eq("")
    reason.loc[available & ((relative_end <= 0) | (relative_start >= clip_duration))] = "event_outside_clip"

    nyquist = sample_rate / 2.0
    low_hz = (numeric_columns["LowFreqHz"] - frequency_margin_hz).clip(lower=20.0)
    high_hz = (numeric_columns["HighFreqHz"] + frequency_margin_hz).clip(upper=nyquist - 20.0)
    available = positive_mask & reason.eq("")
    reason.loc[available & (high_hz <= low_hz)] = "invalid_event_frequency_bounds"

    eligible_mask = positive_mask & reason.eq("")
    donors = {
        label: frame.loc[eligible_mask & species.eq(label)]
        for label in POSITIVE_SPECIES
    }
    ineligible = reason.ne("")
    audit = pd.DataFrame(
        {
            "input_row": frame.index[ineligible],
            "status": "ineligible_donor",
            "reason": reason.loc[ineligible].to_numpy(),
        }
    ).to_dict("records")
    return donors, backgrounds_by_domain, audit


def output_row(
    donor: pd.Series,
    background: pd.Series,
    mixture_index: int,
    filename: str,
    local_path: Path,
    published_path: str,
    donor_path: Path,
    background_path: Path,
    donor_domain: str,
    background_domain: str,
    low_hz: float,
    high_hz: float,
    metrics: dict[str, float],
    seed: int,
    clip_duration: float,
) -> dict[str, Any]:
    row = donor.to_dict()
    species = normalize_text(donor.get("ClassSpecies"))
    row.update(
        {
            "Soundfile": filename,
            "Provider": background.get("Provider", ""),
            "Dataset": background.get("Dataset", ""),
            "ClassSpecies": species,
            "KW": int(species == "KW"),
            "AnnotationLevel": "ControlledMixture",
            "FileBeginSec": metrics["mixed_event_start_sec"],
            "FileEndSec": metrics["mixed_event_end_sec"],
            "CenterSec": (
                metrics["mixed_event_start_sec"] + metrics["mixed_event_end_sec"]
            )
            / 2.0,
            "ClipStartSec": 0.0,
            "ClipEndSec": clip_duration,
            "clip_filename": filename,
            "clip_path": published_path,
            "local_clip_path": str(local_path),
            "Generated": True,
            "mixture_index": mixture_index,
            "mixture_seed": seed,
            "mixture_method": "annotation_time_frequency_mask_controlled_snr",
            "mixture_target_snr_db": metrics["target_snr_db"],
            "mixture_measured_snr_db": metrics["measured_snr_db"],
            "mixture_signal_gain": metrics["signal_gain"],
            "mixture_peak_scale": metrics["mixture_scale"],
            "mixture_snr_band_low_hz": low_hz,
            "mixture_snr_band_high_hz": high_hz,
            "mixture_call_segment_start_sec": metrics["placement_start_sec"],
            "mixture_call_segment_end_sec": metrics["placement_end_sec"],
            "donor_input_row": int(donor.name),
            "donor_clip_path": str(donor_path),
            "donor_soundfile": donor.get("Soundfile", ""),
            "donor_provider": donor.get("Provider", ""),
            "donor_dataset": donor.get("Dataset", ""),
            "donor_domain": donor_domain,
            "background_input_row": int(background.name),
            "background_clip_path": str(background_path),
            "background_soundfile": background.get("Soundfile", ""),
            "background_provider": background.get("Provider", ""),
            "background_dataset": background.get("Dataset", ""),
            "background_domain": background_domain,
            "signal_band_rms_before_gain": metrics["signal_band_rms_before_gain"],
            "background_band_rms": metrics["background_band_rms"],
        }
    )
    return row


def atomic_write_audio(path: Path, waveform: np.ndarray, sample_rate: int, audio_format: str) -> None:
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    subtype = "PCM_16" if audio_format == "wav" else None
    sf.write(temporary, waveform, sample_rate, subtype=subtype)
    os.replace(temporary, path)


def write_outputs(
    manifest_path: Path,
    audit_path: Path,
    rows: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> None:
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)


def write_kaggle_metadata(output_dir: Path, dataset_id: str, title: str, license_name: str) -> None:
    if dataset_id.count("/") != 1:
        raise ValueError("--kaggle-dataset-id must have the form owner/dataset-slug")
    metadata = {"title": title, "id": dataset_id, "licenses": [{"name": license_name}]}
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite a clip_path prefix; may be supplied more than once.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--num-examples", type=int, default=30000)
    parser.add_argument(
        "--species-counts",
        default=None,
        help="Optional exact counts, for example KW=10000,HW=10000,AB=10000.",
    )
    parser.add_argument("--domain-columns", default="Provider,Dataset")
    parser.add_argument(
        "--different-domain-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require background domain to differ from donor domain (default: true).",
    )
    parser.add_argument(
        "--annotation-levels",
        default=None,
        help="Optional comma-separated AnnotationLevel allow-list.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--snr-db-min", type=float, default=-12.0)
    parser.add_argument("--snr-db-max", type=float, default=12.0)
    parser.add_argument("--time-margin-sec", type=float, default=0.15)
    parser.add_argument("--frequency-margin-hz", type=float, default=100.0)
    parser.add_argument("--stft-n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--peak-limit", type=float, default=0.99)
    parser.add_argument("--audio-format", choices=("wav", "flac"), default="wav")
    parser.add_argument("--audio-cache-items", type=int, default=256)
    parser.add_argument("--max-attempts-per-example", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-input-rows", type=int, default=None)
    parser.add_argument("--kaggle-dataset-id", default=DEFAULT_KAGGLE_DATASET_ID)
    parser.add_argument("--kaggle-title", default="DCLDE Controlled Background Mixtures")
    parser.add_argument("--kaggle-license", default="CC0-1.0")
    parser.add_argument(
        "--publish-action",
        choices=("none", "create", "version"),
        default="none",
        help="Optionally run the Kaggle CLI after generation (default: none).",
    )
    parser.add_argument("--version-message", default="Update controlled background mixtures")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_examples < 1:
        raise ValueError("--num-examples must be positive")
    if args.sample_rate < 1000 or args.clip_duration <= 0:
        raise ValueError("--sample-rate and --clip-duration must be positive")
    if args.snr_db_min > args.snr_db_max:
        raise ValueError("--snr-db-min cannot exceed --snr-db-max")
    if args.time_margin_sec < 0 or args.frequency_margin_hz < 0:
        raise ValueError("Time/frequency margins cannot be negative")
    if args.stft_n_fft < 64 or args.stft_hop_length < 1:
        raise ValueError("Invalid STFT dimensions")
    if args.stft_hop_length >= args.stft_n_fft:
        raise ValueError("--stft-hop-length must be smaller than --stft-n-fft")
    if not 0 < args.peak_limit <= 1:
        raise ValueError("--peak-limit must be in (0, 1]")
    if args.audio_cache_items < 1 or args.max_attempts_per_example < 1:
        raise ValueError("Cache size and maximum attempts must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = Path(args.output_manifest) if args.output_manifest else output_dir / "controlled_mixture_manifest.csv"
    audit_path = output_dir / "generation_audit.csv"
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    rewrites = parse_path_rewrites(args.path_rewrite)
    domain_columns = parse_csv_values(args.domain_columns)
    species_counts = parse_species_counts(args.species_counts, args.num_examples)
    annotation_levels = (
        {value.casefold() for value in parse_csv_values(args.annotation_levels)}
        if args.annotation_levels
        else None
    )

    frame = pd.read_csv(manifest_path, low_memory=False)
    required = {"clip_path", "ClassSpecies", "Ecotype", *domain_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")
    if args.max_input_rows is not None:
        if args.max_input_rows < 1:
            raise ValueError("--max-input-rows must be positive")
        frame = frame.iloc[: args.max_input_rows].copy()

    donors, backgrounds_by_domain, audit = eligible_rows(
        frame,
        domain_columns,
        args.clip_duration,
        args.sample_rate,
        args.frequency_margin_hz,
        annotation_levels,
    )
    for species, count in species_counts.items():
        if count and donors[species].empty:
            raise ValueError(f"No eligible annotated {species} donor rows")
    if not backgrounds_by_domain:
        raise ValueError("No BKG or UndBio background rows were found")

    schedule = [species for species, count in species_counts.items() for _ in range(count)]
    np.random.default_rng(args.seed).shuffle(schedule)
    existing_rows: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    if output_manifest.is_file() and not args.no_resume:
        existing = pd.read_csv(output_manifest, low_memory=False)
        existing_rows = existing.to_dict("records")
        if "mixture_index" in existing.columns:
            completed_indices = set(existing["mixture_index"].dropna().astype(int))
    rows = existing_rows
    cache = AudioCache(args.sample_rate, args.clip_duration, args.audio_cache_items)
    all_background_domains = sorted(backgrounds_by_domain)
    published_root = f"/kaggle/input/datasets/{args.kaggle_dataset_id}/clips"
    failures = Counter()

    print("\nControlled call/background mixture generation")
    print("=" * 72)
    print(f"Input manifest:        {manifest_path}")
    print(f"Eligible donors:       " + ", ".join(f"{key}={len(value):,}" for key, value in donors.items()))
    print(f"Background clips:      {sum(map(len, backgrounds_by_domain.values())):,}")
    print(f"Background domains:    {len(backgrounds_by_domain):,}")
    print(f"Requested mixtures:    {len(schedule):,} ({species_counts})")
    print(f"Target SNR range:      {args.snr_db_min:g}..{args.snr_db_max:g} dB")
    print(f"Different domain:      {args.different_domain_background}")
    print(f"Already completed:     {len(completed_indices):,}")
    print("=" * 72 + "\n")

    for mixture_index, species in enumerate(schedule):
        if mixture_index in completed_indices:
            continue
        success = False
        example_rng = np.random.default_rng(np.random.SeedSequence([args.seed, mixture_index]))
        donor_frame = donors[species]
        for attempt in range(1, args.max_attempts_per_example + 1):
            try:
                donor = donor_frame.iloc[int(example_rng.integers(0, len(donor_frame)))]
                donor_domain = canonical_domain(donor, domain_columns)
                allowed_domains = [
                    domain
                    for domain in all_background_domains
                    if not args.different_domain_background or domain != donor_domain
                ]
                if not allowed_domains:
                    raise ValueError("no_background_domain_after_exclusion")
                background_domain = allowed_domains[int(example_rng.integers(0, len(allowed_domains)))]
                background_indices = backgrounds_by_domain[background_domain]
                background = frame.loc[
                    background_indices[int(example_rng.integers(0, len(background_indices)))]
                ]
                if normalize_text(background.get("Soundfile")) == normalize_text(donor.get("Soundfile")):
                    raise ValueError("donor_and_background_share_soundfile")
                donor_path = resolve_audio_path(donor.get("clip_path"), dataset_root, rewrites)
                background_path = resolve_audio_path(background.get("clip_path"), dataset_root, rewrites)
                donor_audio = cache.load(donor_path)
                background_audio = cache.load(background_path)
                event_start, event_end, low_hz, high_hz = annotation_geometry(
                    donor,
                    args.clip_duration,
                    args.sample_rate,
                    args.frequency_margin_hz,
                )
                signal_segment, event_offset_start, event_offset_end = isolate_annotated_call(
                    donor_audio,
                    args.sample_rate,
                    event_start,
                    event_end,
                    low_hz,
                    high_hz,
                    args.time_margin_sec,
                    args.stft_n_fft,
                    args.stft_hop_length,
                )
                target_snr_db = float(example_rng.uniform(args.snr_db_min, args.snr_db_max))
                mixture, metrics = mix_call_and_background(
                    signal_segment,
                    event_offset_start,
                    event_offset_end,
                    background_audio,
                    args.sample_rate,
                    low_hz,
                    high_hz,
                    target_snr_db,
                    example_rng,
                    args.peak_limit,
                )
                extension = ".wav" if args.audio_format == "wav" else ".flac"
                filename = f"mix_{mixture_index:06d}_{species.lower()}_snr{target_snr_db:+05.1f}db{extension}"
                local_path = clips_dir / filename
                atomic_write_audio(local_path, mixture, args.sample_rate, args.audio_format)
                published_path = f"{published_root}/{filename}"
                rows.append(
                    output_row(
                        donor,
                        background,
                        mixture_index,
                        filename,
                        local_path,
                        published_path,
                        donor_path,
                        background_path,
                        donor_domain,
                        background_domain,
                        low_hz,
                        high_hz,
                        metrics,
                        args.seed,
                        args.clip_duration,
                    )
                )
                audit.append(
                    {
                        "mixture_index": mixture_index,
                        "species": species,
                        "attempt": attempt,
                        "status": "saved",
                        "filename": filename,
                    }
                )
                success = True
                break
            except Exception as error:  # Continue past individual corrupt/missing clips.
                reason = f"{type(error).__name__}: {error}"
                failures[reason] += 1
                audit.append(
                    {
                        "mixture_index": mixture_index,
                        "species": species,
                        "attempt": attempt,
                        "status": "retry",
                        "reason": reason,
                    }
                )
        if not success:
            audit.append(
                {
                    "mixture_index": mixture_index,
                    "species": species,
                    "status": "failed",
                    "reason": "maximum_attempts_exhausted",
                }
            )
        if (mixture_index + 1) % args.checkpoint_every == 0:
            write_outputs(output_manifest, audit_path, rows, audit)
            print(f"Processed {mixture_index + 1:,}/{len(schedule):,}; saved {len(rows):,}")

    write_outputs(output_manifest, audit_path, rows, audit)
    write_kaggle_metadata(output_dir, args.kaggle_dataset_id, args.kaggle_title, args.kaggle_license)
    saved_counts = Counter(normalize_text(row.get("ClassSpecies")) for row in rows)
    summary = {
        "input_manifest": str(manifest_path),
        "requested_species_counts": species_counts,
        "saved_species_counts": dict(saved_counts),
        "saved_total": len(rows),
        "failed_examples": sum(1 for row in audit if row.get("status") == "failed"),
        "retry_reasons": dict(failures.most_common()),
        "arguments": vars(args),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\nGeneration complete")
    print("=" * 72)
    print(f"Saved mixtures:        {len(rows):,} ({dict(saved_counts)})")
    print(f"Manifest:              {output_manifest}")
    print(f"Audit:                 {audit_path}")
    print(f"Kaggle metadata:       {output_dir / 'dataset-metadata.json'}")
    print("Review the audit and listen to a sample from every species/SNR range before training.")

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
            f"  kaggle datasets version -p {output_dir} "
            f"-m \"{args.version_message}\" --dir-mode zip"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
