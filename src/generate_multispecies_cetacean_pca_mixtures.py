#!/usr/bin/env python3
"""Build leak-safe PCA vocalization/background mixtures for the V2 dataset.

The script discovers builder-created ``multispecies_cetacean_manifest.csv``
files, uses annotated TRAIN clips as foreground donors, and uses only ambient
TRAIN clips (``clip_kind=background``) as backgrounds.  A low-rank PCA model
estimates stationary donor noise in the STFT; high positive residuals inside
the annotation time/frequency rectangle form the vocalization mask.  The
isolated foreground is placed in a background clip from another recording and
scaled to a random, band-limited SNR.

Output is a standalone Kaggle-dataset directory containing a generated-only
trainer manifest and lossless 16 kHz FLAC files.  Do not concatenate the input
manifests into the output: ``train_multispecies_cetacean_model.py`` discovers
the original and synthetic Kaggle datasets together.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, istft, resample_poly, sosfiltfilt, stft


MANIFEST_NAME = "multispecies_cetacean_manifest.csv"
DEFAULT_OUTPUT_DIR = "/kaggle/working/multispecies_cetacean_pca_mixtures"
DEFAULT_DATASET_ID = "leonisviridis/multispecies-cetacean-pca-mixtures"
VALID_SOURCE_LABELS = {"Abiotic", "KW", "HW", "UndBio"}
EPSILON = 1e-12
_PROJECTIONS: dict[tuple[int, int], np.ndarray] = {}


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "n/a"} else text


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def truthy(value: Any) -> bool:
    return clean(value).casefold() in {"1", "true", "yes", "y"}


def comma_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_counts(value: str | None, labels: list[str], total: int) -> dict[str, int]:
    if value is None:
        base, remainder = divmod(total, len(labels))
        return {label: base + int(index < remainder) for index, label in enumerate(labels)}
    counts = {label: 0 for label in labels}
    for assignment in value.split(","):
        if "=" not in assignment:
            raise ValueError("--label-counts must look like KW=10000,HW=10000,UndBio=10000")
        label, raw_count = (part.strip() for part in assignment.split("=", 1))
        if label not in counts:
            raise ValueError(f"Unknown donor label in --label-counts: {label!r}")
        counts[label] = int(raw_count)
    if any(count < 0 for count in counts.values()) or sum(counts.values()) < 1:
        raise ValueError("--label-counts must contain non-negative counts with a positive sum")
    return counts


def domain_key(row: pd.Series, columns: list[str]) -> str:
    return " | ".join(
        f"{column.casefold()}={clean(row.get(column)).casefold() or '<missing>'}"
        for column in columns
    )


class ClipReader:
    """Read a clip from either clips.zip or Kaggle's expanded clips directory."""

    def __init__(self, sample_rate: int, samples: int):
        self.sample_rate = sample_rate
        self.samples = samples
        self.archives: dict[Path, zipfile.ZipFile] = {}

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()

    def _read(self, row: pd.Series) -> tuple[np.ndarray, int]:
        manifest_dir = Path(clean(row["_manifest_dir"]))
        member = clean(row.get("archive_member_path")).replace("\\", "/")
        archive_name = clean(row.get("archive_path")) or "clips.zip"
        archive_path = (manifest_dir / archive_name).resolve()
        if archive_path.is_file():
            archive = self.archives.get(archive_path)
            if archive is None:
                archive = zipfile.ZipFile(archive_path)
                self.archives[archive_path] = archive
            with archive.open(member) as source:
                payload = source.read()
            waveform, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            return waveform, int(rate)

        for directory in (archive_path, archive_path.with_suffix("")):
            direct = directory.joinpath(*Path(member).parts)
            if direct.is_file():
                waveform, rate = sf.read(direct, dtype="float32", always_2d=True)
                return waveform, int(rate)

        for column in ("relative_clip_path", "clip_path"):
            candidate = clean(row.get(column)).replace("\\", "/")
            if candidate:
                direct = manifest_dir.joinpath(*Path(candidate).parts)
                if direct.is_file():
                    waveform, rate = sf.read(direct, dtype="float32", always_2d=True)
                    return waveform, int(rate)
        raise FileNotFoundError(f"Missing audio for {clean(row.get('clip_id'))}: {archive_path} / {member}")

    def load(self, row: pd.Series) -> np.ndarray:
        waveform, rate = self._read(row)
        # Builder-created shards are mono. If an external compatible shard is
        # multichannel, use channel 0 to match the V2 extraction policy.
        audio = waveform[:, 0]
        if rate != self.sample_rate:
            divisor = math.gcd(rate, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // divisor, rate // divisor)
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio)))
        return np.asarray(audio[: self.samples], dtype=np.float32)


def randomized_low_rank_reconstruction(centered: np.ndarray, rank: int) -> np.ndarray:
    maximum_rank = min(centered.shape)
    if rank < 1 or rank > maximum_rank:
        raise ValueError(f"PCA rank must be in [1, {maximum_rank}]")
    projected_rank = min(maximum_rank, rank + 4)
    key = (centered.shape[1], projected_rank)
    projection = _PROJECTIONS.get(key)
    if projection is None:
        projection = np.random.default_rng(0).standard_normal(key)
        _PROJECTIONS[key] = projection
    basis = centered @ projection
    for _ in range(2):
        basis = centered @ (centered.T @ basis)
    basis, _ = np.linalg.qr(basis, mode="reduced")
    compressed = basis.T @ centered
    left, singular_values, right = np.linalg.svd(compressed, full_matrices=False)
    left = basis @ left[:, :rank]
    return (left * singular_values[:rank]) @ right[:rank]


def annotation_geometry(
    row: pd.Series, clip_seconds: float, sample_rate: int, frequency_margin_hz: float
) -> tuple[float, float, float, float]:
    clip_start = number(row.get("actual_clip_start_sec"))
    if not math.isfinite(clip_start):
        clip_start = number(row.get("clip_start_requested_sec"))
    if not math.isfinite(clip_start):
        clip_start = number(row.get("ClipStartSec"))
    event_start = number(row.get("FileBeginSec"))
    event_end = number(row.get("FileEndSec"))
    low_hz = number(row.get("LowFreqHz"))
    high_hz = number(row.get("HighFreqHz"))
    if not all(math.isfinite(value) for value in (clip_start, event_start, event_end, low_hz, high_hz)):
        raise ValueError("missing_annotation_geometry")
    relative_start = max(0.0, event_start - clip_start)
    relative_end = min(clip_seconds, event_end - clip_start)
    if relative_end <= relative_start:
        raise ValueError("annotation_outside_extracted_clip")
    nyquist = sample_rate / 2.0
    low_hz = max(20.0, low_hz - frequency_margin_hz)
    high_hz = min(nyquist - 20.0, high_hz + frequency_margin_hz)
    if high_hz <= low_hz:
        raise ValueError("invalid_frequency_bounds")
    return relative_start, relative_end, low_hz, high_hz


def isolate_vocalization(
    waveform: np.ndarray,
    sample_rate: int,
    event_start: float,
    event_end: float,
    low_hz: float,
    high_hz: float,
    time_margin_sec: float,
    n_fft: int,
    hop_length: int,
    percentile: float,
    pca_components: int,
) -> tuple[np.ndarray, float, float]:
    clip_seconds = len(waveform) / sample_rate
    segment_start = max(0.0, event_start - time_margin_sec)
    segment_end = min(clip_seconds, event_end + time_margin_sec)
    frequencies, times, spectrum = stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )
    rectangle = (
        (frequencies[:, None] >= low_hz)
        & (frequencies[:, None] <= high_hz)
        & (times[None, :] >= segment_start)
        & (times[None, :] <= segment_end)
    )
    if not np.any(rectangle):
        raise ValueError("empty_annotation_mask")

    magnitude = np.abs(spectrum).astype(np.float64)
    log_magnitude = np.log1p(magnitude / max(float(np.median(magnitude)), EPSILON))
    observations = log_magnitude.T
    mean_spectrum = observations.mean(axis=0, keepdims=True)
    centered = observations - mean_spectrum
    rank = min(pca_components, centered.shape[0] - 1, centered.shape[1])
    if rank < 1:
        raise ValueError("insufficient_stft_frames_for_pca")
    reconstruction = mean_spectrum + randomized_low_rank_reconstruction(centered, rank)
    residual = np.maximum(observations - reconstruction, 0.0).T
    candidates = residual[rectangle]
    candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
    if candidates.size == 0:
        raise ValueError("pca_mask_has_no_positive_residual")
    threshold = float(np.percentile(candidates, percentile))
    mask = ((residual >= threshold) & rectangle).astype(np.float32)
    sigma_frequency = max(0.5, 30.0 / (sample_rate / n_fft))
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
    isolated = np.pad(isolated, (0, max(0, len(waveform) - len(isolated))))[: len(waveform)]
    start_sample = int(round(segment_start * sample_rate))
    end_sample = int(round(segment_end * sample_rate))
    segment = np.asarray(isolated[start_sample:end_sample], dtype=np.float32)
    if segment.size == 0:
        raise ValueError("empty_isolated_segment")
    return segment, event_start - segment_start, event_end - segment_start


def band_rms(waveform: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if waveform.size < 16:
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


def make_mixture(
    signal: np.ndarray,
    event_offset_start: float,
    event_offset_end: float,
    background: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    target_snr_db: float,
    peak_limit: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(signal) > len(background):
        raise ValueError("isolated_signal_longer_than_background")
    placement = int(rng.integers(0, len(background) - len(signal) + 1))
    signal_end = placement + len(signal)
    event_start = max(placement, min(signal_end - 1, placement + round(event_offset_start * sample_rate)))
    event_end = max(event_start + 1, min(signal_end, placement + round(event_offset_end * sample_rate)))
    relative_start, relative_end = event_start - placement, event_end - placement
    signal_rms = band_rms(signal[relative_start:relative_end], sample_rate, low_hz, high_hz)
    background_rms = band_rms(background[event_start:event_end], sample_rate, low_hz, high_hz)
    if signal_rms <= EPSILON or background_rms <= EPSILON:
        raise ValueError("zero_band_energy")
    gain = background_rms * (10.0 ** (target_snr_db / 20.0)) / signal_rms
    track = np.zeros_like(background)
    track[placement:signal_end] = signal * gain
    mixture = background.astype(np.float64) + track.astype(np.float64)
    peak = float(np.max(np.abs(mixture)))
    peak_scale = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    mixture = np.asarray(mixture * peak_scale, dtype=np.float32)
    return mixture, {
        "target_snr_db": target_snr_db,
        "measured_snr_db": 20.0 * math.log10(max(signal_rms * gain, EPSILON) / background_rms),
        "signal_gain": gain,
        "peak_scale": peak_scale,
        "mixed_event_start_sec": event_start / sample_rate,
        "mixed_event_end_sec": event_end / sample_rate,
        "signal_band_rms": signal_rms,
        "background_band_rms": background_rms,
    }


def discover_rows(roots: list[Path], manifest_name: str) -> tuple[pd.DataFrame, list[Path]]:
    manifests = sorted({path.resolve() for root in roots for path in root.rglob(manifest_name)})
    if not manifests:
        raise FileNotFoundError(f"No {manifest_name} files below {roots}")
    frames: list[pd.DataFrame] = []
    for manifest in manifests:
        frame = pd.read_csv(manifest, low_memory=False)
        required = {
            "clip_id", "clip_kind", "split", "model_source_label", "archive_path",
            "archive_member_path", "source_recording_id", "Provider", "Dataset",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{manifest} missing required columns: {sorted(missing)}")
        frame = frame.loc[frame["split"].fillna("").astype(str).str.casefold().eq("train")].copy()
        if frame.empty:
            continue
        frame["_manifest_path"] = str(manifest)
        frame["_manifest_dir"] = str(manifest.parent)
        frames.append(frame)
    if not frames:
        raise ValueError("No training rows were found")
    result = pd.concat(frames, ignore_index=True, sort=False)
    duplicated = result["clip_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, "clip_id"].astype(str).head(5).tolist()
        raise ValueError(f"Duplicate clip_id values across input shards: {examples}")
    return result, manifests


def load_overlap_actions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    required = {"annotation_id", "recommended_source_action", "recommended_ecotype_action"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return {
        clean(row["annotation_id"]): {
            "source": clean(row["recommended_source_action"]).casefold(),
            "ecotype": clean(row["recommended_ecotype_action"]).casefold(),
        }
        for _, row in frame.iterrows()
    }


def output_row(
    donor: pd.Series,
    background: pd.Series,
    mixture_index: int,
    filename: str,
    metrics: dict[str, float],
    low_hz: float,
    high_hz: float,
    seed: int,
    sample_rate: int,
    sample_count: int,
) -> dict[str, Any]:
    label = clean(donor.get("model_source_label"))
    ecotype = clean(donor.get("clean_ecotype")) if label == "KW" else ""
    clip_id = f"pca_mix_{mixture_index:07d}"
    return {
        "clip_id": clip_id,
        "clip_kind": "synthetic_pca_mixture",
        "negative_subtype": "",
        "model_source_label": label,
        "clean_class_species": clean(donor.get("clean_class_species")),
        "clean_ecotype": ecotype,
        "source_head_eligible": "TRUE",
        "ecotype_head_eligible": "TRUE" if label == "KW" and ecotype else "FALSE",
        "split": "train",
        "storage_key": "synthetic_pca_mixtures",
        "archive_path": "clips.zip",
        "archive_member_path": filename,
        "relative_clip_path": f"clips/{filename}",
        "clip_path": f"clips/{filename}",
        "Provider": clean(background.get("Provider")),
        "Dataset": clean(background.get("Dataset")),
        "source_recording_id": f"synthetic|{clean(donor.get('source_recording_id'))}|{clean(background.get('source_recording_id'))}",
        "AnnotationLevel": "PcaVocalizationMaskMixture",
        "FileBeginSec": metrics["mixed_event_start_sec"],
        "FileEndSec": metrics["mixed_event_end_sec"],
        "LowFreqHz": low_hz,
        "HighFreqHz": high_hz,
        "target_sample_rate": sample_rate,
        "target_sample_count": sample_count,
        "Generated": "TRUE",
        "mixture_index": mixture_index,
        "mixture_seed": seed,
        "mixture_method": "pca_percentile_vocalization_mask_controlled_snr",
        "mixture_target_snr_db": metrics["target_snr_db"],
        "mixture_measured_snr_db": metrics["measured_snr_db"],
        "mixture_signal_gain": metrics["signal_gain"],
        "mixture_peak_scale": metrics["peak_scale"],
        "mixture_snr_band_low_hz": low_hz,
        "mixture_snr_band_high_hz": high_hz,
        "donor_clip_id": clean(donor.get("clip_id")),
        "donor_source_recording_id": clean(donor.get("source_recording_id")),
        "donor_provider": clean(donor.get("Provider")),
        "donor_dataset": clean(donor.get("Dataset")),
        "background_clip_id": clean(background.get("clip_id")),
        "background_source_recording_id": clean(background.get("source_recording_id")),
        "background_provider": clean(background.get("Provider")),
        "background_dataset": clean(background.get("Dataset")),
        "signal_band_rms_before_gain": metrics["signal_band_rms"],
        "background_band_rms": metrics["background_band_rms"],
    }


def atomic_flac(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    temporary = path.with_suffix(".partial.flac")
    sf.write(temporary, waveform, sample_rate, format="FLAC", subtype="PCM_16")
    temporary.replace(path)


def write_metadata(output_dir: Path, dataset_id: str, title: str, license_name: str) -> None:
    if dataset_id.count("/") != 1:
        raise ValueError("--kaggle-dataset-id must be owner/dataset-slug")
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
        "subtitle": "PCA vocalization-mask mixtures for multispecies cetacean training",
        "description": (
            "Synthetic train-only 3-second clips made from annotated vocalization masks "
            "and ambient backgrounds drawn from different source recordings."
        ),
    }
    (output_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", default=None, help="Root searched recursively; repeatable.")
    parser.add_argument("--manifest-name", default=MANIFEST_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--overlap-audit-csv", default=None)
    parser.add_argument("--donor-labels", default="KW,HW,UndBio")
    parser.add_argument("--num-mixtures", type=int, default=30000)
    parser.add_argument("--label-counts", default=None)
    parser.add_argument("--domain-columns", default="Provider,Dataset")
    parser.add_argument("--require-different-domain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--snr-db-min", type=float, default=-12.0)
    parser.add_argument("--snr-db-max", type=float, default=12.0)
    parser.add_argument("--time-margin-sec", type=float, default=0.15)
    parser.add_argument("--frequency-margin-hz", type=float, default=100.0)
    parser.add_argument("--stft-n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--mask-percentile", type=float, default=95.0)
    parser.add_argument("--pca-components", type=int, default=1)
    parser.add_argument("--peak-limit", type=float, default=0.99)
    parser.add_argument("--max-attempts-per-mixture", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-input-rows", type=int, default=None, help="Small randomized input subset for testing.")
    parser.add_argument("--kaggle-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--kaggle-title", default="Multispecies Cetacean PCA Vocalization Mixtures")
    parser.add_argument("--kaggle-license", default="CC0-1.0")
    parser.add_argument("--publish-action", choices=("none", "create", "version"), default="none")
    parser.add_argument("--version-message", default="Update PCA vocalization mixtures")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_mixtures < 1 or args.sample_rate < 1000 or args.clip_seconds <= 0:
        raise ValueError("Mixture count, sample rate, and clip duration must be positive")
    if args.snr_db_min > args.snr_db_max:
        raise ValueError("--snr-db-min cannot exceed --snr-db-max")
    if not 0 <= args.mask_percentile < 100 or args.pca_components < 1:
        raise ValueError("Invalid PCA mask settings")
    if args.stft_hop_length < 1 or args.stft_hop_length >= args.stft_n_fft:
        raise ValueError("Invalid STFT dimensions")
    if not 0 < args.peak_limit <= 1:
        raise ValueError("--peak-limit must be in (0, 1]")
    if args.max_attempts_per_mixture < 1 or args.checkpoint_every < 1:
        raise ValueError("Attempt and checkpoint counts must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    roots = [Path(value) for value in (args.data_root or ["/kaggle/input"])]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_manifest) if args.output_manifest else output_dir / MANIFEST_NAME
    audit_path = output_dir / "pca_mixture_generation_audit.csv"

    frame, manifests = discover_rows(roots, args.manifest_name)
    if args.max_input_rows is not None:
        if args.max_input_rows < 1:
            raise ValueError("--max-input-rows must be positive")
        frame = frame.sample(n=min(args.max_input_rows, len(frame)), random_state=args.seed).reset_index(drop=True)

    labels = comma_values(args.donor_labels)
    invalid = set(labels) - VALID_SOURCE_LABELS
    if invalid or "Abiotic" in labels:
        raise ValueError(f"Donors must be biological labels from KW,HW,UndBio; got {labels}")
    counts = parse_counts(args.label_counts, labels, args.num_mixtures)
    schedule = [label for label, count in counts.items() for _ in range(count)]
    np.random.default_rng(args.seed).shuffle(schedule)
    domains = comma_values(args.domain_columns)
    missing_domains = set(domains) - set(frame.columns)
    if missing_domains:
        raise ValueError(f"Input manifests lack domain columns: {sorted(missing_domains)}")

    actions = load_overlap_actions(Path(args.overlap_audit_csv) if args.overlap_audit_csv else None)
    annotated = frame["clip_kind"].fillna("").astype(str).str.casefold().eq("annotated")
    donor_mask = annotated & frame["model_source_label"].isin(labels)
    if "source_head_eligible" in frame:
        donor_mask &= frame["source_head_eligible"].map(truthy)
    if "ecotype_head_eligible" in frame:
        donor_mask &= ~frame["model_source_label"].eq("KW") | frame["ecotype_head_eligible"].map(truthy)
    if actions:
        safe = frame["clip_id"].map(
            lambda clip_id: actions.get(clean(clip_id), {}).get("source", "keep") == "keep"
        )
        kw_safe = frame.apply(
            lambda row: clean(row.get("model_source_label")) != "KW"
            or actions.get(clean(row.get("clip_id")), {}).get("ecotype", "keep") == "keep",
            axis=1,
        )
        donor_mask &= safe & kw_safe
    donors = {label: frame.loc[donor_mask & frame["model_source_label"].eq(label)].copy() for label in labels}
    backgrounds = frame.loc[
        frame["clip_kind"].fillna("").astype(str).str.casefold().eq("background")
        & frame["model_source_label"].eq("Abiotic")
    ].copy()
    if backgrounds.empty:
        raise ValueError("No ambient training backgrounds (clip_kind=background) were found")
    for label, count in counts.items():
        if count and donors[label].empty:
            raise ValueError(f"No eligible {label} donor rows")
    frame_domains = frame.apply(lambda row: domain_key(row, domains), axis=1)
    for label in labels:
        donors[label]["_domain"] = frame_domains.loc[donors[label].index]
    backgrounds["_domain"] = frame_domains.loc[backgrounds.index]

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    completed: set[int] = set()
    if manifest_path.is_file() and not args.no_resume:
        existing = pd.read_csv(manifest_path, low_memory=False)
        rows = existing.to_dict("records")
        completed = set(existing["mixture_index"].dropna().astype(int)) if "mixture_index" in existing else set()
    if audit_path.is_file() and not args.no_resume:
        audit = pd.read_csv(audit_path, low_memory=False).to_dict("records")

    print("\nMultispecies cetacean PCA vocalization mixtures")
    print("=" * 72)
    print(f"Input manifests:       {len(manifests):,}")
    print(f"Training rows:         {len(frame):,}")
    print("Eligible donors:       " + ", ".join(f"{label}={len(donors[label]):,}" for label in labels))
    print(f"Ambient backgrounds:   {len(backgrounds):,}")
    print(f"Requested mixtures:    {len(schedule):,} ({counts})")
    print(f"Target SNR:            {args.snr_db_min:g}..{args.snr_db_max:g} dB")
    print(f"PCA mask:              components={args.pca_components}, percentile={args.mask_percentile:g}")
    print(f"Different domain:      {args.require_different_domain}")
    print(f"Previously completed:  {len(completed):,}")
    print("=" * 72)

    reader = ClipReader(args.sample_rate, round(args.sample_rate * args.clip_seconds))
    failures: Counter[str] = Counter()
    started = time.monotonic()
    try:
        for mixture_index, label in enumerate(schedule):
            if mixture_index in completed:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, mixture_index]))
            donor_pool = donors[label]
            success = False
            for attempt in range(1, args.max_attempts_per_mixture + 1):
                try:
                    donor = donor_pool.iloc[int(rng.integers(len(donor_pool)))]
                    eligible_backgrounds = backgrounds.loc[
                        backgrounds["source_recording_id"].astype(str).ne(clean(donor.get("source_recording_id")))
                    ]
                    if args.require_different_domain:
                        eligible_backgrounds = eligible_backgrounds.loc[
                            eligible_backgrounds["_domain"].ne(clean(donor.get("_domain")))
                        ]
                    if eligible_backgrounds.empty:
                        raise ValueError("no_eligible_background")
                    background = eligible_backgrounds.iloc[int(rng.integers(len(eligible_backgrounds)))]
                    donor_audio = reader.load(donor)
                    background_audio = reader.load(background)
                    event_start, event_end, low_hz, high_hz = annotation_geometry(
                        donor, args.clip_seconds, args.sample_rate, args.frequency_margin_hz
                    )
                    signal, offset_start, offset_end = isolate_vocalization(
                        donor_audio, args.sample_rate, event_start, event_end, low_hz, high_hz,
                        args.time_margin_sec, args.stft_n_fft, args.stft_hop_length,
                        args.mask_percentile, args.pca_components,
                    )
                    target_snr = float(rng.uniform(args.snr_db_min, args.snr_db_max))
                    mixture, metrics = make_mixture(
                        signal, offset_start, offset_end, background_audio, args.sample_rate,
                        low_hz, high_hz, target_snr, args.peak_limit, rng,
                    )
                    filename = f"pca_mix_{mixture_index:07d}_{label.casefold()}_snr{target_snr:+05.1f}db.flac"
                    atomic_flac(clips_dir / filename, mixture, args.sample_rate)
                    rows.append(
                        output_row(
                            donor,
                            background,
                            mixture_index,
                            filename,
                            metrics,
                            low_hz,
                            high_hz,
                            args.seed,
                            args.sample_rate,
                            round(args.sample_rate * args.clip_seconds),
                        )
                    )
                    audit.append({"mixture_index": mixture_index, "label": label, "attempt": attempt, "status": "saved", "filename": filename})
                    success = True
                    break
                except Exception as error:
                    reason = f"{type(error).__name__}: {error}"
                    failures[reason] += 1
                    audit.append({"mixture_index": mixture_index, "label": label, "attempt": attempt, "status": "retry", "reason": reason})
            if not success:
                audit.append({"mixture_index": mixture_index, "label": label, "status": "failed", "reason": "maximum_attempts_exhausted"})
            if (mixture_index + 1) % args.checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(manifest_path, index=False)
                pd.DataFrame(audit).to_csv(audit_path, index=False)
                elapsed = time.monotonic() - started
                rate = (mixture_index + 1 - len(completed)) / max(elapsed, 1e-6)
                remaining = (len(schedule) - mixture_index - 1) / max(rate, 1e-6)
                print(f"{mixture_index + 1:,}/{len(schedule):,}; saved={len(rows):,}; ETA={remaining / 60:.1f} min")
    finally:
        reader.close()

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    write_metadata(output_dir, args.kaggle_dataset_id, args.kaggle_title, args.kaggle_license)
    summary = {
        "input_manifests": [str(path) for path in manifests],
        "input_training_rows": len(frame),
        "eligible_donors": {label: len(donors[label]) for label in labels},
        "ambient_backgrounds": len(backgrounds),
        "requested_label_counts": counts,
        "saved_label_counts": dict(Counter(clean(row.get("model_source_label")) for row in rows)),
        "saved_total": len(rows),
        "failed_mixtures": sum(1 for row in audit if row.get("status") == "failed"),
        "retry_reasons": dict(failures.most_common()),
        "arguments": vars(args),
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nGeneration complete")
    print(f"Saved mixtures:        {len(rows):,}")
    print(f"Trainer manifest:      {manifest_path}")
    print(f"Audit:                 {audit_path}")
    print("This is a generated-only manifest; keep the original shard datasets attached during training.")
    if args.publish_action == "create":
        subprocess.run(["kaggle", "datasets", "create", "-p", str(output_dir), "--dir-mode", "zip"], check=True)
    elif args.publish_action == "version":
        subprocess.run(["kaggle", "datasets", "version", "-p", str(output_dir), "-m", args.version_message, "--dir-mode", "zip"], check=True)
    else:
        print(f"kaggle datasets create -p {output_dir} --dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
