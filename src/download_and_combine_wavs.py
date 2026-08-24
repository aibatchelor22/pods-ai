#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

"""Download DCLDE validation clips and build one path-based test manifest.

This script is designed for Google Colab. It combines an existing Orcasound
test collection with DCLDE clips described by the no-mixed-label validation
manifest. The resulting CSV contains an explicit WAV path for every sample,
so evaluation code does not need to infer paths from filenames or directories.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


DEFAULT_PRIMARY_MANIFEST = (
    "/content/pods-ai/output/csv/testing_60s_samples.csv"
)
DEFAULT_PRIMARY_WAV_ROOT = "/content/pods-ai/src/output/wav"
DEFAULT_SECONDARY_MANIFEST = (
    "/content/pods-ai/output/csv/"
    "orcasound_60s_validation_manifest_no_mixed_extracted.csv"
)
DEFAULT_SECONDARY_OUTPUT_DIR = (
    "/content/pods-ai/src/output/dclde_60s_validation"
)
DEFAULT_COMBINED_MANIFEST = (
    "/content/pods-ai/src/output/csv/combined_60s_evaluation_manifest.csv"
)

BACKGROUND_LABEL = "other/background"
TARGET_LABELS = {"resident", "transient", "humpback"}
COMBINED_FIELDS = [
    "wav_path",
    "label",
    "original_label",
    "source_dataset",
    "source_manifest",
    "source_row_index",
    "node_name",
    "timestamp",
    "uri",
    "description",
    "notes",
    "source_audio_path",
]


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_evaluation_label(value: object) -> str:
    """Map source labels into the evaluator's four comparison classes."""
    label = _clean(value).casefold()
    aliases = {
        "resident": "resident",
        "srkw": "resident",
        "southern resident": "resident",
        "transient": "transient",
        "tkw": "transient",
        "bigg's": "transient",
        "biggs": "transient",
        "humpback": "humpback",
        "hw": "humpback",
        "other": BACKGROUND_LABEL,
        "background": BACKGROUND_LABEL,
        "bkg": BACKGROUND_LABEL,
        BACKGROUND_LABEL: BACKGROUND_LABEL,
    }
    return aliases.get(label, BACKGROUND_LABEL)


def primary_wav_path(row: dict[str, str], wav_root: Path) -> Path:
    """Reproduce the filename used for the original Orcasound collection."""
    category = _clean(row.get("Category"))
    node_name = _clean(row.get("NodeName")).replace("_", "-")
    timestamp = _clean(row.get("Timestamp"))
    return wav_root / category / f"{node_name}_{timestamp}.wav"


def safe_slug(value: object, max_len: int = 140) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", _clean(value))
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len]


def secondary_wav_path(row: dict[str, str], output_dir: Path) -> Path:
    relative_path = _clean(row.get("relative_clip_path"))
    if relative_path:
        # Treat manifest paths as POSIX paths even when this script is inspected
        # or tested on Windows.
        parts = [part for part in relative_path.replace("\\", "/").split("/") if part]
        if any(part == ".." for part in parts):
            raise ValueError(f"Unsafe relative_clip_path: {relative_path!r}")
        return output_dir.joinpath(*parts)

    primary_label = safe_slug(row.get("primary_label") or "UNKNOWN")
    clip_id = safe_slug(row.get("clip_id") or row.get("Soundfile"))
    if not clip_id:
        raise ValueError("DCLDE row has neither relative_clip_path nor a usable clip ID")
    return output_dir / "no_mixed_clips" / primary_label / f"{clip_id}.wav"


def require_columns(
    fieldnames: Optional[list[str]],
    required: set[str],
    manifest: Path,
) -> None:
    available = set(fieldnames or [])
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{manifest} is missing required columns: {missing}")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def download_primary_manifest_wavs(
    manifest: Path,
    wav_root: Path,
    cache_root: Optional[Path] = None,
) -> None:
    """Download only the testing samples listed in the primary manifest.

    Reuse download_wavs.py's testing-sample implementation so timestamps,
    filenames, cache lookup, and existing-file behavior stay consistent. The
    broader process_testing_csv() entry point is deliberately not called
    because it deletes WAVs that are not listed in the manifest.
    """
    from download_wavs import download_testing_sample, parse_csv

    rows = parse_csv(manifest)
    print(f"Primary download: {len(rows)} manifest rows")
    wav_root.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        print(
            f"[primary {index}/{len(rows)}] "
            f"{row.category} - {row.node_name} - {row.timestamp_pst}"
        )
        download_testing_sample(row, wav_root, cache_root=cache_root)


def find_gcs_copy_command() -> Optional[list[str]]:
    if shutil.which("gsutil"):
        return ["gsutil", "-q", "cp"]
    if shutil.which("gcloud"):
        return ["gcloud", "storage", "cp", "--quiet"]
    return None


def download_or_copy_source(source_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source_path.startswith("gs://"):
        command = find_gcs_copy_command()
        if command is None:
            raise RuntimeError(
                "Neither gsutil nor gcloud is available. In Colab, install the "
                "Google Cloud CLI or make gsutil available before running this script."
            )
        process = subprocess.run(
            [*command, source_path, str(destination)],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"GCS copy failed for {source_path}: "
                f"{process.stderr.strip()[:2000]}"
            )
    else:
        local_source = Path(source_path)
        if not local_source.is_file():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        shutil.copy2(local_source, destination)

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded source is missing or empty: {destination}")


def extract_wav(
    source_audio: Path,
    output_wav: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(source_audio),
            "-t",
            f"{duration_seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.strip()[:2000]}")
    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a usable WAV: {output_wav}")


def build_primary_rows(
    manifest: Path,
    wav_root: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    fieldnames, rows = read_manifest(manifest)
    require_columns(
        fieldnames,
        {"Category", "NodeName", "Timestamp"},
        manifest,
    )
    combined_rows = []
    missing_paths = []

    for index, row in enumerate(rows):
        wav_path = primary_wav_path(row, wav_root)
        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            missing_paths.append(str(wav_path))
            continue

        original_label = _clean(row.get("Category"))
        combined_rows.append(
            {
                "wav_path": str(wav_path.resolve()),
                "label": normalize_evaluation_label(original_label),
                "original_label": original_label,
                "source_dataset": "orcasound_testing_old",
                "source_manifest": str(manifest),
                "source_row_index": str(index),
                "node_name": _clean(row.get("NodeName")),
                "timestamp": _clean(row.get("Timestamp")),
                "uri": _clean(row.get("URI")),
                "description": _clean(row.get("Description")),
                "notes": _clean(row.get("Notes")),
                "source_audio_path": "",
            }
        )

    return combined_rows, missing_paths


def build_secondary_rows(
    manifest: Path,
    output_dir: Path,
    skip_existing: bool,
    max_clips: Optional[int],
) -> tuple[list[dict[str, str]], list[str]]:
    fieldnames, rows = read_manifest(manifest)
    require_columns(
        fieldnames,
        {
            "source_audio_path",
            "window_start_sec",
            "primary_label",
            "comparison_label",
        },
        manifest,
    )
    if max_clips is not None:
        rows = rows[:max_clips]

    combined_rows = []
    failures = []

    for index, row in enumerate(rows):
        try:
            output_wav = secondary_wav_path(row, output_dir)
            if not (skip_existing and output_wav.is_file() and output_wav.stat().st_size > 0):
                source_path = _clean(row.get("source_audio_path"))
                if not source_path:
                    raise ValueError("source_audio_path is empty")

                suffix = Path(source_path.split("?", 1)[0]).suffix or ".audio"
                with tempfile.TemporaryDirectory(prefix="dclde_audio_") as temp_dir:
                    cached_source = Path(temp_dir) / f"source{suffix}"
                    print(f"[{index + 1}/{len(rows)}] Downloading {source_path}")
                    download_or_copy_source(source_path, cached_source)

                    start = float(_clean(row.get("window_start_sec")) or 0.0)
                    end_text = _clean(row.get("window_end_sec"))
                    duration = max(0.001, float(end_text) - start) if end_text else 60.0
                    duration = min(duration, 60.0)
                    extract_wav(cached_source, output_wav, start, duration)
            else:
                print(f"[{index + 1}/{len(rows)}] Reusing {output_wav}")

            original_label = _clean(row.get("primary_label"))
            combined_rows.append(
                {
                    "wav_path": str(output_wav.resolve()),
                    "label": normalize_evaluation_label(row.get("comparison_label")),
                    "original_label": original_label,
                    "source_dataset": "dclde_orcasound_validation",
                    "source_manifest": str(manifest),
                    "source_row_index": str(index),
                    "node_name": "",
                    "timestamp": "",
                    "uri": "",
                    "description": _clean(row.get("Soundfile")),
                    "notes": _clean(row.get("label_source")),
                    "source_audio_path": _clean(row.get("source_audio_path")),
                }
            )
        except Exception as error:
            identifier = _clean(row.get("clip_id") or row.get("Soundfile") or index)
            message = f"DCLDE row {index} ({identifier}): {error}"
            print(f"WARNING: {message}", file=sys.stderr)
            failures.append(message)

    return combined_rows, failures


def write_combined_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COMBINED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download/extract the DCLDE 60-second validation clips in Colab "
            "and combine them with the existing Orcasound test WAVs."
        )
    )
    parser.add_argument("--primary-manifest", default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--primary-wav-root", default=DEFAULT_PRIMARY_WAV_ROOT)
    parser.add_argument(
        "--primary-cache-root",
        default=None,
        help=(
            "Optional existing WAV root used as a cache before downloading "
            "primary clips."
        ),
    )
    parser.add_argument(
        "--skip-primary-download",
        action="store_true",
        help="Do not run download_wavs.py's download process for the primary manifest.",
    )
    parser.add_argument("--secondary-manifest", default=DEFAULT_SECONDARY_MANIFEST)
    parser.add_argument("--secondary-output-dir", default=DEFAULT_SECONDARY_OUTPUT_DIR)
    parser.add_argument("--output-manifest", default=DEFAULT_COMBINED_MANIFEST)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download and overwrite DCLDE WAVs that already exist.",
    )
    parser.add_argument(
        "--max-secondary-clips",
        type=int,
        default=None,
        help="Process only the first N DCLDE rows (useful for a Colab smoke test).",
    )
    args = parser.parse_args()

    if args.max_secondary_clips is not None and args.max_secondary_clips <= 0:
        parser.error("--max-secondary-clips must be positive")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required but was not found")

    primary_manifest = Path(args.primary_manifest)
    secondary_manifest = Path(args.secondary_manifest)
    if not primary_manifest.is_file():
        parser.error(f"primary manifest not found: {primary_manifest}")
    if not secondary_manifest.is_file():
        parser.error(f"secondary manifest not found: {secondary_manifest}")

    primary_wav_root = Path(args.primary_wav_root)
    if not args.skip_primary_download:
        download_primary_manifest_wavs(
            primary_manifest,
            primary_wav_root,
            cache_root=(
                Path(args.primary_cache_root)
                if args.primary_cache_root is not None
                else None
            ),
        )

    primary_rows, missing_primary = build_primary_rows(
        primary_manifest,
        primary_wav_root,
    )
    print(
        f"Primary collection: {len(primary_rows)} usable WAVs, "
        f"{len(missing_primary)} missing"
    )
    for missing_path in missing_primary[:10]:
        print(f"WARNING: missing primary WAV: {missing_path}", file=sys.stderr)
    if len(missing_primary) > 10:
        print(
            f"WARNING: {len(missing_primary) - 10} additional primary WAVs are missing",
            file=sys.stderr,
        )

    secondary_rows, secondary_failures = build_secondary_rows(
        secondary_manifest,
        Path(args.secondary_output_dir),
        skip_existing=not args.no_skip_existing,
        max_clips=args.max_secondary_clips,
    )
    all_rows = [*primary_rows, *secondary_rows]
    output_manifest = Path(args.output_manifest)
    write_combined_manifest(output_manifest, all_rows)

    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print(f"\nCombined manifest: {output_manifest}")
    print(f"Total usable clips: {len(all_rows)}")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")
    print(f"DCLDE extraction failures: {len(secondary_failures)}")

    if missing_primary or secondary_failures:
        print(
            "The manifest contains all successfully located/extracted WAVs; "
            "review the warnings above for omitted rows.",
            file=sys.stderr,
        )
    return 0 if all_rows else 1


if __name__ == "__main__":
    sys.exit(main())
