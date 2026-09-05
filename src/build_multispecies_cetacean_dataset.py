#!/usr/bin/env python3
"""Extract, validate, upload, and resume Multispecies Cetacean V2 shards.

The script consumes the CSV files produced by
``plan_multispecies_cetacean_dataset.py``. It processes one source recording at
a time, uses HTTP seeking for oversized public GCS objects, creates canonical
mono 16 kHz PCM16 FLAC clips, samples deterministic original background
windows, validates every output, and can upload each completed shard as a
private Kaggle dataset before deleting its local copy.
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
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MATCHED_STATUSES = {
    "matched",
    "matched_by_stem",
    "matched_by_dataset",
    "matched_by_dataset_and_stem",
}
MANIFEST_NAME = "multispecies_cetacean_manifest.csv"
QC_SUMMARY_NAME = "multispecies_cetacean_qc_summary.json"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "n/a"} else text


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_columns(
    rows: list[dict[str, str]], required: set[str], description: str
) -> None:
    if not rows:
        raise ValueError(f"{description} is empty")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{description} is missing required columns: {missing}")


def validate_plan_bundle(
    shard_rows: list[dict[str, str]],
    annotation_rows: list[dict[str, str]],
    background_rows: list[dict[str, str]],
) -> None:
    """Reject mixed-generation planner outputs before any audio is downloaded."""
    require_columns(
        shard_rows,
        {"shard_id", "clip_count", "kaggle_dataset_id", "dataset_title"},
        "shard summary",
    )
    require_columns(
        annotation_rows,
        {
            "annotation_id",
            "source_recording_id",
            "shard_id",
            "gcs_path",
            "extraction_mode",
            "relative_clip_path",
            "clip_start_requested_sec",
        },
        "extraction plan",
    )
    require_columns(
        background_rows,
        {
            "source_recording_id",
            "shard_id",
            "gcs_path",
            "extraction_mode",
            "requested_window_count",
            "exclusion_intervals_json",
            "random_seed",
        },
        "background plan",
    )

    shard_ids = [row["shard_id"] for row in shard_rows]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("shard summary contains duplicate shard_id values")
    known = set(shard_ids)
    unknown_annotations = sorted(
        {row["shard_id"] for row in annotation_rows}.difference(known)
    )
    unknown_background = sorted(
        {row["shard_id"] for row in background_rows}.difference(known)
    )
    if unknown_annotations or unknown_background:
        raise ValueError(
            "plan rows reference unknown shards; "
            f"extraction={unknown_annotations}, background={unknown_background}"
        )

    annotation_ids = [row["annotation_id"] for row in annotation_rows]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("extraction plan contains duplicate annotation_id values")

    annotations_per_shard = Counter(row["shard_id"] for row in annotation_rows)
    backgrounds_per_shard: Counter[str] = Counter()
    for row in background_rows:
        count = int(row["requested_window_count"])
        if count < 0:
            raise ValueError("background plan contains a negative requested window count")
        backgrounds_per_shard[row["shard_id"]] += count

    mismatches: list[str] = []
    for row in shard_rows:
        shard_id = row["shard_id"]
        summary_count = int(row["clip_count"])
        plan_count = annotations_per_shard[shard_id] + backgrounds_per_shard[shard_id]
        if plan_count != summary_count:
            mismatches.append(
                f"{shard_id}: summary={summary_count:,}, plans={plan_count:,} "
                f"({annotations_per_shard[shard_id]:,} annotated + "
                f"{backgrounds_per_shard[shard_id]:,} background)"
            )
    if mismatches:
        details = "\n  ".join(mismatches)
        raise ValueError(
            "Planner CSVs are inconsistent, usually because they came from "
            "different planner runs. Re-copy all three CSVs from the same final "
            f"plan directory. Mismatches:\n  {details}"
        )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def read_jsonl_latest(path: Path, key: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: ignoring malformed JSONL line {line_number} in {path}")
                continue
            identifier = clean_text(row.get(key))
            if identifier:
                result[identifier] = row
    return result


def write_csv_union(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    preferred = [
        "clip_id",
        "clip_kind",
        "negative_subtype",
        "model_source_label",
        "clean_class_species",
        "clean_ecotype",
        "source_head_eligible",
        "ecotype_head_eligible",
        "split",
        "storage_key",
        "archive_path",
        "archive_member_path",
        "relative_clip_path",
        "clip_path",
        "Provider",
        "Dataset",
        "Soundfile",
        "source_recording_id",
        "gcs_path",
        "actual_clip_start_sec",
        "clip_duration_sec",
        "output_sample_rate",
        "output_sample_count",
        "output_channels",
    ]
    all_fields: set[str] = set()
    for row in rows:
        all_fields.update(row)
    fields = [field for field in preferred if field in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_command(
    command: list[str],
    *,
    capture: bool = True,
    retries: int = 1,
    retry_delay: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, retries + 1):
        last = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=capture,
        )
        if last.returncode == 0:
            return last
        if attempt < retries:
            time.sleep(retry_delay * attempt)
    assert last is not None
    raise RuntimeError(
        f"Command failed ({last.returncode}): {' '.join(command)}\n"
        f"stdout: {last.stdout[-2000:]}\nstderr: {last.stderr[-4000:]}"
    )


def require_executable(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path)
    if resolved:
        return resolved
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Required executable not found: {name_or_path}")


def public_gcs_url(gcs_path: str) -> str:
    encoded = urllib.parse.quote(gcs_path, safe="/")
    return f"https://storage.googleapis.com/{encoded}"


def download_source(
    gcs_path: str,
    destination: Path,
    expected_size: int,
    timeout: int,
    retries: int,
    chunk_bytes: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = public_gcs_url(gcs_path)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(chunk_bytes)
                        if not chunk:
                            break
                        handle.write(chunk)
            actual_size = partial.stat().st_size
            if expected_size > 0 and actual_size != expected_size:
                raise IOError(
                    f"download size mismatch: expected {expected_size}, got {actual_size}"
                )
            os.replace(partial, destination)
            return destination
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt >= retries:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def probe_audio(ffprobe: str, source: str, retries: int) -> dict[str, object]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        source,
    ]
    result = run_command(command, retries=retries)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio stream found in {source}")
    stream = streams[0]
    duration = parse_float(stream.get("duration"))
    if duration is None:
        duration = parse_float(payload.get("format", {}).get("duration"))
    if duration is None or duration <= 0:
        raise RuntimeError(f"Could not determine positive duration for {source}")
    return {
        "duration_sec": duration,
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def expanded_blocked_intervals(
    intervals: list[list[float]], duration: float, margin: float
) -> list[tuple[float, float]]:
    expanded: list[tuple[float, float]] = []
    for pair in intervals:
        if len(pair) != 2:
            continue
        begin, end = float(pair[0]), float(pair[1])
        if not math.isfinite(begin) or not math.isfinite(end) or end <= begin:
            continue
        expanded.append((max(0.0, begin - margin), min(duration, end + margin)))
    expanded.sort()
    merged: list[tuple[float, float]] = []
    for begin, end in expanded:
        if not merged or begin > merged[-1][1]:
            merged.append((begin, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def allowed_start_intervals(
    duration: float,
    window_seconds: float,
    blocked: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    maximum_start = duration - window_seconds
    if maximum_start < 0:
        return []
    allowed: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, end in blocked:
        latest_start = min(maximum_start, begin - window_seconds)
        if latest_start >= cursor:
            allowed.append((cursor, latest_start))
        cursor = max(cursor, end)
        if cursor > maximum_start:
            break
    if cursor <= maximum_start:
        allowed.append((cursor, maximum_start))
    return [(begin, end) for begin, end in allowed if end >= begin]


def choose_background_starts(
    duration: float,
    window_seconds: float,
    requested: int,
    exclusion_intervals_json: str,
    safety_margin: float,
    minimum_spacing: float,
    seed: int,
) -> list[float]:
    try:
        raw_intervals = json.loads(exclusion_intervals_json or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed exclusion_intervals_json") from exc
    blocked = expanded_blocked_intervals(raw_intervals, duration, safety_margin)
    allowed = allowed_start_intervals(duration, window_seconds, blocked)
    if not allowed or requested <= 0:
        return []

    rng = random.Random(seed)
    lengths = [max(end - begin, 0.0) for begin, end in allowed]
    weights = [length + 1e-6 for length in lengths]
    selected: list[float] = []
    attempts = max(5000, requested * 1000)
    for _ in range(attempts):
        if len(selected) >= requested:
            break
        index = rng.choices(range(len(allowed)), weights=weights, k=1)[0]
        begin, end = allowed[index]
        candidate = begin if end <= begin else rng.uniform(begin, end)
        if all(abs(candidate - existing) >= minimum_spacing for existing in selected):
            selected.append(candidate)

    # Maximum-cardinality deterministic fallback for tight intervals such as
    # 60-second Orcasound recordings. Greedily taking the earliest available
    # start maximizes the number of points under a minimum-spacing constraint.
    if len(selected) < requested:
        fallback: list[float] = []
        last = -math.inf
        for begin, end in allowed:
            point = max(begin, last + minimum_spacing)
            while point <= end + 1e-9:
                fallback.append(point)
                last = point
                point += minimum_spacing
        if len(fallback) > requested:
            selected = rng.sample(fallback, requested)
        else:
            selected = fallback

    return sorted(round(value, 6) for value in selected[:requested])


def extract_flac(
    ffmpeg: str,
    source: str,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    sample_count: int,
    output_path: Path,
    remote: bool,
    retries: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.flac")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if remote:
        command.extend(
            [
                "-rw_timeout",
                "120000000",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
            ]
        )
    command.extend(
        [
            "-ss",
            f"{start_seconds:.9f}",
            "-i",
            source,
            "-t",
            f"{duration_seconds:.9f}",
            "-map",
            "0:a:0",
            "-af",
            (
                "pan=mono|c0=c0,"
                f"aresample={sample_rate}:resampler=soxr:precision=28,"
                f"apad=whole_len={sample_count},atrim=end_sample={sample_count}"
            ),
            "-c:a",
            "flac",
            "-compression_level",
            "5",
            "-sample_fmt",
            "s16",
            str(temporary),
        ]
    )
    try:
        run_command(command, retries=retries, retry_delay=3.0)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create {temporary}")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_and_measure_flac(
    path: Path, expected_rate: int, expected_frames: int
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "numpy and soundfile are required. In Kaggle run: pip install -q soundfile"
        ) from exc

    info = sf.info(str(path))
    if info.samplerate != expected_rate:
        raise RuntimeError(f"{path}: sample rate {info.samplerate}, expected {expected_rate}")
    if info.channels != 1:
        raise RuntimeError(f"{path}: channels {info.channels}, expected 1")
    if info.frames != expected_frames:
        raise RuntimeError(f"{path}: frames {info.frames}, expected {expected_frames}")
    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != expected_rate or len(samples) != expected_frames:
        raise RuntimeError(f"{path}: decoded shape/rate mismatch")
    if not np.isfinite(samples).all():
        raise RuntimeError(f"{path}: contains non-finite samples")
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-12))
    return {
        "output_sample_rate": int(rate),
        "output_sample_count": int(len(samples)),
        "output_channels": 1,
        "output_bytes": path.stat().st_size,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "dc_offset": float(np.mean(samples, dtype=np.float64)),
        "clipped_percent": float(np.mean(np.abs(samples) >= 0.9999) * 100.0),
        "silence_percent": float(np.mean(np.abs(samples) < 1e-5) * 100.0),
    }


def sanitized_annotation_row(row: dict[str, str]) -> dict[str, object]:
    excluded = {
        "FilePath",
        "clip_path",
        "relative_clip_path",
        "source_extension",
        "extraction_mode",
    }
    return {
        key: value
        for key, value in row.items()
        if key not in excluded and not key.casefold().startswith("unnamed:")
    }


def make_annotated_manifest_row(
    plan_row: dict[str, str],
    actual_start: float,
    source_info: dict[str, object],
    qc: dict[str, object],
) -> dict[str, object]:
    clip_id = plan_row["annotation_id"]
    relative_path = plan_row["relative_clip_path"].replace("\\", "/")
    archive_member_path = str(PurePosixPath(relative_path).relative_to("clips"))
    requested = float(plan_row["clip_start_requested_sec"])
    return {
        **sanitized_annotation_row(plan_row),
        "clip_id": clip_id,
        "clip_kind": "annotated",
        "negative_subtype": "",
        "archive_path": "clips.zip",
        "archive_member_path": archive_member_path,
        "relative_clip_path": relative_path,
        "clip_path": relative_path,
        "requested_clip_start_sec": requested,
        "actual_clip_start_sec": actual_start,
        "boundary_adjustment_sec": actual_start - requested,
        "source_duration_sec": source_info["duration_sec"],
        "source_sample_rate": source_info["sample_rate"],
        "source_channels": source_info["channels"],
        **qc,
    }


def make_background_manifest_row(
    plan_row: dict[str, str],
    index: int,
    actual_start: float,
    source_info: dict[str, object],
    qc: dict[str, object],
    relative_path: str,
) -> dict[str, object]:
    clip_id = PurePosixPath(relative_path).stem
    archive_member_path = str(PurePosixPath(relative_path).relative_to("clips"))
    return {
        "clip_id": clip_id,
        "clip_kind": "background",
        "negative_subtype": plan_row["negative_subtype"],
        "model_source_label": "Abiotic",
        "clean_class_species": "AB",
        "clean_ecotype": "",
        "ClassSpecies": "AB",
        "Ecotype": "",
        "source_head_eligible": "TRUE",
        "ecotype_head_eligible": "FALSE",
        "split": plan_row["split"],
        "storage_key": plan_row["storage_key"],
        "kaggle_dataset_id": plan_row["kaggle_dataset_id"],
        "archive_path": "clips.zip",
        "archive_member_path": archive_member_path,
        "relative_clip_path": relative_path,
        "clip_path": relative_path,
        "Provider": plan_row["Provider"],
        "Dataset": plan_row["Dataset"],
        "Soundfile": plan_row["Soundfile"],
        "source_recording_id": plan_row["source_recording_id"],
        "gcs_path": plan_row["gcs_path"],
        "background_window_index": index,
        "requested_clip_start_sec": actual_start,
        "actual_clip_start_sec": actual_start,
        "boundary_adjustment_sec": 0.0,
        "clip_duration_sec": plan_row["window_duration_sec"],
        "source_duration_sec": source_info["duration_sec"],
        "source_sample_rate": source_info["sample_rate"],
        "source_channels": source_info["channels"],
        "background_policy": plan_row["background_policy"],
        "safety_margin_sec": plan_row["safety_margin_sec"],
        "minimum_spacing_sec": plan_row["minimum_spacing_sec"],
        **qc,
    }


def source_cache_path(cache_dir: Path, recording_id: str, gcs_path: str) -> Path:
    suffix = PurePosixPath(gcs_path).suffix or ".audio"
    return cache_dir / f"{stable_id('source', recording_id)}{suffix}"


def ensure_free_space(path: Path, required_bytes: int, minimum_free_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    required = required_bytes + minimum_free_bytes
    if free < required:
        raise RuntimeError(
            f"Insufficient free space: {free / 1024**3:.2f} GiB free, "
            f"{required / 1024**3:.2f} GiB required"
        )


def expected_clip_path(shard_dir: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    if not parts or ".." in parts:
        raise ValueError(f"Unsafe relative clip path: {relative_path}")
    path = shard_dir.joinpath(*parts)
    resolved_shard = shard_dir.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_shard and resolved_shard not in resolved_path.parents:
        raise ValueError(f"Clip path escapes shard directory: {relative_path}")
    return path


def process_shard(
    shard: dict[str, str],
    annotation_rows: list[dict[str, str]],
    background_rows: list[dict[str, str]],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, object]:
    shard_id = shard["shard_id"]
    shard_dir = paths["shards"] / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = paths["manifests"] / f"{shard_id}.jsonl"
    failure_path = paths["failures"] / f"{shard_id}.jsonl"
    shortfall_path = paths["shortfalls"] / f"{shard_id}.csv"
    existing = read_jsonl_latest(jsonl_path, "clip_id")

    annotations_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    backgrounds_by_source: dict[str, dict[str, str]] = {}
    for row in annotation_rows:
        annotations_by_source[row["source_recording_id"]].append(row)
    for row in background_rows:
        backgrounds_by_source[row["source_recording_id"]] = row
    source_ids = sorted(set(annotations_by_source) | set(backgrounds_by_source))
    if args.max_recordings is not None:
        source_ids = source_ids[: args.max_recordings]

    created = 0
    skipped = 0
    positive_failures = 0
    source_failures = 0
    shortfalls: list[dict[str, object]] = []
    print(
        f"\n[{shard_id}] sources={len(source_ids):,}, "
        f"annotated={len(annotation_rows):,}, "
        f"background_requested={sum(int(r['requested_window_count']) for r in background_rows):,}"
    )

    for source_number, recording_id in enumerate(source_ids, start=1):
        positive_rows = annotations_by_source.get(recording_id, [])
        background_row = backgrounds_by_source.get(recording_id)
        representative = positive_rows[0] if positive_rows else background_row
        assert representative is not None
        gcs_path = representative["gcs_path"]
        source_size = int(float(representative.get("source_size_bytes") or 0))
        extraction_mode = representative["extraction_mode"]
        remote = extraction_mode == "remote_seek"
        source_url = public_gcs_url(gcs_path)
        local_source: Path | None = None

        try:
            if remote:
                source_input = source_url
            else:
                ensure_free_space(
                    paths["root"], source_size, int(args.minimum_free_gb * 1024**3)
                )
                local_source = source_cache_path(paths["sources"], recording_id, gcs_path)
                if not local_source.exists() or (
                    source_size > 0 and local_source.stat().st_size != source_size
                ):
                    download_source(
                        gcs_path,
                        local_source,
                        source_size,
                        args.http_timeout_seconds,
                        args.download_retries,
                        args.download_chunk_mb * 1024**2,
                    )
                source_input = str(local_source)

            source_info = probe_audio(args.ffprobe, source_input, args.ffmpeg_retries)
            source_duration = float(source_info["duration_sec"])
            expected_rate = args.sample_rate
            expected_frames = round(args.sample_rate * args.clip_seconds)

            for positive_number, row in enumerate(positive_rows, start=1):
                clip_id = row["annotation_id"]
                relative_path = row["relative_clip_path"].replace("\\", "/")
                output_path = expected_clip_path(shard_dir, relative_path)
                annotation_end = parse_float(row.get("FileEndSec"))
                if annotation_end is not None and annotation_end > source_duration + 1.0:
                    positive_failures += 1
                    append_jsonl(
                        failure_path,
                        {
                            "kind": "annotated",
                            "clip_id": clip_id,
                            "source_recording_id": recording_id,
                            "reason": "annotation_outside_source_duration",
                            "annotation_end_sec": annotation_end,
                            "source_duration_sec": source_duration,
                        },
                    )
                    if remote and (
                        positive_number == 1
                        or positive_number == len(positive_rows)
                        or positive_number % args.remote_clip_progress_every == 0
                    ):
                        print(
                            f"    remote annotated clips: {positive_number:,}/"
                            f"{len(positive_rows):,} for source {source_number:,}/"
                            f"{len(source_ids):,}"
                        )
                    continue
                requested_start = float(row["clip_start_requested_sec"])
                actual_start = min(
                    max(requested_start, 0.0), max(source_duration - args.clip_seconds, 0.0)
                )
                if clip_id in existing and output_path.exists():
                    try:
                        validate_and_measure_flac(output_path, expected_rate, expected_frames)
                        skipped += 1
                        if remote and (
                            positive_number == 1
                            or positive_number == len(positive_rows)
                            or positive_number % args.remote_clip_progress_every == 0
                        ):
                            print(
                                f"    remote annotated clips: {positive_number:,}/"
                                f"{len(positive_rows):,} for source {source_number:,}/"
                                f"{len(source_ids):,}"
                            )
                        continue
                    except Exception:
                        pass
                extract_flac(
                    args.ffmpeg,
                    source_input,
                    actual_start,
                    args.clip_seconds,
                    expected_rate,
                    expected_frames,
                    output_path,
                    remote,
                    args.ffmpeg_retries,
                )
                qc = validate_and_measure_flac(output_path, expected_rate, expected_frames)
                manifest_row = make_annotated_manifest_row(
                    row, actual_start, source_info, qc
                )
                append_jsonl(jsonl_path, manifest_row)
                existing[clip_id] = manifest_row
                created += 1
                if remote and (
                    positive_number == 1
                    or positive_number == len(positive_rows)
                    or positive_number % args.remote_clip_progress_every == 0
                ):
                    print(
                        f"    remote annotated clips: {positive_number:,}/"
                        f"{len(positive_rows):,} for source {source_number:,}/"
                        f"{len(source_ids):,}"
                    )

            if background_row is not None:
                requested_count = int(background_row["requested_window_count"])
                starts = choose_background_starts(
                    source_duration,
                    float(background_row["window_duration_sec"]),
                    requested_count,
                    background_row.get("exclusion_intervals_json", "[]"),
                    float(background_row["safety_margin_sec"]),
                    float(background_row["minimum_spacing_sec"]),
                    int(background_row["random_seed"]),
                )
                if len(starts) < requested_count:
                    shortfalls.append(
                        {
                            "shard_id": shard_id,
                            "source_recording_id": recording_id,
                            "Provider": background_row["Provider"],
                            "Dataset": background_row["Dataset"],
                            "Soundfile": background_row["Soundfile"],
                            "requested": requested_count,
                            "selected": len(starts),
                            "shortfall": requested_count - len(starts),
                            "source_duration_sec": source_duration,
                            "background_policy": background_row["background_policy"],
                        }
                    )
                for index, start in enumerate(starts):
                    clip_id = stable_id(
                        "bg", recording_id, background_row["random_seed"], index, start
                    )
                    relative_path = f"clips/Abiotic/{clip_id}.flac"
                    output_path = expected_clip_path(shard_dir, relative_path)
                    if clip_id in existing and output_path.exists():
                        try:
                            validate_and_measure_flac(
                                output_path, expected_rate, expected_frames
                            )
                            skipped += 1
                            continue
                        except Exception:
                            pass
                    extract_flac(
                        args.ffmpeg,
                        source_input,
                        start,
                        args.clip_seconds,
                        expected_rate,
                        expected_frames,
                        output_path,
                        remote,
                        args.ffmpeg_retries,
                    )
                    qc = validate_and_measure_flac(
                        output_path, expected_rate, expected_frames
                    )
                    manifest_row = make_background_manifest_row(
                        background_row,
                        index,
                        start,
                        source_info,
                        qc,
                        relative_path,
                    )
                    append_jsonl(jsonl_path, manifest_row)
                    existing[clip_id] = manifest_row
                    created += 1

        except Exception as exc:
            source_failures += 1
            append_jsonl(
                failure_path,
                {
                    "kind": "source",
                    "source_recording_id": recording_id,
                    "gcs_path": gcs_path,
                    "error": repr(exc),
                    "timestamp": time.time(),
                },
            )
            print(f"WARNING: source failed: {recording_id}: {exc}")
        finally:
            if local_source is not None and local_source.exists():
                local_source.unlink()

        if source_number == 1 or source_number % args.progress_every == 0:
            print(
                f"  {source_number:,}/{len(source_ids):,} sources; "
                f"created={created:,}, resumed={skipped:,}, failures={source_failures:,}"
            )

    manifest_rows = list(read_jsonl_latest(jsonl_path, "clip_id").values())
    manifest_rows.sort(
        key=lambda row: (
            clean_text(row.get("source_recording_id")),
            float(row.get("actual_clip_start_sec") or 0.0),
            clean_text(row.get("clip_id")),
        )
    )
    write_csv_union(shard_dir / MANIFEST_NAME, manifest_rows)
    if shortfalls:
        write_csv_union(shortfall_path, shortfalls)
    elif shortfall_path.exists():
        shortfall_path.unlink()

    label_counts = Counter(clean_text(row.get("model_source_label")) for row in manifest_rows)
    kind_counts = Counter(clean_text(row.get("clip_kind")) for row in manifest_rows)
    qc_summary = {
        "shard_id": shard_id,
        "dataset_id": shard["kaggle_dataset_id"],
        "manifest_rows": len(manifest_rows),
        "label_counts": dict(label_counts),
        "clip_kind_counts": dict(kind_counts),
        "background_shortfall": sum(int(row["shortfall"]) for row in shortfalls),
        "background_shortfall_sources": len(shortfalls),
        "source_failures_this_run": source_failures,
        "annotated_failures_this_run": positive_failures,
        "created_this_run": created,
        "resumed_this_run": skipped,
    }
    atomic_write_json(shard_dir / QC_SUMMARY_NAME, qc_summary)

    if source_failures or positive_failures:
        raise RuntimeError(
            f"{shard_id} has {source_failures} source failures and "
            f"{positive_failures} annotated-clip failures; refusing upload"
        )
    if args.max_recordings is None:
        expected_annotations = len(annotation_rows)
        actual_annotations = kind_counts["annotated"]
        if actual_annotations != expected_annotations:
            raise RuntimeError(
                f"{shard_id}: expected {expected_annotations} annotated clips, "
                f"manifest contains {actual_annotations}"
            )
    return qc_summary


def write_dataset_metadata(
    shard_dir: Path,
    shard: dict[str, str],
    license_name: str,
    public: bool,
) -> None:
    privacy = "public" if public else "private"
    payload = {
        "title": shard["dataset_title"],
        "id": shard["kaggle_dataset_id"],
        "licenses": [{"name": license_name}],
        "subtitle": "Original annotated and ambient 3-second cetacean audio clips",
        "description": (
            "A storage shard of the Multispecies Cetacean V2 dataset. "
            "Audio is lossless mono 16 kHz PCM16 FLAC stored in clips.zip. "
            "The manifest provides archive_path and archive_member_path for "
            "random access without unpacking the complete shard. This shard is "
            f"intended to be {privacy} and is identified by {shard['shard_id']}."
        ),
    }
    atomic_write_json(shard_dir / "dataset-metadata.json", payload)


def kaggle_dataset_exists(kaggle: str, dataset_id: str) -> bool:
    result = subprocess.run(
        [kaggle, "datasets", "status", dataset_id],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def verify_kaggle_dataset_files(kaggle: str, dataset_id: str) -> str:
    """Verify that the uploaded archive and manifest are visible on Kaggle."""
    result = run_command(
        [
            kaggle,
            "datasets",
            "files",
            dataset_id,
            "--page-size",
            "200",
        ]
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    normalized = output.casefold().replace("\\", "/")
    expanded_listing = "clips/" in normalized
    missing = []
    # An expanded archive can contain tens of thousands of members, so the
    # first API page may not yet contain the top-level manifest/QC filenames.
    if not expanded_listing:
        missing.extend(
            name
            for name in (MANIFEST_NAME, QC_SUMMARY_NAME)
            if name.casefold() not in normalized
        )
    if "clips.zip" not in normalized and not expanded_listing:
        missing.append("clips.zip or expanded clips/")
    if missing:
        raise RuntimeError(
            f"Kaggle dataset {dataset_id} is ready but is missing expected files: "
            f"{missing}. File listing:\n{output[-4000:]}"
        )
    return output


def upload_and_verify(
    kaggle: str,
    shard_dir: Path,
    dataset_id: str,
    public: bool,
    status_timeout_seconds: int,
    adopt_existing: bool,
) -> str:
    if kaggle_dataset_exists(kaggle, dataset_id):
        if not adopt_existing:
            raise RuntimeError(
                f"Dataset already exists: {dataset_id}. Use --adopt-existing-dataset "
                "only after confirming it is the intended completed shard."
            )
        print(f"Adopting existing Kaggle dataset: {dataset_id}")
    else:
        command = [
            kaggle,
            "datasets",
            "create",
            "-p",
            str(shard_dir),
            "-q",
            "-t",
            "-r",
            "zip",
        ]
        if public:
            command.append("--public")
        run_command(command, capture=True)

    deadline = time.monotonic() + status_timeout_seconds
    last_output = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            [kaggle, "datasets", "status", dataset_id],
            check=False,
            text=True,
            capture_output=True,
        )
        last_output = (result.stdout + "\n" + result.stderr).strip()
        normalized = last_output.casefold()
        if result.returncode == 0 and any(
            marker in normalized for marker in ("ready", "complete", "active")
        ):
            verify_kaggle_dataset_files(kaggle, dataset_id)
            return last_output
        if any(marker in normalized for marker in ("error", "failed")):
            raise RuntimeError(f"Kaggle dataset processing failed: {last_output}")
        time.sleep(15)
    raise TimeoutError(
        f"Timed out waiting for Kaggle dataset {dataset_id}; last status: {last_output}"
    )


def safe_remove_shard(shards_root: Path, shard_dir: Path) -> None:
    root = shards_root.resolve()
    target = shard_dir.resolve()
    if target == root or root not in target.parents or not target.name:
        raise RuntimeError(f"Refusing unsafe shard deletion: {target}")
    shutil.rmtree(target)


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"shards": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("shards", {})
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument(
        "--work-dir", default="/kaggle/working/multispecies_cetacean_build"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-all", action="store_true")
    selection.add_argument("--shard-id", action="append")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--adopt-existing-dataset", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument(
        "--max-recordings",
        type=int,
        help="Testing only; partial shards cannot be uploaded.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--kaggle", default="kaggle")
    parser.add_argument("--ffmpeg-retries", type=int, default=3)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--http-timeout-seconds", type=int, default=180)
    parser.add_argument("--download-chunk-mb", type=int, default=8)
    parser.add_argument("--minimum-free-gb", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--remote-clip-progress-every",
        type=int,
        default=10,
        help="Print progress every N annotated clips within remote-seek sources.",
    )
    parser.add_argument("--status-timeout-seconds", type=int, default=1200)
    parser.add_argument("--license-name", default="CC-BY-4.0")
    args = parser.parse_args()

    if args.max_recordings is not None and args.upload:
        parser.error("--max-recordings cannot be combined with --upload")
    if args.delete_after_upload and not args.upload:
        parser.error("--delete-after-upload requires --upload")
    if args.public and not args.upload:
        parser.error("--public requires --upload")
    if args.sample_rate <= 0 or args.clip_seconds <= 0:
        parser.error("sample rate and clip seconds must be positive")
    if args.remote_clip_progress_every <= 0:
        parser.error("--remote-clip-progress-every must be positive")

    args.ffmpeg = require_executable(args.ffmpeg)
    args.ffprobe = require_executable(args.ffprobe)
    if args.upload:
        args.kaggle = require_executable(args.kaggle)

    plan_dir = Path(args.plan_dir)
    work_root = Path(args.work_dir)
    paths = {
        "root": work_root,
        "shards": work_root / "shards",
        "sources": work_root / "temporary_sources",
        "manifests": work_root / "manifest_journal",
        "failures": work_root / "failures",
        "shortfalls": work_root / "background_shortfalls",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    shard_rows = read_csv_rows(plan_dir / "multispecies_cetacean_shard_summary.csv")
    annotation_rows = read_csv_rows(
        plan_dir / "multispecies_cetacean_extraction_plan.csv"
    )
    background_rows = read_csv_rows(
        plan_dir / "multispecies_cetacean_background_plan.csv"
    )
    validate_plan_bundle(shard_rows, annotation_rows, background_rows)
    shards_by_id = {row["shard_id"]: row for row in shard_rows}
    if args.run_all:
        selected_ids = [row["shard_id"] for row in shard_rows]
    else:
        selected_ids = args.shard_id or []
        unknown = sorted(set(selected_ids).difference(shards_by_id))
        if unknown:
            parser.error(f"unknown shard ids: {unknown}")
    if args.max_shards is not None:
        selected_ids = selected_ids[: args.max_shards]

    annotations_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    backgrounds_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotation_rows:
        annotations_by_shard[row["shard_id"]].append(row)
    for row in background_rows:
        backgrounds_by_shard[row["shard_id"]].append(row)

    state_path = work_root / "multispecies_cetacean_build_state.json"
    state = load_state(state_path)
    state_shards: dict[str, dict[str, object]] = state["shards"]  # type: ignore[assignment]

    for position, shard_id in enumerate(selected_ids, start=1):
        shard = shards_by_id[shard_id]
        prior = state_shards.get(shard_id, {})
        if prior.get("status") == "uploaded":
            print(f"[{position}/{len(selected_ids)}] already uploaded; skipping {shard_id}")
            continue

        # A new Kaggle session normally starts with an empty /kaggle/working.
        # Recover across session timeouts by checking durable remote shards
        # before rebuilding anything locally.
        if (
            args.upload
            and args.adopt_existing_dataset
            and kaggle_dataset_exists(args.kaggle, shard["kaggle_dataset_id"])
        ):
            print(
                f"[{position}/{len(selected_ids)}] checking existing dataset "
                f"before build: {shard['kaggle_dataset_id']}"
            )
            status_output = upload_and_verify(
                args.kaggle,
                paths["shards"] / shard_id,
                shard["kaggle_dataset_id"],
                args.public,
                args.status_timeout_seconds,
                True,
            )
            state_shards[shard_id] = {
                **prior,
                "status": "uploaded",
                "dataset_id": shard["kaggle_dataset_id"],
                "adopted_at": time.time(),
                "kaggle_status": status_output,
            }
            atomic_write_json(state_path, state)
            print(f"Existing verified shard adopted; skipping build: {shard_id}")
            continue

        state_shards[shard_id] = {
            **prior,
            "status": "building",
            "dataset_id": shard["kaggle_dataset_id"],
            "started_or_resumed_at": time.time(),
        }
        atomic_write_json(state_path, state)
        try:
            summary = process_shard(
                shard,
                annotations_by_shard[shard_id],
                backgrounds_by_shard[shard_id],
                args,
                paths,
            )
            shard_dir = paths["shards"] / shard_id
            write_dataset_metadata(
                shard_dir, shard, args.license_name, args.public
            )
            state_shards[shard_id] = {
                **state_shards[shard_id],
                "status": "partial_test" if args.max_recordings is not None else "validated",
                "summary": summary,
                "validated_at": time.time(),
            }
            atomic_write_json(state_path, state)

            if args.upload:
                print(
                    f"Uploading {shard_id} to Kaggle dataset "
                    f"{shard['kaggle_dataset_id']}..."
                )
                status_output = upload_and_verify(
                    args.kaggle,
                    shard_dir,
                    shard["kaggle_dataset_id"],
                    args.public,
                    args.status_timeout_seconds,
                    args.adopt_existing_dataset,
                )
                state_shards[shard_id] = {
                    **state_shards[shard_id],
                    "status": "uploaded",
                    "uploaded_at": time.time(),
                    "kaggle_status": status_output,
                }
                atomic_write_json(state_path, state)
                print(
                    f"Kaggle upload ready and verified: "
                    f"{shard['kaggle_dataset_id']}"
                )
                if args.delete_after_upload:
                    safe_remove_shard(paths["shards"], shard_dir)
                    state_shards[shard_id]["local_shard_deleted"] = True
                    atomic_write_json(state_path, state)
                    print(f"Uploaded, verified, and removed local shard: {shard_id}")
        except Exception as exc:
            state_shards[shard_id] = {
                **state_shards.get(shard_id, {}),
                "status": "failed",
                "failed_at": time.time(),
                "error": repr(exc),
            }
            atomic_write_json(state_path, state)
            print(f"ERROR: shard failed: {shard_id}: {exc}", file=sys.stderr)
            return 1

    # Preserve a portable combined manifest outside deletable shard directories.
    combined: dict[str, dict[str, object]] = {}
    for jsonl_path in sorted(paths["manifests"].glob("*.jsonl")):
        combined.update(read_jsonl_latest(jsonl_path, "clip_id"))
    if combined:
        write_csv_union(
            work_root / "multispecies_cetacean_combined_manifest.csv",
            sorted(
                combined.values(),
                key=lambda row: (
                    clean_text(row.get("split")),
                    clean_text(row.get("storage_key")),
                    clean_text(row.get("clip_id")),
                ),
            ),
        )
    print(f"\nBuild state: {state_path}")
    print(f"Portable combined manifest: {work_root / 'multispecies_cetacean_combined_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
