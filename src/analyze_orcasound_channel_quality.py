#!/usr/bin/env python3
"""Audit channel quality in Orcasound source recordings from a dataset plan.

The script reads the source-recording and extraction plans, streams each selected
recording through FFmpeg one channel at a time, and writes channel-level metrics,
recording-level flags, and a concise text report. It does not modify source audio.
It intentionally uses only the Python standard library plus NumPy (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


SOURCE_PLAN_NAME = "multispecies_cetacean_source_recording_plan.csv"
EXTRACTION_PLAN_NAME = "multispecies_cetacean_extraction_plan.csv"


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dbfs(value: float, floor: float = -160.0) -> float:
    if not math.isfinite(value) or value <= 0:
        return floor
    return max(20.0 * math.log10(value), floor)


def public_audio_source(value: str) -> str:
    value = clean(value).replace("\\", "/")
    local = Path(value)
    if local.is_file():
        return str(local.resolve())
    if value.startswith(("https://", "http://")):
        return value
    if value.startswith("gs://"):
        value = value[5:]
    if not value:
        raise ValueError("Source plan contains a blank gcs_path")
    return "https://storage.googleapis.com/" + quote(value, safe="/")


def remote_input_options(source: str) -> list[str]:
    if not source.startswith(("https://", "http://")):
        return []
    return [
        "-rw_timeout", "120000000",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
    ]


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def probe_audio(ffprobe: str, source: str, timeout: int) -> dict[str, float | int]:
    command = [ffprobe, "-v", "error", *remote_input_options(source), "-select_streams", "a:0"]
    command += [
        "-show_entries", "stream=sample_rate,channels,duration:format=duration",
        "-of", "json", source,
    ]
    result = run(command, timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("No audio stream found")
    stream = streams[0]
    duration = finite_float(stream.get("duration"))
    if duration is None:
        duration = finite_float(payload.get("format", {}).get("duration"))
    if duration is None or duration <= 0:
        raise RuntimeError("Could not determine a positive duration")
    return {
        "duration_sec": duration,
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def decode_channel(
    ffmpeg: str,
    source: str,
    channel: int,
    sample_rate: int,
    timeout: int,
) -> np.ndarray:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    command += remote_input_options(source)
    command += [
        "-i", source,
        "-map", "0:a:0",
        "-af", f"pan=mono|c0=c{channel},aresample={sample_rate}:resampler=soxr:precision=28",
        "-vn", "-sn", "-dn",
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    result = run(command, timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=False)
    if not len(samples):
        raise RuntimeError("FFmpeg decoded zero samples")
    return samples


def annotation_metrics(
    samples: np.ndarray,
    intervals: list[tuple[float, float]],
    sample_rate: int,
) -> tuple[float | None, float | None]:
    levels: list[float] = []
    for begin, end in intervals:
        left = max(0, min(len(samples), round(begin * sample_rate)))
        right = max(left, min(len(samples), round(end * sample_rate)))
        if right - left < max(16, round(0.02 * sample_rate)):
            continue
        segment = samples[left:right].astype(np.float64)
        levels.append(dbfs(float(np.sqrt(np.mean(segment * segment)))))
    if not levels:
        return None, None
    median = float(np.median(levels))
    overall = dbfs(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))))
    return median, median - overall


def measure_channel(
    samples: np.ndarray,
    expected_duration: float,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> dict[str, float | int]:
    values = samples.astype(np.float64)
    decoded_duration = len(samples) / sample_rate
    rms = float(np.sqrt(np.mean(values * values)))
    peak = float(np.max(np.abs(values)))
    frame_samples = max(1, round(0.1 * sample_rate))
    usable = len(values) - (len(values) % frame_samples)
    if usable:
        frames = values[:usable].reshape(-1, frame_samples)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        active_rms = float(np.percentile(frame_rms, 90))
        near_silent_frames = float(np.mean(frame_rms <= 10 ** (-75.0 / 20.0)) * 100.0)
    else:
        active_rms = rms
        near_silent_frames = 100.0 if rms <= 10 ** (-75.0 / 20.0) else 0.0
    annotated_rms, annotation_contrast = annotation_metrics(samples, intervals, sample_rate)
    return {
        "decoded_samples": len(samples),
        "decoded_duration_sec": decoded_duration,
        "duration_ratio": decoded_duration / expected_duration,
        "rms_dbfs": dbfs(rms),
        "active_rms_dbfs": dbfs(active_rms),
        "peak_dbfs": dbfs(peak),
        "dc_offset": float(np.mean(values)),
        "clipped_percent": float(np.mean(np.abs(values) >= 0.999) * 100.0),
        "near_silent_frame_percent": near_silent_frames,
        "annotated_interval_count": len(intervals),
        "median_annotated_rms_dbfs": annotated_rms if annotated_rms is not None else "",
        "annotation_contrast_db": annotation_contrast if annotation_contrast is not None else "",
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=Path("/kaggle/input/datasets/leonisviridis/orca-detector-misc/multispecies_cetacean_plan_v4"),
    )
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--extraction-plan", type=Path)
    parser.add_argument("--provider", default="OrcaSound")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/orcasound_channel_audit"))
    parser.add_argument("--max-files", type=int, help="Randomly audit at most this many recordings.")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--analysis-sample-rate", type=int, default=16000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    if args.analysis_sample_rate < 1000 or args.timeout_seconds < 1:
        parser.error("Invalid sample rate or timeout")
    return args


def main() -> int:
    args = parse_args()
    ffmpeg = shutil.which(args.ffmpeg)
    ffprobe = shutil.which(args.ffprobe)
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be installed and on PATH")
    source_plan = args.source_plan or args.plan_dir / SOURCE_PLAN_NAME
    extraction_plan = args.extraction_plan or args.plan_dir / EXTRACTION_PLAN_NAME
    if not source_plan.is_file():
        raise FileNotFoundError(source_plan)
    if not extraction_plan.is_file():
        raise FileNotFoundError(extraction_plan)

    biological_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(extraction_plan):
        if clean(row.get("Provider")).casefold() != args.provider.casefold():
            continue
        if clean(row.get("model_source_label")) == "Abiotic":
            continue
        begin = finite_float(row.get("FileBeginSec"))
        end = finite_float(row.get("FileEndSec"))
        if begin is not None and end is not None and end > begin:
            biological_intervals[clean(row.get("source_recording_id"))].append((begin, end))

    sources = [
        row for row in read_csv(source_plan)
        if clean(row.get("Provider")).casefold() == args.provider.casefold()
        and clean(row.get("audio_match_status")).casefold() == "matched"
    ]
    sources.sort(key=lambda row: clean(row.get("source_recording_id")))
    if args.max_files is not None and len(sources) > args.max_files:
        sources = random.Random(args.seed).sample(sources, args.max_files)
        sources.sort(key=lambda row: clean(row.get("source_recording_id")))
    if not sources:
        raise ValueError(f"No matched {args.provider} recordings found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    channel_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    print(f"Auditing {len(sources):,} {args.provider} recordings")
    for source_index, row in enumerate(sources, start=1):
        recording_id = clean(row.get("source_recording_id"))
        try:
            source = public_audio_source(clean(row.get("gcs_path")))
            probe = probe_audio(ffprobe, source, args.timeout_seconds)
            decoded: list[np.ndarray] = []
            metrics: list[dict[str, Any]] = []
            intervals = biological_intervals.get(recording_id, [])
            for channel in range(int(probe["channels"])):
                samples = decode_channel(
                    ffmpeg, source, channel, args.analysis_sample_rate, args.timeout_seconds
                )
                decoded.append(samples)
                measurement = measure_channel(
                    samples, float(probe["duration_sec"]), args.analysis_sample_rate, intervals
                )
                metrics.append(measurement)
                channel_rows.append({
                    "source_recording_id": recording_id,
                    "Provider": clean(row.get("Provider")),
                    "Dataset": clean(row.get("Dataset")),
                    "Soundfile": clean(row.get("Soundfile")),
                    "gcs_path": clean(row.get("gcs_path")),
                    "source_sample_rate": probe["sample_rate"],
                    "source_channels": probe["channels"],
                    "source_duration_sec": probe["duration_sec"],
                    "channel_index": channel,
                    **measurement,
                })

            flags: list[str] = []
            usable: list[int] = []
            for channel, measurement in enumerate(metrics):
                if float(measurement["duration_ratio"]) < 0.99:
                    flags.append(f"channel_{channel}_incomplete")
                if (
                    float(measurement["rms_dbfs"]) <= -75.0
                    or float(measurement["near_silent_frame_percent"]) >= 95.0
                ):
                    flags.append(f"channel_{channel}_near_silent")
                else:
                    usable.append(channel)
                if float(measurement["clipped_percent"]) > 1.0:
                    flags.append(f"channel_{channel}_clipping")

            correlation: float | str = ""
            rms_difference: float | str = ""
            if len(metrics) >= 2:
                rms_difference = float(metrics[1]["rms_dbfs"]) - float(metrics[0]["rms_dbfs"])
                count = min(len(decoded[0]), len(decoded[1]))
                if count and np.std(decoded[0][:count]) > 0 and np.std(decoded[1][:count]) > 0:
                    correlation = float(np.corrcoef(decoded[0][:count], decoded[1][:count])[0, 1])
                    if correlation > 0.999:
                        flags.append("channels_nearly_identical")
                if abs(float(rms_difference)) > 20.0:
                    flags.append("channel_level_difference_over_20db")

            candidate: int | str = ""
            if len(usable) == 1:
                candidate = usable[0]
            elif usable:
                def candidate_score(channel: int) -> float:
                    annotated = finite_float(metrics[channel]["median_annotated_rms_dbfs"])
                    return annotated if annotated is not None else float(metrics[channel]["active_rms_dbfs"])
                candidate = max(usable, key=candidate_score)
                if len(usable) > 1 and candidate != 0:
                    flags.append("channel_1_has_higher_signal_level_review_manually")

            summary_rows.append({
                "source_recording_id": recording_id,
                "Dataset": clean(row.get("Dataset")),
                "Soundfile": clean(row.get("Soundfile")),
                "source_channels": probe["channels"],
                "candidate_channel_for_review": candidate,
                "channel_1_minus_0_rms_db": rms_difference,
                "channel_0_1_correlation": correlation,
                "quality_flags": "|".join(sorted(set(flags))),
            })
        except Exception as exc:
            failures.append({"source_recording_id": recording_id, "error": repr(exc)})
            print(f"  ERROR {recording_id}: {exc}")
        if source_index % 25 == 0 or source_index == len(sources):
            print(f"  {source_index:,}/{len(sources):,}; failures={len(failures):,}")

    write_csv(args.output_dir / "orcasound_channel_metrics.csv", channel_rows)
    write_csv(args.output_dir / "orcasound_channel_recording_summary.csv", summary_rows)
    write_csv(args.output_dir / "orcasound_channel_failures.csv", failures)
    channel_counts = Counter(int(row["source_channels"]) for row in summary_rows)
    flagged = sum(bool(row["quality_flags"]) for row in summary_rows)
    candidate_counts = Counter(str(row["candidate_channel_for_review"]) for row in summary_rows)
    report = [
        "ORCASOUND CHANNEL QUALITY AUDIT",
        "================================",
        f"Requested recordings:       {len(sources):,}",
        f"Successfully audited:       {len(summary_rows):,}",
        f"Failures:                   {len(failures):,}",
        f"Flagged for review:         {flagged:,}",
        f"Source channel counts:      {dict(sorted(channel_counts.items()))}",
        f"Candidate channel counts:   {dict(sorted(candidate_counts.items()))}",
        "",
        "Candidate channels are diagnostic suggestions, not automatic replacements.",
        "Review flagged channel-1 cases acoustically before changing channel policy.",
    ]
    (args.output_dir / "orcasound_channel_quality_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(report))
    print(f"Outputs: {args.output_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
