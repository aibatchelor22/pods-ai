#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Extract 3-second multispecies hard negatives from long recordings.

The script joins ``best_event_matches.csv`` (FP decisions),
``best_stable_events.csv`` (peak/event metadata), and ``window_predictions.csv``
(the exact peak-window probabilities). It discovers source recordings from the
DCLDE annotation inventory, downloads one recording at a time, saves the peak
window as a BKG WAV, appends a training-compatible manifest, and removes the
temporary source recording. The output directory includes Kaggle dataset
metadata and can be uploaded with the Kaggle CLI.

KW candidates include false stable events. Independent window-score mining also
finds background windows with high KW, HW, or AB probabilities. Overlapping
class candidates are merged into one clip and tagged with every class for which
they are a hard negative.

Evaluator FPs can be real but unannotated calls. Review candidates whenever
possible; an optional review CSV can restrict extraction to approved candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np


DEFAULT_MATCHES = "/kaggle/working/dclde_long_stable_events/best_event_matches.csv"
DEFAULT_EVENTS = "/kaggle/working/dclde_long_stable_events/best_stable_events.csv"
DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_ANNOTATIONS = (
    "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/"
    "dclde_2027_killer_whales/Annotations.csv"
)
DEFAULT_OUTPUT = "/kaggle/working/dclde_long_fp_hard_negatives"
DEFAULT_GCS_BASE = "noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales"
DEFAULT_GCS_PROVIDERS = (
    "dfo_crp,dfo_wdlp,onc,orcasound,scripps,simres,smru,uaf,vfpa"
)
DEFAULT_DATASET_ID = "leonisviridis/dclde-long-recording-hard-negatives"
ACCEPTED_REVIEW_DECISIONS = {
    "approved", "background", "hard_negative", "hard-negative", "keep", "yes", "true", "1"
}
WINDOW_SCORE_COLUMNS = (
    "kw_probability",
    "species_probability_background",
    "species_probability_KW",
    "species_probability_HW",
    "species_probability_AB",
    "ecotype_probability_SRKW",
    "ecotype_probability_NRKW",
    "ecotype_probability_TKW",
    "ecotype_probability_OKW",
    "ecotype_probability_SAR",
)
PROVENANCE_COLUMNS = (
    "hard_negative_source",
    "hard_negative_for",
    "hard_negative_primary_class",
    "hard_negative_score",
    "source_candidate_id",
    "source_event_id",
    "source_audio_path",
    "source_peak_time_sec",
    "source_event_start_sec",
    "source_event_end_sec",
    "source_event_peak_score",
    "source_event_topk_mean",
    "source_predicted_ecotype",
    "source_ecotype_confidence",
    "source_provider",
    "source_dataset",
    "review_decision",
    "relative_clip_path",
    "local_clip_path",
    *WINDOW_SCORE_COLUMNS,
)


@dataclass(frozen=True)
class Candidate:
    event_id: str
    soundfile: str
    provider: str
    dataset: str
    peak_time_sec: float
    event_start_sec: float
    event_end_sec: float
    peak_score: float
    event_score: float
    predicted_ecotype: str
    ecotype_confidence: float
    window_start_sec: float
    window_end_sec: float
    window_row: dict[str, str]
    review_decision: str
    candidate_id: str = ""
    hard_negative_for: tuple[str, ...] = ("KW",)
    primary_class: str = "KW"
    candidate_score: float = 0.0
    source_kind: str = "stable_event_false_positive"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"CSV has no header: {path}")
        return fields, list(reader)


def require_fields(path: Path, fields: list[str], required: set[str]) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def parse_float(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field!r} for {source}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field!r} for {source}")
    return value


def false_positive_event_ids(path: Path) -> set[str]:
    fields, rows = read_csv(path)
    require_fields(path, fields, {"status", "event_id"})
    return {
        (row.get("event_id") or "").strip()
        for row in rows
        if (row.get("status") or "").strip().upper() == "FP"
        and (row.get("event_id") or "").strip()
    }


def load_reviews(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    fields, rows = read_csv(path)
    event_field = next(
        (
            field
            for field in ("candidate_id", "source_candidate_id", "event_id", "source_event_id")
            if field in fields
        ),
        None,
    )
    decision_field = next(
        (field for field in ("decision", "review_decision", "approved") if field in fields), None
    )
    if event_field is None or decision_field is None:
        raise ValueError(
            f"{path} must contain candidate_id/source_candidate_id/event_id/source_event_id "
            "and decision/review_decision"
        )
    return {
        (row.get(event_field) or "").strip(): (row.get(decision_field) or "").strip()
        for row in rows
        if (row.get(event_field) or "").strip()
    }


def load_event_rows(path: Path, fp_ids: set[str]) -> list[dict[str, str]]:
    fields, rows = read_csv(path)
    required = {
        "event_id", "soundfile", "provider", "dataset", "peak_time_sec",
        "start_sec", "end_sec", "peak_score", "event_topk_mean",
        "predicted_ecotype", "ecotype_confidence",
    }
    require_fields(path, fields, required)
    selected = [row for row in rows if (row.get("event_id") or "").strip() in fp_ids]
    missing = fp_ids - {(row.get("event_id") or "").strip() for row in selected}
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise ValueError(f"{len(missing)} FP event IDs are absent from {path}: {examples}")
    return selected


def truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes", "y"}


def load_window_rows(
    path: Path, soundfiles: Optional[set[str]] = None
) -> dict[str, list[dict[str, str]]]:
    fields, rows = read_csv(path)
    required = {
        "Soundfile", "window_start_sec", "window_end_sec", "species_ground_truth",
        "ground_truth_ambiguous", *WINDOW_SCORE_COLUMNS,
    }
    require_fields(path, fields, required)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        soundfile = (row.get("Soundfile") or "").strip()
        if soundfile and (soundfiles is None or soundfile in soundfiles):
            grouped[soundfile].append(row)
    for soundfile, values in grouped.items():
        values.sort(key=lambda row: float(row["window_start_sec"]))
    return grouped


def window_candidate_id(soundfile: str, start_sec: float) -> str:
    """Return a stable review/resume identifier for one cached window."""
    return f"window::{soundfile}::{int(round(start_sec * 1000)):010d}ms"


def row_context(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def closest_peak_window(
    event: dict[str, str], windows: list[dict[str, str]], tolerance_sec: float = 0.51
) -> dict[str, str]:
    if not windows:
        raise ValueError(f"No cached windows for {(event.get('soundfile') or '').strip()}")
    peak_time = parse_float(event, "peak_time_sec", event["event_id"])
    distances = [
        abs((float(row["window_start_sec"]) + float(row["window_end_sec"])) / 2.0 - peak_time)
        for row in windows
    ]
    index = int(np.argmin(distances))
    if distances[index] > tolerance_sec:
        raise ValueError(
            f"Closest cached window center for {event['event_id']} is "
            f"{distances[index]:.3f}s from its peak"
        )
    return windows[index]


def prepare_candidates(
    event_rows: list[dict[str, str]],
    windows_by_file: dict[str, list[dict[str, str]]],
    reviews: dict[str, str],
    require_reviewed: bool,
    minimum_event_score: float,
    minimum_peak_separation_sec: float,
    max_clips_per_source: Optional[int],
    max_total_clips: Optional[int],
    seed: int,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    audit = []
    eligible = []
    for event in event_rows:
        event_id = (event.get("event_id") or "").strip()
        soundfile = (event.get("soundfile") or "").strip()
        decision = reviews.get(event_id, "")
        normalized_decision = decision.casefold()
        if decision and normalized_decision not in ACCEPTED_REVIEW_DECISIONS:
            audit.append({"event_id": event_id, "soundfile": soundfile, "status": "skipped_review_rejected"})
            continue
        if require_reviewed and normalized_decision not in ACCEPTED_REVIEW_DECISIONS:
            audit.append({"event_id": event_id, "soundfile": soundfile, "status": "skipped_unapproved"})
            continue
        event_score = parse_float(event, "event_topk_mean", event_id)
        if event_score < minimum_event_score:
            audit.append({"event_id": event_id, "soundfile": soundfile, "status": "skipped_low_score"})
            continue
        window = closest_peak_window(event, windows_by_file.get(soundfile, []))
        actual_species = (window.get("species_ground_truth") or "").strip()
        if actual_species != "background" or truthy(window.get("ground_truth_ambiguous")):
            audit.append(
                {
                    "event_id": event_id,
                    "soundfile": soundfile,
                    "status": "skipped_nonbackground_window_truth",
                    "species_ground_truth": actual_species,
                }
            )
            continue
        eligible.append(
            Candidate(
                event_id=event_id,
                soundfile=soundfile,
                provider=(event.get("provider") or "").strip(),
                dataset=(event.get("dataset") or "").strip(),
                peak_time_sec=parse_float(event, "peak_time_sec", event_id),
                event_start_sec=parse_float(event, "start_sec", event_id),
                event_end_sec=parse_float(event, "end_sec", event_id),
                peak_score=parse_float(event, "peak_score", event_id),
                event_score=event_score,
                predicted_ecotype=(event.get("predicted_ecotype") or "").strip(),
                ecotype_confidence=parse_float(event, "ecotype_confidence", event_id),
                window_start_sec=parse_float(window, "window_start_sec", event_id),
                window_end_sec=parse_float(window, "window_end_sec", event_id),
                window_row=window,
                review_decision=decision,
                candidate_id=window_candidate_id(
                    soundfile, parse_float(window, "window_start_sec", event_id)
                ),
                hard_negative_for=("KW",),
                primary_class="KW",
                candidate_score=max(
                    event_score,
                    parse_float(window, "species_probability_KW", event_id),
                ),
                source_kind="stable_event_false_positive",
            )
        )

    # Highest-confidence event wins when peak clips are too close. This avoids
    # saving many nearly identical overlapping examples from the same burst.
    selected = []
    by_file: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in eligible:
        by_file[candidate.soundfile].append(candidate)
    rng = random.Random(seed)
    file_names = list(by_file)
    rng.shuffle(file_names)
    for soundfile in file_names:
        kept = []
        for candidate in sorted(
            by_file[soundfile], key=lambda item: (-item.event_score, -item.peak_score, item.peak_time_sec)
        ):
            if any(
                abs(candidate.peak_time_sec - other.peak_time_sec) < minimum_peak_separation_sec
                for other in kept
            ):
                audit.append(
                    {"event_id": candidate.event_id, "soundfile": soundfile, "status": "skipped_near_duplicate"}
                )
                continue
            kept.append(candidate)
        if max_clips_per_source is not None and len(kept) > max_clips_per_source:
            for candidate in kept[max_clips_per_source:]:
                audit.append(
                    {"event_id": candidate.event_id, "soundfile": soundfile, "status": "skipped_source_limit"}
                )
            kept = kept[:max_clips_per_source]
        kept.sort(key=lambda item: item.peak_time_sec)
        selected.extend(kept)
    if max_total_clips is not None and len(selected) > max_total_clips:
        rng.shuffle(selected)
        rejected = selected[max_total_clips:]
        selected = selected[:max_total_clips]
        for candidate in rejected:
            audit.append(
                {"event_id": candidate.event_id, "soundfile": candidate.soundfile, "status": "skipped_total_limit"}
            )
    return selected, audit


def prepare_multispecies_candidates(
    windows_by_file: dict[str, list[dict[str, str]]],
    kw_event_candidates: list[Candidate],
    reviews: dict[str, str],
    require_reviewed: bool,
    thresholds: dict[str, float],
    minimum_peak_separation_sec: float,
    max_clips_per_class_per_source: Optional[int],
    max_clips_per_source: Optional[int],
    max_total_clips: Optional[int],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Select high-scoring background windows for KW, HW, and AB.

    Stable-event KW false positives are retained even when their species-KW
    probability is below the window threshold. Class/source quotas are applied
    before the final per-source quota so one noisy head cannot consume the
    complete mining budget.
    """
    audit: list[dict[str, Any]] = []
    event_by_window = {candidate.candidate_id: candidate for candidate in kw_event_candidates}
    candidates: dict[str, Candidate] = {}

    for soundfile, windows in windows_by_file.items():
        for window in windows:
            start = parse_float(window, "window_start_sec", soundfile)
            end = parse_float(window, "window_end_sec", soundfile)
            candidate_id = window_candidate_id(soundfile, start)
            event_candidate = event_by_window.get(candidate_id)
            actual_species = (window.get("species_ground_truth") or "").strip()
            if actual_species != "background" or truthy(window.get("ground_truth_ambiguous")):
                if event_candidate is not None:
                    audit.append({
                        "candidate_id": candidate_id,
                        "event_id": event_candidate.event_id,
                        "soundfile": soundfile,
                        "status": "skipped_nonbackground_window_truth",
                        "species_ground_truth": actual_species,
                    })
                continue

            scores = {
                label: parse_float(window, f"species_probability_{label}", candidate_id)
                for label in ("KW", "HW", "AB")
            }
            targets = {label for label, score in scores.items() if score >= thresholds[label]}
            if event_candidate is not None:
                targets.add("KW")
            if not targets:
                continue

            decision = reviews.get(candidate_id, "")
            if not decision and event_candidate is not None:
                decision = reviews.get(event_candidate.event_id, event_candidate.review_decision)
            normalized_decision = decision.casefold()
            if decision and normalized_decision not in ACCEPTED_REVIEW_DECISIONS:
                audit.append({
                    "candidate_id": candidate_id,
                    "event_id": event_candidate.event_id if event_candidate else "",
                    "soundfile": soundfile,
                    "status": "skipped_review_rejected",
                })
                continue
            if require_reviewed and normalized_decision not in ACCEPTED_REVIEW_DECISIONS:
                audit.append({
                    "candidate_id": candidate_id,
                    "event_id": event_candidate.event_id if event_candidate else "",
                    "soundfile": soundfile,
                    "status": "skipped_unapproved",
                })
                continue

            primary = max(targets, key=lambda label: (scores[label], label))
            candidate_score = max(scores[label] for label in targets)
            if event_candidate is not None:
                candidate_score = max(candidate_score, event_candidate.event_score)
                candidate = replace(
                    event_candidate,
                    review_decision=decision,
                    hard_negative_for=tuple(sorted(targets)),
                    primary_class=primary,
                    candidate_score=candidate_score,
                    source_kind="stable_event_fp+window_score",
                )
            else:
                center = (start + end) / 2.0
                candidate = Candidate(
                    event_id="",
                    soundfile=soundfile,
                    provider=row_context(window, "provider", "Provider"),
                    dataset=row_context(window, "dataset", "Dataset"),
                    peak_time_sec=center,
                    event_start_sec=start,
                    event_end_sec=end,
                    peak_score=candidate_score,
                    event_score=candidate_score,
                    predicted_ecotype="",
                    ecotype_confidence=0.0,
                    window_start_sec=start,
                    window_end_sec=end,
                    window_row=window,
                    review_decision=decision,
                    candidate_id=candidate_id,
                    hard_negative_for=tuple(sorted(targets)),
                    primary_class=primary,
                    candidate_score=candidate_score,
                    source_kind="window_score_false_positive",
                )
            candidates[candidate_id] = candidate

    # Apply independent per-class limits, then take their union. A window that
    # fools multiple heads consumes one WAV but remains tagged for every head.
    quota_selected: set[str] = set()
    by_class_source: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates.values():
        for label in candidate.hard_negative_for:
            by_class_source[(label, candidate.soundfile)].append(candidate)
    for (label, soundfile), values in by_class_source.items():
        ranked = sorted(
            values,
            key=lambda item: (
                -float(item.window_row[f"species_probability_{label}"]),
                -item.candidate_score,
                item.window_start_sec,
            ),
        )
        kept = (
            ranked
            if max_clips_per_class_per_source is None
            else ranked[:max_clips_per_class_per_source]
        )
        quota_selected.update(item.candidate_id for item in kept)
        for item in ranked[len(kept):]:
            audit.append({
                "candidate_id": item.candidate_id,
                "event_id": item.event_id,
                "soundfile": soundfile,
                "hard_negative_for": label,
                "status": "skipped_class_source_limit",
            })

    selected: list[Candidate] = []
    by_file: dict[str, list[Candidate]] = defaultdict(list)
    for candidate_id in quota_selected:
        by_file[candidates[candidate_id].soundfile].append(candidates[candidate_id])
    for soundfile in sorted(by_file):
        kept: list[Candidate] = []
        for candidate in sorted(
            by_file[soundfile],
            key=lambda item: (-item.candidate_score, item.window_start_sec),
        ):
            nearby_index = next(
                (
                    index
                    for index, other in enumerate(kept)
                    if abs(candidate.peak_time_sec - other.peak_time_sec)
                    < minimum_peak_separation_sec
                ),
                None,
            )
            if nearby_index is not None:
                other = kept[nearby_index]
                merged_targets = tuple(
                    sorted(set(other.hard_negative_for) | set(candidate.hard_negative_for))
                )
                kept[nearby_index] = replace(other, hard_negative_for=merged_targets)
                audit.append({
                    "candidate_id": candidate.candidate_id,
                    "event_id": candidate.event_id,
                    "soundfile": soundfile,
                    "status": "merged_near_duplicate",
                    "merged_into": other.candidate_id,
                })
                continue
            kept.append(candidate)
        if max_clips_per_source is not None and len(kept) > max_clips_per_source:
            for candidate in kept[max_clips_per_source:]:
                audit.append({
                    "candidate_id": candidate.candidate_id,
                    "event_id": candidate.event_id,
                    "soundfile": soundfile,
                    "status": "skipped_source_limit",
                })
            kept = kept[:max_clips_per_source]
        selected.extend(kept)

    selected.sort(key=lambda item: (-item.candidate_score, item.soundfile, item.window_start_sec))
    if max_total_clips is not None and len(selected) > max_total_clips:
        for candidate in selected[max_total_clips:]:
            audit.append({
                "candidate_id": candidate.candidate_id,
                "event_id": candidate.event_id,
                "soundfile": candidate.soundfile,
                "status": "skipped_total_limit",
            })
        selected = selected[:max_total_clips]
    selected.sort(key=lambda item: (item.soundfile, item.window_start_sec))
    return selected, audit


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def interval_overlaps_positive_annotation(
    start: float, end: float, annotations: list[Any], buffer_sec: float
) -> bool:
    for annotation in annotations:
        if getattr(annotation, "species", "") == "background":
            continue
        expanded_start = max(0.0, float(annotation.start_sec) - buffer_sec)
        expanded_end = float(annotation.end_sec) + buffer_sec
        if start < expanded_end and end > expanded_start:
            return True
    return False


def extract_audio_clip(
    source_path: Path, start_sec: float, duration_sec: float, target_sample_rate: int
) -> np.ndarray:
    import librosa
    import soundfile as sf

    with sf.SoundFile(source_path) as audio_file:
        source_rate = int(audio_file.samplerate)
        start_frame = max(0, int(round(start_sec * source_rate)))
        frame_count = max(1, int(round(duration_sec * source_rate)))
        audio_file.seek(min(start_frame, len(audio_file)))
        audio = audio_file.read(frame_count, dtype="float32", always_2d=True)
    waveform = (
        np.zeros(0, dtype=np.float32)
        if not audio.size
        else audio.mean(axis=1).astype(np.float32, copy=False)
    )
    if source_rate != target_sample_rate and len(waveform):
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=target_sample_rate
        )
    target_length = int(round(duration_sec * target_sample_rate))
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    return waveform[:target_length].astype(np.float32, copy=False)


def build_manifest_row(
    template: dict[str, str],
    candidate: Candidate,
    clip_name: str,
    local_clip_path: Path,
    published_clip_path: str,
    source_audio: str,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(template)
    row.update(
        {
            "ClassSpecies": "BKG",
            "KW": 0,
            "KW_certain": "",
            "Ecotype": "",
            "AnnotationLevel": "HardNegativeLongFP",
            "FileBeginSec": candidate.window_start_sec,
            "FileEndSec": candidate.window_end_sec,
            "CenterSec": (candidate.window_start_sec + candidate.window_end_sec) / 2.0,
            "ClipStartSec": candidate.window_start_sec,
            "ClipEndSec": candidate.window_end_sec,
            "clip_filename": clip_name,
            "clip_path": published_clip_path,
            "Generated": True,
            "hard_negative_source": candidate.source_kind,
            "hard_negative_for": ";".join(candidate.hard_negative_for),
            "hard_negative_primary_class": candidate.primary_class,
            "hard_negative_score": candidate.candidate_score,
            "source_candidate_id": candidate.candidate_id,
            "source_event_id": candidate.event_id,
            "source_audio_path": source_audio,
            "source_peak_time_sec": candidate.peak_time_sec,
            "source_event_start_sec": candidate.event_start_sec,
            "source_event_end_sec": candidate.event_end_sec,
            "source_event_peak_score": candidate.peak_score,
            "source_event_topk_mean": candidate.event_score,
            "source_predicted_ecotype": candidate.predicted_ecotype,
            "source_ecotype_confidence": candidate.ecotype_confidence,
            "source_provider": candidate.provider,
            "source_dataset": candidate.dataset,
            "review_decision": candidate.review_decision,
            "relative_clip_path": f"clips/{clip_name}",
            "local_clip_path": str(local_clip_path),
        }
    )
    for column in WINDOW_SCORE_COLUMNS:
        row[column] = candidate.window_row.get(column, "")
    return row


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def existing_candidate_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    fields, rows = read_csv(path)
    id_fields = [
        field for field in ("source_candidate_id", "source_event_id") if field in fields
    ]
    if not id_fields:
        return set()
    return {
        value
        for row in rows
        for field in id_fields
        if (value := (row.get(field) or "").strip())
    }


def write_kaggle_metadata(
    output_dir: Path, dataset_id: str, title: str, license_name: str
) -> None:
    if "/" not in dataset_id:
        raise ValueError("--kaggle-dataset-id must have the form owner/dataset-slug")
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-matches", default=DEFAULT_MATCHES)
    parser.add_argument("--stable-events", default=DEFAULT_EVENTS)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--review-csv", default=None)
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate_review.csv without indexing or downloading audio.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--minimum-event-score", type=float, default=0.0)
    parser.add_argument(
        "--kw-hard-negative-threshold", type=float, default=0.80,
        help="Minimum species-KW probability for independent window mining.",
    )
    parser.add_argument(
        "--humpback-hard-negative-threshold", type=float, default=0.80,
        help="Minimum species-HW probability for independent window mining.",
    )
    parser.add_argument(
        "--ab-hard-negative-threshold", type=float, default=0.80,
        help="Minimum species-AB probability for independent window mining.",
    )
    parser.add_argument("--minimum-peak-separation-sec", type=float, default=3.0)
    parser.add_argument("--annotation-buffer-sec", type=float, default=3.0)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--max-clips-per-class-per-source", type=int, default=20,
        help="Maximum selected windows for each target class in one recording.",
    )
    parser.add_argument(
        "--max-clips-per-source", type=int, default=60,
        help="Final maximum across KW, HW, and AB for one recording.",
    )
    parser.add_argument("--max-total-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--gcs-base", default=DEFAULT_GCS_BASE)
    parser.add_argument("--gcs-providers", default=DEFAULT_GCS_PROVIDERS)
    parser.add_argument("--max-file-gb", type=float, default=20.0)
    parser.add_argument("--kaggle-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--kaggle-title", default="DCLDE Long-Recording False-Positive Hard Negatives"
    )
    parser.add_argument("--kaggle-license", default="CC0-1.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    numeric_nonnegative = (
        args.minimum_event_score,
        args.minimum_peak_separation_sec,
        args.annotation_buffer_sec,
    )
    if any(value < 0 for value in numeric_nonnegative):
        raise ValueError("Score/separation/annotation-buffer values cannot be negative")
    if not 0 <= args.minimum_event_score <= 1:
        raise ValueError("--minimum-event-score must be between 0 and 1")
    class_thresholds = {
        "KW": args.kw_hard_negative_threshold,
        "HW": args.humpback_hard_negative_threshold,
        "AB": args.ab_hard_negative_threshold,
    }
    if any(not 0 <= value <= 1 for value in class_thresholds.values()):
        raise ValueError("All hard-negative thresholds must be between 0 and 1")
    if args.clip_duration <= 0 or args.sample_rate <= 0 or args.max_file_gb <= 0:
        raise ValueError("Clip duration/sample rate/max file size must be positive")
    if args.max_clips_per_source is not None and args.max_clips_per_source < 1:
        raise ValueError("--max-clips-per-source must be positive")
    if (
        args.max_clips_per_class_per_source is not None
        and args.max_clips_per_class_per_source < 1
    ):
        raise ValueError("--max-clips-per-class-per-source must be positive")
    if args.max_total_clips is not None and args.max_total_clips < 1:
        raise ValueError("--max-total-clips must be positive")

    match_path = Path(args.event_matches)
    event_path = Path(args.stable_events)
    window_path = Path(args.window_predictions)
    for path in (match_path, event_path, window_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else output_dir / "hard_negative_manifest.csv"
    )
    review_path = Path(args.review_csv) if args.review_csv else None

    fp_ids = false_positive_event_ids(match_path)
    event_rows = load_event_rows(event_path, fp_ids)
    windows_by_file = load_window_rows(window_path)
    reviews = load_reviews(review_path)
    kw_event_candidates, event_audit = prepare_candidates(
        event_rows,
        windows_by_file,
        {},
        False,
        args.minimum_event_score,
        args.minimum_peak_separation_sec,
        None,
        None,
        args.seed,
    )
    candidates, audit = prepare_multispecies_candidates(
        windows_by_file=windows_by_file,
        kw_event_candidates=kw_event_candidates,
        reviews=reviews,
        require_reviewed=args.require_reviewed,
        thresholds=class_thresholds,
        minimum_peak_separation_sec=args.minimum_peak_separation_sec,
        max_clips_per_class_per_source=args.max_clips_per_class_per_source,
        max_clips_per_source=args.max_clips_per_source,
        max_total_clips=args.max_total_clips,
    )
    audit = event_audit + audit
    if not reviews:
        print(
            "WARNING: no --review-csv was supplied. High-scoring background windows may include "
            "real but unannotated calls; audit clips before training."
        )
    if not candidates:
        raise ValueError(
            "No KW/HW/AB candidates remain after score/review/ground-truth filters"
        )

    review_rows = [
        {
            "candidate_id": candidate.candidate_id,
            "event_id": candidate.event_id,
            "soundfile": candidate.soundfile,
            "provider": candidate.provider,
            "dataset": candidate.dataset,
            "peak_time_sec": candidate.peak_time_sec,
            "window_start_sec": candidate.window_start_sec,
            "window_end_sec": candidate.window_end_sec,
            "peak_score": candidate.peak_score,
            "event_topk_mean": candidate.event_score,
            "hard_negative_for": ";".join(candidate.hard_negative_for),
            "hard_negative_primary_class": candidate.primary_class,
            "hard_negative_score": candidate.candidate_score,
            "species_probability_KW": candidate.window_row.get("species_probability_KW", ""),
            "species_probability_HW": candidate.window_row.get("species_probability_HW", ""),
            "species_probability_AB": candidate.window_row.get("species_probability_AB", ""),
            "predicted_ecotype": candidate.predicted_ecotype,
            "species_ground_truth": candidate.window_row.get("species_ground_truth", ""),
            "decision": candidate.review_decision,
        }
        for candidate in candidates
    ]
    review_fields = list(review_rows[0])
    write_rows(output_dir / "candidate_review.csv", review_rows, review_fields)
    if args.dry_run:
        audit_fields = sorted({key for row in audit for key in row})
        if audit_fields:
            write_rows(output_dir / "extraction_audit.csv", audit, audit_fields)
        print(f"Dry run complete: {len(candidates):,} candidates")
        print(f"Review file: {output_dir / 'candidate_review.csv'}")
        print("Fill the decision column, then rerun with --review-csv and --require-reviewed.")
        return 0

    import soundfile as sf
    import evaluate_dclde_long_recordings as base

    annotation_fields, annotation_rows = base.read_csv(args.annotations)
    sound_column = base.find_column(
        annotation_fields, ("Soundfile", "soundfile", "filename", "file")
    )
    selected_soundfiles = {candidate.soundfile for candidate in candidates}
    filtered_annotation_rows = [
        row
        for row in annotation_rows
        if base.normalize_text(row.get(sound_column)) in selected_soundfiles
    ]
    filtered_table = (annotation_fields, filtered_annotation_rows)
    inventory_args = SimpleNamespace(
        max_files=None,
        seed=args.seed,
        gcs_base=args.gcs_base,
        gcs_providers=args.gcs_providers,
        max_file_gb=args.max_file_gb,
    )
    recordings, gcs_sanity = base.load_recordings(inventory_args, filtered_table)
    recording_by_name = {recording.soundfile: recording for recording in recordings}
    annotations_by_file = base.load_all_labeled_annotations(filtered_table, recordings)
    template_by_file = {}
    for row in filtered_annotation_rows:
        template_by_file.setdefault(base.normalize_text(row.get(sound_column)), row)

    completed = set() if args.no_resume else existing_candidate_ids(output_manifest)
    existing_rows = []
    if output_manifest.is_file() and not args.no_resume:
        _, existing_rows = read_csv(output_manifest)
    published_root = f"/kaggle/input/datasets/{args.kaggle_dataset_id}"
    saved_rows = []
    saved_count = 0
    failed_count = 0
    candidates_by_file: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.candidate_id in completed or (
            candidate.event_id and candidate.event_id in completed
        ):
            audit.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "event_id": candidate.event_id,
                    "soundfile": candidate.soundfile,
                    "status": "skipped_resume",
                }
            )
        else:
            candidates_by_file[candidate.soundfile].append(candidate)

    for file_index, soundfile in enumerate(sorted(candidates_by_file), start=1):
        recording = recording_by_name.get(soundfile)
        if recording is None:
            for candidate in candidates_by_file[soundfile]:
                audit.append(
                    {"event_id": candidate.event_id, "soundfile": soundfile, "status": "source_not_found"}
                )
            failed_count += len(candidates_by_file[soundfile])
            continue
        try:
            with base.materialize_audio(recording.audio_source, soundfile, args.temp_dir) as source_path:
                duration_sec, _, _ = base.audio_metadata(source_path)
                for candidate in candidates_by_file[soundfile]:
                    start = min(max(0.0, candidate.window_start_sec), max(0.0, duration_sec - args.clip_duration))
                    end = start + args.clip_duration
                    if interval_overlaps_positive_annotation(
                        start,
                        end,
                        annotations_by_file.get(soundfile, []),
                        args.annotation_buffer_sec,
                    ):
                        audit.append(
                            {
                                "candidate_id": candidate.candidate_id,
                                "event_id": candidate.event_id,
                                "soundfile": soundfile,
                                "status": "skipped_annotation_buffer",
                            }
                        )
                        continue
                    clip_name = (
                        f"longfp_{candidate.primary_class.lower()}_"
                        f"{safe_name(Path(soundfile).stem)}_"
                        f"{int(round(start * 1000)):010d}ms_"
                        f"s{int(round(candidate.event_score * 1000)):03d}.wav"
                    )
                    local_clip_path = clips_dir / clip_name
                    waveform = extract_audio_clip(
                        source_path, start, args.clip_duration, args.sample_rate
                    )
                    sf.write(local_clip_path, waveform, args.sample_rate)
                    published_clip_path = f"{published_root}/clips/{clip_name}"
                    saved_rows.append(
                        build_manifest_row(
                            template_by_file.get(soundfile, {}),
                            candidate,
                            clip_name,
                            local_clip_path,
                            published_clip_path,
                            recording.audio_source,
                        )
                    )
                    audit.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "event_id": candidate.event_id,
                            "soundfile": soundfile,
                            "hard_negative_for": ";".join(candidate.hard_negative_for),
                            "status": "saved",
                        }
                    )
                    saved_count += 1
            print(
                f"[{file_index}/{len(candidates_by_file)}] {soundfile}: "
                f"candidates={len(candidates_by_file[soundfile])}"
            )
        except Exception as error:
            failed_count += len(candidates_by_file[soundfile])
            for candidate in candidates_by_file[soundfile]:
                audit.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "event_id": candidate.event_id,
                        "soundfile": soundfile,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            print(f"ERROR {soundfile}: {type(error).__name__}: {error}")

    all_rows = existing_rows + saved_rows
    manifest_fields = list(annotation_fields)
    for field in (
        "ClassSpecies", "KW", "KW_certain", "Ecotype", "AnnotationLevel",
        "FileBeginSec", "FileEndSec", "CenterSec", "ClipStartSec", "ClipEndSec",
        "clip_filename", "clip_path", "Generated", *PROVENANCE_COLUMNS,
    ):
        if field not in manifest_fields:
            manifest_fields.append(field)
    write_rows(output_manifest, all_rows, manifest_fields)
    audit_fields = sorted({key for row in audit for key in row})
    write_rows(output_dir / "extraction_audit.csv", audit, audit_fields)
    write_kaggle_metadata(
        output_dir, args.kaggle_dataset_id, args.kaggle_title, args.kaggle_license
    )
    summary = {
        "input_false_positive_events": len(fp_ids),
        "eligible_candidates": len(candidates),
        "eligible_candidates_by_target": {
            label: sum(label in candidate.hard_negative_for for candidate in candidates)
            for label in ("KW", "HW", "AB")
        },
        "new_clips_saved": saved_count,
        "new_clips_saved_by_target": {
            label: sum(
                label in str(row.get("hard_negative_for", "")).split(";")
                for row in saved_rows
            )
            for label in ("KW", "HW", "AB")
        },
        "total_manifest_rows": len(all_rows),
        "failed_candidates": failed_count,
        "audit_status_counts": dict(Counter(row.get("status", "") for row in audit)),
        "gcs_sanity": gcs_sanity,
        "arguments": vars(args),
    }
    (output_dir / "mining_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nLong-recording multispecies hard-negative extraction")
    print("=" * 64)
    print(f"Input FP events:       {len(fp_ids):,}")
    print(f"Eligible candidates:   {len(candidates):,}")
    print(
        "Candidate targets:     "
        + ", ".join(
            f"{label}={sum(label in candidate.hard_negative_for for candidate in candidates):,}"
            for label in ("KW", "HW", "AB")
        )
    )
    print(f"New 3-second WAVs:     {saved_count:,}")
    print(f"Manifest rows total:   {len(all_rows):,}")
    print(f"Output directory:      {output_dir}")
    print(f"Training manifest:     {output_manifest}")
    print("\nReview extraction_audit.csv and listen to candidates before training.")
    print("To create the Kaggle dataset:")
    print(f"  kaggle datasets create -p {output_dir} --dir-mode zip")
    print("For a later update:")
    print(
        f"  kaggle datasets version -p {output_dir} -m "
        '"Add long-recording false-positive hard negatives" --dir-mode zip'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
