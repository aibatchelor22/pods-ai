#!/usr/bin/env python3
"""Clean annotations and plan leakage-safe, size-bounded audio extraction shards.

This is the first stage of the Multispecies Cetacean V2 dataset build. It does
not download audio. It creates an auditable master annotation table, assigns
entire source recordings to train/validation/test, plans provider-balanced
original background windows, inventories the public GCS objects, and packs
recordings into extraction jobs that fit Kaggle storage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


VALID_CLASSES = {"AB", "KW", "HW", "UndBio"}
VALID_ECOTYPES = {"NRKW", "SRKW", "OKW", "SAR", "TKW"}
VALID_LEVELS = {"Call", "Detection", "File"}
MODEL_LABELS = ("Abiotic", "KW", "HW", "UndBio")
MISSING_TEXT = {"", "na", "n/a", "nan", "none", "null"}
DEFAULT_GCS_ROOT = (
    "noaa-passive-bioacoustic/dclde/2027/"
    "dclde_2027_killer_whales"
)
PROVIDER_GCS_FOLDERS = {
    "dfo_crp": "dfo_crp",
    "dfo_wdlp": "dfo_wdlp",
    "jasco_vfpa": "vfpa",
    "jasco_vfpa_onc": "vfpa",
    "onc": "onc",
    "orcasound": "orcasound",
    "simres": "simres",
    "sio": "scripps",
    "smruconsulting": "smru",
    "uaf_ngos": "uaf",
}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in MISSING_TEXT else text


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def normalized_file_ok(value: Any) -> bool:
    return clean_text(value).casefold() in {"true", "1", "yes", "y"}


def normalized_kw_certainty(value: Any) -> str:
    text = clean_text(value)
    if text in {"0", "0.0"}:
        return "0"
    if text in {"1", "1.0"}:
        return "1"
    return ""


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "unknown"


def slug_component(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value or "unknown"


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64)


def stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def annotation_key(
    row: dict[str, str], begin: float | None, end: float | None
) -> tuple[object, ...]:
    return (
        clean_text(row.get("Provider")).casefold(),
        clean_text(row.get("Dataset")).casefold(),
        clean_text(row.get("Soundfile")).casefold(),
        begin,
        end,
        parse_float(row.get("LowFreqHz")),
        parse_float(row.get("HighFreqHz")),
        clean_text(row.get("ClassSpecies")),
        clean_text(row.get("Ecotype")),
        clean_text(row.get("AnnotationLevel")),
    )


def process_annotations(
    source_rows: Iterable[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    seen_annotations: set[tuple[object, ...]] = set()
    audited: list[dict[str, str]] = []
    eligible: list[dict[str, str]] = []

    for source_index, source in enumerate(source_rows, start=2):
        row = dict(source)
        class_species = clean_text(source.get("ClassSpecies"))
        annotation_level = clean_text(source.get("AnnotationLevel"))
        ecotype = clean_text(source.get("Ecotype"))
        begin = parse_float(source.get("FileBeginSec"))
        end = parse_float(source.get("FileEndSec"))
        duration = None if begin is None or end is None else end - begin
        kw_certain = normalized_kw_certainty(source.get("KW_certain"))
        soundfile = clean_text(source.get("Soundfile"))
        provider = clean_text(source.get("Provider"))
        dataset = clean_text(source.get("Dataset"))

        duplicate_key = annotation_key(source, begin, end)
        exact_duplicate = duplicate_key in seen_annotations
        seen_annotations.add(duplicate_key)

        reasons: list[str] = []
        if not soundfile:
            reasons.append("missing_soundfile")
        if not normalized_file_ok(source.get("FileOk")):
            reasons.append("file_not_ok")
        if class_species not in VALID_CLASSES:
            reasons.append("unknown_class_species")
        if annotation_level not in VALID_LEVELS:
            reasons.append("unknown_annotation_level")
        elif annotation_level == "File":
            reasons.append("file_level_annotation")
        if begin is None or end is None:
            reasons.append("invalid_time_bounds")
        elif duration is None or duration <= 0:
            reasons.append("zero_or_negative_duration")
        if class_species == "KW" and kw_certain == "0":
            reasons.append("uncertain_kw")
        if exact_duplicate:
            reasons.append("exact_duplicate_annotation")

        model_label = {
            "AB": "Abiotic",
            "KW": "KW",
            "HW": "HW",
            "UndBio": "UndBio",
        }.get(class_species, "")
        valid_ecotype = class_species == "KW" and ecotype in VALID_ECOTYPES
        source_recording_id = "|".join((provider, dataset, soundfile))
        annotation_id = stable_id(
            "ann",
            provider,
            dataset,
            soundfile,
            begin,
            end,
            row.get("LowFreqHz", ""),
            row.get("HighFreqHz", ""),
            class_species,
            ecotype,
        )
        is_eligible = not reasons

        row.update(
            {
                "source_csv_row": str(source_index),
                "annotation_id": annotation_id,
                "source_recording_id": source_recording_id,
                "clean_class_species": class_species
                if class_species in VALID_CLASSES
                else "",
                "model_source_label": model_label,
                "clean_ecotype": ecotype if valid_ecotype else "",
                "annotation_duration_sec": ""
                if duration is None
                else format(duration, ".9g"),
                "is_file_level": bool_text(annotation_level == "File"),
                "is_zero_or_negative_duration": bool_text(
                    duration is not None and duration <= 0
                ),
                "is_uncertain_kw": bool_text(
                    class_species == "KW" and kw_certain == "0"
                ),
                "is_exact_duplicate_annotation": bool_text(exact_duplicate),
                "source_head_eligible": bool_text(is_eligible),
                "ecotype_head_eligible": bool_text(is_eligible and valid_ecotype),
                "exclusion_reasons": "|".join(reasons),
            }
        )
        audited.append(row)
        if is_eligible:
            eligible.append(row)

    return audited, eligible


def provider_folder(provider: str) -> str:
    key = provider.strip().casefold()
    return PROVIDER_GCS_FOLDERS.get(
        key, slug_component(provider).replace("-", "_")
    )


def inventory_gcs(
    providers: Iterable[str], root: str, quiet: bool
) -> tuple[
    dict[tuple[str, str], list[dict[str, object]]],
    dict[tuple[str, str], list[dict[str, object]]],
    list[dict[str, str]],
]:
    try:
        import gcsfs  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gcsfs is required for inventory. In Kaggle run: pip install -q gcsfs"
        ) from exc

    fs = gcsfs.GCSFileSystem(token="anon")
    by_provider_basename: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_provider_stem: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    failures: list[dict[str, str]] = []

    folder_providers: dict[str, list[str]] = defaultdict(list)
    for provider in sorted(set(providers)):
        folder_providers[provider_folder(provider)].append(provider)

    for folder, annotation_providers in sorted(folder_providers.items()):
        search_root = f"{root.rstrip('/')}/{folder}/audio"
        if not quiet:
            print(
                "Indexing bucket folder for "
                + ", ".join(annotation_providers)
                + " ..."
            )
        try:
            found = fs.find(search_root, detail=True)
            if isinstance(found, list):
                found = {path: fs.info(path) for path in found}
            count = 0
            for path, info in found.items():
                suffix = PurePosixPath(path).suffix.casefold()
                if suffix not in {".wav", ".flac", ".mp3", ".aif", ".aiff"}:
                    continue
                size = int((info or {}).get("size", 0) or 0)
                item = {"gcs_path": str(path), "source_size_bytes": size}
                basename = PurePosixPath(path).name.casefold()
                stem = PurePosixPath(path).stem.casefold()
                for provider in annotation_providers:
                    by_provider_basename[(provider.casefold(), basename)].append(item)
                    by_provider_stem[(provider.casefold(), stem)].append(item)
                count += 1
            if not quiet:
                print(f"  indexed {count:,} audio objects")
        except Exception as exc:  # preserve other providers and report failure
            for provider in annotation_providers:
                failures.append(
                    {
                        "Provider": provider,
                        "search_root": search_root,
                        "error": repr(exc),
                    }
                )
            print(
                "WARNING: inventory failed for "
                + ", ".join(annotation_providers)
                + f": {exc}"
            )

    return by_provider_basename, by_provider_stem, failures


def choose_audio_match(
    provider: str,
    dataset: str,
    soundfile: str,
    index: dict[tuple[str, str], list[dict[str, object]]],
    stem_index: dict[tuple[str, str], list[dict[str, object]]],
) -> tuple[str, int, str]:
    candidates = index.get((provider.casefold(), soundfile.casefold()), [])
    matched_by_stem = False
    if not candidates:
        candidates = stem_index.get(
            (provider.casefold(), PurePosixPath(soundfile).stem.casefold()), []
        )
        matched_by_stem = bool(candidates)
    if not candidates:
        return "", 0, "missing"
    if len(candidates) == 1:
        item = candidates[0]
        status = "matched_by_stem" if matched_by_stem else "matched"
        return str(item["gcs_path"]), int(item["source_size_bytes"]), status

    dataset_token = slug_component(dataset).replace("-", "")
    narrowed = [
        item
        for item in candidates
        if dataset_token
        and dataset_token
        in slug_component(str(item["gcs_path"])).replace("-", "")
    ]
    if len(narrowed) == 1:
        item = narrowed[0]
        return (
            str(item["gcs_path"]),
            int(item["source_size_bytes"]),
            "matched_by_dataset_and_stem" if matched_by_stem else "matched_by_dataset",
        )
    return "", 0, "ambiguous"


def assign_split(
    recording_id: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    value = stable_fraction(seed, recording_id)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def assign_recording_splits(
    grouped: dict[str, list[dict[str, str]]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    search_trials: int,
) -> dict[str, str]:
    """Select a deterministic recording split with balanced evaluation labels.

    Each provider is optimized separately so all providers remain represented
    near the requested proportions. Multiple seeded recording-level assignments
    are evaluated within each provider, balancing source labels, KW ecotypes,
    datasets, annotation counts, and recording counts. Audio from a source
    recording is never divided.
    """
    if search_trials < 1:
        raise ValueError("search_trials must be at least 1")
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    dataset_totals: Counter[tuple[str, str]] = Counter()
    for rows in grouped.values():
        dataset_totals[(rows[0]["Provider"], rows[0]["Dataset"])] += len(rows)

    provider_features: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)
    for recording_id, rows in grouped.items():
        label_counts = Counter(row["model_source_label"] for row in rows)
        ecotype_counts = Counter(
            row["clean_ecotype"] for row in rows if row["clean_ecotype"]
        )
        first = rows[0]
        features: dict[str, int] = {"annotations": len(rows), "recordings": 1}
        for label in MODEL_LABELS:
            features[f"label:{label}"] = label_counts[label]
        for ecotype in sorted(VALID_ECOTYPES):
            features[f"ecotype:{ecotype}"] = ecotype_counts[ecotype]
        if dataset_totals[(first["Provider"], first["Dataset"])] >= 100:
            features[f"dataset:{first['Dataset']}"] = len(rows)
        provider_features[first["Provider"]].append((recording_id, features))

    all_assignments: dict[str, str] = {}
    split_names = tuple(fractions)
    for provider, recording_features in sorted(provider_features.items()):
        recording_features.sort(key=lambda item: item[0])
        feature_totals: Counter[str] = Counter()
        for _, features in recording_features:
            feature_totals.update(features)
        best_score = math.inf
        best_assignments: dict[str, str] = {}
        best_trial = -1

        for trial in range(search_trials):
            rng = random.Random(
                f"{seed}|multispecies-cetacean-split|{provider}|{trial}"
            )
            assignments: dict[str, str] = {}
            split_features = {split: Counter() for split in split_names}
            for recording_id, features in recording_features:
                value = rng.random()
                if value < train_fraction:
                    split = "train"
                elif value < train_fraction + validation_fraction:
                    split = "validation"
                else:
                    split = "test"
                assignments[recording_id] = split
                split_features[split].update(features)

            grouped_errors: dict[str, list[float]] = defaultdict(list)
            for feature, total in feature_totals.items():
                if total <= 0:
                    continue
                if feature.startswith("label:"):
                    group_name = "labels"
                elif feature.startswith("ecotype:"):
                    group_name = "ecotypes"
                elif feature.startswith("dataset:"):
                    group_name = "datasets"
                else:
                    group_name = "totals"
                for split, fraction in fractions.items():
                    expected = total * fraction
                    observed = split_features[split][feature]
                    grouped_errors[group_name].append(
                        ((observed - expected) / expected) ** 2
                    )

            group_weights = {
                "totals": 0.15,
                "labels": 0.30,
                "ecotypes": 0.35,
                "datasets": 0.20,
            }
            score = sum(
                group_weights[name] * (sum(values) / len(values))
                for name, values in grouped_errors.items()
                if values
            )
            core_errors = grouped_errors["labels"] + grouped_errors["ecotypes"]
            if core_errors:
                score += 0.25 * max(core_errors)
            if grouped_errors["datasets"]:
                score += 0.10 * max(grouped_errors["datasets"])
            if score < best_score:
                best_score = score
                best_assignments = assignments
                best_trial = trial

        all_assignments.update(best_assignments)
        print(
            f"Selected split trial {best_trial} of {search_trials} for "
            f"{provider} (balance score {best_score:.8f})"
        )

    return all_assignments


def make_dataset_names(split: str, number: int) -> tuple[str, str]:
    split_title = {"train": "Training", "validation": "Validation", "test": "Test"}[split]
    title = f"Multispecies Cetacean V2 {split_title} Original {number:02d}"
    slug_split = {"train": "train", "validation": "validation", "test": "test"}[split]
    slug = f"multispecies-cetacean-v2-{slug_split}-original-{number:02d}"
    return title, slug


def pack_shards(
    recordings: list[dict[str, object]],
    target_output_bytes: int,
    max_working_bytes: int,
    safety_bytes: int,
    remote_buffer_bytes: int,
    owner: str,
) -> list[dict[str, object]]:
    all_shards: list[dict[str, object]] = []

    for split in ("train", "validation", "test"):
        split_records = [row for row in recordings if row["split"] == split]
        split_records.sort(
            key=lambda row: (
                int(row["estimated_output_bytes"]),
                int(row["source_size_bytes"]),
            ),
            reverse=True,
        )
        bins: list[dict[str, object]] = []

        for record in split_records:
            source_bytes = int(record["source_size_bytes"])
            output_bytes = int(record["estimated_output_bytes"])
            local_working_source = source_bytes
            if output_bytes + source_bytes + safety_bytes > max_working_bytes:
                record["extraction_mode"] = "remote_seek"
                local_working_source = remote_buffer_bytes
            else:
                record["extraction_mode"] = "download_then_extract"
            record["working_source_bytes"] = local_working_source

            selected: dict[str, object] | None = None
            for shard in bins:
                new_output = int(shard["estimated_output_bytes"]) + output_bytes
                new_largest = max(
                    int(shard["largest_working_source_bytes"]), local_working_source
                )
                if (
                    new_output <= target_output_bytes
                    and new_output + new_largest + safety_bytes <= max_working_bytes
                ):
                    selected = shard
                    break

            if selected is None:
                if output_bytes + local_working_source + safety_bytes > max_working_bytes:
                    raise RuntimeError(
                        "A single recording's extracted clips cannot fit the configured "
                        f"working limit: {record['source_recording_id']}"
                    )
                selected = {
                    "split": split,
                    "records": [],
                    "estimated_output_bytes": 0,
                    "largest_working_source_bytes": 0,
                }
                bins.append(selected)

            selected["records"].append(record)  # type: ignore[index,union-attr]
            selected["estimated_output_bytes"] = (
                int(selected["estimated_output_bytes"]) + output_bytes
            )
            selected["largest_working_source_bytes"] = max(
                int(selected["largest_working_source_bytes"]), local_working_source
            )

        for number, shard in enumerate(bins):
            title, slug = make_dataset_names(split, number)
            shard_id = f"{split}_original_{number:02d}"
            storage_key = shard_id
            dataset_id = f"{owner}/{slug}" if owner else slug
            for record in shard["records"]:  # type: ignore[union-attr]
                record["shard_id"] = shard_id
                record["storage_key"] = storage_key
                record["kaggle_dataset_id"] = dataset_id
            shard.update(
                {
                    "shard_id": shard_id,
                    "storage_key": storage_key,
                    "dataset_title": title,
                    "kaggle_dataset_id": dataset_id,
                    "recording_count": len(shard["records"]),  # type: ignore[arg-type]
                    "clip_count": sum(
                        int(row["planned_clip_count"])
                        for row in shard["records"]  # type: ignore[union-attr]
                    ),
                    "estimated_peak_working_bytes": (
                        int(shard["estimated_output_bytes"])
                        + int(shard["largest_working_source_bytes"])
                        + safety_bytes
                    ),
                }
            )
            all_shards.append(shard)

    return all_shards


def merged_annotation_intervals(
    rows: Iterable[dict[str, str]],
) -> list[list[float]]:
    intervals: list[tuple[float, float]] = []
    for row in rows:
        begin = parse_float(row.get("FileBeginSec"))
        end = parse_float(row.get("FileEndSec"))
        if begin is not None and end is not None and end > begin:
            intervals.append((begin, end))
    intervals.sort()
    merged: list[list[float]] = []
    for begin, end in intervals:
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def allocate_background_requests(
    recordings: list[dict[str, object]],
    target_per_provider: int,
    max_per_recording: int,
    orcasound_max_per_recording: int,
    seed: int,
    fractions: dict[str, float],
) -> None:
    """Allocate requested background windows broadly across source recordings."""
    for record in recordings:
        record["ambient_windows_requested"] = 0

    if target_per_provider <= 0:
        return

    providers = sorted({str(record["Provider"]) for record in recordings})
    for provider in providers:
        for split, fraction in fractions.items():
            target = round(target_per_provider * fraction)
            candidates = [
                record
                for record in recordings
                if record["Provider"] == provider
                and record["split"] == split
                and record.get("background_policy")
                and (
                    str(record.get("audio_match_status", "")).startswith("matched")
                    or record.get("audio_match_status") == "inventory_skipped"
                )
            ]
            candidates.sort(
                key=lambda record: (
                    int(record["annotation_count"]),
                    stable_fraction(
                        seed,
                        f"background|{provider}|{split}|{record['source_recording_id']}",
                    ),
                )
            )
            cap = (
                orcasound_max_per_recording
                if provider.casefold() == "orcasound"
                else max_per_recording
            )
            remaining = target
            while remaining > 0 and candidates:
                added = False
                for record in candidates:
                    current = int(record["ambient_windows_requested"])
                    if current >= cap:
                        continue
                    record["ambient_windows_requested"] = current + 1
                    remaining -= 1
                    added = True
                    if remaining == 0:
                        break
                if not added:
                    break


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def human_gb(value: int) -> str:
    return f"{value / 1024**3:.2f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument(
        "--output-dir", default="/kaggle/working/multispecies_cetacean_plan"
    )
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument(
        "--split-search-trials",
        type=int,
        default=512,
        help="Deterministic candidate recording splits evaluated for balance.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--estimated-clip-bytes",
        type=int,
        default=100_000,
        help="Conservative per-clip storage estimate for 16 kHz PCM16 FLAC.",
    )
    parser.add_argument("--target-shard-output-gb", type=float, default=4.0)
    parser.add_argument("--max-working-gb", type=float, default=14.0)
    parser.add_argument("--working-safety-gb", type=float, default=2.0)
    parser.add_argument("--remote-seek-buffer-gb", type=float, default=0.25)
    parser.add_argument(
        "--ambient-target-per-provider",
        type=int,
        default=2000,
        help="Requested original background windows per provider across all splits.",
    )
    parser.add_argument(
        "--ambient-max-per-recording",
        type=int,
        default=20,
        help="Maximum requested background windows from a non-Orcasound recording.",
    )
    parser.add_argument(
        "--orcasound-ambient-max-per-recording",
        type=int,
        default=4,
        help="Maximum windows from each 60-second Orcasound AB-only recording.",
    )
    parser.add_argument(
        "--ambient-safety-margin-seconds",
        type=float,
        default=60.0,
        help="Exclusion margin on both sides of annotations in long recordings.",
    )
    parser.add_argument(
        "--ambient-min-spacing-seconds",
        type=float,
        default=15.0,
        help="Minimum spacing between background windows from one recording.",
    )
    parser.add_argument(
        "--disable-ambient-background",
        action="store_true",
        help="Plan event-centered annotated clips only.",
    )
    parser.add_argument("--kaggle-owner", default=os.environ.get("KAGGLE_USERNAME", ""))
    parser.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
    parser.add_argument(
        "--skip-gcs-inventory",
        action="store_true",
        help="Create a preliminary plan without matching or sizing remote audio.",
    )
    parser.add_argument("--quiet-gcs", action="store_true")
    args = parser.parse_args()

    fractions = args.train_fraction + args.validation_fraction + args.test_fraction
    if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-9):
        parser.error("train, validation, and test fractions must sum to 1")
    if args.clip_seconds <= 0 or args.sample_rate <= 0:
        parser.error("clip seconds and sample rate must be positive")
    if (
        args.ambient_target_per_provider < 0
        or args.ambient_max_per_recording < 1
        or args.orcasound_ambient_max_per_recording < 1
        or args.ambient_safety_margin_seconds < 0
        or args.ambient_min_spacing_seconds < args.clip_seconds
    ):
        parser.error("invalid ambient-background sampling settings")

    annotations_path = Path(args.annotations_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        source_rows = list(reader)

    required = {
        "Soundfile",
        "Dataset",
        "FileBeginSec",
        "FileEndSec",
        "ClassSpecies",
        "KW_certain",
        "Ecotype",
        "Provider",
        "AnnotationLevel",
        "FileOk",
    }
    if missing := required.difference(original_fields):
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    audited, eligible = process_annotations(source_rows)
    derived_fields = [field for field in audited[0] if field not in original_fields]
    write_csv(
        output_dir / "multispecies_cetacean_master_annotation_audit.csv",
        original_fields + derived_fields,
        audited,
    )
    write_csv(
        output_dir / "multispecies_cetacean_master_cleaned_annotations.csv",
        original_fields + derived_fields,
        eligible,
    )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        grouped[row["source_recording_id"]].append(row)

    audited_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audited:
        if row["Soundfile"] and row["Provider"] and row["Dataset"]:
            audited_grouped[row["source_recording_id"]].append(row)

    planning_grouped = dict(grouped)
    if not args.disable_ambient_background:
        for recording_id, rows in audited_grouped.items():
            classes = {
                clean_text(row.get("ClassSpecies"))
                for row in rows
                if clean_text(row.get("ClassSpecies"))
            }
            if (
                rows[0]["Provider"].casefold() == "orcasound"
                and classes == {"AB"}
                and all(normalized_file_ok(row.get("FileOk")) for row in rows)
                and recording_id not in planning_grouped
            ):
                # File-level AB rows do not create event-centered clips, but an
                # AB-only 60-second recording is a useful original negative.
                planning_grouped[recording_id] = [rows[0]]

    split_assignments = assign_recording_splits(
        planning_grouped,
        args.seed,
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
        args.split_search_trials,
    )

    if args.skip_gcs_inventory:
        audio_index: dict[tuple[str, str], list[dict[str, object]]] = {}
        audio_stem_index: dict[tuple[str, str], list[dict[str, object]]] = {}
        inventory_failures: list[dict[str, str]] = []
    else:
        audio_index, audio_stem_index, inventory_failures = inventory_gcs(
            (rows[0]["Provider"] for rows in planning_grouped.values()),
            args.gcs_root,
            args.quiet_gcs,
        )

    recordings: list[dict[str, object]] = []
    missing_audio: list[dict[str, object]] = []
    for recording_id, split_rows in planning_grouped.items():
        positive_rows = grouped.get(recording_id, [])
        all_rows = audited_grouped[recording_id]
        first = split_rows[0]
        split = split_assignments[recording_id]
        label_counts = Counter(row["model_source_label"] for row in positive_rows)
        ecotype_counts = Counter(
            row["clean_ecotype"] for row in positive_rows if row["clean_ecotype"]
        )
        annotation_classes = {
            clean_text(row.get("ClassSpecies"))
            for row in all_rows
            if clean_text(row.get("ClassSpecies"))
        }
        all_files_ok = all(normalized_file_ok(row.get("FileOk")) for row in all_rows)
        has_file_level = any(
            clean_text(row.get("AnnotationLevel")) == "File" for row in all_rows
        )
        background_policy = ""
        if not args.disable_ambient_background and all_files_ok:
            if first["Provider"].casefold() == "orcasound":
                if annotation_classes == {"AB"}:
                    background_policy = "orcasound_abiotic_only"
            elif not has_file_level:
                background_policy = "standard_annotation_margin"
        if args.skip_gcs_inventory:
            gcs_path, source_size, match_status = "", 0, "inventory_skipped"
        else:
            gcs_path, source_size, match_status = choose_audio_match(
                first["Provider"],
                first["Dataset"],
                first["Soundfile"],
                audio_index,
                audio_stem_index,
            )
        record: dict[str, object] = {
            "source_recording_id": recording_id,
            "Provider": first["Provider"],
            "Dataset": first["Dataset"],
            "Soundfile": first["Soundfile"],
            "split": split,
            "annotation_count": len(positive_rows),
            **{f"count_{label}": label_counts[label] for label in MODEL_LABELS},
            **{f"count_ecotype_{label}": ecotype_counts[label] for label in sorted(VALID_ECOTYPES)},
            "gcs_path": gcs_path,
            "source_size_bytes": source_size,
            "audio_match_status": match_status,
            "background_policy": background_policy,
            "ambient_windows_requested": 0,
            "planned_clip_count": len(positive_rows),
            "estimated_output_bytes": len(positive_rows) * args.estimated_clip_bytes,
        }
        if match_status not in {
            "matched",
            "matched_by_stem",
            "matched_by_dataset",
            "matched_by_dataset_and_stem",
            "inventory_skipped",
        }:
            missing_audio.append(record)
        recordings.append(record)

    matched_recordings = [
        row
        for row in recordings
        if row["audio_match_status"]
        in {
            "matched",
            "matched_by_stem",
            "matched_by_dataset",
            "matched_by_dataset_and_stem",
            "inventory_skipped",
        }
    ]
    allocate_background_requests(
        matched_recordings,
        0 if args.disable_ambient_background else args.ambient_target_per_provider,
        args.ambient_max_per_recording,
        args.orcasound_ambient_max_per_recording,
        args.seed,
        {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
    )
    for record in recordings:
        planned_count = int(record["annotation_count"]) + int(
            record["ambient_windows_requested"]
        )
        record["planned_clip_count"] = planned_count
        record["estimated_output_bytes"] = planned_count * args.estimated_clip_bytes

    extraction_recordings = [
        row for row in matched_recordings if int(row["planned_clip_count"]) > 0
    ]
    shards = pack_shards(
        extraction_recordings,
        int(args.target_shard_output_gb * 1024**3),
        int(args.max_working_gb * 1024**3),
        int(args.working_safety_gb * 1024**3),
        int(args.remote_seek_buffer_gb * 1024**3),
        args.kaggle_owner,
    )

    recording_fields = [
        "source_recording_id",
        "Provider",
        "Dataset",
        "Soundfile",
        "split",
        "annotation_count",
        *[f"count_{label}" for label in MODEL_LABELS],
        *[f"count_ecotype_{label}" for label in sorted(VALID_ECOTYPES)],
        "gcs_path",
        "source_size_bytes",
        "audio_match_status",
        "background_policy",
        "ambient_windows_requested",
        "planned_clip_count",
        "estimated_output_bytes",
        "extraction_mode",
        "working_source_bytes",
        "shard_id",
        "storage_key",
        "kaggle_dataset_id",
    ]
    write_csv(
        output_dir / "multispecies_cetacean_source_recording_plan.csv",
        recording_fields,
        recordings,
    )
    if missing_audio:
        write_csv(
            output_dir / "multispecies_cetacean_unmatched_audio.csv",
            recording_fields,
            missing_audio,
        )
    if inventory_failures:
        write_csv(
            output_dir / "multispecies_cetacean_inventory_failures.csv",
            ["Provider", "search_root", "error"],
            inventory_failures,
        )

    by_recording = {str(row["source_recording_id"]): row for row in recordings}
    extraction_rows: list[dict[str, object]] = []
    for row in eligible:
        record = by_recording[row["source_recording_id"]]
        if not record.get("shard_id"):
            continue
        begin = float(row["FileBeginSec"])
        end = float(row["FileEndSec"])
        requested_start = max(0.0, ((begin + end) / 2.0) - args.clip_seconds / 2.0)
        extension = PurePosixPath(row["Soundfile"]).suffix.casefold()
        relative_path = (
            f"clips/{safe_component(row['model_source_label'])}/"
            f"{row['annotation_id']}.flac"
        )
        extraction_rows.append(
            {
                **row,
                "split": record["split"],
                "shard_id": record["shard_id"],
                "storage_key": record["storage_key"],
                "kaggle_dataset_id": record["kaggle_dataset_id"],
                "gcs_path": record["gcs_path"],
                "source_size_bytes": record["source_size_bytes"],
                "source_extension": extension,
                "extraction_mode": record["extraction_mode"],
                "clip_start_requested_sec": format(requested_start, ".9g"),
                "clip_duration_sec": format(args.clip_seconds, ".9g"),
                "target_sample_rate": args.sample_rate,
                "target_sample_count": round(args.clip_seconds * args.sample_rate),
                "relative_clip_path": relative_path,
            }
        )

    extraction_derived = [
        "split",
        "shard_id",
        "storage_key",
        "kaggle_dataset_id",
        "gcs_path",
        "source_size_bytes",
        "source_extension",
        "extraction_mode",
        "clip_start_requested_sec",
        "clip_duration_sec",
        "target_sample_rate",
        "target_sample_count",
        "relative_clip_path",
    ]
    write_csv(
        output_dir / "multispecies_cetacean_extraction_plan.csv",
        original_fields + derived_fields + extraction_derived,
        extraction_rows,
    )

    background_rows: list[dict[str, object]] = []
    for record in recordings:
        requested = int(record["ambient_windows_requested"])
        if requested <= 0 or not record.get("shard_id"):
            continue
        recording_id = str(record["source_recording_id"])
        policy = str(record["background_policy"])
        source_rows = audited_grouped[recording_id]
        intervals = (
            []
            if policy == "orcasound_abiotic_only"
            else merged_annotation_intervals(source_rows)
        )
        negative_subtype = (
            "orcasound_abiotic_only_recording"
            if policy == "orcasound_abiotic_only"
            else "ambient_background"
        )
        background_rows.append(
            {
                "source_recording_id": recording_id,
                "Provider": record["Provider"],
                "Dataset": record["Dataset"],
                "Soundfile": record["Soundfile"],
                "split": record["split"],
                "shard_id": record["shard_id"],
                "storage_key": record["storage_key"],
                "kaggle_dataset_id": record["kaggle_dataset_id"],
                "gcs_path": record["gcs_path"],
                "source_size_bytes": record["source_size_bytes"],
                "extraction_mode": record["extraction_mode"],
                "background_policy": policy,
                "model_source_label": "Abiotic",
                "negative_subtype": negative_subtype,
                "requested_window_count": requested,
                "window_duration_sec": args.clip_seconds,
                "safety_margin_sec": (
                    0.0
                    if policy == "orcasound_abiotic_only"
                    else args.ambient_safety_margin_seconds
                ),
                "minimum_spacing_sec": args.ambient_min_spacing_seconds,
                "random_seed": int(
                    stable_fraction(args.seed, f"background|{recording_id}")
                    * (2**31 - 1)
                ),
                "blocked_annotation_count": len(source_rows),
                "exclusion_intervals_json": json.dumps(
                    intervals, separators=(",", ":")
                ),
                "relative_clip_directory": "clips/Abiotic",
            }
        )
    background_fields = [
        "source_recording_id",
        "Provider",
        "Dataset",
        "Soundfile",
        "split",
        "shard_id",
        "storage_key",
        "kaggle_dataset_id",
        "gcs_path",
        "source_size_bytes",
        "extraction_mode",
        "background_policy",
        "model_source_label",
        "negative_subtype",
        "requested_window_count",
        "window_duration_sec",
        "safety_margin_sec",
        "minimum_spacing_sec",
        "random_seed",
        "blocked_annotation_count",
        "exclusion_intervals_json",
        "relative_clip_directory",
    ]
    write_csv(
        output_dir / "multispecies_cetacean_background_plan.csv",
        background_fields,
        background_rows,
    )

    shard_rows: list[dict[str, object]] = []
    for shard in shards:
        shard_rows.append(
            {
                key: value
                for key, value in shard.items()
                if key != "records"
            }
        )
    shard_fields = [
        "shard_id",
        "split",
        "storage_key",
        "dataset_title",
        "kaggle_dataset_id",
        "recording_count",
        "clip_count",
        "estimated_output_bytes",
        "largest_working_source_bytes",
        "estimated_peak_working_bytes",
    ]
    write_csv(
        output_dir / "multispecies_cetacean_shard_summary.csv",
        shard_fields,
        shard_rows,
    )

    split_summary: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        split_rows = [row for row in extraction_rows if row["split"] == split]
        split_recordings = [
            row for row in extraction_recordings if row["split"] == split
        ]
        counts = Counter(str(row["model_source_label"]) for row in split_rows)
        ambient_count = sum(
            int(row["ambient_windows_requested"]) for row in split_recordings
        )
        split_summary.append(
            {
                "split": split,
                "recordings": len(split_recordings),
                "annotations": len(split_rows),
                "ambient_background": ambient_count,
                "planned_clips": len(split_rows) + ambient_count,
                "count_Abiotic_annotated": counts["Abiotic"],
                "count_Abiotic": counts["Abiotic"] + ambient_count,
                **{
                    f"count_{label}": counts[label]
                    for label in MODEL_LABELS
                    if label != "Abiotic"
                },
            }
        )
    write_csv(
        output_dir / "multispecies_cetacean_split_summary.csv",
        [
            "split",
            "recordings",
            "annotations",
            "ambient_background",
            "planned_clips",
            "count_Abiotic_annotated",
            *[f"count_{label}" for label in MODEL_LABELS],
        ],
        split_summary,
    )

    settings = vars(args).copy()
    settings["annotations_csv"] = str(annotations_path)
    settings["output_dir"] = str(output_dir)
    with (output_dir / "multispecies_cetacean_plan_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)

    exclusion_counts = Counter()
    for row in audited:
        for reason in row["exclusion_reasons"].split("|"):
            if reason:
                exclusion_counts[reason] += 1

    print("\n# Multispecies Cetacean V2 planning report")
    print(f"Source annotation rows:       {len(audited):,}")
    print(f"Eligible annotation rows:     {len(eligible):,}")
    print(f"Excluded annotation rows:     {len(audited) - len(eligible):,}")
    print(f"Planned source recordings:    {len(recordings):,}")
    print(f"Matched source recordings:    {len(matched_recordings):,}")
    print(f"Unmatched/ambiguous audio:     {len(missing_audio):,}")
    print(
        "Requested ambient windows:   "
        f"{sum(int(r['ambient_windows_requested']) for r in matched_recordings):,}"
    )
    print(
        "Orcasound AB-only sources:   "
        f"{sum(r.get('background_policy') == 'orcasound_abiotic_only' for r in matched_recordings):,}"
    )
    print(f"Planned extraction shards:    {len(shards):,}")
    print(f"Estimated extracted storage:  {human_gb(sum(int(r['estimated_output_bytes']) for r in matched_recordings))}")
    print("\nSplit counts:")
    for row in split_summary:
        print(
            f"  {row['split']:<10} recordings={int(row['recordings']):,} "
            f"annotated={int(row['annotations']):,} "
            f"ambient={int(row['ambient_background']):,} "
            f"planned={int(row['planned_clips']):,} "
            + " ".join(
                f"{label}={int(row[f'count_{label}']):,}" for label in MODEL_LABELS
            )
        )
    print("\nExclusions:")
    for reason, count in exclusion_counts.most_common():
        print(f"  {reason}: {count:,}")
    print(f"\nPlan written to: {output_dir}")
    print("Review the split, shard, and unmatched-audio reports before extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
