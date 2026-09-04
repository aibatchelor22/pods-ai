#!/usr/bin/env python3
"""Clean annotations and plan leakage-safe, size-bounded audio extraction shards.

This is the first stage of the Multispecies Cetacean V2 dataset build. It does
not download audio. It creates an auditable master annotation table, assigns
entire source recordings to train/validation/test, inventories the public GCS
objects, and packs recordings into extraction jobs that fit Kaggle storage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
    return slug_component(provider).replace("-", "_")


def inventory_gcs(
    providers: Iterable[str], root: str, quiet: bool
) -> tuple[dict[tuple[str, str], list[dict[str, object]]], list[dict[str, str]]]:
    try:
        import gcsfs  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gcsfs is required for inventory. In Kaggle run: pip install -q gcsfs"
        ) from exc

    fs = gcsfs.GCSFileSystem(token="anon")
    by_provider_basename: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    failures: list[dict[str, str]] = []

    for provider in sorted(set(providers)):
        folder = provider_folder(provider)
        search_root = f"{root.rstrip('/')}/{folder}/audio"
        if not quiet:
            print(f"Indexing provider {provider} ...")
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
                key = (provider.casefold(), PurePosixPath(path).name.casefold())
                by_provider_basename[key].append(
                    {"gcs_path": str(path), "source_size_bytes": size}
                )
                count += 1
            if not quiet:
                print(f"  indexed {count:,} audio objects")
        except Exception as exc:  # preserve other providers and report failure
            failures.append(
                {
                    "Provider": provider,
                    "search_root": search_root,
                    "error": repr(exc),
                }
            )
            print(f"WARNING: inventory failed for provider {provider}: {exc}")

    return by_provider_basename, failures


def choose_audio_match(
    provider: str,
    dataset: str,
    soundfile: str,
    index: dict[tuple[str, str], list[dict[str, object]]],
) -> tuple[str, int, str]:
    candidates = index.get((provider.casefold(), soundfile.casefold()), [])
    if not candidates:
        return "", 0, "missing"
    if len(candidates) == 1:
        item = candidates[0]
        return str(item["gcs_path"]), int(item["source_size_bytes"]), "matched"

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
            "matched_by_dataset",
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
                        int(row["annotation_count"])
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

    if args.skip_gcs_inventory:
        audio_index: dict[tuple[str, str], list[dict[str, object]]] = {}
        inventory_failures: list[dict[str, str]] = []
    else:
        audio_index, inventory_failures = inventory_gcs(
            (rows[0]["Provider"] for rows in grouped.values()),
            args.gcs_root,
            args.quiet_gcs,
        )

    recordings: list[dict[str, object]] = []
    missing_audio: list[dict[str, object]] = []
    for recording_id, rows in grouped.items():
        first = rows[0]
        split = assign_split(
            recording_id,
            args.seed,
            args.train_fraction,
            args.validation_fraction,
        )
        label_counts = Counter(row["model_source_label"] for row in rows)
        ecotype_counts = Counter(
            row["clean_ecotype"] for row in rows if row["clean_ecotype"]
        )
        if args.skip_gcs_inventory:
            gcs_path, source_size, match_status = "", 0, "inventory_skipped"
        else:
            gcs_path, source_size, match_status = choose_audio_match(
                first["Provider"], first["Dataset"], first["Soundfile"], audio_index
            )
        record: dict[str, object] = {
            "source_recording_id": recording_id,
            "Provider": first["Provider"],
            "Dataset": first["Dataset"],
            "Soundfile": first["Soundfile"],
            "split": split,
            "annotation_count": len(rows),
            **{f"count_{label}": label_counts[label] for label in MODEL_LABELS},
            **{f"count_ecotype_{label}": ecotype_counts[label] for label in sorted(VALID_ECOTYPES)},
            "gcs_path": gcs_path,
            "source_size_bytes": source_size,
            "audio_match_status": match_status,
            "estimated_output_bytes": len(rows) * args.estimated_clip_bytes,
        }
        if match_status not in {"matched", "matched_by_dataset", "inventory_skipped"}:
            missing_audio.append(record)
        recordings.append(record)

    matched_recordings = [
        row
        for row in recordings
        if row["audio_match_status"]
        in {"matched", "matched_by_dataset", "inventory_skipped"}
    ]
    shards = pack_shards(
        matched_recordings,
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
        counts = Counter(str(row["model_source_label"]) for row in split_rows)
        split_summary.append(
            {
                "split": split,
                "recordings": len(
                    {str(row["source_recording_id"]) for row in split_rows}
                ),
                "annotations": len(split_rows),
                **{f"count_{label}": counts[label] for label in MODEL_LABELS},
            }
        )
    write_csv(
        output_dir / "multispecies_cetacean_split_summary.csv",
        [
            "split",
            "recordings",
            "annotations",
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
    print(f"Eligible source recordings:   {len(recordings):,}")
    print(f"Matched source recordings:    {len(matched_recordings):,}")
    print(f"Unmatched/ambiguous audio:     {len(missing_audio):,}")
    print(f"Planned extraction shards:    {len(shards):,}")
    print(f"Estimated extracted storage:  {human_gb(sum(int(r['estimated_output_bytes']) for r in matched_recordings))}")
    print("\nSplit counts:")
    for row in split_summary:
        print(
            f"  {row['split']:<10} recordings={int(row['recordings']):,} "
            f"clips={int(row['annotations']):,} "
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
