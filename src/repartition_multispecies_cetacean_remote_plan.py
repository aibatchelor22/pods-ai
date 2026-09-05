#!/usr/bin/env python3
"""Repartition remote-seek work into Kaggle-session-safe dataset shards.

This script consumes the three CSV files used by
``build_multispecies_cetacean_dataset.py`` plus the source-recording plan. It
does not download or alter audio. Ordinary sources remain grouped into their
existing shards unless their original shard also contained remote sources; in
that case the ordinary remainder receives a ``*_local_*`` shard name. Each
remote recording is divided into independent shards containing no more than
``--remote-clips-per-shard`` planned clips.

The resulting directory can be passed directly to the existing builder with
``--plan-dir``. Use a fresh builder ``--work-dir`` because shard identifiers
and dataset membership change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SHARD_FILE = "multispecies_cetacean_shard_summary.csv"
EXTRACTION_FILE = "multispecies_cetacean_extraction_plan.csv"
BACKGROUND_FILE = "multispecies_cetacean_background_plan.csv"
SOURCE_FILE = "multispecies_cetacean_source_recording_plan.csv"


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        # Some historical planner exports contain an unnamed dataframe index.
        fields = [field for field in (reader.fieldnames or []) if clean(field)]
        rows = []
        for raw in reader:
            rows.append({field: clean(raw.get(field)) for field in fields})
    return fields, rows


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    field_list = list(dict.fromkeys(field for field in fields if clean(field)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in field_list})


def int_value(value: Any, default: int = 0) -> int:
    text = clean(value)
    if not text:
        return default
    return int(float(text))


def float_value(value: Any, default: float = 0.0) -> float:
    text = clean(value)
    if not text:
        return default
    return float(text)


def derive_slug_prefix(shard_rows: list[dict[str, str]]) -> str:
    row = shard_rows[0]
    dataset_id = row["kaggle_dataset_id"]
    slug = dataset_id.split("/", 1)[-1]
    suffix = "-" + row["shard_id"].replace("_", "-")
    return slug[: -len(suffix)] if slug.endswith(suffix) else slug + "-remote-safe"


def changed_local_shard_id(original: str) -> str:
    if "_original_" in original:
        return original.replace("_original_", "_local_", 1)
    return original + "_local"


def split_order(value: str) -> tuple[int, str]:
    order = {"train": 0, "validation": 1, "val": 1, "test": 2}
    return order.get(value.casefold(), 9), value


def validate_input_bundle(
    shard_rows: list[dict[str, str]],
    extraction_rows: list[dict[str, str]],
    background_rows: list[dict[str, str]],
) -> None:
    """Apply the same fundamental count checks as the audio builder."""
    shard_ids = [row.get("shard_id", "") for row in shard_rows]
    if any(not shard_id for shard_id in shard_ids):
        raise ValueError("Shard summary contains a blank shard_id")
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("Shard summary contains duplicate shard_id values")

    known = set(shard_ids)
    unknown = sorted(
        {
            row.get("shard_id", "")
            for row in extraction_rows + background_rows
        }.difference(known)
    )
    if unknown:
        raise ValueError(f"Plan rows reference unknown shards: {unknown}")

    annotation_ids = [row.get("annotation_id", "") for row in extraction_rows]
    if any(not value for value in annotation_ids):
        raise ValueError("Extraction plan contains a blank annotation_id")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Extraction plan contains duplicate annotation_id values")

    annotation_counts = Counter(row["shard_id"] for row in extraction_rows)
    background_counts: Counter[str] = Counter()
    for row in background_rows:
        count = int_value(row.get("requested_window_count"))
        if count < 0:
            raise ValueError("Background plan contains a negative requested count")
        background_counts[row["shard_id"]] += count

    mismatches = []
    for row in shard_rows:
        shard_id = row["shard_id"]
        summary_count = int_value(row.get("clip_count"))
        actual_count = annotation_counts[shard_id] + background_counts[shard_id]
        if summary_count != actual_count:
            mismatches.append(
                f"{shard_id}: summary={summary_count:,}, plans={actual_count:,} "
                f"({annotation_counts[shard_id]:,} annotated + "
                f"{background_counts[shard_id]:,} background)"
            )
    if mismatches:
        raise ValueError(
            "Input plan files are from different planner generations. Copy all "
            "three builder CSVs from the same final plan directory before "
            "repartitioning. Mismatches:\n  " + "\n  ".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, help="Existing complete plan directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for the repartitioned plan.")
    parser.add_argument(
        "--remote-clips-per-shard",
        type=int,
        default=100,
        help=(
            "Maximum annotated plus requested background clips in a remote "
            "partition (default: 100)."
        ),
    )
    parser.add_argument(
        "--kaggle-owner",
        help="Kaggle owner; defaults to the owner in the original shard summary.",
    )
    parser.add_argument(
        "--dataset-slug-prefix",
        help="Dataset slug prefix; defaults to the prefix inferred from the original plan.",
    )
    parser.add_argument(
        "--dataset-title-prefix",
        default="Multispecies Cetacean V2",
        help="Human-readable title prefix for newly named shards.",
    )
    args = parser.parse_args()

    if args.remote_clips_per_shard <= 0:
        parser.error("--remote-clips-per-shard must be positive")

    plan_dir = Path(args.plan_dir)
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == plan_dir.resolve():
        parser.error("--output-dir must differ from --plan-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_fields, shard_rows = read_csv(plan_dir / SHARD_FILE)
    extraction_fields, extraction_rows = read_csv(plan_dir / EXTRACTION_FILE)
    background_fields, background_rows = read_csv(plan_dir / BACKGROUND_FILE)
    source_fields, source_rows = read_csv(plan_dir / SOURCE_FILE)

    if not shard_rows or not extraction_rows or not source_rows:
        raise ValueError("The source plan bundle is empty or incomplete")
    validate_input_bundle(shard_rows, extraction_rows, background_rows)

    original_shards = {row["shard_id"]: row for row in shard_rows}
    source_lookup = {
        row["source_recording_id"]: row
        for row in source_rows
        if clean(row.get("source_recording_id"))
    }

    owner = args.kaggle_owner or shard_rows[0]["kaggle_dataset_id"].split("/", 1)[0]
    slug_prefix = args.dataset_slug_prefix or derive_slug_prefix(shard_rows)

    annotations_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    backgrounds_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in extraction_rows:
        annotations_by_source[row["source_recording_id"]].append(row)
    for row in background_rows:
        backgrounds_by_source[row["source_recording_id"]].append(row)

    used_sources = set(annotations_by_source) | set(backgrounds_by_source)
    missing_sources = sorted(used_sources.difference(source_lookup))
    for source_id in missing_sources:
        representative = (
            annotations_by_source[source_id][0]
            if annotations_by_source[source_id]
            else backgrounds_by_source[source_id][0]
        )
        source_size = int_value(representative.get("source_size_bytes"))
        extraction_mode = clean(representative.get("extraction_mode"))
        source_lookup[source_id] = {
            "source_recording_id": source_id,
            "Provider": representative.get("Provider", ""),
            "Dataset": representative.get("Dataset", ""),
            "Soundfile": representative.get("Soundfile", ""),
            "split": representative.get("split", ""),
            "annotation_count": 0,
            "count_Abiotic": 0,
            "count_KW": 0,
            "count_HW": 0,
            "count_UndBio": 0,
            "count_ecotype_NRKW": 0,
            "count_ecotype_OKW": 0,
            "count_ecotype_SAR": 0,
            "count_ecotype_SRKW": 0,
            "count_ecotype_TKW": 0,
            "gcs_path": representative.get("gcs_path", ""),
            "source_size_bytes": source_size,
            "audio_match_status": "matched",
            "estimated_output_bytes": 0,
            "extraction_mode": extraction_mode,
            # This value is informational. The builder reads extraction mode
            # and source size directly from the clip/background plans.
            "working_source_bytes": source_size,
            "shard_id": representative.get("shard_id", ""),
            "storage_key": representative.get("storage_key", ""),
            "kaggle_dataset_id": representative.get("kaggle_dataset_id", ""),
        }

    remote_sources: set[str] = set()
    inconsistent_modes: list[str] = []
    for source_id in used_sources:
        modes = {
            clean(row.get("extraction_mode"))
            for row in annotations_by_source[source_id] + backgrounds_by_source[source_id]
        }
        modes.discard("")
        if len(modes) > 1:
            inconsistent_modes.append(f"{source_id}: {sorted(modes)}")
        if "remote_seek" in modes:
            remote_sources.add(source_id)
    if inconsistent_modes:
        raise ValueError("Inconsistent extraction modes:\n  " + "\n  ".join(inconsistent_modes))

    changed_original_shards = {
        source_lookup[source_id]["shard_id"] for source_id in remote_sources
    }

    # shard_id -> assigned rows and provenance
    assignments: dict[str, dict[str, Any]] = {}

    def assignment(shard_id: str, split: str, original_shard_id: str) -> dict[str, Any]:
        if shard_id not in assignments:
            assignments[shard_id] = {
                "shard_id": shard_id,
                "split": split,
                "original_shard_id": original_shard_id,
                "annotations": [],
                "backgrounds": [],
            }
        return assignments[shard_id]

    # Keep ordinary sources together. Only shards whose membership changes are renamed.
    for source_id in sorted(used_sources):
        if source_id in remote_sources:
            continue
        source = source_lookup[source_id]
        original_id = source["shard_id"]
        new_id = (
            changed_local_shard_id(original_id)
            if original_id in changed_original_shards
            else original_id
        )
        target = assignment(new_id, source["split"], original_id)
        target["annotations"].extend(annotations_by_source[source_id])
        target["backgrounds"].extend(backgrounds_by_source[source_id])

    # Split every remote source independently. Ordinals are stable within each split.
    remote_by_split: dict[str, list[str]] = defaultdict(list)
    for source_id in remote_sources:
        remote_by_split[source_lookup[source_id]["split"]].append(source_id)

    remote_partition_count = 0
    for split in sorted(remote_by_split, key=split_order):
        for source_ordinal, source_id in enumerate(sorted(remote_by_split[split])):
            source = source_lookup[source_id]
            original_id = source["shard_id"]
            annotations = sorted(
                annotations_by_source[source_id],
                key=lambda row: (
                    float_value(row.get("clip_start_requested_sec")),
                    row.get("annotation_id", ""),
                ),
            )
            backgrounds = backgrounds_by_source[source_id]
            background_count = sum(
                int_value(row.get("requested_window_count")) for row in backgrounds
            )
            if background_count > args.remote_clips_per_shard:
                raise ValueError(
                    f"Remote source {source_id} requests {background_count} background clips, "
                    f"which exceeds --remote-clips-per-shard={args.remote_clips_per_shard}. "
                    "Background sampling cannot be divided without changing its deterministic "
                    "selection; increase the limit."
                )

            chunks: list[tuple[list[dict[str, str]], list[dict[str, str]]]] = []
            first_capacity = args.remote_clips_per_shard - background_count
            if backgrounds or annotations:
                first_annotations = annotations[:first_capacity]
                chunks.append((first_annotations, backgrounds))
                annotations = annotations[first_capacity:]
            while annotations:
                chunks.append((annotations[: args.remote_clips_per_shard], []))
                annotations = annotations[args.remote_clips_per_shard :]

            for partition_index, (ann_chunk, bg_chunk) in enumerate(chunks):
                shard_id = (
                    f"{split}_remote_{source_ordinal:03d}_p{partition_index:02d}"
                )
                target = assignment(shard_id, split, original_id)
                target["annotations"].extend(ann_chunk)
                target["backgrounds"].extend(bg_chunk)
                target["remote_source_id"] = source_id
                target["remote_partition_index"] = partition_index
                target["remote_partition_count"] = len(chunks)
                remote_partition_count += 1

    # Derive the planner's nominal FLAC size and safety reserve from original summaries.
    bytes_per_clip_values = [
        int_value(row.get("estimated_output_bytes")) / int_value(row.get("clip_count"))
        for row in shard_rows
        if int_value(row.get("clip_count")) > 0
    ]
    bytes_per_clip = round(statistics.median(bytes_per_clip_values))
    safety_values = [
        max(
            0,
            int_value(row.get("estimated_peak_working_bytes"))
            - int_value(row.get("estimated_output_bytes"))
            - int_value(row.get("largest_working_source_bytes")),
        )
        for row in shard_rows
    ]
    safety_bytes = round(statistics.median(safety_values)) if safety_values else 0

    new_extractions: list[dict[str, Any]] = []
    new_backgrounds: list[dict[str, Any]] = []
    new_sources: list[dict[str, Any]] = []
    new_shards: list[dict[str, Any]] = []

    assignment_items = sorted(
        assignments.values(),
        key=lambda item: (split_order(item["split"]), item["shard_id"]),
    )

    for item in assignment_items:
        shard_id = item["shard_id"]
        split = item["split"]
        original_id = item["original_shard_id"]
        original_summary = original_shards[original_id]
        is_unchanged = shard_id == original_id

        if is_unchanged:
            dataset_id = original_summary["kaggle_dataset_id"]
            dataset_title = original_summary["dataset_title"]
        else:
            slug = f"{slug_prefix}-{shard_id.replace('_', '-')}"
            dataset_id = f"{owner}/{slug}"
            readable = shard_id.replace("_", " ").title()
            dataset_title = f"{args.dataset_title_prefix} {readable}"

        annotations = item["annotations"]
        backgrounds = item["backgrounds"]
        background_count = sum(
            int_value(row.get("requested_window_count")) for row in backgrounds
        )
        clip_count = len(annotations) + background_count
        if clip_count <= 0:
            continue

        annotations_for_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        backgrounds_for_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in annotations:
            annotations_for_source[row["source_recording_id"]].append(row)
        for row in backgrounds:
            backgrounds_for_source[row["source_recording_id"]].append(row)
        source_ids = sorted(set(annotations_for_source) | set(backgrounds_for_source))
        largest_working = max(
            int_value(source_lookup[source_id].get("working_source_bytes"))
            for source_id in source_ids
        )
        estimated_output = clip_count * bytes_per_clip

        new_shards.append(
            {
                "shard_id": shard_id,
                "split": split,
                "storage_key": shard_id,
                "dataset_title": dataset_title,
                "kaggle_dataset_id": dataset_id,
                "recording_count": len(source_ids),
                "clip_count": clip_count,
                "estimated_output_bytes": estimated_output,
                "largest_working_source_bytes": largest_working,
                "estimated_peak_working_bytes": (
                    estimated_output + largest_working + safety_bytes
                ),
            }
        )

        for row in annotations:
            updated = dict(row)
            updated.update(
                shard_id=shard_id,
                storage_key=shard_id,
                kaggle_dataset_id=dataset_id,
            )
            new_extractions.append(updated)
        for row in backgrounds:
            updated = dict(row)
            updated.update(
                shard_id=shard_id,
                storage_key=shard_id,
                kaggle_dataset_id=dataset_id,
            )
            new_backgrounds.append(updated)

        for source_id in source_ids:
            base = dict(source_lookup[source_id])
            source_annotations = annotations_for_source[source_id]
            source_background_count = sum(
                int_value(row.get("requested_window_count"))
                for row in backgrounds_for_source[source_id]
            )
            class_counts = Counter(row.get("model_source_label", "") for row in source_annotations)
            ecotype_counts = Counter(row.get("clean_ecotype", "") for row in source_annotations)
            source_clip_count = len(source_annotations) + source_background_count
            base.update(
                annotation_count=len(source_annotations),
                count_Abiotic=class_counts["Abiotic"],
                count_KW=class_counts["KW"],
                count_HW=class_counts["HW"],
                count_UndBio=class_counts["UndBio"],
                count_ecotype_NRKW=ecotype_counts["NRKW"],
                count_ecotype_OKW=ecotype_counts["OKW"],
                count_ecotype_SAR=ecotype_counts["SAR"],
                count_ecotype_SRKW=ecotype_counts["SRKW"],
                count_ecotype_TKW=ecotype_counts["TKW"],
                estimated_output_bytes=source_clip_count * bytes_per_clip,
                shard_id=shard_id,
                storage_key=shard_id,
                kaggle_dataset_id=dataset_id,
            )
            new_sources.append(base)

    # Confirm the exact count contract enforced by the audio builder.
    annotation_counts = Counter(row["shard_id"] for row in new_extractions)
    background_counts: Counter[str] = Counter()
    for row in new_backgrounds:
        background_counts[row["shard_id"]] += int_value(row["requested_window_count"])
    for row in new_shards:
        expected = annotation_counts[row["shard_id"]] + background_counts[row["shard_id"]]
        if expected != int_value(row["clip_count"]):
            raise AssertionError(f"Count mismatch for {row['shard_id']}: {expected} != {row['clip_count']}")

    if len(new_extractions) != len(extraction_rows):
        raise AssertionError("Repartitioning changed the number of annotated clips")
    if sum(background_counts.values()) != sum(
        int_value(row["requested_window_count"]) for row in background_rows
    ):
        raise AssertionError("Repartitioning changed the number of requested background clips")

    write_csv(output_dir / SHARD_FILE, shard_fields, new_shards)
    write_csv(output_dir / EXTRACTION_FILE, extraction_fields, new_extractions)
    write_csv(output_dir / BACKGROUND_FILE, background_fields, new_backgrounds)
    write_csv(output_dir / SOURCE_FILE, source_fields, new_sources)

    protected = {SHARD_FILE, EXTRACTION_FILE, BACKGROUND_FILE, SOURCE_FILE}
    for path in plan_dir.iterdir():
        if (
            path.is_file()
            and path.name not in protected
            and path.name.startswith("multispecies_cetacean_")
            and path.suffix.casefold() in {".csv", ".json", ".txt"}
        ):
            shutil.copy2(path, output_dir / path.name)

    remote_sizes = [
        int_value(row["clip_count"])
        for row in new_shards
        if "_remote_" in row["shard_id"]
    ]
    report = {
        "source_plan_directory": str(plan_dir),
        "remote_clips_per_shard": args.remote_clips_per_shard,
        "remote_source_recordings": len(remote_sources),
        "changed_original_shards": sorted(changed_original_shards),
        "output_shards": len(new_shards),
        "remote_output_shards": len(remote_sizes),
        "largest_remote_output_shard_clips": max(remote_sizes, default=0),
        "annotated_clips": len(new_extractions),
        "requested_background_clips": sum(background_counts.values()),
        "total_planned_clips": len(new_extractions) + sum(background_counts.values()),
        "bytes_per_clip_estimate": bytes_per_clip,
        "working_safety_bytes_estimate": safety_bytes,
    }
    with (output_dir / "multispecies_cetacean_remote_repartition_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nRepartitioned plan written to: {output_dir}")
    print("Use a fresh --work-dir when running the audio builder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
