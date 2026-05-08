#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Shared helpers for reading and appending manual_samples.csv rows."""

import csv
from pathlib import Path

CSV_FIELDNAMES = [
    "Category",
    "NodeName",
    "Timestamp",
    "URI",
    "Description",
    "Notes",
    "Confidence",
]


def load_existing_uris(manual_samples_path: Path) -> set[str]:
    """Load existing manual-sample URIs from a CSV file path."""
    if not manual_samples_path.exists():
        return set()

    try:
        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                (row.get("URI") or "").strip()
                for row in reader
                if (row.get("URI") or "").strip()
            }
    except (OSError, UnicodeError, csv.Error) as exc:
        print(
            f"Warning: Failed to read existing manual sample URIs from "
            f"{manual_samples_path}: {exc}"
        )
        return set()


def append_manual_samples(
    manual_samples_path: Path,
    rows: list[dict],
    existing_uris: set[str],
) -> tuple[int, int]:
    """Append rows to manual_samples.csv, skipping URIs already present."""
    rows_to_append = []
    duplicates = 0

    for row in rows:
        uri = (row.get("URI") or "").strip()
        if uri in existing_uris:
            duplicates += 1
            continue
        existing_uris.add(uri)
        rows_to_append.append({field: row.get(field, "") for field in CSV_FIELDNAMES})

    if not rows_to_append:
        return 0, duplicates

    manual_samples_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manual_samples_path.exists() or manual_samples_path.stat().st_size == 0
    with open(manual_samples_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(rows_to_append)

    return len(rows_to_append), duplicates
