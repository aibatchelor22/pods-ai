#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

"""Evaluate models using explicit WAV paths from a combined test manifest.

This is a path-based Colab front end for compare_new_models_experimantal_2.py.
All model, threshold, aggregation, and smoothing options from that script are
preserved. Replace its ``--testing-csv``/``--wav-dir`` arguments with the single
``--testing-manifest`` argument defined here.
"""

import csv
import sys
from pathlib import Path
from typing import Optional

import compare_new_models_experimantal_2 as evaluator


DEFAULT_TESTING_MANIFEST = (
    "/content/pods-ai/src/output/csv/combined_60s_evaluation_manifest.csv"
)


def load_path_based_samples(
    testing_manifest: Path,
    max_samples: Optional[int] = None,
    category_filter: Optional[str] = None,
) -> list[evaluator.TestSample]:
    """Load normalized labels and explicit WAV paths from the combined CSV."""
    samples = []

    with testing_manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted({"wav_path", "label"} - fieldnames)
        if missing:
            raise ValueError(
                f"{testing_manifest} is missing required columns: {missing}"
            )

        for row_index, row in enumerate(reader):
            label = (row.get("label") or "").strip()
            wav_path_text = (row.get("wav_path") or "").strip()
            if not label:
                raise ValueError(f"Row {row_index} has an empty label")
            if not wav_path_text:
                raise ValueError(f"Row {row_index} has an empty wav_path")
            if category_filter is not None and label != category_filter:
                continue

            source_dataset = (row.get("source_dataset") or "combined").strip()
            source_row = (row.get("source_row_index") or str(row_index)).strip()
            sample = evaluator.TestSample(
                category=label,
                node_name=(row.get("node_name") or source_dataset).strip(),
                timestamp=(row.get("timestamp") or f"row_{source_row}").strip(),
                uri=(row.get("uri") or "").strip(),
                description=(row.get("description") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
            # TestSample is intentionally reused so every existing evaluator path
            # remains unchanged. Its dataclass has no slots, so the normalized
            # manifest path can be attached to each instance.
            sample.wav_path = Path(wav_path_text).expanduser()
            samples.append(sample)

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


def find_manifest_wav(
    sample: evaluator.TestSample,
    _unused_wav_dir: Path,
) -> Optional[Path]:
    """Return the explicit manifest path instead of inferring a filename."""
    wav_path = getattr(sample, "wav_path", None)
    if wav_path is not None and wav_path.is_file():
        return wav_path
    return None


def _extract_testing_manifest(arguments: list[str]) -> tuple[Path, list[str]]:
    """Consume --testing-manifest while leaving all evaluator options intact."""
    cleaned = []
    manifest: Optional[str] = None
    index = 0

    while index < len(arguments):
        argument = arguments[index]
        if argument == "--testing-manifest":
            if index + 1 >= len(arguments):
                raise ValueError("--testing-manifest requires a path")
            manifest = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--testing-manifest="):
            manifest = argument.split("=", 1)[1]
            index += 1
            continue
        if argument in {"--testing-csv", "--wav-dir"} or argument.startswith(
            ("--testing-csv=", "--wav-dir=")
        ):
            raise ValueError(
                f"{argument.split('=', 1)[0]} is not used by this evaluator; "
                "provide --testing-manifest instead"
            )
        cleaned.append(argument)
        index += 1

    return Path(manifest or DEFAULT_TESTING_MANIFEST), cleaned


def main() -> int:
    try:
        testing_manifest, evaluator_arguments = _extract_testing_manifest(sys.argv[1:])
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if not testing_manifest.is_file() and not any(
        argument in {"-h", "--help"} for argument in evaluator_arguments
    ):
        print(f"Error: testing manifest not found: {testing_manifest}", file=sys.stderr)
        return 2

    # The base evaluator still parses --testing-csv and --wav-dir internally.
    # Its sample loader and path resolver are replaced before main() runs, so
    # neither value is used to infer a WAV filename.
    evaluator.load_test_samples = load_path_based_samples
    evaluator.find_wav_file = find_manifest_wav
    sys.argv = [
        sys.argv[0],
        "--testing-csv",
        str(testing_manifest),
        "--wav-dir",
        ".",
        *evaluator_arguments,
    ]

    print(f"Reading labels and WAV paths from: {testing_manifest}")
    return evaluator.main()


if __name__ == "__main__":
    sys.exit(main())
