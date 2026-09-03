#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Evaluate binary/species agreement as a second-chance KW detector.

This is CPU-only post-processing. It reads ``window_predictions.csv`` created
by ``evaluate_dclde_long_recordings.py`` and does not load the neural network
or audio. For each window it can require:

* a minimum binary-KW probability;
* a minimum species-head KW probability;
* a minimum species-KW minus species-background probability margin; and
* an optional minimum product of the binary and species-KW probabilities.

The component probabilities are smoothed before these gates are applied.
Adjacent passing windows form events, which are scored using the same temporal
collar and uncertain-annotation rules as the paper-style evaluator. The report
selects maximum-recall operating points at fixed false-positive/hour budgets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

import evaluate_dclde_long_recordings_paper_style as paper


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_ANNOTATIONS = (
    "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/"
    "dclde_2027_killer_whales/Annotations.csv"
)
DEFAULT_OUTPUT = "/kaggle/working/dclde_long_second_chance"
ACTIVATION_EPSILON = float(np.nextafter(0.0, 1.0))


def parse_values(value: str, item_type: type, name: str) -> list[Any]:
    return paper.parse_csv_list(value, item_type, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--binary-thresholds",
        default="0,0.70,0.80,0.85,0.875,0.90,0.925,0.95",
        help="Comma-separated binary-KW gates; 0 disables this gate.",
    )
    parser.add_argument(
        "--species-kw-thresholds",
        default="0,0.70,0.80,0.85,0.875,0.90,0.925,0.95",
        help="Comma-separated species-KW gates; 0 disables this gate.",
    )
    parser.add_argument(
        "--background-margins",
        default="-1,-0.10,0,0.10,0.20",
        help=(
            "Require p(species KW)-p(species background) >= margin. "
            "Use -1 to disable the margin gate."
        ),
    )
    parser.add_argument(
        "--joint-score-thresholds",
        default="0",
        help=(
            "Optional minimum p(binary KW)*p(species KW); 0 disables this gate."
        ),
    )
    parser.add_argument("--moving-average-windows", default="1,2,3")
    parser.add_argument("--fp-per-hour-budgets", default="5,20")
    parser.add_argument("--primary-fp-budget", type=float, default=20.0)
    parser.add_argument("--collar-sec", type=float, default=1.5)
    parser.add_argument("--ignore-collar-sec", type=float, default=None)
    parser.add_argument(
        "--missing-kw-confidence",
        choices=("confirmed", "ignore", "exclude"),
        default="confirmed",
    )
    parser.add_argument("--ecotype-top-k", type=int, default=2)
    parser.add_argument("--ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--print-top", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> dict[str, list[Any]]:
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.seed < 0:
        raise ValueError("--seed cannot be negative")
    if args.collar_sec < 0:
        raise ValueError("--collar-sec cannot be negative")
    ignore_collar = args.collar_sec if args.ignore_collar_sec is None else args.ignore_collar_sec
    if ignore_collar < 0:
        raise ValueError("--ignore-collar-sec cannot be negative")
    if args.ecotype_top_k < 1 or not 0 <= args.ecotype_threshold <= 1:
        raise ValueError("Ecotype top-k must be positive and threshold within 0..1")
    if args.bootstrap_replicates < 0 or args.print_top < 0:
        raise ValueError("Bootstrap replicates and print-top cannot be negative")
    if args.primary_fp_budget < 0:
        raise ValueError("--primary-fp-budget cannot be negative")

    values = {
        "binary_thresholds": parse_values(
            args.binary_thresholds, float, "binary thresholds"
        ),
        "species_thresholds": parse_values(
            args.species_kw_thresholds, float, "species-KW thresholds"
        ),
        "background_margins": parse_values(
            args.background_margins, float, "background margins"
        ),
        "joint_thresholds": parse_values(
            args.joint_score_thresholds, float, "joint-score thresholds"
        ),
        "moving_widths": parse_values(
            args.moving_average_windows, int, "moving-average windows"
        ),
        "budgets": parse_values(args.fp_per_hour_budgets, float, "FP/hour budgets"),
    }
    for name in ("binary_thresholds", "species_thresholds", "joint_thresholds"):
        if any(not 0 <= value <= 1 for value in values[name]):
            raise ValueError(f"{name} must be within 0..1")
    if any(not -1 <= value <= 1 for value in values["background_margins"]):
        raise ValueError("Background margins must be within -1..1")
    if any(value < 1 for value in values["moving_widths"]):
        raise ValueError("Moving-average widths must be positive")
    if any(value < 0 for value in values["budgets"]):
        raise ValueError("FP/hour budgets cannot be negative")
    values["ignore_collar_sec"] = [float(ignore_collar)]
    return values


def configuration_kind(
    binary_threshold: float,
    species_threshold: float,
    background_margin: float,
    joint_threshold: float,
) -> str:
    binary_enabled = binary_threshold > 0
    species_enabled = species_threshold > 0
    margin_enabled = background_margin > -1
    joint_enabled = joint_threshold > 0
    if binary_enabled and (species_enabled or margin_enabled or joint_enabled):
        return "second_chance"
    if binary_enabled:
        return "binary_only"
    if species_enabled or margin_enabled:
        return "species_only"
    if joint_enabled:
        return "joint_only"
    return "ungated"


def build_gated_scores(
    cached: Any,
    moving_width: int,
    binary_threshold: float,
    species_threshold: float,
    background_margin: float,
    joint_threshold: float,
) -> tuple[np.ndarray, str]:
    binary = paper.centered_moving_average(
        np.asarray(cached.kw_probabilities, dtype=np.float64), moving_width
    )
    species_kw = paper.centered_moving_average(
        np.asarray(
            cached.species_probabilities[:, paper.SPECIES.index("KW")],
            dtype=np.float64,
        ),
        moving_width,
    )
    species_background = paper.centered_moving_average(
        np.asarray(
            cached.species_probabilities[:, paper.SPECIES.index("background")],
            dtype=np.float64,
        ),
        moving_width,
    )
    return gate_smoothed_components(
        binary,
        species_kw,
        species_background,
        binary_threshold,
        species_threshold,
        background_margin,
        joint_threshold,
    )


def gate_smoothed_components(
    binary: np.ndarray,
    species_kw: np.ndarray,
    species_background: np.ndarray,
    binary_threshold: float,
    species_threshold: float,
    background_margin: float,
    joint_threshold: float,
) -> tuple[np.ndarray, str]:
    joint = binary * species_kw
    active = (
        (binary >= binary_threshold)
        & (species_kw >= species_threshold)
        & ((species_kw - species_background) >= background_margin)
        & (joint >= joint_threshold)
    )

    # Use the single active head as the within-event ranking score for a pure
    # baseline. For an actual two-head configuration, use their product.
    kind = configuration_kind(
        binary_threshold, species_threshold, background_margin, joint_threshold
    )
    if kind == "binary_only":
        ranking_score = binary
    elif kind == "species_only":
        ranking_score = species_kw
    else:
        ranking_score = joint
    return np.where(active, ranking_score, 0.0), kind


def precompute_smoothed_components(
    recordings: list[Any], moving_widths: list[int]
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Smooth each head output once per width rather than once per grid row."""
    result = {}
    for width in moving_widths:
        by_file = {}
        for cached in recordings:
            binary = paper.centered_moving_average(
                np.asarray(cached.kw_probabilities, dtype=np.float64), width
            )
            species_kw = paper.centered_moving_average(
                np.asarray(
                    cached.species_probabilities[:, paper.SPECIES.index("KW")],
                    dtype=np.float64,
                ),
                width,
            )
            species_background = paper.centered_moving_average(
                np.asarray(
                    cached.species_probabilities[
                        :, paper.SPECIES.index("background")
                    ],
                    dtype=np.float64,
                ),
                width,
            )
            by_file[cached.recording.soundfile] = (
                binary,
                species_kw,
                species_background,
            )
        result[width] = by_file
    return result


def evaluate_configuration(
    recordings: list[Any],
    smoothed_components: dict[
        int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ],
    truths_by_file: dict[str, list[Any]],
    ignore_by_file: dict[str, list[Any]],
    moving_width: int,
    binary_threshold: float,
    species_threshold: float,
    background_margin: float,
    joint_threshold: float,
    args: argparse.Namespace,
    ignore_collar_sec: float,
    audio_hours: float,
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    scores_by_file = {}
    kinds = set()
    for cached in recordings:
        components = smoothed_components[moving_width][cached.recording.soundfile]
        scores, kind = gate_smoothed_components(
            *components,
            binary_threshold,
            species_threshold,
            background_margin,
            joint_threshold,
        )
        scores_by_file[cached.recording.soundfile] = scores
        kinds.add(kind)
    if len(kinds) != 1:
        raise AssertionError(f"Inconsistent configuration kinds: {kinds}")
    kind = kinds.pop()
    source = f"second_chance_{kind}"
    row, events, matches = paper.evaluate_threshold(
        recordings,
        scores_by_file,
        truths_by_file,
        ignore_by_file,
        source,
        moving_width,
        ACTIVATION_EPSILON,
        args.collar_sec,
        ignore_collar_sec,
        args.ecotype_top_k,
        args.ecotype_threshold,
        audio_hours,
    )
    row.update(
        {
            "configuration_kind": kind,
            "binary_threshold": binary_threshold,
            "species_kw_threshold": species_threshold,
            "background_margin": background_margin,
            "joint_score_threshold": joint_threshold,
        }
    )
    # This generic field is used only as a deterministic final tie-breaker by
    # the shared operating-point selector.
    row["threshold"] = max(binary_threshold, species_threshold, joint_threshold)
    return row, events, matches


def metric_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        -1.0 if row.get("recall") is None else float(row["recall"]),
        -1.0 if row.get("precision") is None else float(row["precision"]),
        -math.inf
        if row.get("false_positives_per_hour") is None
        else -float(row["false_positives_per_hour"]),
        -float(row.get("binary_threshold", 0.0)),
        -float(row.get("species_kw_threshold", 0.0)),
    )


def select_for_budget(
    rows: list[dict[str, Any]], budget: float, kind: Optional[str] = None
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if kind is None or row["configuration_kind"] == kind
    ]
    eligible = [
        row
        for row in candidates
        if row["false_positives_per_hour"] is not None
        and row["false_positives_per_hour"] <= budget
    ]
    if eligible:
        selected = max(eligible, key=metric_sort_key)
        constraint_met = True
    else:
        selected = min(
            candidates,
            key=lambda row: (
                math.inf
                if row["false_positives_per_hour"] is None
                else row["false_positives_per_hour"],
                -metric_sort_key(row)[0],
            ),
        )
        constraint_met = False
    return {
        "fp_per_hour_budget": budget,
        "budget_constraint_met": constraint_met,
        **selected,
    }


def print_operating_point(label: str, row: dict[str, Any]) -> None:
    print(
        f"{label}: recall={row['recall']}, precision={row['precision']}, "
        f"F1={row['f1']}, FP/h={row['false_positives_per_hour']}, "
        f"MA={row['moving_average_windows']}, binary={row['binary_threshold']}, "
        f"species_KW={row['species_kw_threshold']}, "
        f"margin={row['background_margin']}, joint={row['joint_score_threshold']}"
    )


def main() -> int:
    args = parse_args()
    grid = validate_args(args)
    ignore_collar_sec = grid["ignore_collar_sec"][0]
    window_path = Path(args.window_predictions)
    if not window_path.is_file():
        raise FileNotFoundError(f"Window predictions not found: {window_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Delay the heavier audio/Transformers dependency chain until after CLI
    # parsing so --help remains immediate. No model or audio is loaded below.
    import evaluate_dclde_long_recordings as base
    import evaluate_dclde_long_recordings_improved as diagnostic

    failed_path = (
        Path(args.failed_files)
        if args.failed_files
        else window_path.parent / "failed_files.csv"
    )
    recordings, cache_sanity = diagnostic.load_cached_recordings(
        window_path, diagnostic.load_failed_soundfiles(failed_path)
    )
    available_recordings = len(recordings)
    if args.max_files is not None and args.max_files < len(recordings):
        recordings = random.Random(args.seed).sample(recordings, args.max_files)
        cache_sanity["available_recordings_before_random_subset"] = available_recordings
        cache_sanity["random_subset_recordings"] = len(recordings)
        cache_sanity["random_subset_seed"] = args.seed
    if not recordings:
        raise ValueError("No cached recordings remain")

    annotation_table = base.read_csv(args.annotations)
    truths_by_file, ignore_by_file, annotation_sanity = paper.load_confidence_annotations(
        base,
        annotation_table,
        [cached.recording for cached in recordings],
        args.missing_kw_confidence,
    )
    audio_hours = sum(cached.recording.duration_sec for cached in recordings) / 3600.0
    cached_windows = sum(len(cached.starts) for cached in recordings)
    total_configurations = (
        len(grid["moving_widths"])
        * len(grid["binary_thresholds"])
        * len(grid["species_thresholds"])
        * len(grid["background_margins"])
        * len(grid["joint_thresholds"])
    )
    print(f"Cached recordings:       {len(recordings):,}")
    print(f"Cached windows:          {cached_windows:,}")
    print(f"Audio hours:             {audio_hours:.3f}")
    print(f"Confirmed KW events:     {annotation_sanity.get('confirmed', 0):,}")
    print(f"KW ignore intervals:     {annotation_sanity.get('ignore', 0):,}")
    print(f"Grid configurations:     {total_configurations:,}")
    print("Precomputing moving-average component scores...")
    smoothed_components = precompute_smoothed_components(
        recordings, grid["moving_widths"]
    )

    results = []
    completed = 0
    for moving_width in grid["moving_widths"]:
        for binary_threshold in grid["binary_thresholds"]:
            for species_threshold in grid["species_thresholds"]:
                for margin in grid["background_margins"]:
                    for joint_threshold in grid["joint_thresholds"]:
                        row, _, _ = evaluate_configuration(
                            recordings,
                            smoothed_components,
                            truths_by_file,
                            ignore_by_file,
                            moving_width,
                            binary_threshold,
                            species_threshold,
                            margin,
                            joint_threshold,
                            args,
                            ignore_collar_sec,
                            audio_hours,
                        )
                        results.append(row)
                        completed += 1
                        if completed % 50 == 0 or completed == total_configurations:
                            print(f"Completed {completed:,}/{total_configurations:,}")

    paper.write_rows(output_dir / "second_chance_grid_results.csv", results)
    operating_points = []
    kinds = sorted({row["configuration_kind"] for row in results})
    for budget in grid["budgets"]:
        operating_points.append(select_for_budget(results, budget))
        for kind in kinds:
            operating_points.append(select_for_budget(results, budget, kind))
    paper.write_rows(output_dir / "second_chance_operating_points.csv", operating_points)

    selected = select_for_budget(results, args.primary_fp_budget)
    selected_row, selected_events, selected_matches = evaluate_configuration(
        recordings,
        smoothed_components,
        truths_by_file,
        ignore_by_file,
        int(selected["moving_average_windows"]),
        float(selected["binary_threshold"]),
        float(selected["species_kw_threshold"]),
        float(selected["background_margin"]),
        float(selected["joint_score_threshold"]),
        args,
        ignore_collar_sec,
        audio_hours,
    )
    paper.write_rows(
        output_dir / "selected_second_chance_events.csv",
        [asdict(event) for event in selected_events],
        list(paper.ContinuousEvent.__dataclass_fields__),
    )
    paper.write_rows(
        output_dir / "selected_second_chance_matches.csv",
        [asdict(match) for match in selected_matches],
        list(paper.ContinuousMatch.__dataclass_fields__),
    )
    ecotype_overall, ecotype_rows, ecotype_matrix = paper.ecotype_metrics(
        selected_matches
    )
    paper.write_rows(output_dir / "selected_ecotype_metrics.csv", ecotype_rows)
    paper.write_ecotype_matrix(
        output_dir / "selected_ecotype_confusion_matrix.csv", ecotype_matrix
    )
    paper.write_rows(
        output_dir / "selected_provider_metrics.csv",
        paper.grouped_metrics(recordings, selected_matches, "provider"),
    )
    paper.write_rows(
        output_dir / "selected_dataset_metrics.csv",
        paper.grouped_metrics(recordings, selected_matches, "dataset"),
    )
    paper.write_rows(
        output_dir / "selected_recording_metrics.csv",
        paper.recording_metrics(recordings, selected_matches),
    )
    confidence_intervals = paper.bootstrap_intervals(
        recordings, selected_matches, args.bootstrap_replicates, args.seed
    )
    paper.write_rows(
        output_dir / "selected_bootstrap_confidence_intervals.csv",
        confidence_intervals,
    )

    budget_eligible = [
        row
        for row in results
        if row["false_positives_per_hour"] is not None
        and row["false_positives_per_hour"] <= args.primary_fp_budget
    ]
    top_pool = budget_eligible or results
    top_rows = sorted(top_pool, key=metric_sort_key, reverse=True)[: args.print_top]
    paper.write_rows(output_dir / "top_recall_configurations.csv", top_rows)
    summary = {
        "method": (
            "Moving-average component probabilities followed by binary-KW, "
            "species-KW, species-background-margin, and optional joint-score gates; "
            "contiguous-window events with one-to-one temporal-collar matching"
        ),
        "selection": f"maximum recall at <= {args.primary_fp_budget:g} FP/hour",
        "selected_configuration": selected_row,
        "selected_ecotype_metrics": ecotype_overall,
        "audio_hours": audio_hours,
        "recordings": len(recordings),
        "cached_windows": cached_windows,
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "grid": grid,
        "arguments": vars(args),
    }
    (output_dir / "second_chance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nSelected second-chance operating point")
    print("=" * 72)
    print_operating_point(
        f"Maximum recall at <= {args.primary_fp_budget:g} FP/h", selected_row
    )
    print(f"Ecotype accuracy: {ecotype_overall['accuracy']}")
    print(f"Ecotype macro F1: {ecotype_overall['macro_f1']}")
    print("\nBest configurations at operational FP/hour budgets")
    for budget in grid["budgets"]:
        best = select_for_budget(results, budget)
        print_operating_point(f"  Overall <= {budget:g} FP/h", best)
        for kind in ("binary_only", "species_only", "second_chance"):
            if kind in kinds:
                best_kind = select_for_budget(results, budget, kind)
                print_operating_point(f"    {kind}", best_kind)
    print(f"\nReports saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
