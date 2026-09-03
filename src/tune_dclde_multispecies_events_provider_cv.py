#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Tune species-specific long-audio events with leave-one-provider-out validation.

This is CPU-only post-processing. It reads ``window_predictions.csv`` created by
``evaluate_dclde_long_recordings.py`` and searches separate stable-event settings
for KW, HW, and AB. For each held-out provider, settings are selected using all
other providers and then evaluated once on the held-out provider.

The cross-validated metrics are the generalization estimate. The script also
writes one final configuration selected using all providers; use that frozen
configuration on a genuinely independent test collection.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

import evaluate_dclde_long_recordings as base
import evaluate_dclde_long_recordings_improved as cache_tools
import evaluate_dclde_long_recordings_multispecies_events as event_eval


DEFAULT_WINDOWS = "/kaggle/working/dclde_long_evaluation/window_predictions.csv"
DEFAULT_ANNOTATIONS = base.DEFAULT_ANNOTATIONS
DEFAULT_OUTPUT = "/kaggle/working/dclde_multispecies_provider_cv"
SPECIES = event_eval.SPECIES


def number_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        parsed = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return sorted(set(parsed))


def optional_float(value: Optional[float], fallback: float) -> float:
    return fallback if value is None else value


def provider_name(cached: Any) -> str:
    return cached.recording.provider or "unknown"


def safe_divide(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def metric_dict(tp: int, fp: int, fn: int, audio_hours: float) -> dict[str, Any]:
    denominator = 2 * tp + fp + fn
    return {
        "audio_hours": float(audio_hours),
        "ground_truth_events": int(tp + fn),
        "predicted_events": int(tp + fp),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": float(2 * tp / denominator) if denominator else None,
        "false_positives_per_hour": safe_divide(fp, audio_hours),
        "false_negatives_per_hour": safe_divide(fn, audio_hours),
    }


def match_counts(matches: Iterable[event_eval.EventMatch], species: str) -> tuple[int, int, int]:
    matches = list(matches)
    tp = sum(match.status == "TP" and match.true_species == species for match in matches)
    fp = sum(match.predicted_species == species and match.status != "TP" for match in matches)
    fn = sum(match.true_species == species and match.status != "TP" for match in matches)
    return int(tp), int(fp), int(fn)


def add_counts(items: Iterable[tuple[int, int, int]]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for item_tp, item_fp, item_fn in items:
        tp += item_tp
        fp += item_fp
        fn += item_fn
    return tp, fp, fn


def config_id(config: event_eval.SpeciesConfig) -> str:
    return (
        f"{config.species}_st{config.start_threshold:g}_su{config.support_threshold:g}"
        f"_co{config.continuation_threshold:g}_n{config.minimum_support_windows}"
        f"_r{config.support_radius_windows}_g{config.maximum_gap_windows}"
        f"_ps{config.peak_suppression_windows}_k{config.event_top_k}"
        f"_et{config.event_score_threshold:g}"
    )


def build_grid(args: argparse.Namespace, species: str) -> list[event_eval.SpeciesConfig]:
    source = args.kw_score_source if species == "KW" else "species"
    configs = []
    for values in itertools.product(
        args.start_thresholds,
        args.support_thresholds,
        args.continuation_thresholds,
        args.minimum_support_windows,
        args.support_radius_windows,
        args.maximum_gap_windows,
        args.peak_suppression_windows,
        args.event_top_k,
        args.event_score_thresholds,
    ):
        start, support, continuation, minimum, radius, gap, suppression, top_k, event = values
        if not 0 <= continuation <= support <= start <= 1:
            continue
        if not 0 <= event <= 1:
            continue
        if minimum > 2 * radius + 1:
            continue
        configs.append(
            event_eval.SpeciesConfig(
                species=species,
                score_source=source,
                start_threshold=float(start),
                support_threshold=float(support),
                continuation_threshold=float(continuation),
                minimum_support_windows=int(minimum),
                support_radius_windows=int(radius),
                maximum_gap_windows=int(gap),
                peak_suppression_windows=int(suppression),
                event_top_k=int(top_k),
                event_score_threshold=float(event),
            )
        )
    unique = {config_id(config): config for config in configs}
    return [unique[key] for key in sorted(unique)]


def evaluate_config_by_provider(
    recordings: list[Any],
    truths_by_file: dict[str, list[event_eval.TruthEvent]],
    providers: list[str],
    species: str,
    config: event_eval.SpeciesConfig,
    collar_sec: float,
    ecotype_threshold: float,
) -> dict[str, tuple[int, int, int]]:
    counts = {provider: (0, 0, 0) for provider in providers}
    for cached in recordings:
        provider = provider_name(cached)
        predictions = event_eval.form_events(cached, config, ecotype_threshold)
        truths = [
            truth
            for truth in truths_by_file.get(cached.recording.soundfile, [])
            if truth.species == species
        ]
        matches = event_eval.match_events(predictions, truths, collar_sec)
        counts[provider] = add_counts((counts[provider], match_counts(matches, species)))
    return counts


def selection_key(
    metrics: dict[str, Any],
    config: event_eval.SpeciesConfig,
    objective: str,
    fp_budget: float,
) -> tuple[Any, ...]:
    recall = metrics["recall"] if metrics["recall"] is not None else -1.0
    precision = metrics["precision"] if metrics["precision"] is not None else -1.0
    f1 = metrics["f1"] if metrics["f1"] is not None else -1.0
    fp_hour = metrics["false_positives_per_hour"]
    fp_hour = math.inf if fp_hour is None else fp_hour
    feasible = fp_hour <= fp_budget
    conservative = (
        config.event_score_threshold,
        config.start_threshold,
        config.support_threshold,
        config.minimum_support_windows,
        config.peak_suppression_windows,
        -config.maximum_gap_windows,
    )
    if objective == "max_f1":
        return (f1, recall, precision, -fp_hour, *conservative)
    # Always prefer a configuration meeting the operational budget. Within the
    # budget, maximize recall; if no candidate meets it, minimize FP/hour first.
    if feasible:
        return (1, recall, f1, precision, -fp_hour, *conservative)
    return (0, -fp_hour, recall, f1, precision, *conservative)


def select_config(
    candidates: list[dict[str, Any]], objective: str, fp_budget: float
) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: selection_key(item["metrics"], item["config"], objective, fp_budget),
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prefix_fields(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-predictions", default=DEFAULT_WINDOWS)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-files", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collar-sec", type=float, default=1.5)
    parser.add_argument("--ecotype-threshold", type=float, default=0.0)
    parser.add_argument("--cross-class-suppression-sec", type=float, default=0.0)
    parser.add_argument(
        "--kw-score-source",
        choices=("binary_kw", "species_kw", "max_ecotype_composite"),
        default="max_ecotype_composite",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("recall_at_fp_budget", "max_f1"),
        default="recall_at_fp_budget",
    )
    parser.add_argument("--target-fp-per-hour", type=float, default=20.0)
    parser.add_argument("--kw-target-fp-per-hour", type=float, default=None)
    parser.add_argument("--hw-target-fp-per-hour", type=float, default=None)
    parser.add_argument("--ab-target-fp-per-hour", type=float, default=None)
    parser.add_argument(
        "--start-thresholds", type=lambda value: number_list(value, float, "start thresholds"),
        default=number_list("0.80,0.85,0.90", float, "start thresholds"),
    )
    parser.add_argument(
        "--support-thresholds", type=lambda value: number_list(value, float, "support thresholds"),
        default=number_list("0.45,0.55,0.65", float, "support thresholds"),
    )
    parser.add_argument(
        "--continuation-thresholds",
        type=lambda value: number_list(value, float, "continuation thresholds"),
        default=number_list("0.35", float, "continuation thresholds"),
    )
    parser.add_argument(
        "--minimum-support-windows",
        type=lambda value: number_list(value, int, "minimum support windows"),
        default=number_list("1,2,3", int, "minimum support windows"),
    )
    parser.add_argument(
        "--support-radius-windows",
        type=lambda value: number_list(value, int, "support radius windows"),
        default=number_list("1", int, "support radius windows"),
    )
    parser.add_argument(
        "--maximum-gap-windows",
        type=lambda value: number_list(value, int, "maximum gap windows"),
        default=number_list("0", int, "maximum gap windows"),
    )
    parser.add_argument(
        "--peak-suppression-windows",
        type=lambda value: number_list(value, int, "peak suppression windows"),
        default=number_list("1,2", int, "peak suppression windows"),
    )
    parser.add_argument(
        "--event-top-k", type=lambda value: number_list(value, int, "event top-k"),
        default=number_list("2", int, "event top-k"),
    )
    parser.add_argument(
        "--event-score-thresholds",
        type=lambda value: number_list(value, float, "event score thresholds"),
        default=number_list("0.65,0.75,0.85", float, "event score thresholds"),
    )
    parser.add_argument("--maximum-grid-configurations", type=int, default=5000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.collar_sec < 0 or args.cross_class_suppression_sec < 0:
        raise ValueError("Collar and suppression durations cannot be negative")
    if not 0 <= args.ecotype_threshold <= 1:
        raise ValueError("--ecotype-threshold must be between 0 and 1")
    budgets = [
        args.target_fp_per_hour,
        args.kw_target_fp_per_hour,
        args.hw_target_fp_per_hour,
        args.ab_target_fp_per_hour,
    ]
    if any(value is not None and value < 0 for value in budgets):
        raise ValueError("FP/hour budgets cannot be negative")
    integer_lists = (
        args.minimum_support_windows,
        args.support_radius_windows,
        args.peak_suppression_windows,
        args.event_top_k,
    )
    if any(value < 1 for values in integer_lists for value in values):
        raise ValueError("Support, radius, suppression, and top-k values must be positive")
    if any(value < 0 for value in args.maximum_gap_windows):
        raise ValueError("Maximum gap values cannot be negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    window_path = Path(args.window_predictions)
    if not window_path.is_file():
        raise FileNotFoundError(f"Window predictions not found: {window_path}")
    annotation_source = args.annotations
    if not annotation_source.startswith(base.REMOTE_SCHEMES) and not Path(annotation_source).is_file():
        raise FileNotFoundError(f"Annotations not found: {annotation_source}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_path = Path(args.failed_files) if args.failed_files else window_path.parent / "failed_files.csv"
    recordings, cache_sanity = cache_tools.load_cached_recordings(
        window_path, cache_tools.load_failed_soundfiles(failed_path)
    )
    recordings = cache_tools.subset_recordings(recordings, args.max_files, args.seed)
    if not recordings:
        raise ValueError("No cached recordings remain after filtering")
    providers = sorted({provider_name(cached) for cached in recordings})
    if len(providers) < 2:
        raise ValueError("Provider-held-out validation requires at least two providers")
    truths_by_file, annotation_sanity = event_eval.load_truth_events(
        base.read_csv(annotation_source), recordings
    )
    hours = defaultdict(float)
    recordings_by_provider = defaultdict(list)
    for cached in recordings:
        provider = provider_name(cached)
        hours[provider] += cached.recording.duration_sec / 3600.0
        recordings_by_provider[provider].append(cached)
    total_hours = sum(hours.values())

    grids = {species: build_grid(args, species) for species in SPECIES}
    for species, grid in grids.items():
        if not grid:
            raise ValueError(f"No valid configurations for {species}")
        if len(grid) > args.maximum_grid_configurations:
            raise ValueError(
                f"{species} grid has {len(grid):,} configurations; "
                f"increase --maximum-grid-configurations or narrow the grid"
            )
    budgets = {
        "KW": optional_float(args.kw_target_fp_per_hour, args.target_fp_per_hour),
        "HW": optional_float(args.hw_target_fp_per_hour, args.target_fp_per_hour),
        "AB": optional_float(args.ab_target_fp_per_hour, args.target_fp_per_hour),
    }
    print(f"Cached recordings: {len(recordings):,}")
    print(f"Providers:         {len(providers):,} ({', '.join(providers)})")
    print(f"Audio hours:       {total_hours:.3f}")
    print("Grid sizes:        " + ", ".join(f"{s}={len(grids[s]):,}" for s in SPECIES))

    stats: dict[str, list[dict[str, Any]]] = {}
    grid_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[str, dict[str, event_eval.SpeciesConfig]] = {
        provider: {} for provider in providers
    }
    final_configs: dict[str, event_eval.SpeciesConfig] = {}

    for species in SPECIES:
        species_stats = []
        for index, config in enumerate(grids[species], start=1):
            provider_counts = evaluate_config_by_provider(
                recordings,
                truths_by_file,
                providers,
                species,
                config,
                args.collar_sec,
                args.ecotype_threshold,
            )
            species_stats.append({"config": config, "provider_counts": provider_counts})
            if index == 1 or index % 25 == 0 or index == len(grids[species]):
                print(f"{species}: evaluated {index:,}/{len(grids[species]):,} configurations")
        stats[species] = species_stats

        for held_out in providers:
            candidates = []
            train_hours = total_hours - hours[held_out]
            for item in species_stats:
                train_counts = add_counts(
                    counts for provider, counts in item["provider_counts"].items()
                    if provider != held_out
                )
                validation_counts = item["provider_counts"][held_out]
                train_metrics = metric_dict(*train_counts, train_hours)
                validation_metrics = metric_dict(*validation_counts, hours[held_out])
                candidates.append(
                    {
                        "config": item["config"],
                        "metrics": train_metrics,
                        "validation_metrics": validation_metrics,
                    }
                )
            selected = select_config(candidates, args.selection_objective, budgets[species])
            selected_by_fold[held_out][species] = selected["config"]
            selected_id = config_id(selected["config"])
            for candidate in candidates:
                grid_rows.append(
                    {
                        "held_out_provider": held_out,
                        "species": species,
                        "selected": config_id(candidate["config"]) == selected_id,
                        "config_id": config_id(candidate["config"]),
                        **asdict(candidate["config"]),
                        **prefix_fields("training", candidate["metrics"]),
                        **prefix_fields("held_out", candidate["validation_metrics"]),
                    }
                )

        all_candidates = []
        for item in species_stats:
            all_counts = add_counts(item["provider_counts"].values())
            all_candidates.append(
                {
                    "config": item["config"],
                    "metrics": metric_dict(*all_counts, total_hours),
                }
            )
        final_configs[species] = select_config(
            all_candidates, args.selection_objective, budgets[species]
        )["config"]

    selected_rows = []
    isolated_cv_totals = {species: (0, 0, 0) for species in SPECIES}
    for held_out in providers:
        for species in SPECIES:
            config = selected_by_fold[held_out][species]
            item = next(
                item for item in stats[species] if config_id(item["config"]) == config_id(config)
            )
            train_counts = add_counts(
                counts for provider, counts in item["provider_counts"].items()
                if provider != held_out
            )
            held_counts = item["provider_counts"][held_out]
            isolated_cv_totals[species] = add_counts((isolated_cv_totals[species], held_counts))
            selected_rows.append(
                {
                    "held_out_provider": held_out,
                    "species": species,
                    "fp_per_hour_budget": budgets[species],
                    "config_id": config_id(config),
                    **asdict(config),
                    **prefix_fields("training", metric_dict(*train_counts, total_hours - hours[held_out])),
                    **prefix_fields("held_out", metric_dict(*held_counts, hours[held_out])),
                }
            )

    # Reconstruct predictions using only each fold's training-selected settings,
    # then apply the established joint, class-aware matching on the held-out data.
    cv_events = []
    cv_matches = []
    for held_out in providers:
        configs = selected_by_fold[held_out]
        for cached in recordings_by_provider[held_out]:
            file_events = []
            for species in SPECIES:
                file_events.extend(
                    event_eval.form_events(cached, configs[species], args.ecotype_threshold)
                )
            file_events = event_eval.suppress_cross_class_events(
                file_events, args.cross_class_suppression_sec
            )
            file_truths = truths_by_file.get(cached.recording.soundfile, [])
            cv_events.extend(file_events)
            cv_matches.extend(event_eval.match_events(file_events, file_truths, args.collar_sec))

    isolated_rows = [
        {"species": species, **metric_dict(*isolated_cv_totals[species], total_hours)}
        for species in SPECIES
    ]
    joint_species_rows = event_eval.per_species_metrics(cv_matches, total_hours)
    joint_overall = event_eval.overall_metrics(cv_matches, total_hours)
    provider_joint_species_rows = event_eval.grouped_metrics(
        recordings, cv_matches, "provider"
    )
    provider_joint_overall_rows = event_eval.grouped_overall_metrics(
        recordings, cv_matches, "provider"
    )
    ecotype_overall, ecotype_rows, ecotype_matrix = event_eval.ecotype_metrics(cv_matches)
    actual_labels, predicted_labels, species_matrix = event_eval.species_confusion(cv_matches)

    write_rows(output_dir / "provider_held_out_selected_metrics.csv", selected_rows)
    write_rows(output_dir / "provider_cv_grid_results.csv", grid_rows)
    write_rows(output_dir / "cross_validated_isolated_species_metrics.csv", isolated_rows)
    write_rows(output_dir / "cross_validated_joint_species_metrics.csv", joint_species_rows)
    write_rows(
        output_dir / "cross_validated_provider_species_metrics.csv",
        provider_joint_species_rows,
    )
    write_rows(
        output_dir / "cross_validated_provider_overall_metrics.csv",
        provider_joint_overall_rows,
    )
    write_rows(output_dir / "cross_validated_events.csv", [asdict(event) for event in cv_events])
    write_rows(output_dir / "cross_validated_event_matches.csv", [asdict(match) for match in cv_matches])
    write_rows(output_dir / "cross_validated_ecotype_metrics.csv", ecotype_rows)
    event_eval.write_matrix(
        output_dir / "cross_validated_species_confusion_matrix.csv",
        actual_labels,
        predicted_labels,
        species_matrix,
        "actual_species",
    )
    event_eval.write_matrix(
        output_dir / "cross_validated_ecotype_confusion_matrix.csv",
        list(event_eval.ECOTYPES),
        [*event_eval.ECOTYPES, "unknown"],
        ecotype_matrix,
        "actual_ecotype",
    )
    final_payload = {
        "selection_objective": args.selection_objective,
        "fp_per_hour_budgets": budgets,
        "configs": {species: asdict(config) for species, config in final_configs.items()},
        "warning": (
            "These final settings use all providers for tuning. Freeze them before "
            "evaluating a genuinely independent test collection."
        ),
    }
    (output_dir / "final_selected_configs.json").write_text(
        json.dumps(final_payload, indent=2), encoding="utf-8"
    )
    summary = {
        "arguments": vars(args),
        "cache_sanity": cache_sanity,
        "annotation_sanity": annotation_sanity,
        "providers": providers,
        "provider_audio_hours": dict(hours),
        "grid_sizes": {species: len(grid) for species, grid in grids.items()},
        "cross_validated_isolated_species_metrics": isolated_rows,
        "cross_validated_joint_species_metrics": joint_species_rows,
        "cross_validated_joint_overall_metrics": joint_overall,
        "cross_validated_provider_species_metrics": provider_joint_species_rows,
        "cross_validated_provider_overall_metrics": provider_joint_overall_rows,
        "cross_validated_ecotype_metrics": ecotype_overall,
        "final_selected_configs": final_payload,
    }
    (output_dir / "provider_cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nCross-validated held-out-provider performance (joint class-aware matching)")
    print("Species      Truth     Pred       TP       FP       FN        P        R       F1      FP/h")
    for row in joint_species_rows:
        def shown(value: Any) -> str:
            return "None" if value is None else f"{value:.4f}"
        print(
            f"{row['species']:<9} {row['ground_truth_events']:>8,} {row['predicted_events']:>8,} "
            f"{row['true_positives']:>8,} {row['false_positives']:>8,} "
            f"{row['false_negatives']:>8,} {shown(row['precision']):>8} "
            f"{shown(row['recall']):>8} {shown(row['f1']):>8} "
            f"{shown(row['false_positives_per_hour']):>9}"
        )
    print("\nCombined cross-validated performance")
    print(f"Detection F1 ignoring species: {joint_overall['detection_f1_ignoring_species']}")
    print(f"Class-aware micro F1:          {joint_overall['class_aware_micro_f1']}")
    print(f"Class-aware macro F1:          {joint_overall['class_aware_macro_f1']}")
    print(f"Total class-aware FP/hour:     {joint_overall['total_false_positives_per_hour']}")
    print("\nFinal settings selected on all providers (for a future independent test set)")
    for species in SPECIES:
        print(f"{species}: {config_id(final_configs[species])}")
    print(f"\nSaved reports to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
