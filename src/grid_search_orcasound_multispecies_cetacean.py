#!/usr/bin/env python3
"""Tune V2 trigger/source/ecotype aggregation on cached Orcasound windows.

Run ``evaluate_orcasound_multispecies_cetacean.py`` once to create the window
cache. This script performs no neural-network inference and can search many
aggregation settings on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

import evaluate_orcasound_multispecies_cetacean as evaluation


def floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(not 0 <= item <= 1 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated probabilities in [0,1]")
    return result


def integers(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return result


def strings(allowed: set[str]) -> Callable[[str], list[str]]:
    def parse(value: str) -> list[str]:
        result = [item.strip() for item in value.split(",") if item.strip()]
        invalid = set(result) - allowed
        if not result or invalid:
            raise argparse.ArgumentTypeError(f"Expected values from {sorted(allowed)}; invalid={sorted(invalid)}")
        return result
    return parse


def booleans(value: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    result: list[bool] = []
    for item in value.split(","):
        key = item.strip().casefold()
        if key not in mapping:
            raise argparse.ArgumentTypeError("Boolean lists use true,false")
        result.append(mapping[key])
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one Boolean")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-cache",
        default="orcasound_multispecies_cetacean_evaluation/window_predictions.csv",
    )
    parser.add_argument("--output-dir", default="orcasound_multispecies_cetacean_grid_search")
    parser.add_argument("--trigger-thresholds", type=floats, default=floats("0.5,0,0.25,0.75,0.9,0.97"))
    parser.add_argument(
        "--trigger-combinations",
        type=strings({"gate", "product"}),
        default=["gate", "product"],
    )
    parser.add_argument(
        "--ecotype-modes",
        type=strings({"srkw_tkw_conditional", "raw"}),
        default=["srkw_tkw_conditional", "raw"],
    )
    parser.add_argument("--kw-source-thresholds", type=floats, default=floats("0,0.25,0.5,0.75"))
    parser.add_argument("--hw-source-thresholds", type=floats, default=floats("0,0.25,0.5,0.75"))
    parser.add_argument("--resident-ecotype-thresholds", type=floats, default=floats("0,0.5,0.7"))
    parser.add_argument("--transient-ecotype-thresholds", type=floats, default=floats("0,0.5,0.7"))
    parser.add_argument("--humpback-thresholds", type=floats, default=floats("0.4,0.2,0.3,0.5,0.6,0.75"))
    parser.add_argument("--resident-thresholds", type=floats, default=floats("0.2,0.02,0.05,0.1,0.3,0.4,0.6,0.8"))
    parser.add_argument("--transient-thresholds", type=floats, default=floats("0.2,0.02,0.05,0.1,0.3,0.4,0.6,0.8"))
    parser.add_argument("--top-ks", type=integers, default=integers("2,1,3"))
    parser.add_argument("--humpback-min-windows-values", type=integers, default=integers("2,1,3"))
    parser.add_argument("--resident-min-windows-values", type=integers, default=integers("2,1,3,4"))
    parser.add_argument("--transient-min-windows-values", type=integers, default=integers("2,1,3,4"))
    parser.add_argument("--smoothing-values", type=booleans, default=[False])
    parser.add_argument(
        "--max-grid-runs",
        type=int,
        default=25000,
        help="Randomized subset of the Cartesian grid; 0 requests the complete grid.",
    )
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument(
        "--resident-max-fp-rate",
        type=float,
        default=0.30,
        help="Maximum resident false-positive rate for resident-sensitive selection.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def dimensions(args: argparse.Namespace) -> list[tuple[str, list[Any]]]:
    return [
        ("trigger_threshold", args.trigger_thresholds),
        ("trigger_combination", args.trigger_combinations),
        ("ecotype_mode", args.ecotype_modes),
        ("kw_source_threshold", args.kw_source_thresholds),
        ("hw_source_threshold", args.hw_source_thresholds),
        ("resident_ecotype_threshold", args.resident_ecotype_thresholds),
        ("transient_ecotype_threshold", args.transient_ecotype_thresholds),
        ("humpback_threshold", args.humpback_thresholds),
        ("resident_threshold", args.resident_thresholds),
        ("transient_threshold", args.transient_thresholds),
        ("top_k", args.top_ks),
        ("humpback_min_windows", args.humpback_min_windows_values),
        ("resident_min_windows", args.resident_min_windows_values),
        ("transient_min_windows", args.transient_min_windows_values),
        ("smoothing", args.smoothing_values),
    ]


def configuration(values: tuple[Any, ...], names: list[str]) -> evaluation.AggregationConfig:
    return evaluation.AggregationConfig(**dict(zip(names, values)))


def configurations(args: argparse.Namespace) -> tuple[Iterable[evaluation.AggregationConfig], int, int]:
    items = dimensions(args)
    names = [name for name, _ in items]
    value_lists = [values for _, values in items]
    total = math.prod(len(values) for values in value_lists)
    if args.max_grid_runs < 0:
        raise ValueError("--max-grid-runs cannot be negative")
    if args.max_grid_runs == 0 or args.max_grid_runs >= total:
        return (configuration(values, names) for values in product(*value_lists)), total, total

    target = args.max_grid_runs
    rng = np.random.default_rng(args.seed)
    selected: set[tuple[Any, ...]] = {tuple(values[0] for values in value_lists)}
    while len(selected) < target:
        selected.add(tuple(values[int(rng.integers(len(values)))] for values in value_lists))
    ordered = sorted(selected, key=lambda values: tuple(str(value) for value in values))
    # Put the explicit control (the first value in every CLI list) first.
    control = tuple(values[0] for values in value_lists)
    ordered.remove(control)
    ordered.insert(0, control)
    return (configuration(values, names) for values in ordered), total, len(ordered)


def result_row(run: int, config: evaluation.AggregationConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"run": run, **asdict(config)}
    row.update(
        evaluated=metrics["evaluated"],
        correct=metrics["correct"],
        accuracy=metrics["accuracy"],
        whale_macro_f1=metrics["whale_macro_f1"],
        all_class_macro_f1=metrics["all_class_macro_f1"],
    )
    for label in evaluation.EVALUATION_LABELS:
        prefix = "other" if label == "other/background" else label
        for metric_name, value in metrics["per_class"][label].items():
            row[f"{prefix}_{metric_name}"] = value
    return row


def config_from_result(row: pd.Series) -> evaluation.AggregationConfig:
    values: dict[str, Any] = {}
    for name, field in evaluation.AggregationConfig.__dataclass_fields__.items():
        value = row[name]
        if field.type is bool or name == "smoothing":
            value = bool(value) if not isinstance(value, str) else value.casefold() == "true"
        elif name in {"top_k", "humpback_min_windows", "resident_min_windows", "transient_min_windows"}:
            value = int(value)
        values[name] = value
    return evaluation.AggregationConfig(**values)


def save_selected(
    name: str,
    row: pd.Series,
    samples: list[evaluation.CachedSample],
    output_dir: Path,
) -> None:
    config = config_from_result(row)
    predictions, matrix, metrics = evaluation.evaluate_cached_samples(samples, config)
    predictions.to_csv(output_dir / f"{name}_clip_predictions.csv", index=False)
    matrix.to_csv(output_dir / f"{name}_confusion_matrix.csv")
    payload = {"aggregation": asdict(config), "metrics": metrics}
    (output_dir / f"{name}_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n# {name.replace('_', ' ').title()}")
    for key, value in asdict(config).items():
        print(f"{key:31s}: {value}")
    evaluation.print_report(name, matrix, metrics)


def main() -> int:
    args = parse_args()
    cache_path = Path(args.window_cache)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Window cache not found: {cache_path}. Run evaluate_orcasound_multispecies_cetacean.py first."
        )
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if not 0 <= args.resident_max_fp_rate <= 1:
        raise ValueError("--resident-max-fp-rate must be in [0,1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "grid_search_results.csv"
    cache = pd.read_csv(cache_path, low_memory=False)
    samples = evaluation.load_cached_samples(cache)
    iterator, complete_grid, selected_runs = configurations(args)
    print(f"Cached recordings:     {len(samples):,}")
    print(f"Cached windows:        {len(cache):,}")
    print(f"Complete grid size:    {complete_grid:,}")
    print(f"Configurations tested: {selected_runs:,}")
    if selected_runs < complete_grid:
        print(f"Search mode:           reproducible randomized grid subset (seed={args.seed})")
    else:
        print("Search mode:           exhaustive Cartesian grid")

    rows: list[dict[str, Any]] = []
    for run, config in enumerate(iterator, start=1):
        _, _, metrics = evaluation.evaluate_cached_samples(samples, config)
        rows.append(result_row(run, config, metrics))
        if run % args.checkpoint_every == 0 or run == selected_runs:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            best = max(rows, key=lambda row: (row["whale_macro_f1"], row["accuracy"]))
            print(
                f"{run:,}/{selected_runs:,}; best whale macro F1={best['whale_macro_f1']:.4f}, "
                f"accuracy={best['accuracy']:.4f}"
            )

    results = pd.DataFrame(rows)
    overall = results.sort_values(
        ["whale_macro_f1", "accuracy", "all_class_macro_f1"], ascending=False
    ).iloc[0]
    eligible = results.loc[results["resident_false_positive_rate"] <= args.resident_max_fp_rate]
    if eligible.empty:
        print("No configuration met the resident FP-rate constraint; using the lowest-FP configuration.")
        resident = results.sort_values(
            ["resident_false_positive_rate", "resident_recall", "resident_f1"],
            ascending=[True, False, False],
        ).iloc[0]
    else:
        resident = eligible.sort_values(
            ["resident_recall", "resident_f1", "whale_macro_f1", "accuracy"], ascending=False
        ).iloc[0]
    results.sort_values(["whale_macro_f1", "accuracy"], ascending=False).to_csv(output_path, index=False)
    save_selected("best_overall", overall, samples, output_dir)
    save_selected("best_resident_sensitive", resident, samples, output_dir)
    selection = {
        "complete_grid_size": complete_grid,
        "tested_configurations": selected_runs,
        "seed": args.seed,
        "resident_max_fp_rate": args.resident_max_fp_rate,
        "best_overall": overall.to_dict(),
        "best_resident_sensitive": resident.to_dict(),
    }
    (output_dir / "selected_configurations.json").write_text(
        json.dumps(selection, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSaved grid results and selected reports to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
