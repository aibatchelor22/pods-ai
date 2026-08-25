#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Grid-search hierarchical DCLDE 60-second species/ecotype inference.

Model inference is performed once and all raw species/ecotype probabilities are
cached. Species is decided first. An ecotype is emitted only when the predicted
species is killer whale (KW). Ground truth comes from the manifest's
``primary_label`` column, so this remains a single-label 60-second evaluation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

import grid_search_dclde_60s_multispecies as common


DEFAULT_OUTPUT_DIR = "/kaggle/working/dclde_60s_full_taxonomy_grid_search"
SPECIES = ("background", "KW", "HW", "AB")
POSITIVE_SPECIES = ("KW", "HW", "AB")
ECOTYPES = ("NRKW", "SRKW", "OKW", "SAR", "TKW")
HIERARCHICAL_LABELS = ("background", "HW", "AB", *ECOTYPES, "KW/unknown")
NOT_KW = "not-KW"
UNKNOWN_ECOTYPE = "unknown"
SCORE_NAMES = (*SPECIES, *ECOTYPES)
CACHE_FIELDS = (
    "clip_id",
    "window_index",
    *(f"species_{label}" for label in SPECIES),
    *(f"ecotype_{label}" for label in ECOTYPES),
)


@dataclass(frozen=True)
class GridConfig:
    run: int
    smoothing: bool
    top_k: int
    kw_threshold: float
    hw_threshold: float
    ab_threshold: float
    ecotype_threshold: float
    species_min_windows: int
    ecotype_min_windows: int


def parse_list(value: str, item_type: type, name: str) -> list[Any]:
    try:
        result = [item_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not result:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return sorted(set(result))


def build_configs(args: argparse.Namespace) -> list[GridConfig]:
    dimensions = (
        common.parse_smoothing(args.smoothing_values),
        parse_list(args.top_k_values, int, "top-k values"),
        parse_list(args.kw_thresholds, float, "KW thresholds"),
        parse_list(args.hw_thresholds, float, "HW thresholds"),
        parse_list(args.ab_thresholds, float, "AB thresholds"),
        parse_list(args.ecotype_thresholds, float, "ecotype thresholds"),
        parse_list(args.species_min_windows, int, "species minimum windows"),
        parse_list(args.ecotype_min_windows, int, "ecotype minimum windows"),
    )
    return [
        GridConfig(run, *values)
        for run, values in enumerate(itertools.product(*dimensions), start=1)
    ]


def normalize_primary_label(value: str) -> tuple[str, str]:
    label = str(value or "").strip()
    aliases = {
        "BKG": "background",
        "UndBio": "background",
        "other/background": "background",
        "humpback": "HW",
    }
    label = aliases.get(label, label)
    if label in ECOTYPES:
        return "KW", label
    if label in SPECIES:
        return label, ""
    raise ValueError(f"Unsupported primary_label: {value!r}")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        required = {"clip_id", "clip_path", "primary_label"}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        rows = list(reader)
    ids = [row.get("clip_id", "").strip() for row in rows]
    if any(not clip_id for clip_id in ids):
        raise ValueError("Manifest contains an empty clip_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate clip_id values")
    for row in rows:
        species, ecotype = normalize_primary_label(row["primary_label"])
        row["actual_species"] = species
        row["actual_ecotype"] = ecotype
    return fields, rows


def select_rows(
    rows: list[dict[str, str]], max_samples: Optional[int], seed: int
) -> list[dict[str, str]]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    return random.Random(seed).sample(rows, max_samples)


def read_cache(path: Path, expected_windows: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    grouped: dict[str, list[tuple[int, list[float]]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or []) != CACHE_FIELDS:
            raise ValueError(
                f"Existing cache has an incompatible schema: {path}. "
                "Use --refresh-cache to rebuild it."
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                values = [
                    *(float(row[f"species_{label}"]) for label in SPECIES),
                    *(float(row[f"ecotype_{label}"]) for label in ECOTYPES),
                ]
                item = (int(row["window_index"]), values)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid cache value at row {row_number}") from error
            grouped.setdefault(row["clip_id"], []).append(item)
    result = {}
    expected_indices = list(range(expected_windows))
    for clip_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        if [item[0] for item in items] != expected_indices:
            continue
        result[clip_id] = np.asarray([item[1] for item in items], dtype=np.float32)
    return result


def validate_cache_metadata(
    path: Path, cache_exists: bool, model_path: str, expected_windows: int
) -> None:
    if not cache_exists:
        return
    if not path.is_file():
        raise ValueError(f"Cache exists without metadata: {path}; use --refresh-cache")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("model_path") != model_path:
        raise ValueError(
            "Cached probabilities came from a different model; use a new output "
            "directory or --refresh-cache"
        )
    if metadata.get("expected_windows") != expected_windows:
        raise ValueError("Cached window count differs from --expected-windows")


def write_cache_metadata(
    path: Path, model_path: str, expected_windows: int, manifest: Path
) -> None:
    path.write_text(
        json.dumps(
            {
                "model_path": model_path,
                "expected_windows": expected_windows,
                "manifest": str(manifest),
                "score_names": SCORE_NAMES,
                "hierarchy": "Ecotype is reported only when predicted species is KW",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def cache_inference(
    rows: list[dict[str, str]],
    resolved_paths: dict[str, Path],
    cached: dict[str, np.ndarray],
    cache_path: Path,
    predictor: Any,
    expected_windows: int,
    log_every: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    pending = [row for row in rows if row["clip_id"] not in cached]
    failures = []
    write_header = not cache_path.is_file()
    with cache_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CACHE_FIELDS)
        if write_header:
            writer.writeheader()
        for index, row in enumerate(pending, start=1):
            clip_id = row["clip_id"]
            audio_path = resolved_paths[clip_id]
            try:
                scores = np.asarray(
                    predictor.predict_full_window_scores(audio_path), dtype=np.float32
                )
                if scores.shape != (expected_windows, len(SCORE_NAMES)):
                    raise ValueError(
                        f"expected score shape {(expected_windows, len(SCORE_NAMES))}, "
                        f"received {scores.shape}"
                    )
                for window_index, values in enumerate(scores):
                    output = {"clip_id": clip_id, "window_index": window_index}
                    output.update(
                        {
                            f"species_{label}": float(values[class_index])
                            for class_index, label in enumerate(SPECIES)
                        }
                    )
                    output.update(
                        {
                            f"ecotype_{label}": float(values[len(SPECIES) + class_index])
                            for class_index, label in enumerate(ECOTYPES)
                        }
                    )
                    writer.writerow(output)
                file.flush()
                cached[clip_id] = scores
            except Exception as error:
                failures.append(
                    {
                        "clip_id": clip_id,
                        "clip_path": str(audio_path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if index % log_every == 0 or index == len(pending):
                print(f"Inferred {index:,}/{len(pending):,} uncached clips")
    return cached, failures


def build_candidate_cache(
    scores_by_smoothing: dict[bool, np.ndarray], configs: list[GridConfig]
) -> dict[tuple[bool, int, float, int, int], np.ndarray]:
    cache = {}
    for config in configs:
        class_settings = (
            (1, config.kw_threshold, config.species_min_windows),
            (2, config.hw_threshold, config.species_min_windows),
            (3, config.ab_threshold, config.species_min_windows),
            *(
                (len(SPECIES) + index, config.ecotype_threshold, config.ecotype_min_windows)
                for index in range(len(ECOTYPES))
            ),
        )
        for class_index, threshold, minimum_windows in class_settings:
            key = (
                config.smoothing,
                class_index,
                threshold,
                config.top_k,
                minimum_windows,
            )
            if key not in cache:
                cache[key] = common.candidate_array(
                    scores_by_smoothing[config.smoothing][:, :, class_index],
                    threshold,
                    config.top_k,
                    minimum_windows,
                )
    return cache


def choose_candidates(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    means = np.column_stack([value[:, 0] for value in values])
    counts = np.column_stack([value[:, 1] for value in values])
    confidence = np.max(means, axis=1)
    tied = means == confidence[:, None]
    winner = np.argmax(np.where(tied, counts, -1), axis=1)
    unavailable = ~np.isfinite(confidence)
    confidence[unavailable] = 0.0
    winner[unavailable] = -1
    return winner, confidence


def predict_config(
    config: GridConfig,
    candidates: dict[tuple[bool, int, float, int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    species_parameters = (
        (1, config.kw_threshold),
        (2, config.hw_threshold),
        (3, config.ab_threshold),
    )
    species_values = [
        candidates[(
            config.smoothing,
            class_index,
            threshold,
            config.top_k,
            config.species_min_windows,
        )]
        for class_index, threshold in species_parameters
    ]
    species_winner, species_confidence = choose_candidates(species_values)
    # Candidate positions are KW/HW/AB; unavailable rows fall back to background.
    predicted_species = species_winner + 1
    predicted_species[species_winner < 0] = 0

    ecotype_values = [
        candidates[(
            config.smoothing,
            len(SPECIES) + index,
            config.ecotype_threshold,
            config.top_k,
            config.ecotype_min_windows,
        )]
        for index in range(len(ECOTYPES))
    ]
    predicted_ecotype, ecotype_confidence = choose_candidates(ecotype_values)
    # Hierarchical safeguard: suppress every ecotype unless species == KW.
    not_predicted_kw = predicted_species != SPECIES.index("KW")
    predicted_ecotype[not_predicted_kw] = -1
    ecotype_confidence[not_predicted_kw] = 0.0
    return (
        predicted_species,
        species_confidence,
        predicted_ecotype,
        ecotype_confidence,
    )


def square_confusion(actual: np.ndarray, predicted: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def per_class_metrics(
    matrix: np.ndarray, labels: tuple[str, ...], macro_labels: Optional[set[str]] = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = int(matrix.sum())
    rows = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        rows.append(
            {
                "label": label,
                "support": tp + fn,
                "predicted_count": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "false_negative_rate": fn / (tp + fn) if tp + fn else None,
            }
        )
    included = [
        row["f1"]
        for row in rows
        if row["f1"] is not None
        and row["support"] > 0
        and (macro_labels is None or row["label"] in macro_labels)
    ]
    return (
        {
            "evaluated": total,
            "correct": int(np.trace(matrix)),
            "accuracy": float(np.trace(matrix) / total) if total else None,
            "macro_f1": float(np.mean(included)) if included else None,
        },
        rows,
    )


def hierarchical_ids(
    predicted_species: np.ndarray, predicted_ecotype: np.ndarray
) -> np.ndarray:
    output = np.empty(len(predicted_species), dtype=np.int64)
    # Hierarchical order: background=0, HW=1, AB=2, ecotypes=3..7, unknown=8.
    output[predicted_species == SPECIES.index("background")] = 0
    output[predicted_species == SPECIES.index("HW")] = 1
    output[predicted_species == SPECIES.index("AB")] = 2
    kw = predicted_species == SPECIES.index("KW")
    output[kw] = len(HIERARCHICAL_LABELS) - 1
    known_ecotype = kw & (predicted_ecotype >= 0)
    output[known_ecotype] = predicted_ecotype[known_ecotype] + 3
    return output


def ecotype_end_to_end_metrics(
    actual_species: np.ndarray,
    actual_ecotype: np.ndarray,
    predicted_species: np.ndarray,
    predicted_ecotype: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, tuple[str, ...]]:
    mask = (actual_species == SPECIES.index("KW")) & (actual_ecotype >= 0)
    actual = actual_ecotype[mask]
    prediction = predicted_ecotype[mask].copy()
    predicted_kw = predicted_species[mask] == SPECIES.index("KW")
    # Five ecotypes plus unknown and not-KW outcomes.
    prediction[predicted_kw & (prediction < 0)] = len(ECOTYPES)
    prediction[~predicted_kw] = len(ECOTYPES) + 1
    labels = (*ECOTYPES, UNKNOWN_ECOTYPE, NOT_KW)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    if len(actual):
        np.add.at(matrix, (actual, prediction), 1)
    overall, rows = per_class_metrics(matrix, labels, set(ECOTYPES))
    conditional = predicted_kw
    overall["conditional_accuracy_when_species_kw"] = (
        float(np.mean(prediction[conditional] == actual[conditional]))
        if np.any(conditional)
        else None
    )
    overall["species_kw_recall_for_ecotype_rows"] = (
        float(np.mean(predicted_kw)) if len(predicted_kw) else None
    )
    return overall, rows, matrix, labels


def actual_hierarchical_ids(actual_species: np.ndarray, actual_ecotype: np.ndarray) -> np.ndarray:
    return hierarchical_ids(actual_species, actual_ecotype)


def group_scores(
    actual_hierarchical: np.ndarray,
    predicted_hierarchical: np.ndarray,
    groups: np.ndarray,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        samples = int(mask.sum())
        if samples < minimum_samples:
            continue
        matrix = square_confusion(
            actual_hierarchical[mask], predicted_hierarchical[mask], len(HIERARCHICAL_LABELS)
        )
        overall, _ = per_class_metrics(
            matrix, HIERARCHICAL_LABELS, set(HIERARCHICAL_LABELS[:-1])
        )
        rows.append({"group": group, "samples": samples, **overall})
    return rows


def mean_available(values: list[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return float(np.mean(available)) if available else None


def minimum_available(values: list[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return min(available) if available else None


def evaluate_config(
    config: GridConfig,
    candidates: dict[tuple[bool, int, float, int, int], np.ndarray],
    actual_species: np.ndarray,
    actual_ecotype: np.ndarray,
    actual_hierarchical: np.ndarray,
    groups: np.ndarray,
    minimum_group_samples: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    predictions = predict_config(config, candidates)
    predicted_species, _, predicted_ecotype, _ = predictions
    species_matrix = square_confusion(actual_species, predicted_species, len(SPECIES))
    species_overall, species_rows = per_class_metrics(species_matrix, SPECIES)
    ecotype_overall, _, _, _ = ecotype_end_to_end_metrics(
        actual_species, actual_ecotype, predicted_species, predicted_ecotype
    )
    predicted_hierarchical = hierarchical_ids(predicted_species, predicted_ecotype)
    hierarchical_matrix = square_confusion(
        actual_hierarchical, predicted_hierarchical, len(HIERARCHICAL_LABELS)
    )
    hierarchical_overall, hierarchical_rows = per_class_metrics(
        hierarchical_matrix, HIERARCHICAL_LABELS, set(HIERARCHICAL_LABELS[:-1])
    )
    groups_rows = group_scores(
        actual_hierarchical,
        predicted_hierarchical,
        groups,
        minimum_group_samples,
    )
    row = {
        **asdict(config),
        "species_accuracy": species_overall["accuracy"],
        "species_macro_f1": species_overall["macro_f1"],
        "ecotype_end_to_end_accuracy": ecotype_overall["accuracy"],
        "ecotype_end_to_end_macro_f1": ecotype_overall["macro_f1"],
        "ecotype_conditional_accuracy": ecotype_overall[
            "conditional_accuracy_when_species_kw"
        ],
        "species_kw_recall_for_ecotype_rows": ecotype_overall[
            "species_kw_recall_for_ecotype_rows"
        ],
        "hierarchical_accuracy": hierarchical_overall["accuracy"],
        "hierarchical_macro_f1": hierarchical_overall["macro_f1"],
        "mean_group_hierarchical_macro_f1": mean_available(
            [item["macro_f1"] for item in groups_rows]
        ),
        "minimum_group_hierarchical_macro_f1": minimum_available(
            [item["macro_f1"] for item in groups_rows]
        ),
        "groups_evaluated": len(groups_rows),
    }
    for prefix, metric_rows in (
        ("species", species_rows),
        ("hierarchical", hierarchical_rows),
    ):
        for metric_row in metric_rows:
            label = metric_row["label"].replace("/", "_")
            row[f"{prefix}_{label}_f1"] = metric_row["f1"]
            row[f"{prefix}_{label}_precision"] = metric_row["precision"]
            row[f"{prefix}_{label}_recall"] = metric_row["recall"]
    return row, predictions


def ranking_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return -math.inf
    return float(value)


def write_rows(
    path: Path, rows: list[dict[str, Any]], fields: Optional[list[str]] = None
) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(
    path: Path, matrix: np.ndarray, row_labels: tuple[str, ...], column_labels: tuple[str, ...]
) -> None:
    write_rows(
        path,
        [
            {
                "actual_label": actual_label,
                **{
                    predicted_label: int(matrix[row_index, column_index])
                    for column_index, predicted_label in enumerate(column_labels)
                },
            }
            for row_index, actual_label in enumerate(row_labels)
        ],
        ["actual_label", *column_labels],
    )


def make_predictor(model_path: str, batch_size: int, device: Optional[str]) -> Any:
    import librosa
    import torch
    from compare_new_models_experimantal_2 import MultiSpeciesWindowPredictor, SAMPLE_RATE
    from multispecies_train_model import ECOTYPE_LABELS, SPECIES_LABELS

    class FullHeadWindowScorePredictor(MultiSpeciesWindowPredictor):
        def predict_full_window_scores(self, wav_path: Path) -> list[list[float]]:
            audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
            windows = self._window_audio(audio)
            result = []
            with torch.inference_mode():
                for start in range(0, len(windows), self.batch_size):
                    inputs = self.feature_extractor(
                        windows[start:start + self.batch_size],
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    outputs = self.model(**inputs)
                    _, species_logits, ecotype_logits = outputs["logits"]
                    species_probs = torch.softmax(species_logits, dim=-1).cpu().numpy()
                    ecotype_probs = torch.softmax(ecotype_logits, dim=-1).cpu().numpy()
                    for species_row, ecotype_row in zip(species_probs, ecotype_probs):
                        result.append(
                            [
                                *(float(species_row[SPECIES_LABELS[label]]) for label in SPECIES),
                                *(float(ecotype_row[ECOTYPE_LABELS[label]]) for label in ECOTYPES),
                            ]
                        )
            return result

    return FullHeadWindowScorePredictor(
        model_path=model_path,
        threshold=0.25,
        class_thresholds={"humpback": 0.475, "resident": 0.05, "transient": 0.20},
        aggregation_mode="topk_mean",
        use_smoothing=False,
        top_k=2,
        class_min_windows={"humpback": 2, "resident": 2, "transient": 3},
        min_num_positive_calls_threshold=3,
        batch_size=batch_size,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", default=common.DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-windows", type=int, default=29)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--group-column", default="auto")
    parser.add_argument("--minimum-group-samples", type=int, default=20)
    parser.add_argument(
        "--ranking-metric",
        choices=(
            "hierarchical_macro_f1",
            "mean_group_hierarchical_macro_f1",
            "minimum_group_hierarchical_macro_f1",
            "species_macro_f1",
            "ecotype_end_to_end_macro_f1",
        ),
        default="mean_group_hierarchical_macro_f1",
    )
    parser.add_argument("--top-results", type=int, default=25)
    parser.add_argument("--kw-thresholds", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--hw-thresholds", default="0.35,0.475,0.60")
    parser.add_argument("--ab-thresholds", default="0.35,0.50,0.65")
    parser.add_argument("--ecotype-thresholds", default="0.0,0.20,0.30")
    parser.add_argument("--top-k-values", default="2,3")
    parser.add_argument("--species-min-windows", default="1,2,3")
    parser.add_argument("--ecotype-min-windows", default="1,2,3")
    parser.add_argument("--smoothing-values", default="off")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positive_values = (
        args.batch_size,
        args.expected_windows,
        args.log_every,
        args.minimum_group_samples,
        args.top_results,
    )
    if min(positive_values) < 1:
        raise ValueError("Batch/window/log/group/top-result values must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    configs = build_configs(args)
    for config in configs:
        thresholds = (
            config.kw_threshold,
            config.hw_threshold,
            config.ab_threshold,
            config.ecotype_threshold,
        )
        if any(not 0 <= threshold <= 1 for threshold in thresholds):
            raise ValueError("All thresholds must be between 0 and 1")
        if min(config.top_k, config.species_min_windows, config.ecotype_min_windows) < 1:
            raise ValueError("Top-k and minimum-window values must be positive")

    import evaluate_dclde_60s_multispecies_kaggle as dclde

    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "full_window_probability_cache.csv"
    metadata_path = output_dir / "full_window_probability_cache_metadata.json"
    if args.refresh_cache:
        for path in (cache_path, metadata_path):
            if path.exists():
                path.unlink()

    fields, all_rows = read_manifest(manifest_path)
    rows = select_rows(all_rows, args.max_samples, args.seed)
    selected_ids = {row["clip_id"] for row in rows}
    group_column = common.find_group_column(fields, args.group_column)
    print(f"Manifest rows selected: {len(rows):,}")
    print(f"Grid configurations:    {len(configs):,}")
    print(f"Group column:           {group_column or 'none'}")

    validate_cache_metadata(
        metadata_path, cache_path.is_file(), args.model_path, args.expected_windows
    )
    cached = read_cache(cache_path, args.expected_windows)
    cached = {key: value for key, value in cached.items() if key in selected_ids}
    pending = [row for row in rows if row["clip_id"] not in cached]
    failures = []
    if pending:
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        resolved = {
            row["clip_id"]: dclde.resolve_clip_path(row, dataset_root) for row in pending
        }
        missing = [(key, path) for key, path in resolved.items() if not path.is_file()]
        if missing:
            examples = "\n".join(f"  {key}: {path}" for key, path in missing[:20])
            raise FileNotFoundError(
                f"Preflight found {len(missing):,} missing WAV files. First examples:\n{examples}"
            )
        predictor = make_predictor(args.model_path, args.batch_size, args.device)
        write_cache_metadata(
            metadata_path, args.model_path, args.expected_windows, manifest_path
        )
        cached, failures = cache_inference(
            rows,
            resolved,
            cached,
            cache_path,
            predictor,
            args.expected_windows,
            args.log_every,
        )
    write_rows(
        output_dir / "inference_failures.csv",
        failures,
        ["clip_id", "clip_path", "error"],
    )

    successful_rows = [row for row in rows if row["clip_id"] in cached]
    if not successful_rows:
        raise RuntimeError("No selected clips have usable cached probabilities")
    scores = np.stack([cached[row["clip_id"]] for row in successful_rows])
    actual_species = np.asarray(
        [SPECIES.index(row["actual_species"]) for row in successful_rows], dtype=np.int64
    )
    actual_ecotype = np.asarray(
        [ECOTYPES.index(row["actual_ecotype"]) if row["actual_ecotype"] else -1 for row in successful_rows],
        dtype=np.int64,
    )
    actual_hierarchical = actual_hierarchical_ids(actual_species, actual_ecotype)
    groups = np.asarray(
        [
            ((row.get(group_column) or "unknown").strip() or "unknown")
            if group_column
            else "all"
            for row in successful_rows
        ],
        dtype=object,
    )
    scores_by_smoothing = {False: scores, True: common.smooth_scores(scores)}
    candidates = build_candidate_cache(scores_by_smoothing, configs)

    started = time.perf_counter()
    results = []
    for index, config in enumerate(configs, start=1):
        result, _ = evaluate_config(
            config,
            candidates,
            actual_species,
            actual_ecotype,
            actual_hierarchical,
            groups,
            args.minimum_group_samples,
        )
        results.append(result)
        if index % 250 == 0 or index == len(configs):
            print(f"Aggregated {index:,}/{len(configs):,} configurations")

    ranking_metric = args.ranking_metric
    if all(ranking_value(row, ranking_metric) == -math.inf for row in results):
        print(
            f"WARNING: {ranking_metric} unavailable; falling back to hierarchical_macro_f1",
            file=sys.stderr,
        )
        ranking_metric = "hierarchical_macro_f1"
    ranked = sorted(
        results,
        key=lambda row: (
            ranking_value(row, ranking_metric),
            ranking_value(row, "hierarchical_macro_f1"),
            ranking_value(row, "species_macro_f1"),
        ),
        reverse=True,
    )
    write_rows(output_dir / "grid_results.csv", results)
    write_rows(output_dir / "ranked_grid_results.csv", ranked)

    best = ranked[0]
    best_config = next(config for config in configs if config.run == best["run"])
    best_predictions = predict_config(best_config, candidates)
    predicted_species, species_confidence, predicted_ecotype, ecotype_confidence = best_predictions
    species_matrix = square_confusion(actual_species, predicted_species, len(SPECIES))
    species_overall, species_rows = per_class_metrics(species_matrix, SPECIES)
    ecotype_overall, ecotype_rows, ecotype_matrix, ecotype_matrix_labels = (
        ecotype_end_to_end_metrics(
            actual_species, actual_ecotype, predicted_species, predicted_ecotype
        )
    )
    predicted_hierarchical = hierarchical_ids(predicted_species, predicted_ecotype)
    hierarchical_matrix = square_confusion(
        actual_hierarchical, predicted_hierarchical, len(HIERARCHICAL_LABELS)
    )
    hierarchical_overall, hierarchical_rows = per_class_metrics(
        hierarchical_matrix, HIERARCHICAL_LABELS, set(HIERARCHICAL_LABELS[:-1])
    )
    best_groups = group_scores(
        actual_hierarchical,
        predicted_hierarchical,
        groups,
        args.minimum_group_samples,
    )
    write_rows(output_dir / "best_species_metrics.csv", species_rows)
    write_rows(output_dir / "best_ecotype_end_to_end_metrics.csv", ecotype_rows)
    write_rows(output_dir / "best_hierarchical_metrics.csv", hierarchical_rows)
    write_rows(output_dir / "best_group_metrics.csv", best_groups)
    write_matrix(output_dir / "best_species_confusion_matrix.csv", species_matrix, SPECIES, SPECIES)
    write_matrix(
        output_dir / "best_ecotype_end_to_end_confusion_matrix.csv",
        ecotype_matrix,
        ecotype_matrix_labels,
        ecotype_matrix_labels,
    )
    write_matrix(
        output_dir / "best_hierarchical_confusion_matrix.csv",
        hierarchical_matrix,
        HIERARCHICAL_LABELS,
        HIERARCHICAL_LABELS,
    )
    write_rows(
        output_dir / "best_predictions.csv",
        [
            {
                "clip_id": row["clip_id"],
                "primary_label": row["primary_label"],
                "contains_labels": row.get("contains_labels", ""),
                "group": groups[index],
                "actual_species": SPECIES[actual_species[index]],
                "predicted_species": SPECIES[predicted_species[index]],
                "species_confidence": float(species_confidence[index]),
                "actual_ecotype": (
                    ECOTYPES[actual_ecotype[index]] if actual_ecotype[index] >= 0 else ""
                ),
                "predicted_ecotype": (
                    ECOTYPES[predicted_ecotype[index]] if predicted_ecotype[index] >= 0 else ""
                ),
                "ecotype_confidence": float(ecotype_confidence[index]),
                "actual_hierarchical_label": HIERARCHICAL_LABELS[
                    actual_hierarchical[index]
                ],
                "predicted_hierarchical_label": HIERARCHICAL_LABELS[
                    predicted_hierarchical[index]
                ],
                "species_correct": bool(actual_species[index] == predicted_species[index]),
                "hierarchical_correct": bool(
                    actual_hierarchical[index] == predicted_hierarchical[index]
                ),
            }
            for index, row in enumerate(successful_rows)
        ],
    )
    summary = {
        "selected_configuration": asdict(best_config),
        "selection_metric_requested": args.ranking_metric,
        "selection_metric_used": ranking_metric,
        "best_result": best,
        "best_species_overall": species_overall,
        "best_ecotype_end_to_end_overall": ecotype_overall,
        "best_hierarchical_overall": hierarchical_overall,
        "manifest": str(manifest_path),
        "model_path": args.model_path,
        "selected_manifest_rows": len(rows),
        "successfully_inferred_rows": len(successful_rows),
        "group_column": group_column,
        "grid_configurations": len(configs),
        "arguments": vars(args),
    }
    (output_dir / "best_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nTop {min(args.top_results, len(ranked))} configurations")
    print("=" * 110)
    for rank, row in enumerate(ranked[:args.top_results], start=1):
        print(
            f"{rank:>3}. run={row['run']:<5} rank={ranking_value(row, ranking_metric):.4f} "
            f"hier_F1={ranking_value(row, 'hierarchical_macro_f1'):.4f} "
            f"species_F1={ranking_value(row, 'species_macro_f1'):.4f} "
            f"ecotype_F1={ranking_value(row, 'ecotype_end_to_end_macro_f1'):.4f} "
            f"smooth={'on' if row['smoothing'] else 'off'} top_k={row['top_k']} "
            f"species_thresholds=(KW={row['kw_threshold']}, HW={row['hw_threshold']}, "
            f"AB={row['ab_threshold']}) eco_threshold={row['ecotype_threshold']} "
            f"mins=({row['species_min_windows']}, {row['ecotype_min_windows']})"
        )
    print(f"\nUsable clips: {len(successful_rows):,}/{len(rows):,}")
    print(f"Aggregation time: {time.perf_counter() - started:.1f}s")
    print(f"Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
