#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Compare multiple models on a test set of audio samples.

Usage:
    python compare_models.py [options]

Loads a test set from testing_samples.csv, then runs each enabled model
(fastai, orcahello, podsai (AST), oldpodsai (Wav2Vec2)) on the corresponding
60-second WAV files and
reports correct identifications, false positives, and false negatives per model.

A "correct" identification means:
  - Model predicted "resident" (SRKW) when the label is "resident".
  - Model predicted anything other than "resident" when the label is not "resident".

A "false positive" means the model predicted "resident" when the correct label is not "resident".
A "false negative" means the model predicted something other than "resident" when the label is "resident".
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from run_inference import run_inference

RESIDENT_LABEL = "resident"
MATRIX_CELL_PADDING = 2
PODSAI_MODEL_ID = "davethaler/whale-call-detector"
# renovate: datasource=git-refs depName=https://huggingface.co/davethaler/whale-call-detector versioning=git.
PODSAI_MODEL_REVISION = "d1eedf5c614268da7551039a84dfc35d317168b9"
OLD_PODSAI_MODEL_REVISION = "cef82c6e9ee661646ea0c583aeb68f4f7ec6d9d8"
# Maps user-facing model names to inference backends. oldpodsai reuses podsai
# inference with a different pinned model revision.
MODEL_TYPE_TO_INFERENCE_TYPE = {
    "fastai": "fastai",
    "orcahello": "orcahello",
    "podsai": "podsai",
    "oldpodsai": "podsai",
}


@dataclass
class TestSample:
    """A single detection row used as a test sample."""

    category: str
    node_name: str
    timestamp: str
    uri: str
    description: str
    notes: str


@dataclass
class ModelResult:
    """Accumulated results for a single model across all test samples."""

    model_type: str
    total: int = 0
    correct: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    skipped: int = 0
    predict_times: list[float] = field(default_factory=list)
    # Maps actual_label -> {predicted_label -> count} for each evaluated sample.
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def evaluated(self) -> int:
        """Number of samples actually evaluated (not skipped)."""
        return self.total - self.skipped

    @property
    def accuracy(self) -> Optional[float]:
        """Fraction of evaluated samples correctly identified."""
        if self.evaluated == 0:
            return None
        return self.correct / self.evaluated

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Fraction of evaluated samples that are false positives."""
        if self.evaluated == 0:
            return None
        return self.false_positives / self.evaluated

    @property
    def false_negative_rate(self) -> Optional[float]:
        """Fraction of evaluated samples that are false negatives."""
        if self.evaluated == 0:
            return None
        return self.false_negatives / self.evaluated

    @property
    def avg_predict_time(self) -> Optional[float]:
        """Average time in seconds spent in predict() method per WAV file."""
        if not self.predict_times:
            return None
        return sum(self.predict_times) / len(self.predict_times)


def load_test_samples(testing_csv: Path, max_samples: Optional[int] = None,
                      category_filter: Optional[str] = None) -> list[TestSample]:
    """
    Load test samples from testing_samples.csv.

    Args:
        testing_csv: Path to testing_samples.csv.
        max_samples: Maximum number of samples to load. If None, load all samples.
        category_filter: If specified, only load samples matching this category.
                        If None, load samples from all categories.

    Returns:
        List of TestSample objects, or an empty list on error.
    """
    samples = []
    try:
        with open(testing_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row.get("Category", "")

                # Skip if category filter is specified and doesn't match.
                if category_filter is not None and category != category_filter:
                    continue

                samples.append(TestSample(
                    category=category,
                    node_name=row.get("NodeName", ""),
                    timestamp=row.get("Timestamp", ""),
                    uri=row.get("URI", ""),
                    description=row.get("Description", ""),
                    notes=row.get("Notes", ""),
                ))

                # Stop if we've reached the maximum.
                if max_samples is not None and len(samples) >= max_samples:
                    break

    except (OSError, csv.Error, UnicodeDecodeError) as e:
        print(f"Error reading {testing_csv}: {e}", file=sys.stderr)
    return samples


def find_wav_file(sample: TestSample, wav_dir: Path) -> Optional[Path]:
    """
    Find the 60-second WAV file for a testing sample.

    WAV files are saved by download_wavs.py as:
        <wav_dir>/<category>/<node_name_with_dashes>_<timestamp>.wav

    Args:
        sample: The testing sample.
        wav_dir: Root directory of testing WAV files.

    Returns:
        Path to the WAV file, or None if not found.
    """
    node_name_in_filename = sample.node_name.replace("_", "-")
    wav_filename = f"{node_name_in_filename}_{sample.timestamp}.wav"
    wav_path = wav_dir / sample.category / wav_filename
    if wav_path.exists():
        return wav_path
    return None


def is_resident_prediction(global_prediction_label: str, model_type: str) -> bool:
    """
    Determine whether a model's prediction corresponds to "resident" (SRKW).

    All model types (fastai, orcahello, podsai, oldpodsai) use "resident" as the
    positive class label, so the check is the same regardless of model type.

    Args:
        global_prediction_label: The model's predicted class label.
        model_type: The model type ('fastai', 'orcahello', or 'podsai').

    Returns:
        True if the prediction is "resident"; False otherwise.
    """
    return global_prediction_label == RESIDENT_LABEL


def evaluate_model(
    model_type: str,
    model_path: Optional[str],
    samples: list[TestSample],
    wav_dir: Path,
    model_revision: Optional[str] = None,
    result_model_type: Optional[str] = None,
) -> ModelResult:
    """
    Run a model against all test samples and accumulate results.

    Args:
        model_type: One of 'fastai', 'orcahello', 'podsai', or 'oldpodsai'.
                    'oldpodsai' is mapped to 'podsai' inference internally.
        model_path: Path to the model (or HuggingFace Hub model ID).
        samples: List of testing samples.
        wav_dir: Root directory containing testing WAV files.
        model_revision: Git commit hash to pin the HuggingFace Hub model revision.
                        Only used when model_path is a Hub model ID (not a local path).
        result_model_type: Optional display name to store in ModelResult.model_type.
                           If omitted, model_type is used.

    Returns:
        ModelResult with counts of correct, false positive, and false negative predictions,
        plus timing information for predict() calls.
    """
    result = ModelResult(model_type=result_model_type or model_type, total=len(samples))

    for sample in samples:
        wav_path = find_wav_file(sample, wav_dir)
        if wav_path is None:
            print(
                f"  [{model_type}] Skipping {sample.category}/{sample.node_name}"
                f"/{sample.timestamp}: WAV not found"
            )
            result.skipped += 1
            continue

        expected_resident = (sample.category == RESIDENT_LABEL)

        try:
            inference_result = run_inference(str(wav_path), model_type=model_type,
                                             model_path=model_path,
                                             model_revision=model_revision)
            predict_time = inference_result.get("predict_time", 0.0)
            result.predict_times.append(predict_time)
        except Exception as e:
            print(f"  [{model_type}] Error on {wav_path.name}: {e}")
            result.skipped += 1
            continue

        predicted_label = inference_result.get("global_prediction_label", "")
        predicted_resident = is_resident_prediction(predicted_label, model_type)

        if predicted_resident == expected_resident:
            result.correct += 1
            status = "correct"
        elif predicted_resident and not expected_resident:
            result.false_positives += 1
            status = "false_positive"
        else:
            result.false_negatives += 1
            status = "false_negative"

        # Update per-class confusion matrix.
        actual_label = sample.category
        if actual_label not in result.confusion_matrix:
            result.confusion_matrix[actual_label] = {}
        preds = result.confusion_matrix[actual_label]
        preds[predicted_label] = preds.get(predicted_label, 0) + 1

        print(
            f"  [{model_type}] {sample.category}/{sample.node_name}/{sample.timestamp}: "
            f"predicted={predicted_label!r} -> {status} ({predict_time:.2f}s)"
        )

    return result


def print_confusion_matrix(result: ModelResult) -> None:
    """
    Print a per-class confusion matrix for a single model result.

    Rows are actual (ground-truth) labels; columns are predicted labels.
    Only labels with at least one non-zero entry in their row (for actuals) or
    column (for predicted) are shown; all-zero rows and columns are omitted.

    Args:
        result: ModelResult whose confusion_matrix to display.
    """
    matrix = result.confusion_matrix
    if not matrix:
        return

    # Collect actual labels (rows) that have at least one non-zero prediction.
    actual_labels = sorted(
        actual for actual in matrix
        if any(count > 0 for count in matrix[actual].values())
    )

    # Collect predicted labels (columns) that have at least one non-zero count.
    predicted_labels = sorted(
        set(
            predicted
            for preds in matrix.values()
            for predicted, count in preds.items()
            if count > 0
        )
    )

    if not actual_labels or not predicted_labels:
        return

    all_labels = sorted(set(actual_labels) | set(predicted_labels))
    col_width = max(len(label) for label in all_labels) + MATRIX_CELL_PADDING
    row_label_width = max(len(label) for label in all_labels) + MATRIX_CELL_PADDING

    print(f"Confusion Matrix for {result.model_type} (rows=actual, cols=predicted):")
    print(f"{'':>{row_label_width}}", end="")
    for label in predicted_labels:
        print(f"{label:>{col_width}}", end="")
    print()

    for actual in actual_labels:
        print(f"{actual:>{row_label_width}}", end="")
        for predicted in predicted_labels:
            count = matrix.get(actual, {}).get(predicted, 0)
            print(f"{count:>{col_width}}", end="")
        print()


def print_summary(results: list[ModelResult]) -> None:
    """
    Print a formatted comparison table for all model results.

    Args:
        results: List of ModelResult objects, one per model.
    """
    print()
    print("=" * 90)
    print("Model Comparison Summary")
    print("=" * 90)
    header = (
        f"{'Model':<15} {'Evaluated':>9} {'Correct':>9} {'Accuracy':>9}"
        f" {'FP':>6} {'FP%':>7} {'FN':>6} {'FN%':>7} {'Avg Time':>10}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        evaluated = r.evaluated
        accuracy = f"{r.accuracy:.1%}" if r.accuracy is not None else "N/A"
        fp_rate = f"{r.false_positive_rate:.1%}" if r.false_positive_rate is not None else "N/A"
        fn_rate = f"{r.false_negative_rate:.1%}" if r.false_negative_rate is not None else "N/A"
        avg_time = f"{r.avg_predict_time:.2f}s" if r.avg_predict_time is not None else "N/A"

        print(
            f"{r.model_type:<15} {evaluated:>9} {r.correct:>9} {accuracy:>9}"
            f" {r.false_positives:>6} {fp_rate:>7} {r.false_negatives:>6} {fn_rate:>7} {avg_time:>10}"
        )
        if r.skipped:
            print(f"  ({r.skipped} skipped due to missing WAV or inference error)")

    print("=" * 90)
    print()
    print("Definitions:")
    print("  Correct      = predicted resident when expected, or non-resident when expected")
    print("  FP (false+)  = predicted resident when correct class was non-resident")
    print("  FN (false-)  = predicted non-resident when correct class was resident")
    print("  Avg Time     = average time spent in model predict() per 60-second WAV file")

    for r in results:
        print()
        print_confusion_matrix(r)


def main() -> int:
    """Entry point for the compare_models CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare model predictions on a test set loaded from testing_samples.csv. "
            "Runs each enabled model against the corresponding 60-second WAV files "
            "and reports correct identifications, false positives, and false negatives."
        )
    )
    parser.add_argument(
        "--testing-csv",
        default="output/csv/testing_samples.csv",
        help="Path to testing_samples.csv (default: output/csv/testing_samples.csv).",
    )
    parser.add_argument(
        "--wav-dir",
        default="output/testing-wav",
        help="Root directory containing testing WAV files (default: output/testing-wav).",
    )
    parser.add_argument(
        "--models",
        default="fastai,orcahello,podsai,oldpodsai",
        help=(
            "Comma-separated list of models to evaluate "
            "(default: fastai,orcahello,podsai,oldpodsai)."
        ),
    )
    parser.add_argument(
        "--fastai-model-path",
        default="model",
        help=(
            "Path to FastAI model directory. "
            "Defaults to model when not specified."
        ),
    )
    parser.add_argument(
        "--orcahello-model-path",
        default="orcasound/orcahello-srkw-detector-v1",
        help=(
            "Path or HuggingFace Hub ID for the OrcaHello model. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 when not specified."
        ),
    )
    parser.add_argument(
        "--podsai-model-path",
        default=PODSAI_MODEL_ID,
        help=(
            "Path to PODS-AI model directory or HuggingFace Hub ID. "
            "Used by both podsai (AST) and oldpodsai (Wav2Vec2). "
            f"Defaults to {PODSAI_MODEL_ID!r} when not specified."
        ),
    )
    parser.add_argument(
        "--podsai-model-revision",
        default=PODSAI_MODEL_REVISION,
        help=(
            "Git commit hash to pin the PODS-AI HuggingFace Hub model revision. "
            "Only used when --podsai-model-path is a Hub model ID. "
            f"Defaults to the pinned revision ({PODSAI_MODEL_REVISION})."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of test samples to process. "
            "If not specified, all samples are processed."
        ),
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Only test samples from this category. "
            "If not specified, all categories are tested. "
            "Categories: water, resident, transient, humpback, vessel, jingle, human."
        ),
    )

    args = parser.parse_args()

    testing_csv = Path(args.testing_csv)
    if not testing_csv.exists():
        print(f"Error: testing CSV not found: {testing_csv}", file=sys.stderr)
        print(
            "Run extract_training_samples.py first to generate testing_samples.csv.",
            file=sys.stderr,
        )
        return 1

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        print(
            "Run download_wavs.py first to download testing WAV files.",
            file=sys.stderr,
        )
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid_models = {"fastai", "orcahello", "podsai", "oldpodsai"}
    for model in models:
        if model not in valid_models:
            print(
                f"Error: unknown model type {model!r}. Valid: {sorted(valid_models)}",
                file=sys.stderr,
            )
            return 1

    model_paths: dict[str, Optional[str]] = {
        "fastai": args.fastai_model_path,
        "orcahello": args.orcahello_model_path,
        "podsai": args.podsai_model_path,
        "oldpodsai": args.podsai_model_path,
    }
    model_revisions: dict[str, Optional[str]] = {
        "fastai": None,
        "orcahello": None,
        "podsai": args.podsai_model_revision,
        "oldpodsai": OLD_PODSAI_MODEL_REVISION,
    }

    # Validate max_samples if specified.
    if args.max_samples is not None and args.max_samples <= 0:
        print(f"Error: --max-samples must be a positive integer, got {args.max_samples}", file=sys.stderr)
        return 1

    samples = load_test_samples(testing_csv, max_samples=args.max_samples,
                                category_filter=args.category)
    if not samples:
        if args.category:
            print(f"Error: no test samples found for category '{args.category}'.", file=sys.stderr)
        else:
            print("Error: no test samples found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(samples)} test samples from {testing_csv}")
    if args.category:
        print(f"  (filtered to category: {args.category})")
    if args.max_samples is not None:
        print(f"  (limited to first {args.max_samples} samples)")
    print(f"WAV directory: {wav_dir}")
    print(f"Models to evaluate: {', '.join(models)}")
    print()

    results = []
    for model_type in models:
        print(f"Evaluating model: {model_type}")
        inference_model_type = MODEL_TYPE_TO_INFERENCE_TYPE[model_type]
        model_result = evaluate_model(
            model_type=inference_model_type,
            model_path=model_paths[model_type],
            samples=samples,
            wav_dir=wav_dir,
            model_revision=model_revisions[model_type],
            result_model_type=model_type,
        )
        results.append(model_result)
        print()

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
