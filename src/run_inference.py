#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Run inference on a wav file and output per-class probabilities.

Usage:
    python run_inference.py sample.wav
    python run_inference.py sample.wav --model podsai --model-path /path/to/podsai-model
    python run_inference.py sample.wav --model fastai --model-path ../model
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from model_inference import get_model_inference

PODSAI_MODEL_ID = "davethaler/whale-call-detector"
# renovate: datasource=git-refs depName=https://huggingface.co/davethaler/whale-call-detector versioning=git.
PODSAI_MODEL_REVISION = "cef82c6e9ee661646ea0c583aeb68f4f7ec6d9d8"
PROPOSED_DESCRIPTION_EXTRA_CLASSES = {"vessel", "human", "jingle"}


def _build_proposed_description(
    global_prediction_label: str,
    local_prediction_labels: list[str],
) -> str:
    """Build a proposed description string from global and local predictions.

    Args:
        global_prediction_label: Predicted class label for the whole file.
        local_prediction_labels: Segment-level predicted class labels.

    Returns:
        Proposed description text beginning with "AI:" and optionally appending
        a dominant non-whale context class from {"vessel", "human", "jingle"}.
    """
    proposed_description = f"AI: {global_prediction_label}"
    if not local_prediction_labels:
        return proposed_description

    most_common = Counter(local_prediction_labels).most_common(1)
    if not most_common:
        return proposed_description
    most_common_label, _ = most_common[0]
    if (
        most_common_label in PROPOSED_DESCRIPTION_EXTRA_CLASSES
        and most_common_label != global_prediction_label
    ):
        proposed_description = f"{proposed_description} and {most_common_label}"
    return proposed_description


def run_inference(wav_path: str, model_type: str = "podsai",
                  model_path: Optional[str] = None,
                  model_revision: Optional[str] = None) -> dict:
    """
    Run inference on a wav file and return per-class probabilities.

    Args:
        wav_path: Path to the wav file.
        model_type: Type of model to use ('podsai', 'fastai', or 'orcahello').
        model_path: Path to the model directory or HuggingFace Hub model ID.
                    Required for podsai. Defaults to './model' for fastai,
                    'orcasound/orcahello-srkw-detector-v1' for orcahello,
                    and PODSAI_MODEL_ID for podsai.
        model_revision: Git commit hash to pin the HuggingFace Hub model revision.
                        Only used when model_path is a Hub model ID (not a local path).
                        Defaults to PODSAI_MODEL_REVISION when model_path is the default
                        PODS-AI Hub model.

    Returns:
        Dictionary with:
            - probabilities: dict mapping class label to probability (0.0-1.0).
              Each value is the mean local_confidence for windows that predicted
              that class and whose confidence exceeds the model's threshold.
            - global_prediction_label: predicted class label for the whole file
            - global_confidence: confidence score (0.0-1.0) for the global prediction
            - proposed_description: text description suitable for manual sample notes
            - predict_time: time in seconds spent in the model's predict() method
    """

    if model_type == "fastai":
        if model_path is None:
            model_path = "./model"
        model = get_model_inference(model_type="fastai", model_path=model_path)

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time
        # local_confidences that exceed the threshold (resident windows).
        resident_prob = float(result.get("global_confidence", 0.0))
        other_prob = round(1.0 - resident_prob, 4)

        probabilities: dict[str, float] = {
            "other": other_prob,
            "resident": round(resident_prob, 4),
        }
        local_prediction_labels = [
            "resident" if int(local_prediction) == 1 else "other"
            for local_prediction in result.get("local_predictions", [])
        ]
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "resident" if global_prediction else "other"
        global_confidence = resident_prob

    elif model_type == "orcahello":
        if model_path is None:
            model_path = "orcasound/orcahello-srkw-detector-v1"
        model = get_model_inference(model_type="orcahello", model_path=model_path)

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time

        # The OrcaHello SRKW detector is a binary classifier (other vs resident).
        resident_prob = float(result.get("global_confidence", 0.0))
        other_prob = round(1.0 - resident_prob, 4)

        probabilities = {
            "other": other_prob,
            "resident": round(resident_prob, 4),
        }
        local_prediction_labels = [
            "resident" if int(local_prediction) == 1 else "other"
            for local_prediction in result.get("local_predictions", [])
        ]
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "resident" if global_prediction else "other"
        global_confidence = resident_prob

    elif model_type == "podsai":
        if model_path is None:
            model_path = PODSAI_MODEL_ID
            if model_revision is None:
                model_revision = PODSAI_MODEL_REVISION

        model = get_model_inference(model_type="podsai", model_path=model_path,
                                    model_revision=model_revision)

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time

        probabilities = result["per_class_probabilities"]
        local_prediction_labels = []
        for local_prediction in result.get("local_predictions", []):
            if isinstance(local_prediction, str):
                local_prediction_labels.append(local_prediction)
            else:
                label = getattr(model, "id2label", {}).get(local_prediction)
                if isinstance(label, str):
                    local_prediction_labels.append(label)
        global_prediction_label = result.get("global_prediction_label", "")
        global_confidence = float(result.get("global_confidence", 0.0))

    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Use 'podsai', 'fastai', or 'orcahello'."
        )

    proposed_description = _build_proposed_description(global_prediction_label, local_prediction_labels)

    return {
        "probabilities": probabilities,
        "global_prediction_label": global_prediction_label,
        "global_confidence": global_confidence,
        "proposed_description": proposed_description,
        "predict_time": predict_time,
    }


def print_results(results: dict, model_type: str) -> None:
    """Print inference results to stdout.

    Args:
        results: Dictionary returned by run_inference().
        model_type: Model type string, printed for context.
    """
    probabilities = results["probabilities"]
    label = results["global_prediction_label"]
    confidence = results["global_confidence"]
    proposed_description = results["proposed_description"]
    predict_time = results.get("predict_time", 0.0)

    print(f"Model type: {model_type}")
    print(f"Global prediction: {label} (confidence: {confidence:.4f})")
    print(f"Proposed description: {proposed_description}")
    print(f"Prediction time: {predict_time:.2f}s")
    print()
    print("Per-class probabilities:")
    for class_name, prob in sorted(probabilities.items()):
        print(f"  {class_name}: {prob:.4f}")


def main() -> int:
    """Entry point for the run_inference CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Run model inference on a wav file and output per-class probabilities."
    )
    parser.add_argument(
        "wav_file",
        help="Path to the wav file to score.",
    )
    parser.add_argument(
        "--model",
        default="podsai",
        help=(
            "Model type to use (default: podsai). "
            "podsai: 7-class model (water, resident, transient, humpback, vessel, jingle, human). "
            "fastai: 2-class model (other, resident). "
            "orcahello: 2-class SRKW detector (other, resident) using the OrcaHello ResNet50 model."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to model directory or HuggingFace Hub model ID. "
            "Required for --model podsai. "
            "Defaults to ./model for --model fastai. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 for --model orcahello. "
            f"Defaults to {PODSAI_MODEL_ID!r} for --model podsai."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Git commit hash to pin the HuggingFace Hub model revision. "
            "Only used when --model-path is a Hub model ID. "
            f"Defaults to the pinned revision ({PODSAI_MODEL_REVISION}) when using "
            f"the default PODS-AI model ({PODSAI_MODEL_ID!r})."
        ),
    )

    args = parser.parse_args()
    wav_path = args.wav_file
    if not Path(wav_path).exists():
        print(f"Error: wav file not found: {wav_path}", file=sys.stderr)
        return 1

    try:
        results = run_inference(wav_path, model_type=args.model, model_path=args.model_path,
                                model_revision=args.model_revision)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Inference failed: {e}", file=sys.stderr)
        return 1

    print_results(results, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
