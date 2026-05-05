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
from pathlib import Path
from typing import Optional

from model_inference import get_model_inference


def run_inference(wav_path: str, model_type: str = "podsai",
                  model_path: Optional[str] = None) -> dict:
    """
    Run inference on a wav file and return per-class probabilities.

    Args:
        wav_path: Path to the wav file.
        model_type: Type of model to use ('podsai', 'fastai', or 'orcahello').
        model_path: Path to the model directory or HuggingFace Hub model ID.
                    Required for podsai. Defaults to './model' for fastai,
                    'orcasound/orcahello-srkw-detector-v1' for orcahello,
                    and 'davethaler/whale-call-detector' for podsai.

    Returns:
        Dictionary with:
            - probabilities: dict mapping class label to probability (0.0-1.0).
              Each value is the mean local_confidence for windows that predicted
              that class and whose confidence exceeds the model's threshold.
            - global_prediction_label: predicted class label for the whole file
            - global_confidence: confidence score (0.0-1.0) for the global prediction
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
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "resident" if global_prediction else "other"
        global_confidence = resident_prob

    elif model_type == "podsai":
        if model_path is None:
            model_path = "davethaler/whale-call-detector"

        model = get_model_inference(model_type="podsai", model_path=model_path)

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time

        probabilities = result["per_class_probabilities"]
        global_prediction_label = result.get("global_prediction_label", "")
        global_confidence = float(result.get("global_confidence", 0.0))

    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Use 'podsai', 'fastai', or 'orcahello'."
        )

    return {
        "probabilities": probabilities,
        "global_prediction_label": global_prediction_label,
        "global_confidence": global_confidence,
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
    predict_time = results.get("predict_time", 0.0)

    print(f"Model type: {model_type}")
    print(f"Global prediction: {label} (confidence: {confidence:.4f})")
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
            "Defaults to orcasound/orcahello-srkw-detector-v1 for --model orcahello."
            "Defaults to davethaler/whale-call-detector for --model podsai."
        ),
    )

    args = parser.parse_args()
    wav_path = args.wav_file
    if not Path(wav_path).exists():
        print(f"Error: wav file not found: {wav_path}", file=sys.stderr)
        return 1

    try:
        results = run_inference(wav_path, model_type=args.model, model_path=args.model_path)
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
