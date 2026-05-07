#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
OrcaHello SRKW Detector inference wrapper.

This module provides an inference wrapper for the OrcaHello SRKW detector model
(orcasound/orcahello-srkw-detector-v1) hosted on HuggingFace Hub.

The model implementation is imported from the orcasound/orcahello submodule at
external/orcahello/InferenceSystem/src/model.  Initialize the submodule before
use with:

    git submodule update --init external/orcahello

Usage:
    from orcahello_inference import OrcaHelloSRKWInference

    model = OrcaHelloSRKWInference("orcasound/orcahello-srkw-detector-v1")
    result = model.predict("sample.wav")
    print(result["global_prediction_label"])  # "whale" or "other"
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from model_inference import ModelInference


# Path to the orcahello submodule's model package.
_ORCAHELLO_SRC = (
    Path(__file__).parent.parent / "external" / "orcahello" / "InferenceSystem" / "src"
)

# Cached tuple of (OrcaHelloSRKWDetectorV1, DetectorInferenceConfig) from submodule.
_ORCAHELLO_IMPORTS: Optional[Tuple[Type, Type]] = None


def _get_orcahello_classes() -> Tuple[Type, Type]:
    """Lazily import OrcaHelloSRKWDetectorV1 and DetectorInferenceConfig from the submodule.

    Adds the submodule's src directory to sys.path on first call.

    Returns:
        Tuple of (OrcaHelloSRKWDetectorV1, DetectorInferenceConfig).

    Raises:
        ImportError: If the orcahello submodule is not initialized or dependencies
                     are missing.
    """
    global _ORCAHELLO_IMPORTS
    if _ORCAHELLO_IMPORTS is not None:
        return _ORCAHELLO_IMPORTS

    if not _ORCAHELLO_SRC.exists():
        raise ImportError(
            f"orcahello submodule not found at {_ORCAHELLO_SRC}. "
            "Run: git submodule update --init external/orcahello"
        )

    src_str = str(_ORCAHELLO_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    try:
        from model import OrcaHelloSRKWDetectorV1, DetectorInferenceConfig  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"Failed to import from orcahello submodule at {_ORCAHELLO_SRC}. "
            f"Ensure the submodule is initialized and all dependencies are installed "
            f"(scipy, torchaudio, pydub, librosa, soundfile). "
            f"Original error: {e}"
        ) from e

    _ORCAHELLO_IMPORTS = (OrcaHelloSRKWDetectorV1, DetectorInferenceConfig)
    return _ORCAHELLO_IMPORTS


class OrcaHelloSRKWInference(ModelInference):
    """Inference wrapper for the OrcaHello SRKW detector model.

    Loads OrcaHelloSRKWDetectorV1 from HuggingFace Hub (default:
    orcasound/orcahello-srkw-detector-v1) via the orcahello submodule.

    The model is a binary classifier (0 = no SRKW, 1 = SRKW detected).
    Unlike the PodsAIInference wrapper which uses Wav2Vec2, this model
    uses a ResNet50 backbone with mel spectrograms.
    """

    DEFAULT_MODEL_PATH = "orcasound/orcahello-srkw-detector-v1"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize the OrcaHello SRKW inference wrapper.

        Args:
            model_path: HuggingFace Hub model ID or local path to model directory.
                        Defaults to "orcasound/orcahello-srkw-detector-v1".
            config: Optional configuration overrides as a nested dict.
                    If None, uses the default DetectorInferenceConfig values.

        Raises:
            ImportError: If the orcahello submodule is not initialized.
            RuntimeError: If the model cannot be loaded from model_path.
        """
        super().__init__(model_path=model_path)

        OrcaHelloSRKWDetectorV1, DetectorInferenceConfig = _get_orcahello_classes()
        self.inference_config = DetectorInferenceConfig.from_dict(config or {})

        print(f"Loading OrcaHello SRKW detector from {model_path}...")
        try:
            self.model = OrcaHelloSRKWDetectorV1.from_pretrained(
                model_path,
                config=self.inference_config.as_dict(),
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Error loading OrcaHello model from {model_path}: {type(e).__name__}: {e}"
            ) from e

        print(
            f"OrcaHello model loaded. Device: {self.model._device}, "
            f"Dtype: {self.model._dtype}"
        )

    def predict(self, wav_file_path: str) -> Dict:
        """Run inference on a wav file.

        Uses the OrcaHello audio preprocessing pipeline (mel spectrograms via
        torchaudio, no fastai_audio dependency) and the ResNet50-based detector.

        Args:
            wav_file_path: Path to the wav file (typically 60 seconds long).

        Returns:
            Dictionary with keys:
                - local_predictions: List of binary predictions (0/1) per segment.
                - local_confidences: List of confidence scores (0.0-1.0) per segment.
                - global_prediction: Overall binary prediction (0 or 1).
                - global_prediction_label: "whale" if global_prediction==1 else "other".
                - global_confidence: Aggregated confidence score (0.0-1.0).
                - hop_duration: Hop duration in seconds between segments.
                - segment_duration: Duration of each segment in seconds.
            Returns dict with empty lists and zero scores on error.
        """
        try:
            result = self.model.detect_srkw_from_file(
                wav_file_path, self.inference_config
            )
        except Exception as e:
            print(f"Error running inference on {wav_file_path}: {type(e).__name__}: {e}")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": 0,
                "global_prediction_label": "other",
                "global_confidence": 0.0,
                "hop_duration": float(self.inference_config.inference.window_hop_s),
                "segment_duration": float(self.inference_config.inference.window_s),
            }

        return {
            "local_predictions": result.local_predictions,
            "local_confidences": result.local_confidences,
            "global_prediction": result.global_prediction,
            "global_prediction_label": "whale" if result.global_prediction == 1 else "other",
            "global_confidence": result.global_confidence,
            "hop_duration": float(self.inference_config.inference.window_hop_s),
            "segment_duration": float(self.inference_config.inference.window_s),
        }


def get_orcahello_srkw_inference(
    model_path: str = OrcaHelloSRKWInference.DEFAULT_MODEL_PATH,
    **kwargs,
) -> OrcaHelloSRKWInference:
    """Factory function to create an OrcaHelloSRKWInference instance.

    Args:
        model_path: HuggingFace Hub model ID or local path.
        **kwargs: Additional arguments passed to OrcaHelloSRKWInference.

    Returns:
        OrcaHelloSRKWInference instance.
    """
    return OrcaHelloSRKWInference(model_path, **kwargs)
