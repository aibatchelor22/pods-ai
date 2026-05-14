#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
PODS-AI model inference wrapper for orca call detection.

This module provides a wrapper around Wav2Vec2 audio classification models
that implements the interface expected by PODS-AI's model_inference system.
"""

import torch
import librosa
import numpy as np
from collections import Counter
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from pathlib import Path
from datetime import datetime

# Import base class to establish inheritance
from model_inference import ModelInference

# Segment grouping size for scaling the positive calls threshold.
# For every SEGMENT_GROUP_SIZE segments, require at least 1 positive prediction.
SEGMENT_GROUP_SIZE = 10


class PodsAIInference(ModelInference):  # Inherit from ModelInference
    """
    Inference wrapper for PODS-AI audio classification models.

    This class implements the interface expected by model_inference.py,
    providing predictions in the format required by extract_training_samples.py.
    """

    def __init__(self, model_path: str, device: Optional[str] = None,
                 threshold: float = 0.5, min_num_positive_calls_threshold: int = 3,
                 model_revision: Optional[str] = None) -> None:
        """
        Initialize the inference model.

        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            threshold: Confidence threshold for positive predictions (default: 0.5)
            min_num_positive_calls_threshold: Minimum positive predictions for global positive classification.
                                             The effective threshold scales with audio length: it requires
                                             at least 1 positive per SEGMENT_GROUP_SIZE (10) segments, but
                                             is capped at min_num_positive_calls_threshold. Formula:
                                             min(ceil(segments/10), min_num_positive_calls_threshold).
                                             Default: use instance value (typically 3).
            model_revision: Git commit hash or tag to pin the HuggingFace Hub model revision.
                           Ignored when model_path is a local directory. (default: None)
        """
        super().__init__(model_path)  # Call parent constructor
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold

        # Auto-detect device. Default to GPU if available, otherwise CPU. Allow override via argument.
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        revision_info = f" @ {model_revision}" if model_revision else ""
        print(f"Loading PODS-AI model from {model_path}{revision_info}...")
        print(f"Using device: {self.device}")

        # Build kwargs for from_pretrained; only pass revision for Hub IDs (not local paths).
        pretrained_kwargs: dict = {}
        if model_revision and not Path(model_path).exists():
            pretrained_kwargs["revision"] = model_revision

        # Load feature extractor and model.
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_path, **pretrained_kwargs
            )
        except Exception as e:
            error_msg = f"Error loading feature extractor from {model_path}: {type(e).__name__}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path, **pretrained_kwargs
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {type(e).__name__}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

        # Get label mapping. This assumes the model was trained with a config that includes id2label and label2id.
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"Model loaded successfully. Label mapping: {self.label2id}")

        # Print model version/date information.
        self._print_model_metadata(model_path)

        # Validate that model has at least one negative/background class.
        # Accept either explicit "other" or treat specific classes as negative.
        negative_classes = {"other", "water", "vessel", "jingle", "human"}
        positive_classes = {"resident", "transient", "humpback"}

        found_negative = negative_classes & set(self.label2id.keys())
        found_positive = positive_classes & set(self.label2id.keys())

        if not found_negative:
            raise ValueError(
                f"Model must include at least one negative/background class (other, water, vessel, jingle, or human). "
                f"Found labels: {list(self.label2id.keys())}. "
                f"Please train the model with at least one negative class to distinguish from whale calls."
            )

        if not found_positive:
            print(
                f"Warning: Model has negative classes but no expected positive classes (resident, transient, humpback). "
                f"This may work but confidence scores may not be meaningful. "
                f"Found labels: {list(self.label2id.keys())}"
            )
        elif found_positive != positive_classes:
            missing = positive_classes - found_positive
            print(
                f"Warning: Model is missing some expected positive classes: {missing}. "
                f"Predictions will be computed from available classes: {found_positive}"
            )

        # Store which classes are considered negative (non-whale)
        self.negative_class_ids = {self.label2id[label] for label in found_negative}
        print(f"Treating classes as negative/background: {found_negative}")

    def _print_model_metadata(self, model_path: str) -> None:
        """
        Print model metadata including version, date, and architecture information.

        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
        """
        config = self.model.config
        metadata_found = False

        # Print model name/path from config.
        if hasattr(config, '_name_or_path') and config._name_or_path:
            print(f"Model name/path: {config._name_or_path}")
            metadata_found = True

        # Print architecture information.
        if hasattr(config, 'architectures') and config.architectures:
            print(f"Model architecture: {', '.join(config.architectures)}")
            metadata_found = True

        # Print HuggingFace Hub commit hash if available.
        if hasattr(config, '_commit_hash') and config._commit_hash:
            print(f"Model commit hash: {config._commit_hash}")
            metadata_found = True

        # Check for local model directory and extract file timestamps.
        model_path_obj = Path(model_path)
        if model_path_obj.exists() and model_path_obj.is_dir():
            # Check config.json modification time.
            config_json_path = model_path_obj / "config.json"
            if config_json_path.exists():
                mtime = datetime.fromtimestamp(config_json_path.stat().st_mtime)
                print(f"Model config last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                metadata_found = True

            # Check pytorch_model.bin modification time.
            model_bin_path = model_path_obj / "pytorch_model.bin"
            if model_bin_path.exists():
                mtime = datetime.fromtimestamp(model_bin_path.stat().st_mtime)
                print(f"Model weights last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                metadata_found = True

            # Check for training_args.bin with training metadata.
            training_args_path = model_path_obj / "training_args.bin"
            if training_args_path.exists():
                try:
                    training_args = torch.load(training_args_path, map_location='cpu', weights_only=False)
                    if hasattr(training_args, 'output_dir'):
                        print(f"Training output directory: {training_args.output_dir}")
                    if hasattr(training_args, 'num_train_epochs'):
                        print(f"Training epochs: {training_args.num_train_epochs}")
                    metadata_found = True
                except Exception as e:
                    # Silently skip if training_args.bin can't be loaded.
                    pass

        if not metadata_found:
            print("Model metadata: No version or date information available.")

    def predict(self, wav_path: str, segment_duration: int = 3, hop_duration: int = 2,
                threshold: Optional[float] = None, min_num_positive_calls_threshold: Optional[int] = None) -> dict[str, object]:
        """
        Run inference on a wav file using sliding window.

        Uses a sliding window approach with configurable segment and hop duration to match
        the orcahello LiveInferenceOrchestrator behavior. Unlike FastAIModel which uses a
        1-second hop and produces per-second outputs, this implementation uses a configurable
        hop_duration (default 2 seconds) for efficiency.

        The timestamp correction logic in extract_training_samples.py automatically adapts
        by inferring hop_duration = audio_duration / len(local_confidences), so the different
        output length is handled transparently.

        Args:
            wav_path: Path to wav file (typically 60 seconds long)
            segment_duration: Duration of each segment in seconds (default: 3)
            hop_duration: Hop size in seconds between segments (default: 2)
                         With hop_duration=2, a 60s audio produces ~29 confidence values.
                         With hop_duration=1, a 60s audio produces ~58 confidence values (matching FastAI).
            threshold: Confidence threshold for positive (non-other) predictions (default: use instance value)
            min_num_positive_calls_threshold: Minimum positive predictions for global positive classification.
                                             The effective threshold scales with audio length: it requires
                                             at least 1 positive per SEGMENT_GROUP_SIZE (10) segments, but
                                             is capped at min_num_positive_calls_threshold. Formula:
                                             min(ceil(segments/10), min_num_positive_calls_threshold).
                                             Default: use instance value (typically 3).

        Returns:
            Dictionary with keys:
                - local_predictions: List of class IDs for each hop_duration interval.
                                    Length = floor((audio_duration - segment_duration) / hop_duration) + 1
                - local_confidences: List of whale-call likelihood scores (0.0-1.0) at each interval.
                                    Computed as 1 - P(negative classes). Used by timestamp correction
                                    to identify the most likely position of whale calls.
                                    local_confidences[i] represents the score at time offset
                                    i * hop_duration seconds from the start.
                - global_prediction: Overall class ID for the entire audio
                - global_prediction_label: Human-readable label for the global prediction
                - global_confidence: Overall confidence score (0.0-1.0) for the global prediction
            Returns dict with empty lists and error values if audio loading fails.
        """
        # Use instance values if not overridden.
        threshold = threshold if threshold is not None else self.threshold
        min_num_positive_calls_threshold = (
            min_num_positive_calls_threshold
            if min_num_positive_calls_threshold is not None
            else self.min_num_positive_calls_threshold
        )

        # Get the primary negative class ID for error returns.
        # This is used when we cannot process the audio and need to return a default negative prediction.
        if "other" in self.label2id:
            primary_negative_id = self.label2id["other"]
        else:
            primary_negative_id = min(self.negative_class_ids)

        # Load audio. Resample to 16kHz and convert to mono. Handle exceptions gracefully.
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            error_msg = f"Error loading audio file {wav_path}: {type(e).__name__}: {e}"
            print(error_msg)
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": self.id2label[primary_negative_id],
                "global_confidence": 0.0,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Handle empty audio. This can happen if the file is corrupted or has no valid audio data.
        if len(audio) == 0:
            print(f"Warning: Audio file {wav_path} is empty")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": self.id2label[primary_negative_id],
                "global_confidence": 0.0,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Calculate segment and hop sizes in samples.
        segment_samples = segment_duration * sr
        hop_samples = hop_duration * sr

        # Calculate total audio duration and number of segments with sliding window.
        audio_duration = len(audio) / sr

        # Generate segment predictions using sliding window with hop_duration-second hop.
        # Each position represents a segment_duration-second window starting at position_index * hop_duration.
        # Store both class predictions and their probabilities.
        segment_class_ids: list[int] = []
        segment_probs: list[np.ndarray] = []

        # Calculate number of positions based on hop_duration.
        # num_positions = how many segment_duration windows fit with hop_duration spacing
        # Last position must start early enough that segment_duration window fits within audio.
        num_positions = int(np.floor((audio_duration - segment_duration) / hop_duration)) + 1

        # Handle very short audio (shorter than segment_duration).
        # In this case, we will just process one segment starting at 0 seconds.
        if num_positions < 1:
            num_positions = 1

        # Collect all segments first, then batch-process them in a single forward pass
        # for better performance (fewer model calls, better hardware utilization).
        segments = []
        for pos_idx in range(num_positions):
            start = pos_idx * hop_samples
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]

            # Pad if necessary (for short audio or last segment).
            if len(segment) < segment_samples:
                padding = segment_samples - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant')
            segments.append(segment)

        with torch.no_grad():
            # Extract features for all segments at once.
            inputs = self.feature_extractor(
                segments,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )

            # Move to device. This is important for performance, especially if using GPU.
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions for all segments in one forward pass.
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Get predicted class (argmax) and store probabilities for each segment.
            predicted_classes = torch.argmax(probs, dim=-1).tolist()
            segment_class_ids = [int(c) for c in predicted_classes]
            segment_probs = [probs[i].cpu().numpy() for i in range(len(segments))]

        # Guard against empty predictions list.
        # This can happen if the audio is too short or if there was an error during processing.
        if not segment_class_ids:
            print(f"Warning: No segments processed for {wav_path}")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": self.id2label[primary_negative_id],
                "global_confidence": 0.0,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Apply rolling average to smooth predictions (matching FastAI behavior).
        # For multi-class, we average the probability distributions.
        n = len(segment_probs)
        smoothed_probs: list[np.ndarray] = []

        for i in range(n):
            if i == 0:
                # First position: use first segment probabilities directly.
                smoothed_probs.append(segment_probs[0])
            elif i == n - 1:
                # Last position: use last segment probabilities directly.
                smoothed_probs.append(segment_probs[-1])
            else:
                # Middle positions: average previous and current segment.
                avg_probs = (segment_probs[i - 1] + segment_probs[i]) / 2.0
                smoothed_probs.append(avg_probs)

        # Get local predictions (class IDs) and call-likelihood confidences.
        # For timestamp correction, we need the likelihood of a whale call being present,
        # not the confidence in the predicted class (which could be a background class).
        local_predictions: list[int] = []
        local_confidences: list[float] = []

        for probs in smoothed_probs:
            predicted_class = int(np.argmax(probs))

            # Compute call-likelihood as 1 - P(negative classes).
            # This represents "how likely is there a whale call" regardless of which class wins.
            negative_prob = sum(probs[class_id] for class_id in self.negative_class_ids)
            call_likelihood = 1.0 - negative_prob

            local_predictions.append(predicted_class)
            local_confidences.append(float(call_likelihood))

        # Determine global prediction based on voting among high-confidence predictions.
        # For positive (whale) classes, we require multiple high-confidence predictions.
        # For negative (background) classes, we use the most common prediction.

        # Filter for positive (whale) predictions with confidence above threshold.
        positive_predictions = [
            (class_id, conf)
            for class_id, conf in zip(local_predictions, local_confidences)
            if class_id not in self.negative_class_ids and conf >= threshold
        ]

        # Scale the positive calls threshold based on the number of segments.
        # For every SEGMENT_GROUP_SIZE segments, require at least 1 positive prediction.
        # Cap at min_num_positive_calls_threshold to avoid requiring too many for very long clips.
        total_segments = len(local_predictions)
        scaled_threshold = max(1, (total_segments + SEGMENT_GROUP_SIZE - 1) // SEGMENT_GROUP_SIZE)
        effective_threshold = min(scaled_threshold, min_num_positive_calls_threshold)

        # If we have enough positive predictions, use majority vote among them.
        if len(positive_predictions) >= effective_threshold:
            # Count votes for each positive class.
            class_votes: dict[int, list[float]] = {}
            for class_id, conf in positive_predictions:
                if class_id not in class_votes:
                    class_votes[class_id] = []
                class_votes[class_id].append(conf)

            # Winner is the class with most votes (ties broken by average confidence).
            global_prediction_id = max(
                class_votes.keys(),
                key=lambda cid: (len(class_votes[cid]), np.mean(class_votes[cid]))
            )
            global_confidence = float(np.mean(class_votes[global_prediction_id]))
        else:
            # Not enough whale predictions - determine which background class is most likely.
            # Filter to only background/negative classes to ensure whale classes can't bypass
            # the effective_threshold requirement.
            from collections import Counter
            background_predictions = [c for c in local_predictions if c in self.negative_class_ids]

            if background_predictions:
                # Get the most common background class.
                class_counts = Counter(background_predictions)
                global_prediction_id = class_counts.most_common(1)[0][0]

                # Compute confidence as the mean probability of the predicted class across all segments.
                global_confidence = float(np.mean([
                    probs[global_prediction_id] for probs in smoothed_probs
                ]))
            else:
                # No background predictions found (all predictions were positive but below threshold).
                # Fall back to a safe background default.
                if "other" in self.label2id:
                    global_prediction_id = self.label2id["other"]
                else:
                    # Use the first negative class (water is typically class 0).
                    global_prediction_id = min(self.negative_class_ids)
                global_confidence = 0.0

        # Convert global prediction ID to label name.
        global_prediction_label = self.id2label[global_prediction_id]

        # Calculate per-class probabilities for display purposes.
        # These represent the mean probability for each class across all windows.
        per_class_probabilities = {}
        for class_id, label in self.id2label.items():
            class_probs = [float(probs[class_id]) for probs in smoothed_probs]
            per_class_probabilities[label] = float(np.mean(class_probs))

        return {
            "local_predictions": local_predictions,
            "local_confidences": local_confidences,
            "local_probs": smoothed_probs,
            "global_prediction": global_prediction_id,
            "global_prediction_label": global_prediction_label,
            "global_confidence": global_confidence,
            "per_class_probabilities": per_class_probabilities,
            "hop_duration": float(hop_duration),
            "segment_duration": float(segment_duration),
        }


def get_podsai_inference(model_path: str, **kwargs) -> PodsAIInference:
    """
    Factory function to create PODS-AI inference instance.

    Args:
        model_path: Path to model directory or HuggingFace Hub model ID
        **kwargs: Additional arguments passed to PodsAIInference, e.g.
                  model_revision (str): pinned git commit hash for Hub models.

    Returns:
        PodsAIInference instance
    """
    return PodsAIInference(model_path, **kwargs)
