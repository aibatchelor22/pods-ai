#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Live 60-second inference adapter for the multispecies-cetacean V2 model.

Implements the ModelInference contract used by LiveInferenceOrchestrator.py.
It evaluates overlapping three-second windows with the custom trigger, source,
and ecotype heads, then emits the existing live labels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt
from torch import nn
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

from model_inference import ModelInference

SAMPLE_RATE = 16_000
TRIGGER_LABELS = {"not_whale": 0, "known_whale": 1}
SOURCE_LABELS = {"Abiotic": 0, "KW": 1, "HW": 2, "UndBio": 3}
ECOTYPE_LABELS = {"NRKW": 0, "SRKW": 1, "OKW": 2, "SAR": 3, "TKW": 4}
OUTPUT_LABELS = ("other/background", "humpback", "resident", "transient")


@dataclass(frozen=True)
class AggregationConfig:
    """Tuned 60-second operating point; all fields are config-map overridable."""

    trigger_threshold: float = 0.90
    trigger_combination: str = "product"
    ecotype_mode: str = "srkw_tkw_conditional"
    kw_source_threshold: float = 0.50
    hw_source_threshold: float = 0.25
    resident_ecotype_threshold: float = 0.70
    transient_ecotype_threshold: float = 0.50
    humpback_threshold: float = 0.50
    resident_threshold: float = 0.05
    transient_threshold: float = 0.50
    top_k: int = 3
    humpback_min_windows: int = 1
    resident_min_windows: int = 2
    transient_min_windows: int = 2
    smoothing: bool = False

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "AggregationConfig":
        source = values or {}
        result = cls(
            **{
                item.name: source[item.name]
                for item in fields(cls)
                if item.name in source
            }
        )
        result.validate()
        return result

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("threshold") and not 0 <= float(value) <= 1:
                raise ValueError(f"{item.name} must be between zero and one")
        if self.trigger_combination not in {"gate", "product"}:
            raise ValueError("trigger_combination must be gate or product")
        if self.ecotype_mode not in {"srkw_tkw_conditional", "raw"}:
            raise ValueError("ecotype_mode must be srkw_tkw_conditional or raw")
        if min(
            self.top_k,
            self.humpback_min_windows,
            self.resident_min_windows,
            self.transient_min_windows,
        ) < 1:
            raise ValueError("top_k and minimum-window settings must be positive")


class MultispeciesCetaceanModel(nn.Module):
    """AST backbone with known-whale, source, and killer-whale ecotype heads."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.config = base_model.config
        self.ast = base_model.audio_spectrogram_transformer
        hidden_size = int(self.config.hidden_size)
        self.trigger_classifier = nn.Linear(hidden_size, len(TRIGGER_LABELS))
        self.source_classifier = nn.Linear(hidden_size, len(SOURCE_LABELS))
        self.ecotype_classifier = nn.Linear(hidden_size, len(ECOTYPE_LABELS))

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        sequence = self.ast(input_values=input_values, return_dict=True).last_hidden_state
        pooled = (
            (sequence[:, 0] + sequence[:, 1]) / 2
            if sequence.shape[1] >= 2
            else sequence[:, 0]
        )
        return (
            self.trigger_classifier(pooled),
            self.source_classifier(pooled),
            self.ecotype_classifier(pooled),
        )


LEGACY_AST_KEY_REPLACEMENTS = (
    ("ast.encoder.layer.", "ast.encoder.layers."),
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
    (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
)


def _read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Any = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older PyTorch
            state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def _remap_legacy_keys(
    state: Mapping[str, torch.Tensor], expected: set[str]
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for original, value in state.items():
        output = original
        if output not in expected:
            candidate = output
            for old, new in LEGACY_AST_KEY_REPLACEMENTS:
                candidate = candidate.replace(old, new)
            output = next(
                (
                    possible
                    for possible in (
                        candidate,
                        candidate.replace("ast.encoder.layers.", "ast.layers."),
                    )
                    if possible in expected
                ),
                original,
            )
        if output in result:
            raise ValueError(f"Checkpoint key collision after remap: {output}")
        result[output] = value
    return result


def _materialize_model(model_path: str, revision: Optional[str]) -> Path:
    local = Path(model_path)
    if local.is_dir():
        return local.resolve()
    return Path(snapshot_download(repo_id=model_path, revision=revision)).resolve()


def _load_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "multispecies_cetacean_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required; this is not a V2 multispecies checkpoint"
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "trigger_labels": TRIGGER_LABELS,
        "source_labels": SOURCE_LABELS,
        "ecotype_labels": ECOTYPE_LABELS,
    }
    for name, labels in expected.items():
        if metadata.get(name) != labels:
            raise ValueError(
                f"Incompatible {name}: {metadata.get(name)!r}; expected {labels!r}"
            )
    return metadata


def _load_checkpoint(
    directory: Path,
) -> tuple[MultispeciesCetaceanModel, Any, dict[str, Any]]:
    metadata = _load_metadata(directory)
    try:
        architecture = AutoConfig.from_pretrained(directory, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "The deployment checkpoint is not standalone. config.json must contain "
            "a valid AST model_type and feature-extractor files must be present."
        ) from exc
    model = MultispeciesCetaceanModel(
        AutoModelForAudioClassification.from_config(architecture)
    )
    weights = next(
        (
            directory / name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (directory / name).is_file()
        ),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"No supported model weights in {directory}")
    state = _remap_legacy_keys(_read_state_dict(weights), set(model.state_dict()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Refusing partially loaded deployment weights; "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    extractor = AutoFeatureExtractor.from_pretrained(directory, local_files_only=True)
    return model, extractor, metadata


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    result = values.copy()
    result[1:-1] = (values[:-2] + values[1:-1]) / 2
    return result


def score_windows(
    probabilities: Mapping[str, np.ndarray], config: AggregationConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Convert three-head probabilities into live class scores and gates."""
    trigger = probabilities["trigger"][:, TRIGGER_LABELS["known_whale"]]
    kw = probabilities["source"][:, SOURCE_LABELS["KW"]]
    hw = probabilities["source"][:, SOURCE_LABELS["HW"]]
    srkw = probabilities["ecotype"][:, ECOTYPE_LABELS["SRKW"]]
    tkw = probabilities["ecotype"][:, ECOTYPE_LABELS["TKW"]]
    if config.ecotype_mode == "srkw_tkw_conditional":
        denominator = srkw + tkw + 1e-12
        srkw_component, tkw_component = srkw / denominator, tkw / denominator
    else:
        srkw_component, tkw_component = srkw, tkw
    multiplier = trigger if config.trigger_combination == "product" else np.ones_like(trigger)
    scores = {
        "humpback": multiplier * hw,
        "resident": multiplier * kw * srkw_component,
        "transient": multiplier * kw * tkw_component,
    }
    gates = {
        "humpback": (trigger >= config.trigger_threshold)
        & (hw >= config.hw_source_threshold),
        "resident": (trigger >= config.trigger_threshold)
        & (kw >= config.kw_source_threshold)
        & (srkw_component >= config.resident_ecotype_threshold),
        "transient": (trigger >= config.trigger_threshold)
        & (kw >= config.kw_source_threshold)
        & (tkw_component >= config.transient_ecotype_threshold),
    }
    if config.smoothing:
        scores = {label: _smooth(values) for label, values in scores.items()}
    return scores, gates


def aggregate_probabilities(
    probabilities: Mapping[str, np.ndarray], config: AggregationConfig
) -> dict[str, Any]:
    """Apply threshold-first local decisions and top-k clip aggregation."""
    scores, gates = score_windows(probabilities, config)
    thresholds = {
        "humpback": config.humpback_threshold,
        "resident": config.resident_threshold,
        "transient": config.transient_threshold,
    }
    minimums = {
        "humpback": config.humpback_min_windows,
        "resident": config.resident_min_windows,
        "transient": config.transient_min_windows,
    }
    local_labels: list[str] = []
    local_confidences: list[float] = []
    for index in range(len(probabilities["trigger"])):
        surviving = {
            label: float(scores[label][index])
            for label in thresholds
            if bool(gates[label][index])
            and float(scores[label][index]) >= thresholds[label]
        }
        if surviving:
            label, confidence = max(surviving.items(), key=lambda item: item[1])
        else:
            label, confidence = "other/background", 0.0
        local_labels.append(label)
        local_confidences.append(confidence)

    candidates: dict[str, tuple[float, int]] = {}
    class_aggregates: dict[str, float] = {}
    for label, threshold in thresholds.items():
        indices = np.flatnonzero(gates[label] & (scores[label] >= threshold))
        if len(indices) < minimums[label]:
            class_aggregates[label] = 0.0
            continue
        selected = indices[np.argsort(scores[label][indices])[::-1]][: config.top_k]
        aggregate = float(np.mean(scores[label][selected]))
        class_aggregates[label] = aggregate
        candidates[label] = (aggregate, len(indices))
    if candidates:
        global_label = max(candidates, key=lambda label: candidates[label])
        global_confidence = candidates[global_label][0]
    else:
        global_label, global_confidence = "other/background", 0.0
    return {
        "local_labels": local_labels,
        "local_confidences": local_confidences,
        "global_label": global_label,
        "global_confidence": global_confidence,
        "class_aggregates": class_aggregates,
    }


class MultispeciesCetaceanInference(ModelInference):
    """Production predictor for a complete, standalone V2 checkpoint."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        model_revision: Optional[str] = None,
        inference_batch_size: int = 8,
        aggregation_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_batch_size = int(inference_batch_size)
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        self.aggregation = AggregationConfig.from_mapping(aggregation_config)
        self.model_directory = _materialize_model(model_path, model_revision)
        self.model, self.feature_extractor, self.metadata = _load_checkpoint(
            self.model_directory
        )
        self.model.to(self.device).eval()
        self.id2label = dict(enumerate(OUTPUT_LABELS))
        self.label2id = {label: index for index, label in self.id2label.items()}
        preprocessing = self.metadata.get("preprocessing", {})
        self.mean_subtract = bool(preprocessing.get("mean_subtract", False))
        self.high_pass_filter = bool(preprocessing.get("high_pass_filter", False))
        self.high_pass_cutoff_hz = float(preprocessing.get("high_pass_cutoff_hz", 50))
        self.high_pass_order = int(preprocessing.get("high_pass_order", 4))

    @staticmethod
    def _read_audio(path: str) -> np.ndarray:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = np.asarray(audio[:, 0], dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
            audio = resample_poly(
                audio, SAMPLE_RATE // divisor, int(sample_rate) // divisor
            ).astype(np.float32)
        return audio

    def _windows(
        self, audio: np.ndarray, segment_duration: float, hop_duration: float
    ) -> list[np.ndarray]:
        segment_samples = int(round(segment_duration * SAMPLE_RATE))
        hop_samples = int(round(hop_duration * SAMPLE_RATE))
        if min(segment_samples, hop_samples) < 1:
            raise ValueError("segment_duration and hop_duration must be positive")
        positions = max(1, 1 + (len(audio) - segment_samples) // hop_samples)
        high_pass_sos = (
            butter(
                self.high_pass_order,
                self.high_pass_cutoff_hz,
                btype="highpass",
                fs=SAMPLE_RATE,
                output="sos",
            )
            if self.high_pass_filter
            else None
        )
        windows: list[np.ndarray] = []
        for index in range(positions):
            start = index * hop_samples
            window = audio[start : start + segment_samples]
            if len(window) < segment_samples:
                window = np.pad(window, (0, segment_samples - len(window)))
            window = np.asarray(window, dtype=np.float32)
            if self.mean_subtract:
                window = window - float(window.mean())
            if high_pass_sos is not None:
                try:
                    window = sosfiltfilt(high_pass_sos, window).astype(np.float32)
                except ValueError:
                    window = sosfilt(high_pass_sos, window).astype(np.float32)
            windows.append(window)
        return windows

    def predict(
        self,
        wav_file_path: str,
        segment_duration: int = 3,
        hop_duration: int = 2,
    ) -> dict[str, Any]:
        windows = self._windows(
            self._read_audio(wav_file_path),
            float(segment_duration),
            float(hop_duration),
        )
        parts: dict[str, list[np.ndarray]] = {
            "trigger": [],
            "source": [],
            "ecotype": [],
        }
        with torch.inference_mode():
            for start in range(0, len(windows), self.inference_batch_size):
                features = self.feature_extractor(
                    windows[start : start + self.inference_batch_size],
                    sampling_rate=SAMPLE_RATE,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(input_values=features["input_values"].to(self.device))
                for name, value in zip(parts, logits):
                    parts[name].append(torch.softmax(value, dim=-1).cpu().numpy())
        probabilities = {
            name: np.concatenate(values) for name, values in parts.items()
        }
        aggregated = aggregate_probabilities(probabilities, self.aggregation)
        global_label = aggregated["global_label"]
        return {
            "local_predictions": aggregated["local_labels"],
            "local_confidences": aggregated["local_confidences"],
            "global_prediction": self.label2id[global_label],
            "global_prediction_label": global_label,
            "global_confidence": aggregated["global_confidence"],
            "per_class_probabilities": aggregated["class_aggregates"],
            "hop_duration": float(hop_duration),
            "segment_duration": float(segment_duration),
        }


def get_multispecies_cetacean_inference(
    model_path: str, **kwargs: Any
) -> MultispeciesCetaceanInference:
    return MultispeciesCetaceanInference(model_path, **kwargs)
