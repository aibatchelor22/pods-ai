#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Experimental full-spectrogram inference for multispecies-cetacean V2.

This adapter follows the optimized PODS-AI AST path: deterministic waveform
preprocessing is applied once to the 60-second waveform, one Kaldi fbank is
computed, and the overlapping three-second feature windows are sliced from it.
Compact mode also removes unnecessary AST time padding. By default it crops
the positional grid to the positions occupied by real three-second audio;
interpolation remains available as an experiment. Compact inference is not
bit-identical to the padded model and must pass evaluation before deployment.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt

from multispecies_cetacean_inference import (
    ECOTYPE_LABELS,
    OUTPUT_LABELS,
    SOURCE_LABELS,
    TRIGGER_LABELS,
    SAMPLE_RATE,
    MultispeciesCetaceanInference,
    aggregate_probabilities,
)

NUM_SPECIAL_TOKENS = 2
DEFAULT_AST_PATCH_SIZE = 16
DEFAULT_AST_FREQUENCY_STRIDE = 10
DEFAULT_AST_TIME_STRIDE = 10


class FullSpectrogramMultispeciesCetaceanInference(
    MultispeciesCetaceanInference
):
    """V2 predictor using one fbank calculation per source recording."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        model_revision: Optional[str] = None,
        inference_batch_size: int = 8,
        aggregation_config: Optional[Mapping[str, Any]] = None,
        compact_ast_frames: bool = True,
        compact_position_embedding_mode: str = "crop",
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            model_revision=model_revision,
            inference_batch_size=inference_batch_size,
            aggregation_config=aggregation_config,
        )
        self.compact_ast_frames = bool(compact_ast_frames)
        if compact_position_embedding_mode not in {"crop", "interpolate"}:
            raise ValueError(
                "compact_position_embedding_mode must be 'crop' or 'interpolate'"
            )
        self.compact_position_embedding_mode = compact_position_embedding_mode
        self._position_embedding_cache: dict[
            tuple[str, int, int], torch.Tensor
        ] = {}
        self._segment_frame_cache: dict[tuple[int, int], int] = {}
        self._original_position_embeddings = (
            self.model.ast.embeddings.position_embeddings.detach().clone()
        )

    def _ensure_ast_position_embeddings(
        self, target_frames: int, num_mel_bins: int
    ) -> None:
        embeddings = self.model.ast.embeddings
        current = embeddings.position_embeddings
        config = self.model.config
        patch_size = int(
            getattr(config, "patch_size", DEFAULT_AST_PATCH_SIZE)
        )
        frequency_stride = int(
            getattr(config, "frequency_stride", DEFAULT_AST_FREQUENCY_STRIDE)
        )
        time_stride = int(
            getattr(config, "time_stride", DEFAULT_AST_TIME_STRIDE)
        )
        configured_frames = int(
            getattr(config, "max_length", self.feature_extractor.max_length)
        )
        configured_mels = int(
            getattr(config, "num_mel_bins", num_mel_bins)
        )
        target_frequency = (
            (num_mel_bins - patch_size) // frequency_stride + 1
        )
        target_time = (target_frames - patch_size) // time_stride + 1
        source_frequency = (
            (configured_mels - patch_size) // frequency_stride + 1
        )
        source_time = (
            (configured_frames - patch_size) // time_stride + 1
        )
        if min(target_frequency, target_time, source_frequency, source_time) < 1:
            raise ValueError("Invalid AST patch geometry for compact inference")

        target_key = (
            self.compact_position_embedding_mode,
            target_frequency,
            target_time,
        )
        target_tokens = target_frequency * target_time + NUM_SPECIAL_TOKENS
        if current.shape[1] == target_tokens:
            return
        cached = self._position_embedding_cache.get(target_key)
        if cached is not None:
            embeddings.position_embeddings = torch.nn.Parameter(
                cached, requires_grad=False
            )
            return

        source = self._original_position_embeddings
        expected_source_tokens = (
            source_frequency * source_time + NUM_SPECIAL_TOKENS
        )
        if source.shape[1] != expected_source_tokens:
            raise RuntimeError(
                "Cannot infer the checkpoint AST positional grid: "
                f"weights have {source.shape[1]} tokens, expected "
                f"{expected_source_tokens}"
            )
        special = source[:, :NUM_SPECIAL_TOKENS, :]
        hidden_size = source.shape[-1]
        patch = source[:, NUM_SPECIAL_TOKENS:, :].reshape(
            1, source_frequency, source_time, hidden_size
        )
        if self.compact_position_embedding_mode == "crop":
            if (
                target_frequency > source_frequency
                or target_time > source_time
            ):
                raise RuntimeError(
                    "Cannot crop AST positional embeddings to a larger grid: "
                    f"source={source_frequency}x{source_time}, "
                    f"target={target_frequency}x{target_time}"
                )
            # Training placed the real short-clip frames at the beginning of
            # the padded time axis. Keep those exact learned positions instead
            # of compressing the entire padded timeline into the short clip.
            patch = patch[:, :target_frequency, :target_time, :]
        else:
            patch = patch.permute(0, 3, 1, 2)
            patch = torch.nn.functional.interpolate(
                patch,
                size=(target_frequency, target_time),
                mode="bilinear",
                align_corners=False,
            )
            patch = patch.permute(0, 2, 3, 1)
        patch = patch.reshape(
            1, target_frequency * target_time, hidden_size
        )
        resized = torch.cat((special, patch), dim=1).detach()
        self._position_embedding_cache[target_key] = resized
        embeddings.position_embeddings = torch.nn.Parameter(
            resized, requires_grad=False
        )

    def _preprocess_full_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply deterministic preprocessing once to the continuous waveform."""
        result = np.asarray(audio, dtype=np.float32)
        if self.mean_subtract:
            result = result - float(result.mean())
        if self.high_pass_filter:
            sos = butter(
                self.high_pass_order,
                self.high_pass_cutoff_hz,
                btype="highpass",
                fs=SAMPLE_RATE,
                output="sos",
            )
            try:
                result = sosfiltfilt(sos, result).astype(np.float32)
            except ValueError:
                result = sosfilt(sos, result).astype(np.float32)
        return np.asarray(result, dtype=np.float32)

    def _compute_input_values(
        self,
        audio: np.ndarray,
        segment_duration: float,
        hop_duration: float,
    ) -> torch.Tensor:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError(
                "Optimized full-spectrogram inference requires torchaudio"
            ) from exc

        segment_samples = int(round(segment_duration * SAMPLE_RATE))
        hop_samples = int(round(hop_duration * SAMPLE_RATE))
        if min(segment_samples, hop_samples) < 1:
            raise ValueError("segment_duration and hop_duration must be positive")
        positions = max(1, 1 + (len(audio) - segment_samples) // hop_samples)

        extractor = self.feature_extractor
        num_mel_bins = int(getattr(extractor, "num_mel_bins", 128))
        max_length = int(getattr(extractor, "max_length", 1024))
        feature_mean = float(getattr(extractor, "mean", -4.2677393))
        feature_std = float(getattr(extractor, "std", 4.5689974))
        do_normalize = bool(getattr(extractor, "do_normalize", True))
        frame_shift_ms = float(getattr(extractor, "hop_length", 10.0))

        waveform = torch.from_numpy(
            self._preprocess_full_audio(audio)
        ).float().unsqueeze(0)
        # Keep this call aligned with ASTFeatureExtractor._extract_fbank_features.
        # In particular, do not subtract the log-mel mean afterward: the AST
        # extractor pads the raw fbank and then applies its configured global
        # mean/std normalization.
        full_fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=SAMPLE_RATE,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=frame_shift_ms,
        )

        frames_per_second = 1000.0 / frame_shift_ms
        hop_frames = round(hop_duration * frames_per_second)
        # A 3-second Kaldi fbank has fewer than 300 frames because complete
        # analysis frames must fit inside the waveform (normally 298 frames at
        # 16 kHz, 25 ms frame length, and 10 ms frame shift). Derive rather than
        # assume that count so slicing matches independently extracted windows.
        frame_cache_key = (segment_samples, num_mel_bins)
        segment_frames = self._segment_frame_cache.get(frame_cache_key)
        if segment_frames is None:
            probe = torch.zeros((1, segment_samples), dtype=torch.float32)
            segment_frames = int(
                torchaudio.compliance.kaldi.fbank(
                    probe,
                    sample_frequency=SAMPLE_RATE,
                    window_type="hanning",
                    num_mel_bins=num_mel_bins,
                    dither=0.0,
                    frame_shift=frame_shift_ms,
                ).shape[0]
            )
            if segment_frames < 1:
                raise RuntimeError("Kaldi fbank produced no frames for one segment")
            self._segment_frame_cache[frame_cache_key] = segment_frames
        target_frames = (
            max(1, min(max_length, segment_frames))
            if self.compact_ast_frames
            else max_length
        )
        if self.compact_ast_frames:
            self._ensure_ast_position_embeddings(target_frames, num_mel_bins)

        windows: list[torch.Tensor] = []
        for index in range(positions):
            start = index * hop_frames
            window = full_fbank[start : start + segment_frames, :]
            if window.shape[0] < target_frames:
                window = torch.nn.functional.pad(
                    window, (0, 0, 0, target_frames - window.shape[0])
                )
            else:
                window = window[:target_frames, :]
            windows.append(window)

        input_values = torch.stack(windows)
        if do_normalize:
            input_values = (
                input_values - feature_mean
            ) / (feature_std * 2.0)
        return input_values

    def predict(
        self,
        wav_file_path: str,
        segment_duration: int = 3,
        hop_duration: int = 2,
    ) -> dict[str, Any]:
        input_values = self._compute_input_values(
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
            for start in range(0, len(input_values), self.inference_batch_size):
                logits = self.model(
                    input_values=input_values[
                        start : start + self.inference_batch_size
                    ].to(self.device)
                )
                for name, values in zip(parts, logits):
                    parts[name].append(
                        torch.softmax(values, dim=-1).cpu().numpy()
                    )
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
            "feature_extraction_mode": (
                "full_spectrogram_compact_"
                + self.compact_position_embedding_mode
                if self.compact_ast_frames
                else "full_spectrogram_padded"
            ),
        }


def get_full_spectrogram_multispecies_cetacean_inference(
    model_path: str, **kwargs: Any
) -> FullSpectrogramMultispeciesCetaceanInference:
    return FullSpectrogramMultispeciesCetaceanInference(model_path, **kwargs)
