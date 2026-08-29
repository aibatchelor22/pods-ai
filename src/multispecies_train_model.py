#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Fine-tune a shared AST backbone with DCLDE multi-task classification heads.

Tasks:
  - KW detection: KW vs not-KW
  - Species: background, KW, HW, AB
  - KW ecotype: NRKW, SRKW, OKW, SAR, TKW

The species labels are read from ClassSpecies:
  KW     -> killer whale
  HW     -> humpback whale
  AB     -> Pacific white-sided dolphin
  UndBio -> background
  BKG    -> background

The ecotype loss is only computed for rows whose ClassSpecies is KW and whose
Ecotype value is one of NRKW, SRKW, OKW, SAR, or TKW.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Collection, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (  # noqa: E402
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
DEFAULT_MAX_DURATION = 3.0
CHECKPOINT_SAVE_LIMIT = 6
DEFAULT_DATALOADER_WORKERS = 2

KW_LABELS = {"not_kw": 0, "kw": 1}
KW_ID2LABEL = {v: k for k, v in KW_LABELS.items()}
SPECIES_LABELS = {"background": 0, "KW": 1, "HW": 2, "AB": 3}
SPECIES_ID2LABEL = {v: k for k, v in SPECIES_LABELS.items()}
ECOTYPE_LABELS = {"NRKW": 0, "SRKW": 1, "OKW": 2, "SAR": 3, "TKW": 4}
ECOTYPE_ID2LABEL = {v: k for k, v in ECOTYPE_LABELS.items()}
BACKGROUND_SPECIES = {"UndBio", "BKG"}
KNOWN_SPECIES = set(SPECIES_LABELS) | BACKGROUND_SPECIES
IGNORE_INDEX = -100


def resolve_path(path: str) -> Path:
    """Resolve absolute paths as-is and repo-relative paths under REPO_ROOT."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def normalize_label(value: Any) -> str:
    """Normalize CSV label values to stable strings."""
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    if normalized.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return normalized


def map_species_label(class_species: str) -> int:
    """Map ClassSpecies to the species-head target."""
    if class_species in BACKGROUND_SPECIES:
        return SPECIES_LABELS["background"]
    if class_species in SPECIES_LABELS:
        return SPECIES_LABELS[class_species]
    raise ValueError(f"Unknown ClassSpecies label: {class_species!r}")


def map_kw_label(class_species: str) -> int:
    """Map ClassSpecies to the KW-vs-not-KW target."""
    return KW_LABELS["kw"] if class_species == "KW" else KW_LABELS["not_kw"]


def map_ecotype_label(class_species: str, ecotype: str) -> int:
    """Map Ecotype to target ID, or IGNORE_INDEX when not applicable."""
    if class_species != "KW":
        return IGNORE_INDEX
    if not ecotype:
        return IGNORE_INDEX
    if ecotype not in ECOTYPE_LABELS:
        raise ValueError(f"Unknown KW Ecotype label: {ecotype!r}")
    return ECOTYPE_LABELS[ecotype]


class DCLDEAudioDataset(Dataset):
    """Lightweight manifest-backed dataset.

    Audio is intentionally loaded later by the collator so preprocessing does
    not materialize AST feature tensors for the whole training set in RAM.
    """

    def __init__(self, frame: pd.DataFrame, domain_columns: Optional[list[str]] = None) -> None:
        self.clip_paths = frame["clip_path"].astype(str).tolist()
        self.kw_labels = frame["kw_labels"].astype(int).tolist()
        self.species_labels = frame["species_labels"].astype(int).tolist()
        self.ecotype_labels = frame["ecotype_labels"].astype(int).tolist()
        self.domain_columns = list(domain_columns or [])
        self.domain_labels = (
            frame["domain_label"].astype(str).tolist()
            if "domain_label" in frame.columns
            else []
        )

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "clip_path": self.clip_paths[index],
            "kw_labels": self.kw_labels[index],
            "species_labels": self.species_labels[index],
            "ecotype_labels": self.ecotype_labels[index],
        }

    def label_values(self, label_column: str) -> list[int]:
        return list(getattr(self, label_column))


def load_manifest(
    manifest_path: str,
    drop_unknown_labels: bool = False,
    domain_columns: Optional[list[str]] = None,
) -> DCLDEAudioDataset:
    """Load a DCLDE manifest into a lightweight PyTorch Dataset."""
    path = resolve_path(manifest_path)
    df = pd.read_csv(path, low_memory=False)

    required_columns = {"clip_path", "ClassSpecies", "Ecotype"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ClassSpecies"] = df["ClassSpecies"].map(normalize_label)
    df["Ecotype"] = df["Ecotype"].map(normalize_label)

    known_species_mask = df["ClassSpecies"].isin(KNOWN_SPECIES)
    if not known_species_mask.all():
        unknown_counts = df.loc[~known_species_mask, "ClassSpecies"].value_counts(dropna=False)
        if not drop_unknown_labels:
            raise ValueError(
                "Unknown ClassSpecies values found. Pass --drop-unknown-labels to skip them: "
                f"{unknown_counts.to_dict()}"
            )
        print(f"Dropping unknown ClassSpecies rows: {unknown_counts.to_dict()}")
        df = df.loc[known_species_mask].copy()

    known_ecotype_mask = (
        (df["ClassSpecies"] != "KW")
        | (df["Ecotype"] == "")
        | df["Ecotype"].isin(ECOTYPE_LABELS)
    )
    if not known_ecotype_mask.all():
        unknown_counts = df.loc[~known_ecotype_mask, "Ecotype"].value_counts(dropna=False)
        if not drop_unknown_labels:
            raise ValueError(
                "Unknown KW Ecotype values found. Pass --drop-unknown-labels to skip them: "
                f"{unknown_counts.to_dict()}"
            )
        print(f"Dropping unknown KW Ecotype rows: {unknown_counts.to_dict()}")
        df = df.loc[known_ecotype_mask].copy()

    df["kw_labels"] = df["ClassSpecies"].map(map_kw_label)
    df["species_labels"] = df["ClassSpecies"].map(map_species_label)
    df["ecotype_labels"] = [
        map_ecotype_label(class_species, ecotype)
        for class_species, ecotype in zip(df["ClassSpecies"], df["Ecotype"])
    ]

    if domain_columns:
        missing_domain_columns = set(domain_columns) - set(df.columns)
        if missing_domain_columns:
            raise ValueError(
                f"{path} missing domain-balance columns: {sorted(missing_domain_columns)}"
            )
        domain_parts = []
        for column in domain_columns:
            values = (
                df[column]
                .map(normalize_label)
                .replace("", "<missing>")
                .str.casefold()
            )
            domain_parts.append(column + "=" + values)
        domain_labels = domain_parts[0]
        for part in domain_parts[1:]:
            domain_labels = domain_labels + " | " + part
        df["domain_label"] = domain_labels

    return DCLDEAudioDataset(df.reset_index(drop=True), domain_columns=domain_columns)


def parse_domain_columns(value: str) -> list[str]:
    """Parse a comma-separated list of manifest columns used as a domain key."""
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise ValueError("--domain-columns must contain at least one manifest column")
    if len(columns) != len(set(columns)):
        raise ValueError(f"--domain-columns contains duplicate names: {columns}")
    return columns


def domain_balanced_sample_weights(
    dataset: DCLDEAudioDataset,
    minimum_domain_samples: int = 25,
) -> torch.DoubleTensor:
    """Balance species, then domains within species, while preserving epoch length.

    Every species receives equal expected sampling probability. Within a species,
    sufficiently populated domains receive equal shares, and rows within each
    species/domain group are sampled uniformly. A minimum effective group size
    prevents a domain represented by only one or two clips from being repeated
    thousands of times in one epoch.
    """
    if not dataset.domain_labels or len(dataset.domain_labels) != len(dataset):
        raise ValueError("Training dataset does not contain domain labels")
    if minimum_domain_samples < 1:
        raise ValueError("minimum_domain_samples must be at least 1")

    group_counts = Counter(zip(dataset.species_labels, dataset.domain_labels))
    domains_by_species: dict[int, set[str]] = {}
    for species_id, domain_label in group_counts:
        domains_by_species.setdefault(species_id, set()).add(domain_label)

    species_count = len(domains_by_species)
    if species_count == 0:
        raise ValueError("Cannot construct domain-balanced weights for an empty dataset")

    domain_mass_sums: dict[int, float] = {}
    for (species_id, _domain_label), group_size in group_counts.items():
        domain_mass_sums[species_id] = domain_mass_sums.get(species_id, 0.0) + min(
            1.0,
            group_size / minimum_domain_samples,
        )

    weights = []
    for species_id, domain_label in zip(dataset.species_labels, dataset.domain_labels):
        group_size = group_counts[(species_id, domain_label)]
        effective_group_size = max(group_size, minimum_domain_samples)
        weights.append(
            1.0
            / (
                species_count
                * domain_mass_sums[species_id]
                * effective_group_size
            )
        )

    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    # Scaling does not change sampling probabilities and makes diagnostics easier.
    weights_tensor /= weights_tensor.mean()
    return weights_tensor


class WaveformAugmenter:
    """Simple waveform augmentations applied during preprocessing."""

    def __init__(
        self,
        sample_rate: int,
        random_gain: bool = False,
        random_gain_prob: float = 1.0,
        gain_db: float = 6.0,
        gain_clipping_mode: str = "clip",
        time_shift: bool = False,
        time_shift_prob: float = 1.0,
        max_shift_ms: float = 250.0,
        time_shift_fade_ms: float = 0.0,
        gaussian_noise: bool = False,
        gaussian_noise_prob: float = 1.0,
        noise_std: float = 0.002,
        noise_scale_mode: str = "absolute",
        random_eq: bool = False,
        random_eq_prob: float = 0.5,
        eq_num_bands_min: int = 1,
        eq_num_bands_max: int = 3,
        eq_gain_db_min: float = -6.0,
        eq_gain_db_max: float = 6.0,
        eq_q_min: float = 0.4,
        eq_q_max: float = 1.2,
        eq_freq_min: float = 100.0,
        eq_freq_max: float = 6000.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.random_gain = random_gain
        self.random_gain_prob = random_gain_prob
        self.gain_db = gain_db
        self.gain_clipping_mode = gain_clipping_mode
        self.time_shift = time_shift
        self.time_shift_prob = time_shift_prob
        self.max_shift = int(sample_rate * max_shift_ms / 1000)
        self.time_shift_fade = int(sample_rate * time_shift_fade_ms / 1000)
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_std = noise_std
        self.noise_scale_mode = noise_scale_mode
        self.random_eq = random_eq
        self.random_eq_prob = random_eq_prob
        self.eq_num_bands_min = eq_num_bands_min
        self.eq_num_bands_max = eq_num_bands_max
        self.eq_gain_db_min = eq_gain_db_min
        self.eq_gain_db_max = eq_gain_db_max
        self.eq_q_min = eq_q_min
        self.eq_q_max = eq_q_max
        self.eq_freq_min = eq_freq_min
        self.eq_freq_max = eq_freq_max

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32, copy=True)
        if self.random_gain and random.random() < self.random_gain_prob:
            gain_db = random.uniform(-self.gain_db, self.gain_db)
            gain = 10 ** (gain_db / 20)
            if self.gain_clipping_mode == "safe":
                peak = float(np.max(np.abs(audio)))
                if peak > 0:
                    gain = min(gain, 1.0 / peak)
            audio *= gain
            if self.gain_clipping_mode == "normalize":
                peak = float(np.max(np.abs(audio)))
                if peak > 1.0:
                    audio /= peak
            elif self.gain_clipping_mode == "soft":
                audio = np.tanh(audio)
        if self.time_shift and self.max_shift > 0 and random.random() < self.time_shift_prob:
            audio = self._zero_pad_shift(
                audio,
                random.randint(-self.max_shift, self.max_shift),
                fade_samples=self.time_shift_fade,
            )
        if self.random_eq and random.random() < self.random_eq_prob:
            audio = self._random_eq(audio)
        if self.gaussian_noise and random.random() < self.gaussian_noise_prob:
            noise_std = self.noise_std
            if self.noise_scale_mode == "rms":
                rms = float(np.sqrt(np.mean(np.square(audio))))
                noise_std *= rms
            audio = audio + np.random.normal(0, noise_std, size=audio.shape)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _zero_pad_shift(
        audio: np.ndarray,
        shift: int,
        fade_samples: int = 0,
    ) -> np.ndarray:
        """Shift audio without wrapping; newly exposed samples are silence."""
        if shift == 0:
            return audio

        shifted = np.zeros_like(audio)
        if shift > 0:
            shifted[shift:] = audio[:-shift]
            fade_len = min(fade_samples, len(shifted) - shift)
            if fade_len > 0:
                shifted[shift : shift + fade_len] *= np.linspace(
                    0.0,
                    1.0,
                    fade_len,
                    endpoint=True,
                    dtype=shifted.dtype,
                )
        else:
            shifted[:shift] = audio[-shift:]
            fade_len = min(fade_samples, len(shifted) + shift)
            if fade_len > 0:
                fade_start = len(shifted) + shift - fade_len
                fade_end = len(shifted) + shift
                shifted[fade_start:fade_end] *= np.linspace(
                    1.0,
                    0.0,
                    fade_len,
                    endpoint=True,
                    dtype=shifted.dtype,
                )
        return shifted

    def _random_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply random peaking EQ bands to one waveform."""
        num_bands = random.randint(self.eq_num_bands_min, self.eq_num_bands_max)
        output = audio
        for _ in range(num_bands):
            frequency_hz = random.uniform(self.eq_freq_min, self.eq_freq_max)
            gain_db = random.uniform(self.eq_gain_db_min, self.eq_gain_db_max)
            q = random.uniform(self.eq_q_min, self.eq_q_max)
            b, a = self._peaking_eq_coefficients(
                frequency_hz=frequency_hz,
                gain_db=gain_db,
                q=q,
            )
            output = lfilter(b, a, output).astype(np.float32, copy=False)
        return output

    def _peaking_eq_coefficients(
        self,
        frequency_hz: float,
        gain_db: float,
        q: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return RBJ cookbook peaking-EQ biquad coefficients."""
        frequency_hz = min(max(frequency_hz, 1.0), (self.sample_rate / 2.0) - 1.0)
        q = max(q, 1e-6)
        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * np.pi * frequency_hz / self.sample_rate
        alpha = np.sin(omega) / (2.0 * q)
        cos_omega = np.cos(omega)

        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / amplitude

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
        return b, a


class DCLDEAudioCollator:
    """Load audio and build AST inputs lazily for one batch."""

    def __init__(
        self,
        feature_extractor: Any,
        max_duration: float,
        augmenter: Optional[WaveformAugmenter] = None,
        mean_subtract: bool = False,
        high_pass_cutoff_hz: Optional[float] = None,
        high_pass_order: int = 4,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.target_length = int(max_duration * SAMPLE_RATE)
        self.augmenter = augmenter
        self.mean_subtract = mean_subtract
        self.high_pass_cutoff_hz = high_pass_cutoff_hz
        self.high_pass_order = high_pass_order
        self.high_pass_sos: Optional[np.ndarray] = None
        if high_pass_cutoff_hz is not None:
            nyquist = SAMPLE_RATE / 2.0
            if not 0.0 < high_pass_cutoff_hz < nyquist:
                raise ValueError(
                    f"high_pass_cutoff_hz must be between 0 and {nyquist}, "
                    f"got {high_pass_cutoff_hz}"
                )
            if high_pass_order < 1:
                raise ValueError(f"high_pass_order must be positive, got {high_pass_order}")
            self.high_pass_sos = butter(
                high_pass_order,
                high_pass_cutoff_hz,
                btype="highpass",
                fs=SAMPLE_RATE,
                output="sos",
            )

    def _high_pass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply configured high-pass filter to one waveform."""
        if self.high_pass_sos is None:
            return audio
        try:
            filtered = sosfiltfilt(self.high_pass_sos, audio)
        except ValueError:
            filtered = sosfilt(self.high_pass_sos, audio)
        return filtered.astype(np.float32, copy=False)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed_audio = []
        for example in examples:
            clip_path = example["clip_path"]
            try:
                audio, _ = librosa.load(clip_path, sr=SAMPLE_RATE, mono=True)
            except Exception as exc:
                raise RuntimeError(f"Failed to load audio clip: {clip_path}") from exc

            if len(audio) > self.target_length:
                audio = audio[: self.target_length]
            elif len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode="constant")

            audio = audio.astype(np.float32, copy=False)
            if self.mean_subtract:
                audio = audio - float(audio.mean())
            if self.high_pass_sos is not None:
                audio = self._high_pass_filter(audio)
            if self.augmenter is not None:
                audio = self.augmenter(audio)
            processed_audio.append(audio)

        batch = self.feature_extractor(
            processed_audio,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        batch["kw_labels"] = torch.tensor(
            [example["kw_labels"] for example in examples],
            dtype=torch.long,
        )
        batch["species_labels"] = torch.tensor(
            [example["species_labels"] for example in examples],
            dtype=torch.long,
        )
        batch["ecotype_labels"] = torch.tensor(
            [example["ecotype_labels"] for example in examples],
            dtype=torch.long,
        )
        return batch


class MultiTaskASTForDCLDE(nn.Module):
    """AST backbone with KW, species, and ecotype classifier heads."""

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        kw_loss_weight: float = 1.0,
        species_loss_weight: float = 1.0,
        ecotype_loss_weight: float = 1.0,
        freeze_backbone: bool = False,
        kw_class_weights: Optional[list[float]] = None,
        species_class_weights: Optional[list[float]] = None,
        ecotype_class_weights: Optional[list[float]] = None,
        initialize_from_config: bool = False,
    ) -> None:
        super().__init__()
        if initialize_from_config:
            config = AutoConfig.from_pretrained(model_name)
            base_model = AutoModelForAudioClassification.from_config(config)
        else:
            base_model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.base_model_name = model_name
        self.config = base_model.config
        self.ast = base_model.audio_spectrogram_transformer
        hidden_size = int(getattr(self.config, "hidden_size"))
        self.dropout = nn.Dropout(dropout)
        self.kw_classifier = nn.Linear(hidden_size, len(KW_LABELS))
        self.species_classifier = nn.Linear(hidden_size, len(SPECIES_LABELS))
        self.ecotype_classifier = nn.Linear(hidden_size, len(ECOTYPE_LABELS))
        self.kw_loss_weight = kw_loss_weight
        self.species_loss_weight = species_loss_weight
        self.ecotype_loss_weight = ecotype_loss_weight
        self.register_buffer(
            "kw_class_weights",
            self._class_weight_tensor(kw_class_weights, len(KW_LABELS), "kw"),
            persistent=False,
        )
        self.register_buffer(
            "species_class_weights",
            self._class_weight_tensor(species_class_weights, len(SPECIES_LABELS), "species"),
            persistent=False,
        )
        self.register_buffer(
            "ecotype_class_weights",
            self._class_weight_tensor(ecotype_class_weights, len(ECOTYPE_LABELS), "ecotype"),
            persistent=False,
        )

        if freeze_backbone:
            print("Freezing AST backbone and training classification heads only.")
            for param in self.ast.parameters():
                param.requires_grad = False

    @staticmethod
    def _class_weight_tensor(
        weights: Optional[list[float]],
        expected_length: int,
        head_name: str,
    ) -> Optional[torch.Tensor]:
        """Validate and convert optional per-class loss weights."""
        if weights is None:
            return None
        if len(weights) != expected_length:
            raise ValueError(
                f"{head_name} class weights must have {expected_length} values, got {len(weights)}"
            )
        if any(weight < 0 for weight in weights):
            raise ValueError(f"{head_name} class weights must be non-negative: {weights}")
        return torch.tensor(weights, dtype=torch.float32)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None) -> None:
        """Allow Trainer gradient checkpointing to reach the AST backbone when available."""
        if hasattr(self.ast, "gradient_checkpointing_enable"):
            self.ast.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the AST backbone when available."""
        if hasattr(self.ast, "gradient_checkpointing_disable"):
            self.ast.gradient_checkpointing_disable()

    def _pool_ast_output(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """Use AST's CLS/distillation-token pooled embedding."""
        if sequence_output.shape[1] >= 2:
            return (sequence_output[:, 0] + sequence_output[:, 1]) / 2.0
        return sequence_output[:, 0]

    def forward(
        self,
        input_values: torch.Tensor,
        kw_labels: Optional[torch.Tensor] = None,
        species_labels: Optional[torch.Tensor] = None,
        ecotype_labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, Any]:
        outputs = self.ast(input_values=input_values, return_dict=True)
        pooled = self.dropout(self._pool_ast_output(outputs.last_hidden_state))

        kw_logits = self.kw_classifier(pooled)
        species_logits = self.species_classifier(pooled)
        ecotype_logits = self.ecotype_classifier(pooled)

        loss = None
        losses = []
        if kw_labels is not None:
            losses.append(
                self.kw_loss_weight
                * nn.functional.cross_entropy(
                    kw_logits,
                    kw_labels.long(),
                    weight=self.kw_class_weights,
                )
            )
        if species_labels is not None:
            losses.append(
                self.species_loss_weight
                * nn.functional.cross_entropy(
                    species_logits,
                    species_labels.long(),
                    weight=self.species_class_weights,
                )
            )
        if ecotype_labels is not None:
            ecotype_mask = ecotype_labels != IGNORE_INDEX
            if torch.any(ecotype_mask):
                losses.append(
                    self.ecotype_loss_weight
                    * nn.functional.cross_entropy(
                        ecotype_logits[ecotype_mask],
                        ecotype_labels[ecotype_mask].long(),
                        weight=self.ecotype_class_weights,
                    )
                )
        if losses:
            loss = torch.stack(losses).sum()

        return {
            "loss": loss,
            "logits": (kw_logits, species_logits, ecotype_logits),
        }


def _find_multitask_weights_path(model_dir: Path) -> Optional[Path]:
    """Return the first supported multi-task checkpoint weight file in a directory."""
    for filename in ("pytorch_model.bin", "model.safetensors"):
        weights_path = model_dir / filename
        if weights_path.exists():
            return weights_path
    return None


def _download_multitask_weights_path(model_name: str) -> Path:
    """Download the first supported multi-task checkpoint weight file from the Hub."""
    errors = []
    for filename in ("pytorch_model.bin", "model.safetensors"):
        try:
            return Path(hf_hub_download(model_name, filename))
        except EntryNotFoundError as exc:
            errors.append(f"{filename}: not found")
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError(
        f"Could not find/download supported weights for {model_name}. "
        f"Tried pytorch_model.bin and model.safetensors. Details: {'; '.join(errors)}"
    )


def load_multitask_checkpoint_files(model_name: str) -> Optional[tuple[dict[str, Any], Path]]:
    """Return saved multitask metadata/state paths for local dirs or Hub repos."""
    local_path = Path(model_name)
    if local_path.exists():
        config_path = local_path / "multitask_config.json"
        weights_path = _find_multitask_weights_path(local_path)
        if config_path.exists() and weights_path is not None:
            with config_path.open("r", encoding="utf-8") as file:
                return json.load(file), weights_path
        return None

    try:
        config_path = Path(hf_hub_download(model_name, "multitask_config.json"))
    except Exception as exc:
        print(
            f"Warning: could not download multitask_config.json from {model_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    weights_path = _download_multitask_weights_path(model_name)

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file), weights_path


LEGACY_AST_KEY_REPLACEMENTS = (
    ("ast.encoder.layer.", "ast.encoder.layers."),
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
    (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
)


def remap_legacy_ast_state_dict_keys(
    state_dict: dict[str, Any],
    expected_keys: Collection[str],
) -> tuple[dict[str, Any], int]:
    """Map legacy Transformers AST parameter names to names expected by this runtime.

    Transformers renamed the AST encoder and attention/MLP parameters. Saved
    multi-task checkpoints use the names from the Transformers version on which
    they were trained. Only apply a translated name when the model built by the
    installed runtime expects it, so legacy runtimes remain supported as well.
    """
    expected = set(expected_keys)
    remapped_state_dict: dict[str, Any] = {}
    original_keys_by_output: dict[str, str] = {}
    remapped_count = 0

    for original_key, value in state_dict.items():
        output_key = original_key
        if original_key not in expected:
            candidate_key = original_key
            for legacy_fragment, current_fragment in LEGACY_AST_KEY_REPLACEMENTS:
                candidate_key = candidate_key.replace(legacy_fragment, current_fragment)
            candidate_keys = (
                candidate_key,
                candidate_key.replace("ast.encoder.layers.", "ast.layers."),
            )
            output_key = next(
                (candidate for candidate in candidate_keys if candidate in expected),
                original_key,
            )

        if output_key in remapped_state_dict:
            previous_key = original_keys_by_output[output_key]
            raise ValueError(
                "Checkpoint contains parameter names that collide after AST "
                f"compatibility remapping: {previous_key!r} and {original_key!r} "
                f"both map to {output_key!r}."
            )

        remapped_state_dict[output_key] = value
        original_keys_by_output[output_key] = original_key
        if output_key != original_key:
            remapped_count += 1

    return remapped_state_dict, remapped_count


def _format_checkpoint_key_mismatch(keys: Collection[str], limit: int = 12) -> str:
    """Format checkpoint key mismatches without printing hundreds of names."""
    key_list = list(keys)
    displayed = ", ".join(repr(key) for key in key_list[:limit])
    if len(key_list) > limit:
        displayed += f", ... ({len(key_list) - limit} more)"
    return displayed


def load_training_model(
    model_name: str,
    dropout: float,
    kw_loss_weight: float,
    species_loss_weight: float,
    ecotype_loss_weight: float,
    freeze_backbone: bool,
    kw_class_weights: Optional[list[float]] = None,
    species_class_weights: Optional[list[float]] = None,
    ecotype_class_weights: Optional[list[float]] = None,
) -> MultiTaskASTForDCLDE:
    """Load either a base AST model or a previously saved DCLDE multi-task model."""
    checkpoint = load_multitask_checkpoint_files(model_name)
    if checkpoint is None:
        return MultiTaskASTForDCLDE(
            model_name=model_name,
            dropout=dropout,
            kw_loss_weight=kw_loss_weight,
            species_loss_weight=species_loss_weight,
            ecotype_loss_weight=ecotype_loss_weight,
            freeze_backbone=freeze_backbone,
            kw_class_weights=kw_class_weights,
            species_class_weights=species_class_weights,
            ecotype_class_weights=ecotype_class_weights,
        )

    metadata, weights_path = checkpoint
    standalone = bool(metadata.get("standalone", False))
    base_model = metadata.get("base_model")
    if not standalone and not base_model:
        raise ValueError(
            f"{model_name} has multitask weights but multitask_config.json is missing base_model."
        )

    print(f"Detected saved DCLDE multi-task checkpoint: {model_name}")
    if standalone:
        initialization_source = model_name
        print("Rebuilding AST architecture from the checkpoint's bundled config.json.")
    else:
        initialization_source = base_model
        print(f"Rebuilding AST backbone from base model: {base_model}")
    model = MultiTaskASTForDCLDE(
        model_name=initialization_source,
        dropout=dropout,
        kw_loss_weight=kw_loss_weight,
        species_loss_weight=species_loss_weight,
        ecotype_loss_weight=ecotype_loss_weight,
        freeze_backbone=freeze_backbone,
        kw_class_weights=kw_class_weights,
        species_class_weights=species_class_weights,
        ecotype_class_weights=ecotype_class_weights,
        initialize_from_config=standalone,
    )
    if weights_path.name.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise ImportError(
                "Loading model.safetensors requires safetensors. "
                "Install it with: pip install safetensors"
            ) from exc
        state_dict = load_safetensors_file(str(weights_path), device="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint {weights_path} did not contain a parameter state dictionary."
        )

    state_dict, remapped_count = remap_legacy_ast_state_dict_keys(
        state_dict,
        model.state_dict().keys(),
    )
    if remapped_count:
        print(
            f"Remapped {remapped_count} legacy AST checkpoint parameter names "
            "for compatibility with the installed Transformers version."
        )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(
                "missing keys: " + _format_checkpoint_key_mismatch(missing_keys)
            )
        if unexpected_keys:
            details.append(
                "unexpected keys: " + _format_checkpoint_key_mismatch(unexpected_keys)
            )
        raise RuntimeError(
            f"Checkpoint {weights_path} is incompatible with the reconstructed model; "
            + "; ".join(details)
            + ". Refusing to run inference with partially loaded weights."
        )
    return model


def parse_class_weights(
    value: Optional[str],
    label2id: dict[str, int],
    argument_name: str,
) -> Optional[list[float]]:
    """Parse class weights as label=value pairs or full ID-order values."""
    if value is None or not value.strip():
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None

    weights = [1.0] * len(label2id)
    if all("=" not in part for part in parts):
        if len(parts) != len(label2id):
            raise ValueError(
                f"{argument_name} expected {len(label2id)} comma-separated weights "
                f"in class-ID order, got {len(parts)}"
            )
        weights = [float(part) for part in parts]
    else:
        for part in parts:
            if "=" not in part:
                raise ValueError(
                    f"{argument_name} must use all label=value entries or all plain values: {value!r}"
                )
            label, weight_text = [piece.strip() for piece in part.split("=", 1)]
            if label in label2id:
                class_id = label2id[label]
            elif label.isdigit() and int(label) in set(label2id.values()):
                class_id = int(label)
            else:
                raise ValueError(
                    f"{argument_name} has unknown class {label!r}. "
                    f"Known classes: {sorted(label2id)}"
                )
            weights[class_id] = float(weight_text)

    if any(weight < 0 for weight in weights):
        raise ValueError(f"{argument_name} weights must be non-negative: {weights}")
    return weights


def validate_probability(value: float, argument_name: str) -> float:
    """Validate a command-line probability."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{argument_name} must be between 0 and 1, got {value}")
    return value


def _f1(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Return an F1 score with sklearn's zero-division behavior fixed."""
    return float(
        precision_recall_fscore_support(
            y_true,
            y_pred,
            zero_division=0,
            **kwargs,
        )[2]
    )


def _class_f1(labels: np.ndarray, predictions: np.ndarray, class_id: int) -> float:
    """Return one-vs-rest F1 for a single class in a multiclass target."""
    scores = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[class_id],
        average=None,
        zero_division=0,
    )[2]
    return float(scores[0]) if len(scores) else 0.0


def _unpack_three(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Handle Trainer outputs as either (a, b, c) or ((a, b, c),)."""
    if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        raise ValueError(f"Expected three prediction/label arrays, got {type(value).__name__}")
    return value[0], value[1], value[2]


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute metrics for all heads and a combined model-selection score."""
    kw_logits, species_logits, ecotype_logits = _unpack_three(eval_pred.predictions)
    kw_labels, species_labels, ecotype_labels = _unpack_three(eval_pred.label_ids)

    kw_predictions = np.argmax(kw_logits, axis=1)
    species_predictions = np.argmax(species_logits, axis=1)
    ecotype_predictions = np.argmax(ecotype_logits, axis=1)

    metrics: dict[str, float] = {
        "kw_accuracy": float(accuracy_score(kw_labels, kw_predictions)),
        "kw_f1": _f1(kw_labels, kw_predictions, average="binary", pos_label=KW_LABELS["kw"]),
        "species_accuracy": float(accuracy_score(species_labels, species_predictions)),
        "species_macro_f1": _f1(species_labels, species_predictions, average="macro"),
    }

    for class_id, class_name in SPECIES_ID2LABEL.items():
        metrics[f"species_f1_{class_name}"] = _class_f1(
            species_labels,
            species_predictions,
            class_id,
        )

    ecotype_mask = ecotype_labels != IGNORE_INDEX
    if np.any(ecotype_mask):
        ecotype_true = ecotype_labels[ecotype_mask]
        ecotype_pred = ecotype_predictions[ecotype_mask]
        metrics["ecotype_accuracy"] = float(accuracy_score(ecotype_true, ecotype_pred))
        metrics["ecotype_macro_f1"] = _f1(ecotype_true, ecotype_pred, average="macro")
        for class_id, class_name in ECOTYPE_ID2LABEL.items():
            metrics[f"ecotype_f1_{class_name}"] = _class_f1(
                ecotype_true,
                ecotype_pred,
                class_id,
            )
        srkw_tkw_labels = [ECOTYPE_LABELS["SRKW"], ECOTYPE_LABELS["TKW"]]
        metrics["ecotype_srkw_tkw_f1"] = _f1(
            ecotype_true,
            ecotype_pred,
            average="macro",
            labels=srkw_tkw_labels,
        )
    else:
        metrics["ecotype_accuracy"] = 0.0
        metrics["ecotype_macro_f1"] = 0.0
        metrics["ecotype_srkw_tkw_f1"] = 0.0

    metrics["combined_score"] = (
        0.4 * metrics["kw_f1"]
        + 0.3 * metrics["species_macro_f1"]
        + 0.3 * metrics["ecotype_srkw_tkw_f1"]
    )
    return metrics


class LazyAudioTrainer(Trainer):
    """Trainer that uses different lazy collators for training and evaluation."""

    def __init__(
        self,
        *args: Any,
        train_collator: DCLDEAudioCollator,
        eval_collator: DCLDEAudioCollator,
        train_sample_weights: Optional[torch.DoubleTensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_collator = train_collator
        self.eval_collator = eval_collator
        self.train_sample_weights = train_sample_weights
        self._final_candidate_evaluated = False
        self._optimizer_scheduler_reported = False

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.train_sample_weights is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.args.data_seed))
            sampler = WeightedRandomSampler(
                weights=self.train_sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=generator,
            )
        else:
            sampler = self._get_train_sampler()
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=self.train_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )
        return self.accelerator.prepare(dataloader)

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Create optimizer/scheduler and print the actual LR setup once."""
        super().create_optimizer_and_scheduler(num_training_steps)
        if self._optimizer_scheduler_reported:
            return
        self._optimizer_scheduler_reported = True
        optimizer_lrs = []
        if self.optimizer is not None:
            optimizer_lrs = [
                float(group.get("lr", 0.0))
                for group in self.optimizer.param_groups
            ]
        scheduler_name = type(self.lr_scheduler).__name__ if self.lr_scheduler is not None else "None"
        unique_lrs = sorted(set(optimizer_lrs))
        formatted_lrs = ", ".join(f"{lr:.3e}" for lr in unique_lrs) if unique_lrs else "N/A"
        print("\n" + "=" * 72)
        print("OPTIMIZER / SCHEDULER")
        print("=" * 72)
        print(f"TrainingArguments learning_rate: {float(self.args.learning_rate):.3e}")
        print(f"Initial optimizer lr group(s):   {formatted_lrs}")
        print(f"Scheduler:                       {scheduler_name}")
        print(f"Scheduler training steps:        {num_training_steps:,}")
        print(f"Warmup ratio:                    {float(self.args.warmup_ratio):.6g}")
        print(f"Warmup steps:                    {int(self.args.get_warmup_steps(num_training_steps)):,}")
        print("=" * 72 + "\n")

    def _metric_key_for_best_model(self) -> str:
        """Return the evaluation metric key used by Trainer for best-model selection."""
        metric_name = self.args.metric_for_best_model
        if metric_name is None:
            metric_name = "loss"
        if not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        return metric_name

    def _is_better_metric(self, current_metric: float, best_metric: Optional[float]) -> bool:
        """Compare a candidate metric against the current best metric."""
        if best_metric is None:
            return True
        if self.args.greater_is_better:
            return current_metric > best_metric
        return current_metric < best_metric

    def _load_best_model(self, *args: Any, **kwargs: Any) -> None:
        """Evaluate the final training state as a best-checkpoint candidate before reload."""
        if not self._final_candidate_evaluated and self.eval_dataset is not None:
            self._final_candidate_evaluated = True
            metric_key = self._metric_key_for_best_model()
            print("Evaluating final training state as a best-model candidate...")
            metrics = self.evaluate(metric_key_prefix="eval")
            final_metric = metrics.get(metric_key)
            if final_metric is None:
                print(
                    f"Warning: final evaluation did not return {metric_key}; "
                    "final state cannot be compared against the best checkpoint."
                )
            elif self._is_better_metric(float(final_metric), self.state.best_metric):
                final_checkpoint = Path(self.args.output_dir) / "checkpoint-final"
                print(
                    f"Final training state is the new best model: "
                    f"{metric_key}={float(final_metric):.6f}"
                )
                self.save_model(str(final_checkpoint))
                self.state.best_metric = float(final_metric)
                self.state.best_model_checkpoint = str(final_checkpoint)
            else:
                print(
                    f"Final training state did not beat best checkpoint: "
                    f"{metric_key}={float(final_metric):.6f}, "
                    f"best={self.state.best_metric:.6f}"
                )

        return super()._load_best_model(*args, **kwargs)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=self._get_eval_sampler(eval_dataset),
            collate_fn=self.eval_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )
        return self.accelerator.prepare(dataloader)


def analyze_dataset(dataset: dict[str, DCLDEAudioDataset]) -> None:
    """Print target distributions for each split."""
    print("\n" + "=" * 72)
    print("DATASET ANALYSIS")
    print("=" * 72)
    for split_name, split_dataset in dataset.items():
        print(f"\n{split_name.upper()} split: {len(split_dataset)} samples")
        for label_column, id2label in (
            ("kw_labels", KW_ID2LABEL),
            ("species_labels", SPECIES_ID2LABEL),
            ("ecotype_labels", ECOTYPE_ID2LABEL),
        ):
            labels = [
                label
                for label in split_dataset.label_values(label_column)
                if label != IGNORE_INDEX
            ]
            counts = Counter(labels)
            print(f"  {label_column}:")
            if not labels:
                print("    no labeled rows")
                continue
            for label_id in sorted(counts):
                print(f"    {id2label[label_id]:12s}: {counts[label_id]:7d}")
    print("=" * 72 + "\n")


def analyze_domain_balancing(
    dataset: DCLDEAudioDataset,
    sample_weights: torch.DoubleTensor,
    minimum_domain_samples: int,
) -> None:
    """Print source and expected sampled distributions for balanced training."""
    probabilities = sample_weights / sample_weights.sum()
    source_species = Counter(dataset.species_labels)
    expected_species: dict[int, float] = {}
    expected_domains: dict[int, dict[str, float]] = {}
    for index, (species_id, domain_label) in enumerate(
        zip(dataset.species_labels, dataset.domain_labels)
    ):
        probability = float(probabilities[index])
        expected_species[species_id] = expected_species.get(species_id, 0.0) + probability
        species_domains = expected_domains.setdefault(species_id, {})
        species_domains[domain_label] = species_domains.get(domain_label, 0.0) + probability

    print("\n" + "=" * 72)
    print("DOMAIN-BALANCED TRAINING SAMPLER")
    print("=" * 72)
    print(f"Domain columns: {', '.join(dataset.domain_columns)}")
    print(f"Minimum effective species/domain group size: {minimum_domain_samples}")
    print(f"Samples per epoch: {len(dataset):,} (sampling with replacement)")
    for species_id in sorted(source_species):
        species_name = SPECIES_ID2LABEL[species_id]
        domains = expected_domains[species_id]
        domain_percentages = [100.0 * value for value in domains.values()]
        print(
            f"  {species_name}: source={source_species[species_id]:,}, "
            f"expected={100.0 * expected_species[species_id]:.2f}%, "
            f"domains={len(domains)}, expected domain range="
            f"{min(domain_percentages):.3f}%..{max(domain_percentages):.3f}% of all samples"
        )
    print("=" * 72 + "\n")


def print_training_geometry(
    training_args: TrainingArguments,
    train_rows: int,
    validation_rows: int,
) -> None:
    """Print batch-size and step-count details before training starts."""
    world_size = max(1, int(getattr(training_args, "world_size", 1)))
    gradient_accumulation_steps = max(1, int(training_args.gradient_accumulation_steps))
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    train_micro_batch = per_device_train_batch_size * world_size
    effective_train_batch = train_micro_batch * gradient_accumulation_steps
    train_batches_per_epoch = math.ceil(train_rows / train_micro_batch) if train_rows else 0
    optimizer_steps_per_epoch = (
        math.ceil(train_batches_per_epoch / gradient_accumulation_steps)
        if train_batches_per_epoch
        else 0
    )
    eval_micro_batch = per_device_eval_batch_size * world_size
    eval_batches = math.ceil(validation_rows / eval_micro_batch) if validation_rows else 0

    print("\n" + "=" * 72)
    print("TRAINING GEOMETRY")
    print("=" * 72)
    print(f"Per-device train batch size:    {per_device_train_batch_size}")
    print(f"Per-device eval batch size:     {per_device_eval_batch_size}")
    print(f"Requested learning rate:        {float(training_args.learning_rate):.3e}")
    print(f"Warmup ratio:                   {float(training_args.warmup_ratio):.6g}")
    print(f"World size/device count:        {world_size}")
    print(f"Gradient accumulation steps:    {gradient_accumulation_steps}")
    print(f"Train rows:                     {train_rows:,}")
    print(f"Validation rows:                {validation_rows:,}")
    print(f"Train dataloader batches/epoch: {train_batches_per_epoch:,}")
    print(f"Optimizer steps/epoch:          {optimizer_steps_per_epoch:,}")
    print(f"Effective train batch/update:   {effective_train_batch:,}")
    print(f"Eval batches/pass:              {eval_batches:,}")
    print("=" * 72 + "\n")


def save_training_curves(trainer: Trainer, output_dir: Path) -> None:
    """Save training and validation curves from Trainer logs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training curve plots.")
        return

    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    combined_steps: list[int] = []
    combined_scores: list[float] = []

    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(int(entry["step"]))
            train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_steps.append(int(entry["step"]))
            eval_losses.append(float(entry["eval_loss"]))
        if "eval_combined_score" in entry:
            combined_steps.append(int(entry["step"]))
            combined_scores.append(float(entry["eval_combined_score"]))

    if train_losses or eval_losses:
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_steps, train_losses, label="Training loss")
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="Validation loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        loss_path = output_dir / "loss_curve.png"
        plt.savefig(loss_path, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curve to {loss_path}")

    if combined_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(combined_steps, combined_scores, label="Combined validation score")
        plt.xlabel("Training step")
        plt.ylabel("Score")
        plt.title("Validation Score")
        plt.legend()
        plt.grid(True)
        score_path = output_dir / "validation_score_curve.png"
        plt.savefig(score_path, bbox_inches="tight")
        plt.close()
        print(f"Saved validation score curve to {score_path}")


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    model: Optional[MultiTaskASTForDCLDE] = None,
) -> None:
    """Write label maps and training configuration next to model artifacts."""
    base_model_name = getattr(model, "base_model_name", args.model_name)
    metadata = {
        "model_type": "podsai_dclde_multitask_ast",
        "base_model": base_model_name,
        "source_model": args.model_name,
        "lr_scheduler_type": args.lr_scheduler_type,
        "kw_label2id": KW_LABELS,
        "kw_id2label": KW_ID2LABEL,
        "species_label2id": SPECIES_LABELS,
        "species_id2label": SPECIES_ID2LABEL,
        "ecotype_label2id": ECOTYPE_LABELS,
        "ecotype_id2label": ECOTYPE_ID2LABEL,
        "ignore_index": IGNORE_INDEX,
        "loss_weights": {
            "kw": args.kw_loss_weight,
            "species": args.species_loss_weight,
            "ecotype": args.ecotype_loss_weight,
        },
        "class_loss_weights": {
            "kw": parse_class_weights(args.kw_class_weights, KW_LABELS, "--kw-class-weights"),
            "species": parse_class_weights(
                args.species_class_weights,
                SPECIES_LABELS,
                "--species-class-weights",
            ),
            "ecotype": parse_class_weights(
                args.ecotype_class_weights,
                ECOTYPE_LABELS,
                "--ecotype-class-weights",
            ),
        },
        "sampling": {
            "domain_balanced": args.domain_balanced_sampling,
            "domain_columns": parse_domain_columns(args.domain_columns),
            "minimum_domain_samples": args.domain_balance_min_samples,
            "strategy": "equal_species_then_equal_domain_with_small_group_floor",
            "samples_per_epoch": "training_manifest_length",
            "replacement": args.domain_balanced_sampling,
        },
        "augmentation": {
            "mean_subtract": args.mean_subtract,
            "high_pass_filter": args.high_pass_filter,
            "high_pass_cutoff_hz": args.high_pass_cutoff_hz,
            "high_pass_order": args.high_pass_order,
            "random_gain": args.random_gain,
            "random_gain_prob": args.random_gain_prob,
            "gain_db": args.gain_db,
            "gain_clipping_mode": args.gain_clipping_mode,
            "time_shift": args.time_shift,
            "time_shift_prob": args.time_shift_prob,
            "max_shift_ms": args.max_shift_ms,
            "time_shift_fade_ms": args.time_shift_fade_ms,
            "gaussian_noise": args.gaussian_noise,
            "gaussian_noise_prob": args.gaussian_noise_prob,
            "noise_std": args.noise_std,
            "noise_scale_mode": args.noise_scale_mode,
            "random_eq": args.random_eq,
            "random_eq_prob": args.random_eq_prob,
            "eq_num_bands_min": args.eq_num_bands_min,
            "eq_num_bands_max": args.eq_num_bands_max,
            "eq_gain_db_min": args.eq_gain_db_min,
            "eq_gain_db_max": args.eq_gain_db_max,
            "eq_q_min": args.eq_q_min,
            "eq_q_max": args.eq_q_max,
            "eq_freq_min": args.eq_freq_min,
            "eq_freq_max": args.eq_freq_max,
        },
        "early_stopping": {
            "enabled": args.early_stopping_patience is not None,
            "patience": args.early_stopping_patience,
            "threshold": args.early_stopping_threshold,
            "metric": "combined_score",
            "greater_is_better": True,
        },
    }
    with (output_dir / "multitask_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def main() -> int:
    """Train the DCLDE multi-task AST model."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a shared AST model for DCLDE KW/species/ecotype tasks."
    )
    parser.add_argument("--train-manifest", required=True, help="Training manifest CSV.")
    parser.add_argument("--val-manifest", required=True, help="Validation manifest CSV.")
    parser.add_argument(
        "--output-dir",
        default="model/dclde_multitask",
        help="Directory to save checkpoints and final artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="davethaler/whale-call-detector",
        help="Existing AST model or Hugging Face model ID to fine-tune.",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "constant_with_warmup", "cosine"),
        default="linear",
        help="Learning-rate scheduler type (default: linear).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for model initialization, dataloader shuffling, and Trainer-controlled "
            "randomness. Change this between fresh one-epoch runs to change the sample order."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20000,
        help="Validate every N optimizer steps (default: 20000).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoints every N optimizer steps (default: same as --eval-steps).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Stop training after this many evaluations without improving combined_score. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum combined_score improvement required to reset early-stopping patience.",
    )
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--save-total-limit", type=int, default=CHECKPOINT_SAVE_LIMIT)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=DEFAULT_DATALOADER_WORKERS,
        help="Worker processes for lazy audio loading during training/eval.",
    )
    parser.add_argument(
        "--preprocessing-workers",
        type=int,
        default=None,
        help="Deprecated; kept for older commands. Lazy preprocessing now uses --dataloader-workers.",
    )
    parser.add_argument(
        "--domain-balanced-sampling",
        action="store_true",
        help=(
            "Enable replacement sampling that gives each species equal expected weight, "
            "then balances recording domains within each species. Disabled by default."
        ),
    )
    parser.add_argument(
        "--domain-columns",
        default="Provider,Dataset",
        help=(
            "Comma-separated training-manifest columns that define a recording domain "
            "when --domain-balanced-sampling is enabled (default: Provider,Dataset)."
        ),
    )
    parser.add_argument(
        "--domain-balance-min-samples",
        type=int,
        default=25,
        help=(
            "Minimum effective size of a species/domain group. This limits repeated "
            "sampling of extremely small domains (default: 25; use 1 for exact balancing)."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kw-loss-weight", type=float, default=1.0)
    parser.add_argument("--species-loss-weight", type=float, default=1.0)
    parser.add_argument("--ecotype-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--kw-class-weights",
        default=None,
        help=(
            "Optional KW head class weights, e.g. 'kw=2.0' or 'not_kw=1,kw=2'. "
            "Unspecified label=value classes default to 1."
        ),
    )
    parser.add_argument(
        "--species-class-weights",
        default=None,
        help=(
            "Optional species head class weights, e.g. 'AB=3.0' or "
            "'background=1,KW=1,HW=1,AB=3'. Unspecified label=value classes default to 1."
        ),
    )
    parser.add_argument(
        "--ecotype-class-weights",
        default=None,
        help=(
            "Optional ecotype head class weights, e.g. 'SRKW=1.5,TKW=1.1'. "
            "Unspecified label=value classes default to 1."
        ),
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--drop-unknown-labels", action="store_true")
    parser.add_argument(
        "--mean-subtract",
        "--mean_subtract",
        action="store_true",
        help=(
            "Subtract each loaded clip's waveform mean before augmentation and "
            "AST feature extraction. Useful for removing DC offset."
        ),
    )
    parser.add_argument(
        "--high-pass-filter",
        "--high_pass_filter",
        action="store_true",
        help=(
            "Apply a Butterworth high-pass filter to each loaded clip before "
            "augmentation and AST feature extraction."
        ),
    )
    parser.add_argument(
        "--high-pass-cutoff-hz",
        "--high_pass_cutoff_hz",
        type=float,
        default=50.0,
        help="High-pass cutoff frequency in Hz when --high-pass-filter is set (default: 50).",
    )
    parser.add_argument(
        "--high-pass-order",
        "--high_pass_order",
        type=int,
        default=4,
        help="Butterworth high-pass filter order when --high-pass-filter is set (default: 4).",
    )
    parser.add_argument("--random-gain", action="store_true")
    parser.add_argument(
        "--random-gain-prob",
        type=float,
        default=1.0,
        help="Probability of applying random gain to each training clip when --random-gain is set.",
    )
    parser.add_argument("--gain-db", type=float, default=6.0)
    parser.add_argument(
        "--gain-clipping-mode",
        choices=("clip", "safe", "normalize", "soft"),
        default="clip",
        help=(
            "How to handle random gain values that exceed [-1, 1]: "
            "'clip' hard-clips, 'safe' caps gain before applying it, "
            "'normalize' rescales only if needed, and 'soft' applies tanh limiting."
        ),
    )
    parser.add_argument("--time-shift", action="store_true")
    parser.add_argument(
        "--time-shift-prob",
        type=float,
        default=1.0,
        help="Probability of applying time shift to each training clip when --time-shift is set.",
    )
    parser.add_argument("--max-shift-ms", type=float, default=250.0)
    parser.add_argument(
        "--time-shift-fade-ms",
        type=float,
        default=0.0,
        help=(
            "Linear fade duration at the zero-padding boundary for time-shift augmentation. "
            "Default 0 disables fading."
        ),
    )
    parser.add_argument("--gaussian-noise", action="store_true")
    parser.add_argument(
        "--gaussian-noise-prob",
        type=float,
        default=1.0,
        help="Probability of adding Gaussian noise to each training clip when --gaussian-noise is set.",
    )
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument(
        "--noise-scale-mode",
        choices=("absolute", "rms"),
        default="absolute",
        help=(
            "How --noise-std is interpreted: 'absolute' uses the value as waveform "
            "standard deviation; 'rms' multiplies it by the clip RMS."
        ),
    )
    parser.add_argument(
        "--random-eq",
        "--random_eq",
        action="store_true",
        help="Apply randomized peaking-EQ bands to training clips.",
    )
    parser.add_argument(
        "--random-eq-prob",
        "--random_eq_prob",
        type=float,
        default=0.5,
        help="Probability of applying random EQ to each training clip when --random-eq is set.",
    )
    parser.add_argument(
        "--eq-num-bands-min",
        "--eq_num_bands_min",
        type=int,
        default=1,
        help="Minimum number of random EQ bands when --random-eq is set.",
    )
    parser.add_argument(
        "--eq-num-bands-max",
        "--eq_num_bands_max",
        type=int,
        default=3,
        help="Maximum number of random EQ bands when --random-eq is set.",
    )
    parser.add_argument(
        "--eq-gain-db-min",
        "--eq_gain_db_min",
        type=float,
        default=-6.0,
        help="Minimum random EQ band gain in dB.",
    )
    parser.add_argument(
        "--eq-gain-db-max",
        "--eq_gain_db_max",
        type=float,
        default=6.0,
        help="Maximum random EQ band gain in dB.",
    )
    parser.add_argument(
        "--eq-q-min",
        "--eq_q_min",
        type=float,
        default=0.4,
        help="Minimum random EQ band Q value.",
    )
    parser.add_argument(
        "--eq-q-max",
        "--eq_q_max",
        type=float,
        default=1.2,
        help="Maximum random EQ band Q value.",
    )
    parser.add_argument(
        "--eq-freq-min",
        "--eq_freq_min",
        type=float,
        default=100.0,
        help="Minimum random EQ band center frequency in Hz.",
    )
    parser.add_argument(
        "--eq-freq-max",
        "--eq_freq_max",
        type=float,
        default=6000.0,
        help="Maximum random EQ band center frequency in Hz.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("COMMAND LINE")
    print("=" * 72)
    print("Raw argv:")
    print("  " + " ".join(sys.argv))
    print("Parsed key settings:")
    print(f"  batch_size={args.batch_size}")
    print(f"  learning_rate={args.learning_rate:.3e}")
    print(f"  lr_scheduler_type={args.lr_scheduler_type}")
    print(f"  warmup_ratio={args.warmup_ratio}")
    print(f"  seed={args.seed}")
    print(f"  epochs={args.epochs}")
    print(f"  output_dir={args.output_dir}")
    print(f"  resume_from_checkpoint={args.resume_from_checkpoint}")
    print(f"  domain_balanced_sampling={args.domain_balanced_sampling}")
    if args.domain_balanced_sampling:
        print(f"  domain_columns={args.domain_columns}")
        print(f"  domain_balance_min_samples={args.domain_balance_min_samples}")
    print("=" * 72 + "\n")

    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.random_gain_prob = validate_probability(args.random_gain_prob, "--random-gain-prob")
    args.time_shift_prob = validate_probability(args.time_shift_prob, "--time-shift-prob")
    args.gaussian_noise_prob = validate_probability(
        args.gaussian_noise_prob,
        "--gaussian-noise-prob",
    )
    args.random_eq_prob = validate_probability(args.random_eq_prob, "--random-eq-prob")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1 when provided")
    if args.domain_balance_min_samples < 1:
        raise ValueError("--domain-balance-min-samples must be at least 1")
    if args.time_shift_fade_ms < 0:
        raise ValueError(f"--time-shift-fade-ms must be non-negative, got {args.time_shift_fade_ms}")
    if args.eq_num_bands_min < 1 or args.eq_num_bands_max < 1:
        raise ValueError("--eq-num-bands-min and --eq-num-bands-max must be positive")
    if args.eq_num_bands_min > args.eq_num_bands_max:
        raise ValueError(
            "--eq-num-bands-min must be less than or equal to --eq-num-bands-max, "
            f"got {args.eq_num_bands_min} > {args.eq_num_bands_max}"
        )
    if args.eq_gain_db_min > args.eq_gain_db_max:
        raise ValueError(
            "--eq-gain-db-min must be less than or equal to --eq-gain-db-max, "
            f"got {args.eq_gain_db_min} > {args.eq_gain_db_max}"
        )
    if args.eq_q_min <= 0.0 or args.eq_q_max <= 0.0:
        raise ValueError("--eq-q-min and --eq-q-max must be positive")
    if args.eq_q_min > args.eq_q_max:
        raise ValueError(
            "--eq-q-min must be less than or equal to --eq-q-max, "
            f"got {args.eq_q_min} > {args.eq_q_max}"
        )
    nyquist = SAMPLE_RATE / 2.0
    if args.eq_freq_min <= 0.0 or args.eq_freq_max <= 0.0:
        raise ValueError("--eq-freq-min and --eq-freq-max must be positive")
    if args.eq_freq_min > args.eq_freq_max:
        raise ValueError(
            "--eq-freq-min must be less than or equal to --eq-freq-max, "
            f"got {args.eq_freq_min} > {args.eq_freq_max}"
        )
    if args.eq_freq_max >= nyquist:
        raise ValueError(
            f"--eq-freq-max must be below Nyquist ({nyquist:g} Hz), got {args.eq_freq_max}"
        )
    if args.high_pass_filter:
        if not 0.0 < args.high_pass_cutoff_hz < nyquist:
            raise ValueError(
                f"--high-pass-cutoff-hz must be between 0 and {nyquist}, "
                f"got {args.high_pass_cutoff_hz}"
            )
        if args.high_pass_order < 1:
            raise ValueError(f"--high-pass-order must be positive, got {args.high_pass_order}")

    kw_class_weights = parse_class_weights(args.kw_class_weights, KW_LABELS, "--kw-class-weights")
    species_class_weights = parse_class_weights(
        args.species_class_weights,
        SPECIES_LABELS,
        "--species-class-weights",
    )
    ecotype_class_weights = parse_class_weights(
        args.ecotype_class_weights,
        ECOTYPE_LABELS,
        "--ecotype-class-weights",
    )
    if kw_class_weights is not None:
        print(f"KW class loss weights: {kw_class_weights}")
    if species_class_weights is not None:
        print(f"Species class loss weights: {species_class_weights}")
    if ecotype_class_weights is not None:
        print(f"Ecotype class loss weights: {ecotype_class_weights}")

    print("Loading manifests...")
    domain_columns = parse_domain_columns(args.domain_columns)
    train_dataset = load_manifest(
        args.train_manifest,
        drop_unknown_labels=args.drop_unknown_labels,
        domain_columns=domain_columns if args.domain_balanced_sampling else None,
    )
    validation_dataset = load_manifest(args.val_manifest, drop_unknown_labels=args.drop_unknown_labels)
    dataset = {"train": train_dataset, "validation": validation_dataset}
    analyze_dataset(dataset)
    train_sample_weights = None
    if args.domain_balanced_sampling:
        train_sample_weights = domain_balanced_sample_weights(
            train_dataset,
            minimum_domain_samples=args.domain_balance_min_samples,
        )
        analyze_domain_balancing(
            train_dataset,
            train_sample_weights,
            minimum_domain_samples=args.domain_balance_min_samples,
        )

    print(f"Loading feature extractor: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    augmenter = None
    if args.random_gain or args.time_shift or args.gaussian_noise or args.random_eq:
        augmenter = WaveformAugmenter(
            sample_rate=SAMPLE_RATE,
            random_gain=args.random_gain,
            random_gain_prob=args.random_gain_prob,
            gain_db=args.gain_db,
            gain_clipping_mode=args.gain_clipping_mode,
            time_shift=args.time_shift,
            time_shift_prob=args.time_shift_prob,
            max_shift_ms=args.max_shift_ms,
            time_shift_fade_ms=args.time_shift_fade_ms,
            gaussian_noise=args.gaussian_noise,
            gaussian_noise_prob=args.gaussian_noise_prob,
            noise_std=args.noise_std,
            noise_scale_mode=args.noise_scale_mode,
            random_eq=args.random_eq,
            random_eq_prob=args.random_eq_prob,
            eq_num_bands_min=args.eq_num_bands_min,
            eq_num_bands_max=args.eq_num_bands_max,
            eq_gain_db_min=args.eq_gain_db_min,
            eq_gain_db_max=args.eq_gain_db_max,
            eq_q_min=args.eq_q_min,
            eq_q_max=args.eq_q_max,
            eq_freq_min=args.eq_freq_min,
            eq_freq_max=args.eq_freq_max,
        )
        print(
            "Training augmentation probabilities: "
            f"random_gain={args.random_gain_prob if args.random_gain else 0.0}, "
            f"time_shift={args.time_shift_prob if args.time_shift else 0.0}, "
            f"gaussian_noise={args.gaussian_noise_prob if args.gaussian_noise else 0.0}, "
            f"random_eq={args.random_eq_prob if args.random_eq else 0.0}; "
            f"gain_clipping_mode={args.gain_clipping_mode}; "
            f"noise_scale_mode={args.noise_scale_mode}; "
            f"time_shift_fade_ms={args.time_shift_fade_ms}"
        )
        if args.random_eq:
            print(
                "Random EQ settings: "
                f"bands={args.eq_num_bands_min}-{args.eq_num_bands_max}, "
                f"gain_db={args.eq_gain_db_min:g}..{args.eq_gain_db_max:g}, "
                f"q={args.eq_q_min:g}..{args.eq_q_max:g}, "
                f"freq_hz={args.eq_freq_min:g}..{args.eq_freq_max:g}"
            )

    if args.preprocessing_workers is not None:
        print("--preprocessing-workers is deprecated and ignored; use --dataloader-workers.")
    print(f"Waveform mean subtraction: {'enabled' if args.mean_subtract else 'disabled'}")
    if args.high_pass_filter:
        print(
            "Waveform high-pass filter: "
            f"enabled, cutoff={args.high_pass_cutoff_hz:g} Hz, order={args.high_pass_order}"
        )
    else:
        print("Waveform high-pass filter: disabled")
    print(
        "Using lazy audio preprocessing: features are generated per batch "
        f"with {args.dataloader_workers} dataloader worker(s)."
    )
    train_collator = DCLDEAudioCollator(
        feature_extractor=feature_extractor,
        max_duration=args.max_duration,
        augmenter=augmenter,
        mean_subtract=args.mean_subtract,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz if args.high_pass_filter else None,
        high_pass_order=args.high_pass_order,
    )
    eval_collator = DCLDEAudioCollator(
        feature_extractor=feature_extractor,
        max_duration=args.max_duration,
        augmenter=None,
        mean_subtract=args.mean_subtract,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz if args.high_pass_filter else None,
        high_pass_order=args.high_pass_order,
    )

    print(f"Loading multi-task AST model from {args.model_name}")
    model = load_training_model(
        model_name=args.model_name,
        dropout=args.dropout,
        kw_loss_weight=args.kw_loss_weight,
        species_loss_weight=args.species_loss_weight,
        ecotype_loss_weight=args.ecotype_loss_weight,
        freeze_backbone=args.freeze_backbone,
        kw_class_weights=kw_class_weights,
        species_class_weights=species_class_weights,
        ecotype_class_weights=ecotype_class_weights,
    )

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.seed,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        label_names=["kw_labels", "species_labels", "ecotype_labels"],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        push_to_hub=args.push_to_hub,
        hub_strategy="all_checkpoints" if args.push_to_hub else "end",
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )
    print_training_geometry(
        training_args=training_args,
        train_rows=len(dataset["train"]),
        validation_rows=len(dataset["validation"]),
    )

    callbacks = []
    if args.early_stopping_patience is not None:
        print(
            "Early stopping enabled: "
            f"patience={args.early_stopping_patience} evaluations, "
            f"threshold={args.early_stopping_threshold}, metric=combined_score"
        )
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = LazyAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        train_collator=train_collator,
        eval_collator=eval_collator,
        train_sample_weights=train_sample_weights,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Evaluating selected best model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    print(f"Saving final model artifacts to {output_dir}")
    trainer.save_model(str(output_dir))
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    feature_extractor.save_pretrained(str(output_dir))
    save_metadata(output_dir, args, model)
    save_training_curves(trainer, output_dir)

    if args.push_to_hub:
        print(f"Pushing feature extractor to Hugging Face Hub: {args.hub_model_id}")
        feature_extractor.push_to_hub(args.hub_model_id)

    print("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
