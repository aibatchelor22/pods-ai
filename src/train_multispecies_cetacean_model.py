#!/usr/bin/env python3
"""Train the Multispecies Cetacean model with an optionally frozen AST.

This first-stage trainer discovers builder-created shard manifests below one or
more Kaggle input roots, reads individual FLAC members directly from each
``clips.zip`` archive, caches one pooled AST embedding per clip, and trains:

* a binary known-whale trigger (Abiotic=negative; KW/HW=positive),
* a four-class source head (Abiotic, KW, HW, UndBio), and
* a five-class killer-whale ecotype head.

UndBio rows are ignored by the binary trigger by default. The source head sees
all four classes. Ecotype loss is computed only for eligible KW annotations.
Use ``--freeze-backbone`` for the first-stage cached-embedding head baseline.
Without it, the script fine-tunes the AST and heads together. Checkpoint
loading, legacy AST key compatibility, and strict backbone validation are
implemented here so the script has no project-local Python dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_scheduler,
)


SAMPLE_RATE = 16_000
IGNORE_INDEX = -100
TRIGGER_LABELS = {"not_whale": 0, "known_whale": 1}
SOURCE_LABELS = {"Abiotic": 0, "KW": 1, "HW": 2, "UndBio": 3}
ECOTYPE_LABELS = {"NRKW": 0, "SRKW": 1, "OKW": 2, "SAR": 3, "TKW": 4}
TRIGGER_ID2LABEL = {value: key for key, value in TRIGGER_LABELS.items()}
SOURCE_ID2LABEL = {value: key for key, value in SOURCE_LABELS.items()}
ECOTYPE_ID2LABEL = {value: key for key, value in ECOTYPE_LABELS.items()}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "n/a"} else text


def bool_value(value: Any) -> bool:
    return clean(value).casefold() in {"1", "true", "yes", "y"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class ArchiveManifestDataset(Dataset):
    """Manifest metadata for clips stored as members of shard ZIP archives."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ArchiveAudioCollator:
    """Load only requested ZIP members and convert waveforms to AST features."""

    def __init__(
        self,
        feature_extractor: Any,
        clip_seconds: float,
        mean_subtract: bool,
        high_pass_cutoff_hz: float | None,
        high_pass_order: int,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.target_samples = round(clip_seconds * SAMPLE_RATE)
        self.mean_subtract = mean_subtract
        self.handles: dict[str, zipfile.ZipFile] = {}
        self.high_pass_sos: np.ndarray | None = None
        if high_pass_cutoff_hz is not None:
            if not 0 < high_pass_cutoff_hz < SAMPLE_RATE / 2:
                raise ValueError("high-pass cutoff must lie between 0 and Nyquist")
            self.high_pass_sos = butter(
                high_pass_order,
                high_pass_cutoff_hz,
                btype="highpass",
                fs=SAMPLE_RATE,
                output="sos",
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["handles"] = {}
        return state

    def _read(self, row: dict[str, Any]) -> np.ndarray:
        audio_file = row.get("audio_file", "")
        member_path = row["archive_member_path"]
        if audio_file:
            try:
                audio, rate = sf.read(audio_file, dtype="float32", always_2d=False)
            except Exception as exc:
                raise RuntimeError(f"Could not read audio file: {audio_file}") from exc
            source_description = audio_file
        else:
            archive_path = row["archive_file"]
            handle = self.handles.get(archive_path)
            if handle is None:
                handle = zipfile.ZipFile(archive_path, "r")
                self.handles[archive_path] = handle
            try:
                with handle.open(member_path, "r") as member:
                    audio_bytes = member.read()
            except Exception as exc:
                raise RuntimeError(f"Could not read {member_path!r} from {archive_path}") from exc
            audio, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            source_description = f"{archive_path}::{member_path}"
        if rate != SAMPLE_RATE:
            raise ValueError(f"{source_description} is {rate} Hz, expected {SAMPLE_RATE}")
        if audio.ndim == 2:
            audio = audio[:, 0]
        if len(audio) > self.target_samples:
            audio = audio[: self.target_samples]
        elif len(audio) < self.target_samples:
            audio = np.pad(audio, (0, self.target_samples - len(audio)))
        audio = audio.astype(np.float32, copy=False)
        if self.mean_subtract:
            audio = audio - float(audio.mean())
        if self.high_pass_sos is not None:
            try:
                audio = sosfiltfilt(self.high_pass_sos, audio).astype(np.float32)
            except ValueError:
                audio = sosfilt(self.high_pass_sos, audio).astype(np.float32)
        return audio

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        audio = [self._read(row) for row in rows]
        features = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        features["trigger_labels"] = torch.tensor(
            [row["trigger_label"] for row in rows], dtype=torch.long
        )
        features["source_labels"] = torch.tensor(
            [row["source_label"] for row in rows], dtype=torch.long
        )
        features["ecotype_labels"] = torch.tensor(
            [row["ecotype_label"] for row in rows], dtype=torch.long
        )
        features["clip_ids"] = [row["clip_id"] for row in rows]
        return features


class FrozenMultispeciesHeads(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.trigger_classifier = nn.Linear(hidden_size, len(TRIGGER_LABELS))
        self.source_classifier = nn.Linear(hidden_size, len(SOURCE_LABELS))
        self.ecotype_classifier = nn.Linear(hidden_size, len(ECOTYPE_LABELS))

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self.dropout(embeddings)
        return (
            self.trigger_classifier(values),
            self.source_classifier(values),
            self.ecotype_classifier(values),
        )


class MultispeciesCetaceanModel(nn.Module):
    """AST backbone plus the three challenge heads."""

    def __init__(self, base_model: nn.Module, dropout: float) -> None:
        super().__init__()
        self.config = base_model.config
        self.ast = base_model.audio_spectrogram_transformer
        hidden_size = int(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.trigger_classifier = nn.Linear(hidden_size, len(TRIGGER_LABELS))
        self.source_classifier = nn.Linear(hidden_size, len(SOURCE_LABELS))
        self.ecotype_classifier = nn.Linear(hidden_size, len(ECOTYPE_LABELS))

    @staticmethod
    def pool(sequence: torch.Tensor) -> torch.Tensor:
        return (
            (sequence[:, 0] + sequence[:, 1]) / 2.0
            if sequence.shape[1] >= 2
            else sequence[:, 0]
        )

    def pooled_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        output = self.ast(input_values=input_values, return_dict=True)
        return self.pool(output.last_hidden_state)

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self.dropout(self.pooled_embeddings(input_values))
        return (
            self.trigger_classifier(values),
            self.source_classifier(values),
            self.ecotype_classifier(values),
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


def remap_legacy_keys(
    state: dict[str, torch.Tensor], expected_keys: Collection[str]
) -> tuple[dict[str, torch.Tensor], int]:
    expected = set(expected_keys)
    result: dict[str, torch.Tensor] = {}
    changed = 0
    for original, value in state.items():
        output = original
        if original not in expected:
            candidate = original
            for old, new in LEGACY_AST_KEY_REPLACEMENTS:
                candidate = candidate.replace(old, new)
            candidates = (
                candidate,
                candidate.replace("ast.encoder.layers.", "ast.layers."),
            )
            output = next((item for item in candidates if item in expected), original)
        if output in result:
            raise ValueError(f"Checkpoint key collision after compatibility remap: {output}")
        result[output] = value
        changed += int(output != original)
    return result, changed


def find_local_weights(directory: Path) -> Path | None:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def download_weights(model_name: str) -> Path:
    errors = []
    for name in ("model.safetensors", "pytorch_model.bin"):
        try:
            return Path(hf_hub_download(model_name, name))
        except EntryNotFoundError:
            errors.append(f"{name}: absent")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError(f"No supported checkpoint weights in {model_name}: {'; '.join(errors)}")


def checkpoint_files(model_name: str) -> tuple[dict[str, Any], Path, str] | None:
    local = Path(model_name)
    config_names = (
        ("multispecies_cetacean_config.json", "new"),
        ("multitask_config.json", "legacy"),
    )
    if local.is_dir():
        weights = find_local_weights(local)
        for name, kind in config_names:
            path = local / name
            if path.is_file() and weights is not None:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle), weights, kind
        return None
    for name, kind in config_names:
        try:
            config_path = Path(hf_hub_download(model_name, name))
        except Exception:
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), download_weights(model_name), kind
    return None


def read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.name.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load this checkpoint") from exc
        state: Any = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint did not contain a state dictionary: {path}")
    return state


def load_model(
    model_name: str, dropout: float, freeze_backbone: bool
) -> tuple[MultispeciesCetaceanModel, dict[str, Any], str]:
    checkpoint = checkpoint_files(model_name)
    if checkpoint is None:
        base = AutoModelForAudioClassification.from_pretrained(model_name)
        model = MultispeciesCetaceanModel(base, dropout)
        feature_source = model_name
        identity = {"model_name": model_name, "weights": None, "kind": "base_ast"}
    else:
        metadata, weights_path, kind = checkpoint
        standalone = bool(metadata.get("standalone", False)) or kind == "new"
        architecture_source = model_name if standalone else clean(metadata.get("base_model"))
        if not architecture_source:
            raise ValueError("Checkpoint metadata is missing base_model")
        if standalone:
            base = AutoModelForAudioClassification.from_config(
                AutoConfig.from_pretrained(architecture_source)
            )
        else:
            base = AutoModelForAudioClassification.from_pretrained(architecture_source)
        model = MultispeciesCetaceanModel(base, dropout)
        state = read_state_dict(weights_path)
        state, remapped = remap_legacy_keys(state, model.state_dict().keys())
        if remapped:
            print(f"Remapped {remapped} legacy AST parameter names")
        if kind == "new":
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "New-format checkpoint is incomplete or incompatible; "
                    f"missing={missing[:12]}, unexpected={unexpected[:12]}"
                )
        else:
            expected_ast = {key for key in model.state_dict() if key.startswith("ast.")}
            checkpoint_ast = {key for key in state if key.startswith("ast.")}
            missing_ast = sorted(expected_ast.difference(checkpoint_ast))
            unexpected_ast = sorted(checkpoint_ast.difference(expected_ast))
            shape_errors = [
                key
                for key in expected_ast & checkpoint_ast
                if state[key].shape != model.state_dict()[key].shape
            ]
            if missing_ast or unexpected_ast or shape_errors:
                raise RuntimeError(
                    "Refusing partial AST load; "
                    f"missing={missing_ast[:8]}, unexpected={unexpected_ast[:8]}, "
                    f"shape_mismatch={shape_errors[:8]}"
                )
            model.load_state_dict(
                {key: state[key] for key in expected_ast}, strict=False
            )
            ecotype_keys = {
                key: value
                for key, value in state.items()
                if key.startswith("ecotype_classifier.")
                and key in model.state_dict()
                and value.shape == model.state_dict()[key].shape
            }
            if len(ecotype_keys) == 2:
                model.load_state_dict(ecotype_keys, strict=False)
                print("Transferred the compatible ecotype head")
            print("Initialized new trigger and four-class source heads")
        feature_source = model_name if standalone else architecture_source
        identity = {
            "model_name": model_name,
            "weights": file_identity(weights_path),
            "kind": kind,
        }
    for parameter in model.ast.parameters():
        parameter.requires_grad = not freeze_backbone
    return model, identity, feature_source


def parse_class_weights(
    value: str | None, label2id: dict[str, int], argument_name: str
) -> list[float] | None:
    if value is None or not value.strip():
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    weights = [1.0] * len(label2id)
    if all("=" not in item for item in parts):
        if len(parts) != len(label2id):
            raise ValueError(f"{argument_name} requires {len(label2id)} weights")
        weights = [float(item) for item in parts]
    else:
        for item in parts:
            if "=" not in item:
                raise ValueError(f"Malformed {argument_name} item: {item!r}")
            label, number = (part.strip() for part in item.split("=", 1))
            if label not in label2id:
                raise ValueError(f"Unknown {argument_name} label: {label!r}")
            weights[label2id[label]] = float(number)
    if any(number < 0 for number in weights):
        raise ValueError(f"{argument_name} cannot contain negative weights")
    return weights


def trigger_label(source: str, undbio_policy: str) -> int:
    if source == "Abiotic":
        return TRIGGER_LABELS["not_whale"]
    if source in {"KW", "HW"}:
        return TRIGGER_LABELS["known_whale"]
    if source == "UndBio":
        if undbio_policy == "ignore":
            return IGNORE_INDEX
        return TRIGGER_LABELS["known_whale" if undbio_policy == "positive" else "not_whale"]
    raise ValueError(f"Unknown source label: {source!r}")


def discover_rows(
    roots: list[Path],
    split: str,
    manifest_name: str,
    undbio_policy: str,
) -> tuple[list[dict[str, Any]], list[Path], list[Path]]:
    manifests = sorted({path.resolve() for root in roots for path in root.rglob(manifest_name)})
    if not manifests:
        raise FileNotFoundError(f"No {manifest_name} files found below: {roots}")
    rows: list[dict[str, Any]] = []
    used_manifests: list[Path] = []
    archives: set[Path] = set()
    clip_ids: set[str] = set()
    required = {
        "clip_id",
        "split",
        "model_source_label",
        "clean_ecotype",
        "archive_path",
        "archive_member_path",
    }
    for manifest in manifests:
        used = False
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{manifest} missing required columns: {sorted(missing)}")
            for raw in reader:
                if clean(raw.get("split")).casefold() != split.casefold():
                    continue
                if "source_head_eligible" in raw and not bool_value(raw["source_head_eligible"]):
                    continue
                source = clean(raw["model_source_label"])
                if source not in SOURCE_LABELS:
                    raise ValueError(f"{manifest}: unknown model_source_label {source!r}")
                clip_id = clean(raw["clip_id"])
                if not clip_id or clip_id in clip_ids:
                    raise ValueError(f"Missing or duplicate clip_id: {clip_id!r}")
                clip_ids.add(clip_id)
                archive = (manifest.parent / clean(raw["archive_path"])).resolve()
                member_path = clean(raw["archive_member_path"]).replace("\\", "/")
                audio_file = ""
                archive_file = ""
                if archive.is_file():
                    archive_file = str(archive)
                    archives.add(archive)
                else:
                    # Kaggle may expose an uploaded ZIP as a directory named
                    # after the archive stem instead of retaining clips.zip.
                    directory_candidates = [archive, archive.with_suffix("")]
                    direct = next(
                        (
                            candidate.joinpath(*Path(member_path).parts)
                            for candidate in directory_candidates
                            if candidate.joinpath(*Path(member_path).parts).is_file()
                        ),
                        None,
                    )
                    if direct is None:
                        raise FileNotFoundError(
                            f"Neither archive nor expanded member was found for "
                            f"{manifest}: {archive} / {member_path}"
                        )
                    audio_file = str(direct.resolve())
                    archives.add(direct.parents[len(Path(member_path).parts) - 1])
                ecotype = clean(raw["clean_ecotype"])
                ecotype_eligible = (
                    source == "KW"
                    and ecotype in ECOTYPE_LABELS
                    and ("ecotype_head_eligible" not in raw or bool_value(raw["ecotype_head_eligible"]))
                )
                rows.append(
                    {
                        "clip_id": clip_id,
                        "archive_file": archive_file,
                        "audio_file": audio_file,
                        "archive_member_path": member_path,
                        "trigger_label": trigger_label(source, undbio_policy),
                        "source_label": SOURCE_LABELS[source],
                        "ecotype_label": ECOTYPE_LABELS[ecotype] if ecotype_eligible else IGNORE_INDEX,
                    }
                )
                used = True
        if used:
            used_manifests.append(manifest)
    if not rows:
        raise ValueError(f"No eligible rows found for split {split!r}")
    return rows, used_manifests, sorted(archives)


def random_subset(rows: list[dict[str, Any]], maximum: int | None, seed: int) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(rows):
        return rows
    if maximum < 1:
        raise ValueError("maximum clip counts must be positive")
    indices = np.random.default_rng(seed).choice(len(rows), size=maximum, replace=False)
    return [rows[index] for index in sorted(indices.tolist())]


def preprocessing_from_checkpoint(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoint_files(args.model_name)
    metadata = checkpoint[0] if checkpoint is not None else {}
    augmentation = metadata.get("preprocessing", metadata.get("augmentation", {}))
    settings = {
        "mean_subtract": bool(augmentation.get("mean_subtract", False)),
        "high_pass_filter": bool(augmentation.get("high_pass_filter", False)),
        "high_pass_cutoff_hz": float(augmentation.get("high_pass_cutoff_hz", 50.0)),
        "high_pass_order": int(augmentation.get("high_pass_order", 4)),
        "clip_seconds": args.clip_seconds,
    }
    if args.mean_subtract is not None:
        settings["mean_subtract"] = args.mean_subtract
    if args.high_pass_filter is not None:
        settings["high_pass_filter"] = args.high_pass_filter
    if args.high_pass_cutoff_hz is not None:
        settings["high_pass_cutoff_hz"] = args.high_pass_cutoff_hz
    if args.high_pass_order is not None:
        settings["high_pass_order"] = args.high_pass_order
    identity = {
        "model_name": args.model_name,
        "weights": file_identity(checkpoint[1]) if checkpoint is not None else None,
        "kind": checkpoint[2] if checkpoint is not None else "base_ast",
    }
    return settings, identity


def cache_signature(
    split: str,
    manifests: list[Path],
    archives: list[Path],
    checkpoint: dict[str, Any],
    preprocessing: dict[str, Any],
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "manifests": [file_identity(path) for path in manifests],
        "archives": [file_identity(path) for path in archives],
        "checkpoint": checkpoint,
        "preprocessing": preprocessing,
        "rows": len(rows),
        "clip_id_digest": hashlib.sha256("\n".join(row["clip_id"] for row in rows).encode()).hexdigest(),
        "seed": seed,
    }


def cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{split}_embeddings.npy",
        cache_dir / f"{split}_labels.npz",
        cache_dir / f"{split}_cache.json",
    )


def load_cache(
    cache_dir: Path, split: str, signature: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    embedding_path, label_path, metadata_path = cache_paths(cache_dir, split)
    if not all(path.exists() for path in (embedding_path, label_path, metadata_path)):
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        if json.load(handle).get("signature") != signature:
            return None
    embeddings = np.load(embedding_path, mmap_mode="r")
    label_file = np.load(label_path, allow_pickle=False)
    labels = {name: label_file[name] for name in label_file.files}
    if len(embeddings) != len(labels["source"]):
        return None
    print(f"Reusing {split} embedding cache: {len(embeddings):,} clips")
    return embeddings, labels


def extract_embeddings(
    split: str,
    dataset: ArchiveManifestDataset,
    model: Any,
    collator: ArchiveAudioCollator,
    device: torch.device,
    cache_dir: Path,
    signature: dict[str, Any],
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    embedding_path, label_path, metadata_path = cache_paths(cache_dir, split)
    hidden_size = int(model.config.hidden_size)
    embeddings = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32, shape=(len(dataset), hidden_size)
    )
    trigger = np.empty(len(dataset), dtype=np.int64)
    source = np.empty(len(dataset), dtype=np.int64)
    ecotype = np.empty(len(dataset), dtype=np.int64)
    clip_ids: list[str] = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    offset = 0
    model.eval()
    print(f"Extracting {split} embeddings: {len(dataset):,} clips")
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            values = batch["input_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model.ast(input_values=values, return_dict=True)
                pooled = model.pool(output.last_hidden_state)
            count = len(values)
            embeddings[offset : offset + count] = pooled.float().cpu().numpy()
            trigger[offset : offset + count] = batch["trigger_labels"].numpy()
            source[offset : offset + count] = batch["source_labels"].numpy()
            ecotype[offset : offset + count] = batch["ecotype_labels"].numpy()
            clip_ids.extend(batch["clip_ids"])
            offset += count
            if batch_index % 250 == 0 or offset == len(dataset):
                print(f"  {split}: {offset:,}/{len(dataset):,}")
    embeddings.flush()
    np.savez_compressed(
        label_path,
        trigger=trigger,
        source=source,
        ecotype=ecotype,
        clip_id=np.asarray(clip_ids, dtype=str),
    )
    atomic_json(
        metadata_path,
        {"signature": signature, "rows": len(dataset), "hidden_size": hidden_size},
    )
    return np.load(embedding_path, mmap_mode="r"), {
        "trigger": trigger,
        "source": source,
        "ecotype": ecotype,
        "clip_id": np.asarray(clip_ids, dtype=str),
    }


def f1(true: np.ndarray, predicted: np.ndarray, average: str, **kwargs: Any) -> float:
    return float(
        precision_recall_fscore_support(
            true, predicted, average=average, zero_division=0, **kwargs
        )[2]
    )


def class_f1(true: np.ndarray, predicted: np.ndarray, class_id: int) -> float:
    return float(
        precision_recall_fscore_support(
            true, predicted, labels=[class_id], average=None, zero_division=0
        )[2][0]
    )


def metrics_for_predictions(
    labels: dict[str, np.ndarray], predictions: dict[str, np.ndarray]
) -> dict[str, float]:
    result: dict[str, float] = {}
    trigger_mask = labels["trigger"] != IGNORE_INDEX
    result["trigger_accuracy"] = float(
        accuracy_score(labels["trigger"][trigger_mask], predictions["trigger"][trigger_mask])
    )
    result["trigger_f1"] = f1(
        labels["trigger"][trigger_mask],
        predictions["trigger"][trigger_mask],
        "binary",
        pos_label=TRIGGER_LABELS["known_whale"],
    )
    result["source_accuracy"] = float(accuracy_score(labels["source"], predictions["source"]))
    result["source_macro_f1"] = f1(labels["source"], predictions["source"], "macro")
    for class_id, name in SOURCE_ID2LABEL.items():
        result[f"source_f1_{name}"] = class_f1(labels["source"], predictions["source"], class_id)
    ecotype_mask = labels["ecotype"] != IGNORE_INDEX
    if np.any(ecotype_mask):
        true = labels["ecotype"][ecotype_mask]
        pred = predictions["ecotype"][ecotype_mask]
        result["ecotype_accuracy"] = float(accuracy_score(true, pred))
        result["ecotype_macro_f1"] = f1(true, pred, "macro")
        for class_id, name in ECOTYPE_ID2LABEL.items():
            result[f"ecotype_f1_{name}"] = class_f1(true, pred, class_id)
    else:
        result["ecotype_accuracy"] = 0.0
        result["ecotype_macro_f1"] = 0.0
    result["combined_score"] = (
        0.3 * result["trigger_f1"]
        + 0.4 * result["source_macro_f1"]
        + 0.3 * result["ecotype_macro_f1"]
    )
    return result


def predict(
    head: FrozenMultispeciesHeads,
    embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(embeddings))), batch_size=batch_size
    )
    parts: list[list[np.ndarray]] = [[], [], []]
    head.eval()
    with torch.inference_mode():
        for (values,) in loader:
            outputs = head(values.to(device, non_blocking=True))
            for index, output in enumerate(outputs):
                parts[index].append(output.cpu().numpy())
    logits = [np.concatenate(group) for group in parts]
    predictions = {
        "trigger": logits[0].argmax(axis=1),
        "source": logits[1].argmax(axis=1),
        "ecotype": logits[2].argmax(axis=1),
        "trigger_logits": logits[0],
        "source_logits": logits[1],
        "ecotype_logits": logits[2],
    }
    return metrics_for_predictions(labels, predictions), predictions


def automatic_weights(values: np.ndarray, classes: int, ignore_index: int | None = None) -> list[float]:
    if ignore_index is not None:
        values = values[values != ignore_index]
    counts = np.bincount(values, minlength=classes)
    if np.any(counts == 0):
        raise ValueError(f"All classes must occur when automatic weights are enabled: {counts.tolist()}")
    weights = len(values) / (classes * counts.astype(np.float64))
    return weights.tolist()


def weight_tensor(values: list[float] | None, device: torch.device) -> torch.Tensor | None:
    return None if values is None else torch.tensor(values, dtype=torch.float32, device=device)


def train_head(
    args: argparse.Namespace,
    head: FrozenMultispeciesHeads,
    train_embeddings: np.ndarray,
    train_labels: dict[str, np.ndarray],
    val_embeddings: np.ndarray,
    val_labels: dict[str, np.ndarray],
    weights: dict[str, list[float] | None],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_embeddings)),
        torch.from_numpy(train_labels["trigger"]),
        torch.from_numpy(train_labels["source"]),
        torch.from_numpy(train_labels["ecotype"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    head.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(head.trigger_classifier.parameters())
                + list(head.source_classifier.parameters()),
                "lr": args.learning_rate,
            },
            {"params": head.ecotype_classifier.parameters(), "lr": args.ecotype_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    tensors = {name: weight_tensor(value, device) for name, value in weights.items()}
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] = {}
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        examples = 0
        for features, trigger, source, ecotype in loader:
            features = features.to(device, non_blocking=True)
            trigger = trigger.to(device, non_blocking=True)
            source = source.to(device, non_blocking=True)
            ecotype = ecotype.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            trigger_logits, source_logits, ecotype_logits = head(features)
            losses = [
                args.source_loss_weight
                * nn.functional.cross_entropy(source_logits, source, weight=tensors["source"])
            ]
            trigger_mask = trigger != IGNORE_INDEX
            if torch.any(trigger_mask):
                losses.append(
                    args.trigger_loss_weight
                    * nn.functional.cross_entropy(
                        trigger_logits[trigger_mask], trigger[trigger_mask], weight=tensors["trigger"]
                    )
                )
            ecotype_mask = ecotype != IGNORE_INDEX
            if torch.any(ecotype_mask):
                losses.append(
                    args.ecotype_loss_weight
                    * nn.functional.cross_entropy(
                        ecotype_logits[ecotype_mask], ecotype[ecotype_mask], weight=tensors["ecotype"]
                    )
                )
            loss = torch.stack(losses).sum()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            examples += len(features)
        metrics, _ = predict(
            head, val_embeddings, val_labels, args.head_batch_size, device
        )
        row = {"epoch": float(epoch), "train_loss": total_loss / max(examples, 1), **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:02d}: loss={row['train_loss']:.5f}, "
            f"trigger_f1={metrics['trigger_f1']:.4f}, "
            f"source_macro_f1={metrics['source_macro_f1']:.4f}, "
            f"ecotype_macro_f1={metrics['ecotype_macro_f1']:.4f}, "
            f"combined={metrics['combined_score']:.4f}"
        )
        if metrics["combined_score"] > best_score + 1e-8:
            best_score = metrics["combined_score"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"Early stopping after {stale} non-improving epochs")
                break
    if not best_state:
        raise RuntimeError("No best head checkpoint was selected")
    return best_state, pd.DataFrame(history)


def multitask_loss(
    logits: tuple[torch.Tensor, ...],
    trigger: torch.Tensor,
    source: torch.Tensor,
    ecotype: torch.Tensor,
    weights: dict[str, torch.Tensor | None],
    args: argparse.Namespace,
) -> torch.Tensor:
    trigger_logits, source_logits, ecotype_logits = logits
    losses = [
        args.source_loss_weight
        * nn.functional.cross_entropy(source_logits, source, weight=weights["source"])
    ]
    trigger_mask = trigger != IGNORE_INDEX
    if torch.any(trigger_mask):
        losses.append(
            args.trigger_loss_weight
            * nn.functional.cross_entropy(
                trigger_logits[trigger_mask],
                trigger[trigger_mask],
                weight=weights["trigger"],
            )
        )
    ecotype_mask = ecotype != IGNORE_INDEX
    if torch.any(ecotype_mask):
        losses.append(
            args.ecotype_loss_weight
            * nn.functional.cross_entropy(
                ecotype_logits[ecotype_mask],
                ecotype[ecotype_mask],
                weight=weights["ecotype"],
            )
        )
    return torch.stack(losses).sum()


def evaluate_audio_model(
    model: nn.Module,
    dataset: ArchiveManifestDataset,
    collator: ArchiveAudioCollator,
    batch_size: int,
    workers: int,
    device: torch.device,
    weight_values: dict[str, list[float] | None],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    weights = {name: weight_tensor(value, device) for name, value in weight_values.items()}
    logit_parts: list[list[np.ndarray]] = [[], [], []]
    label_parts: dict[str, list[np.ndarray]] = {"trigger": [], "source": [], "ecotype": []}
    clip_ids: list[str] = []
    total_loss = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            values = batch["input_values"].to(device, non_blocking=True)
            trigger = batch["trigger_labels"].to(device, non_blocking=True)
            source = batch["source_labels"].to(device, non_blocking=True)
            ecotype = batch["ecotype_labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(values)
                loss = multitask_loss(logits, trigger, source, ecotype, weights, args)
            total_loss += float(loss) * len(values)
            examples += len(values)
            for index, value in enumerate(logits):
                logit_parts[index].append(value.float().cpu().numpy())
            label_parts["trigger"].append(trigger.cpu().numpy())
            label_parts["source"].append(source.cpu().numpy())
            label_parts["ecotype"].append(ecotype.cpu().numpy())
            clip_ids.extend(batch["clip_ids"])
    logits = [np.concatenate(group) for group in logit_parts]
    labels = {name: np.concatenate(group) for name, group in label_parts.items()}
    labels["clip_id"] = np.asarray(clip_ids, dtype=str)
    predictions = {
        "trigger": logits[0].argmax(axis=1),
        "source": logits[1].argmax(axis=1),
        "ecotype": logits[2].argmax(axis=1),
        "trigger_logits": logits[0],
        "source_logits": logits[1],
        "ecotype_logits": logits[2],
    }
    metrics = metrics_for_predictions(labels, predictions)
    metrics["loss"] = total_loss / max(examples, 1)
    return metrics, predictions, labels


def train_audio_model(
    args: argparse.Namespace,
    model: MultispeciesCetaceanModel,
    train_dataset: ArchiveManifestDataset,
    val_dataset: ArchiveManifestDataset,
    collator: ArchiveAudioCollator,
    weight_values: dict[str, list[float] | None],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    loader = DataLoader(
        train_dataset,
        batch_size=args.embedding_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
        num_workers=args.preprocessing_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.preprocessing_workers > 0,
    )
    model.to(device)
    training_model: nn.Module = model
    if args.multi_gpu:
        visible_gpus = torch.cuda.device_count()
        if device.type != "cuda" or visible_gpus < 2:
            print(
                "Multi-GPU requested, but fewer than two CUDA GPUs are visible; "
                "continuing on one device."
            )
        else:
            device_ids = list(range(visible_gpus))
            training_model = nn.DataParallel(model, device_ids=device_ids)
            print(
                f"Using DataParallel across {visible_gpus} GPUs: "
                + ", ".join(torch.cuda.get_device_name(index) for index in device_ids)
            )
    head_parameters = (
        list(model.trigger_classifier.parameters())
        + list(model.source_classifier.parameters())
        + list(model.ecotype_classifier.parameters())
    )
    groups = [{"params": head_parameters, "lr": args.learning_rate}]
    backbone_parameters = [parameter for parameter in model.ast.parameters() if parameter.requires_grad]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    weights = {name: weight_tensor(value, device) for name, value in weight_values.items()}
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] = {}
    best_evaluation: tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        examples = 0
        for batch_index, batch in enumerate(loader, start=1):
            values = batch["input_values"].to(device, non_blocking=True)
            trigger = batch["trigger_labels"].to(device, non_blocking=True)
            source = batch["source_labels"].to(device, non_blocking=True)
            ecotype = batch["ecotype_labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = training_model(values)
                raw_loss = multitask_loss(logits, trigger, source, ecotype, weights, args)
                loss = raw_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += float(raw_loss.detach()) * len(values)
            examples += len(values)
            if batch_index % 250 == 0:
                print(f"  epoch {epoch}: {batch_index:,}/{len(loader):,} batches")
        evaluation = evaluate_audio_model(
            training_model,
            val_dataset,
            collator,
            args.embedding_batch_size,
            args.preprocessing_workers,
            device,
            weight_values,
            args,
        )
        metrics, _, _ = evaluation
        row = {"epoch": float(epoch), "train_loss": total_loss / max(examples, 1), **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train_loss={row['train_loss']:.5f}, "
            f"val_loss={metrics['loss']:.5f}, trigger_f1={metrics['trigger_f1']:.4f}, "
            f"source_macro_f1={metrics['source_macro_f1']:.4f}, "
            f"ecotype_macro_f1={metrics['ecotype_macro_f1']:.4f}, "
            f"combined={metrics['combined_score']:.4f}"
        )
        if metrics["combined_score"] > best_score + 1e-8:
            best_score = metrics["combined_score"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_evaluation = evaluation
            stale = 0
        else:
            stale += 1
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"Early stopping after {stale} non-improving epochs")
                break
    if not best_state or best_evaluation is None:
        raise RuntimeError("No best full-model checkpoint was selected")
    return best_state, pd.DataFrame(history), *best_evaluation


def confusion_frame(
    true: np.ndarray, predicted: np.ndarray, id2label: dict[int, str]
) -> pd.DataFrame:
    ids = sorted(id2label)
    names = [id2label[index] for index in ids]
    return pd.DataFrame(
        confusion_matrix(true, predicted, labels=ids),
        index=[f"actual_{name}" for name in names],
        columns=[f"predicted_{name}" for name in names],
    )


def label_counts(values: np.ndarray, id2label: dict[int, str]) -> dict[str, int]:
    counts = Counter(values.tolist())
    return {name: counts[index] for index, name in id2label.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Root searched recursively for shard manifests; repeatable (default: /kaggle/input).",
    )
    parser.add_argument("--manifest-name", default="multispecies_cetacean_manifest.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/multispecies_cetacean_model",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--ecotype-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=2048)
    parser.add_argument("--preprocessing-workers", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help=(
            "Use all visible CUDA GPUs with PyTorch DataParallel during full-backbone "
            "training. This has no effect in --freeze-backbone mode."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=["linear", "constant_with_warmup", "cosine"],
        default="linear",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument(
        "--undbio-trigger-policy",
        choices=["ignore", "positive", "negative"],
        default="ignore",
    )
    parser.add_argument(
        "--reuse-embedding-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze AST and train heads from one cached embedding per clip.",
    )
    parser.add_argument(
        "--automatic-class-weights", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--trigger-class-weights")
    parser.add_argument("--source-class-weights")
    parser.add_argument("--ecotype-class-weights")
    parser.add_argument("--trigger-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-loss-weight", type=float, default=1.0)
    parser.add_argument("--ecotype-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--mean-subtract", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--high-pass-filter", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--high-pass-cutoff-hz", type=float)
    parser.add_argument("--high-pass-order", type=int)
    args = parser.parse_args()
    args.data_root = args.data_root or ["/kaggle/input"]
    if min(
        args.epochs,
        args.embedding_batch_size,
        args.head_batch_size,
        args.gradient_accumulation_steps,
    ) < 1:
        parser.error("epochs and batch sizes must be positive")
    if (
        args.learning_rate <= 0
        or args.ecotype_learning_rate < 0
        or args.backbone_learning_rate < 0
    ):
        parser.error("learning rates must be non-negative and the new-head LR must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be at least 0 and less than 1")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.clip_seconds <= 0:
        parser.error("--clip-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    roots = [Path(value).expanduser().resolve() for value in args.data_root]
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, train_manifests, train_archives = discover_rows(
        roots, "train", args.manifest_name, args.undbio_trigger_policy
    )
    val_rows, val_manifests, val_archives = discover_rows(
        roots, "validation", args.manifest_name, args.undbio_trigger_policy
    )
    train_rows = random_subset(train_rows, args.max_train_files, args.seed)
    val_rows = random_subset(val_rows, args.max_val_files, args.seed + 1)
    train_dataset = ArchiveManifestDataset(train_rows)
    val_dataset = ArchiveManifestDataset(val_rows)
    preprocessing, _ = preprocessing_from_checkpoint(args)
    model, checkpoint_identity, feature_source = load_model(
        args.model_name, args.dropout, args.freeze_backbone
    )
    hidden_size = int(model.config.hidden_size)
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(feature_source)
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    collator = ArchiveAudioCollator(
        feature_extractor,
        args.clip_seconds,
        preprocessing["mean_subtract"],
        preprocessing["high_pass_cutoff_hz"] if preprocessing["high_pass_filter"] else None,
        preprocessing["high_pass_order"],
    )

    print("\nMultispecies Cetacean training")
    print(f"Base model:             {args.model_name}")
    print(f"Device:                 {device}")
    print(f"Visible CUDA GPUs:      {torch.cuda.device_count()}")
    print(f"Multi-GPU requested:    {args.multi_gpu}")
    print(f"Backbone frozen:        {args.freeze_backbone}")
    print(f"Train/validation clips: {len(train_rows):,} / {len(val_rows):,}")
    print(f"Train shard archives:   {len(train_archives)}")
    print(f"Validation archives:    {len(val_archives)}")
    print(f"UndBio trigger policy:  {args.undbio_trigger_policy}")
    print(f"Mean subtraction:       {preprocessing['mean_subtract']}")
    print(f"High-pass filter:       {preprocessing['high_pass_filter']}")
    if preprocessing["high_pass_filter"]:
        print(
            f"High-pass settings:     {preprocessing['high_pass_cutoff_hz']:g} Hz, "
            f"order {preprocessing['high_pass_order']}"
        )

    train_label_values = {
        name: np.asarray([row[f"{name}_label"] for row in train_rows], dtype=np.int64)
        for name in ("trigger", "source", "ecotype")
    }
    weights = {
        "trigger": parse_class_weights(
            args.trigger_class_weights, TRIGGER_LABELS, "--trigger-class-weights"
        ),
        "source": parse_class_weights(
            args.source_class_weights, SOURCE_LABELS, "--source-class-weights"
        ),
        "ecotype": parse_class_weights(
            args.ecotype_class_weights, ECOTYPE_LABELS, "--ecotype-class-weights"
        ),
    }
    if args.automatic_class_weights:
        if weights["trigger"] is None:
            weights["trigger"] = automatic_weights(train_label_values["trigger"], 2, IGNORE_INDEX)
        if weights["source"] is None:
            weights["source"] = automatic_weights(train_label_values["source"], 4)
        if weights["ecotype"] is None:
            weights["ecotype"] = automatic_weights(train_label_values["ecotype"], 5, IGNORE_INDEX)

    print(f"Train trigger counts:   {label_counts(train_label_values['trigger'][train_label_values['trigger'] != IGNORE_INDEX], TRIGGER_ID2LABEL)}")
    print(f"Train source counts:    {label_counts(train_label_values['source'], SOURCE_ID2LABEL)}")
    print(f"Train ecotype counts:   {label_counts(train_label_values['ecotype'][train_label_values['ecotype'] != IGNORE_INDEX], ECOTYPE_ID2LABEL)}")
    print(f"Class weights:          {weights}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters:   {trainable:,}")

    if args.freeze_backbone:
        train_signature = cache_signature(
            "train", train_manifests, train_archives, checkpoint_identity,
            preprocessing, train_rows, args.seed
        )
        val_signature = cache_signature(
            "validation", val_manifests, val_archives, checkpoint_identity,
            preprocessing, val_rows, args.seed + 1
        )
        train_cache = (
            load_cache(cache_dir, "train", train_signature)
            if args.reuse_embedding_cache else None
        )
        val_cache = (
            load_cache(cache_dir, "validation", val_signature)
            if args.reuse_embedding_cache else None
        )
        model.to(device)
        if train_cache is None:
            train_cache = extract_embeddings(
                "train", train_dataset, model, collator, device, cache_dir,
                train_signature, args.embedding_batch_size, args.preprocessing_workers
            )
        if val_cache is None:
            val_cache = extract_embeddings(
                "validation", val_dataset, model, collator, device, cache_dir,
                val_signature, args.embedding_batch_size, args.preprocessing_workers
            )
        train_embeddings, train_labels = train_cache
        val_embeddings, val_labels = val_cache
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        head = FrozenMultispeciesHeads(hidden_size, args.dropout)
        head.trigger_classifier.load_state_dict(model.trigger_classifier.state_dict())
        head.source_classifier.load_state_dict(model.source_classifier.state_dict())
        head.ecotype_classifier.load_state_dict(model.ecotype_classifier.state_dict())
        best_head_state, history = train_head(
            args, head, train_embeddings, train_labels,
            val_embeddings, val_labels, weights, device
        )
        head.load_state_dict(best_head_state)
        head.to(device)
        best_metrics, predictions = predict(
            head, val_embeddings, val_labels, args.head_batch_size, device
        )
        head.to("cpu")
        model.trigger_classifier.load_state_dict(head.trigger_classifier.state_dict())
        model.source_classifier.load_state_dict(head.source_classifier.state_dict())
        model.ecotype_classifier.load_state_dict(head.ecotype_classifier.state_dict())
        full_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
    else:
        full_state, history, best_metrics, predictions, val_labels = train_audio_model(
            args, model, train_dataset, val_dataset, collator, weights, device
        )
        model.load_state_dict(full_state)
        model.to("cpu")

    torch.save(full_state, output_dir / "pytorch_model.bin")
    history.to_csv(output_dir / "training_history.csv", index=False)
    trigger_mask = val_labels["trigger"] != IGNORE_INDEX
    confusion_frame(
        val_labels["trigger"][trigger_mask], predictions["trigger"][trigger_mask], TRIGGER_ID2LABEL
    ).to_csv(output_dir / "trigger_confusion_matrix.csv")
    confusion_frame(
        val_labels["source"], predictions["source"], SOURCE_ID2LABEL
    ).to_csv(output_dir / "source_confusion_matrix.csv")
    ecotype_mask = val_labels["ecotype"] != IGNORE_INDEX
    confusion_frame(
        val_labels["ecotype"][ecotype_mask], predictions["ecotype"][ecotype_mask], ECOTYPE_ID2LABEL
    ).to_csv(output_dir / "ecotype_confusion_matrix.csv")

    trigger_prob = torch.softmax(torch.from_numpy(predictions["trigger_logits"]), dim=1).numpy()
    source_prob = torch.softmax(torch.from_numpy(predictions["source_logits"]), dim=1).numpy()
    ecotype_prob = torch.softmax(torch.from_numpy(predictions["ecotype_logits"]), dim=1).numpy()
    report: dict[str, Any] = {
        "clip_id": val_labels["clip_id"],
        "trigger_true": val_labels["trigger"],
        "trigger_pred": predictions["trigger"],
        "trigger_probability_known_whale": trigger_prob[:, 1],
        "source_true": val_labels["source"],
        "source_pred": predictions["source"],
        "ecotype_true": val_labels["ecotype"],
        "ecotype_pred": predictions["ecotype"],
    }
    for class_id, name in SOURCE_ID2LABEL.items():
        report[f"source_probability_{name}"] = source_prob[:, class_id]
    for class_id, name in ECOTYPE_ID2LABEL.items():
        report[f"ecotype_probability_{name}"] = ecotype_prob[:, class_id]
    pd.DataFrame(report).to_csv(output_dir / "validation_predictions.csv", index=False)

    configuration = {
        "format": "multispecies_cetacean_model_v1",
        "standalone": True,
        "base_model": args.model_name,
        "base_checkpoint_identity": checkpoint_identity,
        "backbone_frozen": args.freeze_backbone,
        "multi_gpu_requested": args.multi_gpu,
        "visible_cuda_gpus": torch.cuda.device_count(),
        "hidden_size": hidden_size,
        "trainable_parameters": trainable,
        "trigger_labels": TRIGGER_LABELS,
        "source_labels": SOURCE_LABELS,
        "ecotype_labels": ECOTYPE_LABELS,
        "undbio_trigger_policy": args.undbio_trigger_policy,
        "preprocessing": preprocessing,
        "class_weights": weights,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "ecotype_learning_rate": args.ecotype_learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "loss_weights": {
            "trigger": args.trigger_loss_weight,
            "source": args.source_loss_weight,
            "ecotype": args.ecotype_loss_weight,
        },
        "best_validation_metrics": best_metrics,
        "seed": args.seed,
    }
    atomic_json(output_dir / "multispecies_cetacean_config.json", configuration)
    model.config.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print("\nBest validation metrics")
    for name, value in best_metrics.items():
        print(f"{name:28s}: {value:.6f}")
    print(f"Saved outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
