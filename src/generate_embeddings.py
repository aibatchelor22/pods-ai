#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from model_inference import get_model_inference


PODSAI_MODEL_ID = "davethaler/whale-call-detector"
PODSAI_AST_MODEL_REVISION = "db51f75da131de0e53e8080a1f2c5f4b534810aa"

EMBEDDING_CSV_BASE_FIELDS = [
    "manifest_row_index",
    "category",
    "ground_truth_label",
    "node_name",
    "timestamp",
    "uri",
    "description",
    "notes",
    "wav_path",
    "model_type",
    "segment_index",
    "start_time_seconds",
    "duration_seconds",
    "predicted_label",
    "predicted_class_id",
    "local_confidence",
    "global_prediction_label",
    "global_confidence",
]


@dataclass
class TestSample:
    row_index: int
    category: str
    node_name: str
    timestamp: str
    uri: str
    description: str
    notes: str


def load_test_samples(
    testing_csv: Path,
    max_samples: Optional[int] = None,
    category_filter: Optional[str] = None,
) -> list[TestSample]:

    samples = []

    with open(testing_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):
            category = row.get("Category", "")

            if category_filter is not None and category != category_filter:
                continue

            samples.append(
                TestSample(
                    row_index=row_index,
                    category=category,
                    node_name=row.get("NodeName", ""),
                    timestamp=row.get("Timestamp", ""),
                    uri=row.get("URI", ""),
                    description=row.get("Description", ""),
                    notes=row.get("Notes", ""),
                )
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


def find_wav_file(sample: TestSample, wav_dir: Path) -> Optional[Path]:
    node_name_in_filename = sample.node_name.replace("_", "-")

    wav_filename = (
        f"{node_name_in_filename}_{sample.timestamp}.wav"
    )

    wav_path = wav_dir / sample.category / wav_filename

    if wav_path.exists():
        return wav_path

    return None


def _as_tensor(value: Any) -> Any:
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state

    if hasattr(value, "hidden_states") and value.hidden_states:
        return value.hidden_states[-1]

    if isinstance(value, (tuple, list)) and value:
        return value[0]

    return value


def capture_ast_embeddings(model):
    embeddings = []

    ast_module = getattr(
        getattr(model, "model", None),
        "audio_spectrogram_transformer",
        None,
    )

    if ast_module is None:
        raise ValueError(
            "Unable to locate AST module for embedding extraction."
        )

    def hook(_module, _inputs, output):
        tensor = _as_tensor(output)

        if tensor is None:
            return

        if getattr(tensor, "ndim", 0) == 3:
            tensor = tensor[:, 0, :]

        elif getattr(tensor, "ndim", 0) != 2:
            return

        embeddings.extend(
            tensor.detach().cpu().float().tolist()
        )

    handle = ast_module.register_forward_hook(hook)

    return embeddings, handle


def write_embedding_rows(
    embeddings_csv: Path,
    sample: TestSample,
    wav_path: Path,
    inference_result: dict,
):

    embeddings = inference_result["ast_embeddings"]

    if not embeddings:
        return 0

    local_predictions = inference_result.get(
        "local_predictions", []
    )

    local_prediction_labels = inference_result.get(
        "local_prediction_labels", []
    )

    local_confidences = inference_result.get(
        "local_confidences", []
    )

    hop_duration = float(
        inference_result.get("hop_duration", 0.0)
    )

    segment_duration = float(
        inference_result.get("segment_duration", 0.0)
    )

    global_prediction_label = inference_result.get(
        "global_prediction_label", ""
    )

    global_confidence = float(
        inference_result.get("global_confidence", 0.0)
    )

    embedding_width = max(
        len(e) for e in embeddings
    )

    fieldnames = [
        *EMBEDDING_CSV_BASE_FIELDS,
        *[
            f"embedding_{i}"
            for i in range(embedding_width)
        ],
    ]

    write_header = not embeddings_csv.exists()

    embeddings_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with embeddings_csv.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        for segment_index, embedding in enumerate(
            embeddings
        ):

            predicted_class_id = (
                local_predictions[segment_index]
                if segment_index < len(local_predictions)
                else ""
            )

            predicted_label = (
                local_prediction_labels[segment_index]
                if segment_index < len(local_prediction_labels)
                else str(predicted_class_id)
            )

            local_confidence = (
                local_confidences[segment_index]
                if segment_index < len(local_confidences)
                else ""
            )

            row = {
                "manifest_row_index": sample.row_index,
                "category": sample.category,

                # preserved from original implementation
                "ground_truth_label": wav_path.parent.name,

                "node_name": sample.node_name,
                "timestamp": sample.timestamp,
                "uri": sample.uri,
                "description": sample.description,
                "notes": sample.notes,

                "wav_path": str(wav_path),
                "model_type": "podsai",

                "segment_index": segment_index,

                "start_time_seconds":
                    segment_index * hop_duration,

                "duration_seconds":
                    segment_duration,

                "predicted_label":
                    predicted_label,

                "predicted_class_id":
                    predicted_class_id,

                "local_confidence":
                    local_confidence,

                "global_prediction_label":
                    global_prediction_label,

                "global_confidence":
                    global_confidence,
            }

            row.update(
                {
                    f"embedding_{idx}": float(value)
                    for idx, value in enumerate(
                        embedding
                    )
                }
            )

            writer.writerow(row)

    return len(embeddings)


def run_ast_inference(
    model,
    wav_path: str,
):

    embeddings, handle = capture_ast_embeddings(model)

    try:
        result = model.predict(wav_path)
    finally:
        handle.remove()

    local_predictions = result.get(
        "local_predictions", []
    )

    id2label = getattr(
        model,
        "id2label",
        {},
    )

    local_prediction_labels = []

    for prediction in local_predictions:

        if isinstance(prediction, str):
            label = prediction
        else:
            label = id2label.get(
                prediction,
                str(prediction),
            )

        local_prediction_labels.append(label)

    return {
        "ast_embeddings": embeddings,
        "local_predictions":
            local_predictions,
        "local_prediction_labels":
            local_prediction_labels,
        "local_confidences":
            result.get(
                "local_confidences",
                [],
            ),
        "hop_duration":
            result.get(
                "hop_duration",
                2.0,
            ),
        "segment_duration":
            result.get(
                "segment_duration",
                3.0,
            ),
        "global_prediction_label":
            result.get(
                "global_prediction_label",
                "",
            ),
        "global_confidence":
            result.get(
                "global_confidence",
                0.0,
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testing-csv",
        default="../output/csv/testing_60s_samples.csv",
        help="Path to testing_60s_samples.csv (default: output/csv/testing_60s_samples.csv).",
    )

    parser.add_argument(
        "--wav-dir",
        default="output/testing-wav",
        help="Root directory containing testing WAV files (default: output/testing-wav).",
    )

    parser.add_argument(
        "--output-csv",
        default="output/csv/embeddings.csv",
        help="Output file for embeddings (default: output/csv/embeddings.csv).",
    )

    parser.add_argument(
        "--model-path",
        default=PODSAI_MODEL_ID,
    )

    parser.add_argument(
        "--model-revision",
        default=PODSAI_AST_MODEL_REVISION,
    )

    parser.add_argument(
        "--category",
        default=None,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    samples = load_test_samples(
        Path(args.testing_csv),
        max_samples=args.max_samples,
        category_filter=args.category,
    )

    print(f"Loaded {len(samples)} samples")

    output_csv = Path(args.output_csv)

    if output_csv.exists():
        output_csv.unlink()

    model = get_model_inference(
        model_type="podsai",
        model_path=args.model_path,
        model_revision=args.model_revision,
    )

    wav_dir = Path(args.wav_dir)

    for idx, sample in enumerate(samples, start=1):

        if idx % 100 == 0:
            print(
                f"Processed {idx}/{len(samples)} clips"
            )

        wav_path = find_wav_file(
            sample,
            wav_dir,
        )

        if wav_path is None:
            print(
                f"Missing WAV: "
                f"{sample.category}/"
                f"{sample.node_name}/"
                f"{sample.timestamp}"
            )
            continue

        try:
            result = run_ast_inference(
                model,
                str(wav_path),
            )

            write_embedding_rows(
                output_csv,
                sample,
                wav_path,
                result,
            )

        except Exception as e:
            print(
                f"Failed: {wav_path.name}: {e}"
            )

    print(
        f"\nFinished. "
        f"Embeddings written to {output_csv}"
    )


if __name__ == "__main__":
    sys.exit(main())
