#!/usr/bin/env python3

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


# --------------------------------------------------------
# Load a batch of audio clips
# --------------------------------------------------------

def load_audio_batch(paths, sample_rate=16000, duration=3.0):

    target_length = int(sample_rate * duration)

    audio = []

    for path in paths:

        y, sr = librosa.load(
            path,
            sr=sample_rate,
        )

        if len(y) > target_length:
            y = y[:target_length]

        elif len(y) < target_length:
            y = np.pad(
                y,
                (0, target_length - len(y)),
            )

        audio.append(y)

    return audio


# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output_csv",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name
    )

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name
    )

    model.to(device)
    model.eval()

    manifest = pd.read_csv(args.manifest)

    rows = []

    for start in tqdm(
        range(0, len(manifest), args.batch_size)
    ):

        batch = manifest.iloc[
            start:start + args.batch_size
        ]

        audio = load_audio_batch(
            batch["clip_path"].tolist(),
            sample_rate=args.sample_rate,
            duration=args.duration,
        )

        inputs = feature_extractor(
            audio,
            sampling_rate=args.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = model.audio_spectrogram_transformer(
                **inputs
            )

            embeddings = (
                outputs.last_hidden_state[:, 0, :]
                .cpu()
                .numpy()
            )

            logits = model.classifier(
                torch.tensor(
                    embeddings,
                    device=device,
                )
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = probs.max(dim=1)

            prediction = prediction.cpu().numpy()
            confidence = confidence.cpu().numpy()

        for i, (_, row) in enumerate(batch.iterrows()):

            result = {
                "clip_path": row["clip_path"],
                "ground_truth_label": row["label"],
                "predicted_label":
                    model.config.id2label[
                        int(prediction[i])
                    ],
                "confidence":
                    float(confidence[i]),
                "domain":
                    (
                        "old"
                        if "orca-data-part-6-orcasound"
                        in row["clip_path"]
                        else "new"
                    ),
            }

            for j, value in enumerate(embeddings[i]):
                result[f"embedding_{j}"] = float(value)

            rows.append(result)

    df = pd.DataFrame(rows)

    df.to_csv(
        args.output_csv,
        index=False,
    )

    print(df.head())

    print()

    print(
        f"Saved {len(df)} embeddings to "
        f"{args.output_csv}"
    )


if __name__ == "__main__":
    main()