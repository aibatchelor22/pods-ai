#!/usr/bin/env python3
"""Run experimental PODS-AI-style V2 inference on one 60-second WAV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from multispecies_cetacean_inference_full_spectrogram import (
    FullSpectrogramMultispeciesCetaceanInference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_file", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--aggregation-json", type=Path)
    parser.add_argument(
        "--preserve-max-length",
        action="store_true",
        help="Keep padded AST frames while still computing the fbank only once.",
    )
    args = parser.parse_args()
    if not args.wav_file.is_file():
        raise FileNotFoundError(args.wav_file)
    if args.inference_batch_size < 1:
        parser.error("--inference-batch-size must be positive")

    aggregation = {}
    if args.aggregation_json is not None:
        loaded = json.loads(args.aggregation_json.read_text(encoding="utf-8"))
        aggregation = loaded.get("aggregation", loaded)

    predictor = FullSpectrogramMultispeciesCetaceanInference(
        args.model_path,
        device=args.device,
        model_revision=args.model_revision,
        inference_batch_size=args.inference_batch_size,
        aggregation_config=aggregation,
        compact_ast_frames=not args.preserve_max_length,
    )
    device = predictor.device
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = predictor.predict(
        str(args.wav_file), segment_duration=3, hop_duration=2
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result["predict_time"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

