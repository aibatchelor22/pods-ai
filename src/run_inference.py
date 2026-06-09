#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Run inference on a wav file and output per-class probabilities.

Usage:
    python run_inference.py sample.wav
    python run_inference.py sample.wav --model podsai --model-path /path/to/podsai-model
    python run_inference.py sample.wav --model fastai --model-path ../model
    python run_inference.py --node-name rpi_sunset_bay --end-timestamp-str 2025_01_15_12_30_00_PST
"""

import argparse
import math
import shutil
import sys
import time
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import ffmpeg
from pytz import timezone as pytz_tz

from audio_utils import (
    download_from_url,
    get_cached_folders,
    get_difference_between_times_in_seconds,
    get_folders_between_timestamp,
    load_m3u8_with_retry,
)
from model_inference import get_model_inference

PODSAI_MODEL_ID = "davethaler/whale-call-detector"
# renovate: datasource=git-refs depName=https://huggingface.co/davethaler/whale-call-detector versioning=git.
PODSAI_AST_MODEL_REVISION = "d1eedf5c614268da7551039a84dfc35d317168b9"
PODSAI_WAV2VEC2_MODEL_REVISION = "cef82c6e9ee661646ea0c583aeb68f4f7ec6d9d8"
# Preserve the existing exported constant name for compatibility.
PODSAI_MODEL_REVISION = PODSAI_AST_MODEL_REVISION
PROPOSED_DESCRIPTION_EXTRA_CLASSES = {"vessel", "human", "jingle"}
NEGATIVE_LABELS = {"other", "water", "vessel", "jingle", "human"}
PACIFIC_TZ = pytz_tz("US/Pacific")
UTC_TZ = timezone.utc
MIN_SEGMENT_DURATION = 0.001
FLOAT_TOLERANCE = 1e-9


def parse_pst_end_timestamp(timestamp_str: str) -> datetime:
    """Parse PST end timestamp format YYYY_MM_DD_HH_MM_SS_PST."""
    dt = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S_PST")
    return PACIFIC_TZ.localize(dt)


def parse_utc_start_timestamp(timestamp_str: str) -> datetime:
    """Parse UTC start timestamp format YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC_TZ)


def _format_utc_iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_clip_id(start_time_utc: datetime) -> str:
    return start_time_utc.astimezone(PACIFIC_TZ).strftime("%Y_%m_%d_%H_%M_%S_PST")


def _format_pacific_timestamp(dt: datetime) -> str:
    return dt.astimezone(PACIFIC_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def download_60s_audio_from_start_utc(
    node_name: str,
    start_time_utc: datetime,
    tmp_dir: str,
) -> Optional[str]:
    """Download a 60-second clip beginning at start_time_utc."""
    duration_seconds = 60.0
    end_time_utc = start_time_utc + timedelta(seconds=duration_seconds)
    start_unix_time = int(start_time_utc.timestamp())
    end_unix_time = int(end_time_utc.timestamp())

    hydrophone_stream_url = f"https://s3-us-west-2.amazonaws.com/audio-orcasound-net/{node_name}"
    bucket_folder = hydrophone_stream_url.split("https://s3-us-west-2.amazonaws.com/")[1]
    tokens = bucket_folder.split("/")
    s3_bucket = tokens[0]
    folder_name = tokens[1]
    prefix = folder_name + "/hls/"

    try:
        all_hydrophone_folders = get_cached_folders(s3_bucket, prefix=prefix)
        print(f"  Found {len(all_hydrophone_folders)} folders in total for {node_name}")
        valid_folders = get_folders_between_timestamp(all_hydrophone_folders, start_unix_time, end_unix_time)
        print(f"  Found {len(valid_folders)} folders in date range")
        if not valid_folders:
            print(f"  Warning: No folders found for timestamp {start_time_utc}")
            return None
        current_folder = int(valid_folders[0])
    except Exception as e:
        print(f"  ERROR: Failed to query S3 bucket: {e}")
        return None

    stream_url = f"{hydrophone_stream_url}/hls/{current_folder}/live.m3u8"
    try:
        stream_obj = load_m3u8_with_retry(stream_url)
    except Exception as e:
        print(f"  ERROR: Failed to load m3u8 file: {e}")
        return None

    num_total_segments = len(stream_obj.segments)
    if num_total_segments == 0:
        print("  ERROR: No segments found in m3u8 file")
        return None

    target_duration_exact = sum(item.duration for item in stream_obj.segments) / num_total_segments
    target_duration = max(target_duration_exact, MIN_SEGMENT_DURATION)

    time_since_folder_start_for_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)
    time_since_folder_start_for_end = get_difference_between_times_in_seconds(end_unix_time, current_folder)

    segment_start_index = max(
        0,
        math.floor((time_since_folder_start_for_start + FLOAT_TOLERANCE) / target_duration),
    )
    segment_end_index = min(
        num_total_segments,
        math.ceil((time_since_folder_start_for_end - FLOAT_TOLERANCE) / target_duration),
    )
    if segment_end_index <= segment_start_index:
        segment_end_index = min(num_total_segments, segment_start_index + 1)

    print(
        f"Segment: folder={current_folder}, indices=[{segment_start_index}:{segment_end_index}), "
        f"start={_format_utc_iso_z(start_time_utc)}, duration={duration_seconds:.1f}s"
    )

    try:
        file_names = []
        for i in range(segment_start_index, segment_end_index):
            audio_segment = stream_obj.segments[i]
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            download_from_url(audio_url, tmp_dir)
            file_names.append(file_name)

        if not file_names:
            print("  ERROR: No segments were successfully downloaded")
            return None

        clip_id = _build_clip_id(start_time_utc)
        clipname = f"temp_60s_{node_name}_{clip_id}"
        if len(file_names) > 1:
            hls_file = str(Path(tmp_dir) / f"{clipname}.ts")
            with open(hls_file, "wb") as wfd:
                for f in file_names:
                    with open(Path(tmp_dir) / f, "rb") as fd:
                        shutil.copyfileobj(fd, wfd)
        else:
            hls_file = str(Path(tmp_dir) / file_names[0])

        wav_file_path = str(Path(tmp_dir) / f"{clipname}.wav")
        ss_offset = time_since_folder_start_for_start - (segment_start_index * target_duration)
        if ss_offset < 0:
            ss_offset = 0.0

        stream = ffmpeg.input(hls_file, ss=ss_offset)
        stream = ffmpeg.output(
            stream,
            wav_file_path,
            t=duration_seconds,
            acodec="pcm_s16le",
            ar=44100,
            ac=1,
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        print(f"  Downloaded 60s audio: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        print(f"  Warning: Unable to retrieve audio clip: {e}")
        return None


def build_proposed_description(
    global_prediction_label: str,
    local_prediction_labels: list[str],
) -> str:
    """Build a proposed description string from global and local predictions.

    Args:
        global_prediction_label: Predicted class label for the whole file.
        local_prediction_labels: Segment-level predicted class labels.

    Returns:
        Proposed description text beginning with "AI:" and optionally appending
        a dominant non-whale context class from {"vessel", "human", "jingle"}.
    """
    proposed_description = f"AI: {global_prediction_label}"
    if not local_prediction_labels:
        return proposed_description

    most_common = Counter(local_prediction_labels).most_common(1)
    if not most_common:
        return proposed_description
    most_common_label, _ = most_common[0]
    if (
        most_common_label in PROPOSED_DESCRIPTION_EXTRA_CLASSES
        and most_common_label != global_prediction_label
    ):
        proposed_description = f"{proposed_description} and {most_common_label}"
    return proposed_description


def prediction_to_label(prediction: Any, id2label: Optional[dict[int, str]]) -> str:
    """Return a string label for a local/global prediction value."""
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, int) and id2label is not None:
        return id2label.get(prediction, str(prediction))
    return str(prediction)


def calculate_positive_segments(
    local_predictions: list[Any],
    local_confidences: list[Any],
    hop_duration: float,
    segment_duration: float,
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
    threshold: Optional[float] = None,
    start_time_utc: Optional[datetime] = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Count positive segments and optionally include UTC/Pacific timestamps."""
    effective_negative_labels = (
        negative_labels if negative_labels is not None else NEGATIVE_LABELS
    )
    confidence_threshold = float(threshold) if threshold is not None else 0.0
    positive_segments: list[dict[str, Any]] = []

    for idx, (local_prediction, local_confidence) in enumerate(
        zip(local_predictions, local_confidences)
    ):
        label = prediction_to_label(local_prediction, id2label)
        confidence = float(local_confidence)
        if label in effective_negative_labels or confidence < confidence_threshold:
            continue

        start_seconds = idx * float(hop_duration)
        segment_info: dict[str, Any] = {
            "index": idx,
            "label": label,
            "confidence": confidence,
            "start_time_seconds": start_seconds,
            "duration_seconds": float(segment_duration),
        }
        if start_time_utc is not None:
            segment_start_time_utc = start_time_utc + timedelta(seconds=start_seconds)
            segment_info["start_time_utc"] = _format_utc_iso_z(segment_start_time_utc)
            segment_info["start_time_pacific"] = _format_pacific_timestamp(
                segment_start_time_utc
            )
        positive_segments.append(segment_info)

    return len(positive_segments), positive_segments


def run_inference(wav_path: str, model_type: str = "podsai",
                  model_path: Optional[str] = None,
                  model_revision: Optional[str] = None,
                  model_variant: str = "ast",
                  start_time_utc: Optional[datetime] = None) -> dict:
    """
    Run inference on a wav file and return per-class probabilities.

    Args:
        wav_path: Path to the wav file.
        model_type: Type of model to use ('podsai', 'fastai', or 'orcahello').
        model_path: Path to the model directory or HuggingFace Hub model ID.
                    Required for podsai. Defaults to './model' for fastai,
                    'orcasound/orcahello-srkw-detector-v1' for orcahello,
                    and PODSAI_MODEL_ID for podsai.
        model_revision: Git commit hash to pin the HuggingFace Hub model revision.
                        Only used when model_path is a Hub model ID (not a local path).
                        Defaults to PODSAI_MODEL_REVISION when model_path is the default
                        PODS-AI Hub model.
        model_variant: PODS-AI model variant to use when model_type is "podsai"
                       and the default PODS-AI Hub model/revision is being used
                       (model_path and model_revision are not explicitly set).
                       Supported values are "ast" (default) and "wav2vec2".
                       Ignored when model_path or model_revision is explicitly provided.

    Returns:
        Dictionary with:
            - probabilities: dict mapping class label to probability (0.0-1.0).
              Each value is the mean local_confidence for windows that predicted
              that class and whose confidence exceeds the model's threshold.
            - global_prediction_label: predicted class label for the whole file
            - global_confidence: confidence score (0.0-1.0) for the global prediction
            - proposed_description: text description suitable for manual sample notes
            - predict_time: time in seconds spent in the model's predict() method
            - positive_segments_count: number of positive PODS-AI segments above threshold
            - positive_segments: list of positive segment details (label/confidence/timestamps)
    """
    local_predictions: list[Any] = []
    local_confidences: list[Any] = []
    hop_duration = 2.0
    segment_duration = 3.0
    positive_segments_count = 0
    positive_segments: list[dict[str, Any]] = []

    if model_type == "fastai":
        if model_path is None:
            model_path = "./model"
        model = get_model_inference(model_type="fastai", model_path=model_path)

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time
        local_predictions = result.get("local_predictions", [])
        local_confidences = result.get("local_confidences", [])
        hop_duration = float(result.get("hop_duration", 1.0))
        segment_duration = float(result.get("segment_duration", 3.0))
        # local_confidences that exceed the threshold (resident windows).
        resident_prob = float(result.get("global_confidence", 0.0))
        other_prob = round(1.0 - resident_prob, 4)

        probabilities: dict[str, float] = {
            "other": other_prob,
            "resident": round(resident_prob, 4),
        }
        local_prediction_labels = [
            "resident" if int(local_prediction) == 1 else "other"
            for local_prediction in result.get("local_predictions", [])
        ]
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
        local_predictions = result.get("local_predictions", [])
        local_confidences = result.get("local_confidences", [])
        hop_duration = float(result.get("hop_duration", 1.0))
        segment_duration = float(result.get("segment_duration", 2.0))

        # The OrcaHello SRKW detector is a binary classifier (other vs resident).
        resident_prob = float(result.get("global_confidence", 0.0))
        other_prob = round(1.0 - resident_prob, 4)

        probabilities = {
            "other": other_prob,
            "resident": round(resident_prob, 4),
        }
        local_prediction_labels = [
            "resident" if int(local_prediction) == 1 else "other"
            for local_prediction in result.get("local_predictions", [])
        ]
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "resident" if global_prediction else "other"
        global_confidence = resident_prob

    elif model_type == "podsai":
        if model_variant not in {"ast", "wav2vec2"}:
            raise ValueError(
                f"Unknown PODS-AI model variant: {model_variant!r}. Use 'ast' or 'wav2vec2'."
            )
        if model_path is None:
            model_path = PODSAI_MODEL_ID
            if model_revision is None:
                if model_variant == "wav2vec2":
                    model_revision = PODSAI_WAV2VEC2_MODEL_REVISION
                else:
                    model_revision = PODSAI_AST_MODEL_REVISION

        model = get_model_inference(model_type="podsai", model_path=model_path,
                                    model_revision=model_revision)

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time
        local_predictions = result.get("local_predictions", [])
        local_confidences = result.get("local_confidences", [])
        hop_duration = float(result.get("hop_duration", 2.0))
        segment_duration = float(result.get("segment_duration", 3.0))

        probabilities = result["per_class_probabilities"]
        local_prediction_labels = []
        id2label = getattr(model, "id2label", {})
        for local_prediction in local_predictions:
            label = prediction_to_label(local_prediction, id2label)
            if isinstance(label, str):
                local_prediction_labels.append(label)
        global_prediction_label = result.get("global_prediction_label", "")
        global_confidence = float(result.get("global_confidence", 0.0))
        positive_segments_count, positive_segments = calculate_positive_segments(
            local_predictions=local_predictions,
            local_confidences=local_confidences,
            hop_duration=hop_duration,
            segment_duration=segment_duration,
            id2label=id2label,
            threshold=getattr(model, "threshold", None),
            start_time_utc=start_time_utc,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Use 'podsai', 'fastai', or 'orcahello'."
        )

    proposed_description = build_proposed_description(global_prediction_label, local_prediction_labels)

    return {
        "probabilities": probabilities,
        "global_prediction_label": global_prediction_label,
        "global_confidence": global_confidence,
        "proposed_description": proposed_description,
        "predict_time": predict_time,
        "local_predictions": local_predictions,
        "local_confidences": local_confidences,
        "hop_duration": hop_duration,
        "segment_duration": segment_duration,
        "positive_segments_count": positive_segments_count,
        "positive_segments": positive_segments,
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
    proposed_description = results["proposed_description"]
    predict_time = results.get("predict_time", 0.0)

    print(f"Model type: {model_type}")
    print(f"Global prediction: {label} (confidence: {confidence:.4f})")
    print(f"Proposed description: {proposed_description}")
    print(f"Prediction time: {predict_time:.2f}s")
    if model_type == "podsai":
        local_predictions = results.get("local_predictions", [])
        positive_segments = results.get("positive_segments", [])
        positive_segments_count = results.get("positive_segments_count", 0)
        print(f"Positive segments: {positive_segments_count}/{len(local_predictions)}")
        if positive_segments:
            print("Positive segment timestamps:")
            for segment in positive_segments:
                timestamp = segment.get("start_time_pacific")
                if timestamp is None:
                    timestamp = f"+{float(segment.get('start_time_seconds', 0.0)):.1f}s"
                print(
                    f"  {timestamp}: {segment.get('label', '')} "
                    f"(confidence: {float(segment.get('confidence', 0.0)):.3f})"
                )
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
        description=(
            "Run model inference on a wav file and output per-class probabilities. "
            "Provide either a local wav file path, or --node-name plus exactly one of "
            "--end-timestamp-str (PST end time) or --start-timestamp-utc."
        )
    )
    parser.add_argument(
        "wav_file",
        nargs="?",
        help="Path to the wav file to score.",
    )
    parser.add_argument(
        "--node-name",
        default=None,
        help="Feed node name (e.g., rpi_sunset_bay) used with timestamp arguments to download audio.",
    )
    parser.add_argument(
        "--end-timestamp-str",
        default=None,
        help=(
            "PST end timestamp used with --node-name to download audio "
            "(format: YYYY_MM_DD_HH_MM_SS_PST)."
        ),
    )
    parser.add_argument(
        "--start-timestamp-utc",
        default=None,
        help=(
            "UTC start timestamp used with --node-name to download audio "
            "(format: YYYY-MM-DDTHH:MM:SSZ)."
        ),
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
        "--type",
        default="ast",
        choices=("ast", "wav2vec2"),
        help=(
            "PODS-AI model type to use with --model podsai (default: ast). "
            "ast selects the AST-based checkpoint, wav2vec2 selects the older Wav2Vec2 checkpoint."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to model directory or HuggingFace Hub model ID. "
            "Required for --model podsai. "
            "Defaults to ./model for --model fastai. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 for --model orcahello. "
            f"Defaults to {PODSAI_MODEL_ID!r} for --model podsai."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Git commit hash to pin the HuggingFace Hub model revision. "
            "Only used when --model-path is a Hub model ID. "
            f"Defaults to the pinned revision ({PODSAI_MODEL_REVISION}) when using "
            f"the default PODS-AI model ({PODSAI_MODEL_ID!r})."
        ),
    )

    args = parser.parse_args()
    end_timestamp_str = args.end_timestamp_str
    timestamp_args_provided = sum(
        [
            end_timestamp_str is not None,
            args.start_timestamp_utc is not None,
        ]
    )
    if args.node_name is None and timestamp_args_provided > 0:
        print(
            "Error: --node-name is required when using timestamp arguments.",
            file=sys.stderr,
        )
        return 1
    if args.node_name is not None and timestamp_args_provided != 1:
        print(
            "Error: with --node-name, provide exactly one of --end-timestamp-str "
            "or --start-timestamp-utc.",
            file=sys.stderr,
        )
        return 1

    with ExitStack() as stack:
        start_time_utc: Optional[datetime] = None
        if args.wav_file:
            if args.node_name is not None:
                print(
                    "Error: provide either wav_file or --node-name with one timestamp argument, not both.",
                    file=sys.stderr,
                )
                return 1
            wav_path = args.wav_file
            if not Path(wav_path).exists():
                print(f"Error: wav file not found: {wav_path}", file=sys.stderr)
                return 1
        else:
            if args.node_name is None:
                print(
                    "Error: either provide wav_file, or provide --node-name with one timestamp argument.",
                    file=sys.stderr,
                )
                return 1

            temp_dir = stack.enter_context(TemporaryDirectory())
            try:
                if args.start_timestamp_utc is not None:
                    start_time_utc = parse_utc_start_timestamp(args.start_timestamp_utc)
                else:
                    end_time_pst = parse_pst_end_timestamp(end_timestamp_str)
                    start_time_utc = end_time_pst.astimezone(UTC_TZ) - timedelta(seconds=60)
                wav_path = download_60s_audio_from_start_utc(args.node_name, start_time_utc, temp_dir)
            except ValueError as e:
                print(f"Failed to download wav: {e}", file=sys.stderr)
                return 1
            except Exception as e:
                print(f"Failed to download wav: {e}", file=sys.stderr)
                return 1
            if not wav_path:
                print("Error: failed to download wav file.", file=sys.stderr)
                return 1
            if not Path(wav_path).exists():
                print(f"Error: downloaded wav file not found: {wav_path}", file=sys.stderr)
                return 1

        try:
            results = run_inference(
                wav_path,
                model_type=args.model,
                model_path=args.model_path,
                model_revision=args.model_revision,
                model_variant=args.type,
                start_time_utc=start_time_utc,
            )
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
