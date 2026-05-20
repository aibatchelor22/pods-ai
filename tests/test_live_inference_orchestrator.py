# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for PODS-AI LiveInferenceOrchestrator helpers."""

import LiveInferenceOrchestrator as orchestrator


def test_is_positive_label_uses_multiclass_negative_set() -> None:
    """Negative classes should be non-detections, whale classes should be detections."""
    assert orchestrator.is_positive_label("resident")
    assert orchestrator.is_positive_label("transient")
    assert orchestrator.is_positive_label("humpback")

    assert not orchestrator.is_positive_label("other")
    assert not orchestrator.is_positive_label("water")
    assert not orchestrator.is_positive_label("vessel")
    assert not orchestrator.is_positive_label("jingle")
    assert not orchestrator.is_positive_label("human")


def test_build_prediction_list_filters_negative_labels() -> None:
    """Only positive local segment labels should be included in prediction rows."""
    result = {
        "local_predictions": [1, 0, 3, 4],
        "local_confidences": [0.95, 0.10, 0.80, 0.05],
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }
    id2label = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
    }

    predictions = orchestrator.build_prediction_list(result, id2label=id2label)

    assert len(predictions) == 2
    assert predictions[0]["label"] == "resident"
    assert predictions[0]["startTime"] == 0.0
    assert predictions[0]["duration"] == 3.0
    assert predictions[1]["label"] == "humpback"
    assert predictions[1]["startTime"] == 4.0


def test_build_cosmosdb_metadata_includes_multiclass_fields() -> None:
    """Metadata should include global prediction label, positive segment list, and comments."""
    result = {
        "local_predictions": [2, 0],
        "local_confidences": [0.90, 0.20],
        "global_confidence": 0.9,
        "global_prediction_label": "transient",
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }
    id2label = {0: "water", 2: "transient"}

    metadata = orchestrator.build_cosmosdb_metadata(
        audio_uri="audio-uri",
        image_uri="image-uri",
        result=result,
        timestamp_in_iso="2026-01-01T00:00:00Z",
        source_guid="unknown_feed",
        model_id="podsai-model",
        id2label=id2label,
    )

    assert metadata["modelId"] == "podsai-model"
    assert metadata["globalPredictionLabel"] == "transient"
    assert metadata["whaleFoundConfidence"] == 90.0
    assert metadata["location"]["id"] == "unknown_feed"
    assert len(metadata["predictions"]) == 1
    assert metadata["predictions"][0]["label"] == "transient"
    assert metadata["comments"] == "AI: transient"


def test_build_cosmosdb_metadata_comments_appends_dominant_extra_class() -> None:
    """comments field should append the most common vessel/human/jingle label when it differs from the global prediction."""
    result = {
        "local_predictions": ["resident", "vessel", "vessel"],
        "local_confidences": [0.80, 0.70, 0.65],
        "global_confidence": 0.8,
        "global_prediction_label": "resident",
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }

    metadata = orchestrator.build_cosmosdb_metadata(
        audio_uri="audio-uri",
        image_uri="image-uri",
        result=result,
        timestamp_in_iso="2026-01-01T00:00:00Z",
        source_guid="rpi_orcasound_lab",
        model_id="podsai-model",
    )

    assert metadata["comments"] == "AI: resident and vessel"
