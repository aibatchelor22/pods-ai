#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for generate_embeddings.py.

Tests cover:
- load_test_samples() CSV parsing, filtering, and edge cases
- find_wav_file() path construction and existence checks
- _as_tensor() attribute dispatch
- capture_ast_embeddings() hook registration and missing-module error
- write_embedding_rows() CSV output format and append behaviour
- run_ast_inference() label resolution via id2label
"""

import csv
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_testing_csv(path, rows, fieldnames=None):
    """Write a minimal testing CSV and return its Path."""
    if fieldnames is None:
        fieldnames = [
            "Category", "NodeName", "Timestamp",
            "URI", "Description", "Notes",
        ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_rows():
    return [
        {
            "Category": "resident",
            "NodeName": "rpi_orcasound_lab",
            "Timestamp": "2023_08_18_00_59_53_PST",
            "URI": "https://example.com/1",
            "Description": "J pod calls",
            "Notes": "tp_human_only",
        },
        {
            "Category": "humpback",
            "NodeName": "rpi_sunset_bay",
            "Timestamp": "2024_01_01_00_00_00_PST",
            "URI": "https://example.com/2",
            "Description": "Humpback calls",
            "Notes": "tp_human_only",
        },
        {
            "Category": "other",
            "NodeName": "rpi_bush_point",
            "Timestamp": "2024_02_01_12_00_00_PST",
            "URI": "https://example.com/3",
            "Description": "Vessel noise",
            "Notes": "fp_machine_only",
        },
    ]


def _make_sample(
    row_index=0,
    category="resident",
    node_name="rpi_orcasound_lab",
    timestamp="2023_08_18_00_59_53_PST",
    uri="https://example.com/1",
    description="J pod calls",
    notes="tp_human_only",
):
    from generate_embeddings import TestSample
    return TestSample(
        row_index=row_index,
        category=category,
        node_name=node_name,
        timestamp=timestamp,
        uri=uri,
        description=description,
        notes=notes,
    )


def _make_inference_result(
    embeddings=None,
    local_predictions=None,
    local_prediction_labels=None,
    local_confidences=None,
    hop_duration=2.0,
    segment_duration=3.0,
    global_prediction_label="resident",
    global_confidence=0.75,
):
    return {
        "ast_embeddings": embeddings if embeddings is not None else [[0.1, 0.2, 0.3]],
        "local_predictions": local_predictions if local_predictions is not None else [1],
        "local_prediction_labels": (
            local_prediction_labels
            if local_prediction_labels is not None
            else ["resident"]
        ),
        "local_confidences": local_confidences if local_confidences is not None else [0.75],
        "hop_duration": hop_duration,
        "segment_duration": segment_duration,
        "global_prediction_label": global_prediction_label,
        "global_confidence": global_confidence,
    }


# ---------------------------------------------------------------------------
# Tests for load_test_samples()
# ---------------------------------------------------------------------------

class TestLoadTestSamples:
    """Tests for load_test_samples()."""

    def test_loads_all_rows(self, tmp_path):
        """load_test_samples returns one TestSample per CSV row."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", _make_rows())
        samples = load_test_samples(csv_path)

        assert len(samples) == 3

    def test_maps_fields_correctly(self, tmp_path):
        """load_test_samples maps CSV columns to TestSample attributes."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", _make_rows())
        samples = load_test_samples(csv_path)
        first = samples[0]

        assert first.row_index == 0
        assert first.category == "resident"
        assert first.node_name == "rpi_orcasound_lab"
        assert first.timestamp == "2023_08_18_00_59_53_PST"
        assert first.uri == "https://example.com/1"
        assert first.description == "J pod calls"
        assert first.notes == "tp_human_only"

    def test_row_index_increments(self, tmp_path):
        """load_test_samples assigns sequential row_index values."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", _make_rows())
        samples = load_test_samples(csv_path)

        assert [s.row_index for s in samples] == [0, 1, 2]

    def test_max_samples_limits_results(self, tmp_path):
        """load_test_samples stops after max_samples rows."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", _make_rows())
        samples = load_test_samples(csv_path, max_samples=2)

        assert len(samples) == 2

    def test_category_filter_returns_matching_rows(self, tmp_path):
        """load_test_samples returns only rows matching category_filter."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", _make_rows())
        samples = load_test_samples(csv_path, category_filter="humpback")

        assert len(samples) == 1
        assert samples[0].category == "humpback"

    def test_category_filter_no_match_returns_empty(self, tmp_path):
        """load_test_samples returns [] when no rows match category_filter."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", _make_rows())
        samples = load_test_samples(csv_path, category_filter="transient")

        assert samples == []

    def test_category_filter_combined_with_max_samples(self, tmp_path):
        """load_test_samples respects both max_samples and category_filter."""
        from generate_embeddings import load_test_samples

        rows = [
            {
                "Category": "resident",
                "NodeName": f"rpi_{i}",
                "Timestamp": f"2024_01_0{i}_00_00_00_PST",
                "URI": f"https://example.com/{i}",
                "Description": "",
                "Notes": "",
            }
            for i in range(1, 5)
        ]
        csv_path = _write_testing_csv(tmp_path / "testing.csv", rows)
        samples = load_test_samples(
            csv_path, max_samples=2, category_filter="resident"
        )

        assert len(samples) == 2
        assert all(s.category == "resident" for s in samples)

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """load_test_samples returns [] for a CSV with only a header row."""
        from generate_embeddings import load_test_samples

        csv_path = _write_testing_csv(tmp_path / "testing.csv", [])
        samples = load_test_samples(csv_path)

        assert samples == []


# ---------------------------------------------------------------------------
# Tests for find_wav_file()
# ---------------------------------------------------------------------------

class TestFindWavFile:
    """Tests for find_wav_file()."""

    def test_returns_path_when_file_exists(self, tmp_path):
        """find_wav_file returns the expected Path when the WAV file exists."""
        from generate_embeddings import find_wav_file

        sample = _make_sample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
        )

        wav_dir = tmp_path / "wav"
        expected = wav_dir / "resident" / "rpi-orcasound-lab_2023_08_18_00_59_53_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)

        assert result == expected

    def test_returns_none_when_file_missing(self, tmp_path):
        """find_wav_file returns None when the WAV file does not exist."""
        from generate_embeddings import find_wav_file

        sample = _make_sample()
        result = find_wav_file(sample, tmp_path / "wav")

        assert result is None

    def test_replaces_underscores_in_node_name(self, tmp_path):
        """find_wav_file converts underscores to hyphens in the node-name part."""
        from generate_embeddings import find_wav_file

        sample = _make_sample(
            category="humpback",
            node_name="rpi_bush_point",
            timestamp="2024_01_01_00_00_00_PST",
        )

        wav_dir = tmp_path / "wav"
        expected = wav_dir / "humpback" / "rpi-bush-point_2024_01_01_00_00_00_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)

        assert result == expected

    def test_category_used_as_subdirectory(self, tmp_path):
        """find_wav_file places the file under <wav_dir>/<category>/."""
        from generate_embeddings import find_wav_file

        sample = _make_sample(
            category="transient",
            node_name="rpi_port_townsend",
            timestamp="2024_03_01_06_00_00_PST",
        )

        wav_dir = tmp_path / "wav"
        expected = wav_dir / "transient" / "rpi-port-townsend_2024_03_01_06_00_00_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)

        assert result == expected


# ---------------------------------------------------------------------------
# Tests for _as_tensor()
# ---------------------------------------------------------------------------

class TestAsTensor:
    """Tests for _as_tensor()."""

    def test_returns_last_hidden_state_if_present(self):
        """_as_tensor returns .last_hidden_state when present."""
        from generate_embeddings import _as_tensor

        sentinel = object()
        # spec=['last_hidden_state'] ensures only that attribute exists on the mock.
        value = MagicMock(spec=["last_hidden_state"])
        value.last_hidden_state = sentinel

        result = _as_tensor(value)
        assert result is sentinel

    def test_returns_last_hidden_states_entry_if_no_last_hidden_state(self):
        """_as_tensor returns hidden_states[-1] when last_hidden_state absent."""
        from generate_embeddings import _as_tensor

        sentinel = object()
        value = MagicMock(spec=[])
        value.hidden_states = [object(), sentinel]

        result = _as_tensor(value)
        assert result is sentinel

    def test_returns_first_element_of_tuple(self):
        """_as_tensor returns index-0 element for tuples."""
        from generate_embeddings import _as_tensor

        sentinel = object()
        result = _as_tensor((sentinel, object()))
        assert result is sentinel

    def test_returns_first_element_of_list(self):
        """_as_tensor returns index-0 element for lists."""
        from generate_embeddings import _as_tensor

        sentinel = object()
        result = _as_tensor([sentinel, object()])
        assert result is sentinel

    def test_passthrough_for_plain_value(self):
        """_as_tensor returns the value unchanged when no known attribute is found."""
        from generate_embeddings import _as_tensor

        value = 42
        assert _as_tensor(value) == 42

    def test_returns_none_passthrough(self):
        """_as_tensor returns None unchanged."""
        from generate_embeddings import _as_tensor

        assert _as_tensor(None) is None

    def test_empty_tuple_returns_empty_tuple(self):
        """_as_tensor passes through an empty tuple (falsy container)."""
        from generate_embeddings import _as_tensor

        result = _as_tensor(())
        assert result == ()

    def test_empty_list_returns_empty_list(self):
        """_as_tensor passes through an empty list (falsy container)."""
        from generate_embeddings import _as_tensor

        result = _as_tensor([])
        assert result == []


# ---------------------------------------------------------------------------
# Tests for capture_ast_embeddings()
# ---------------------------------------------------------------------------

class TestCaptureAstEmbeddings:
    """Tests for capture_ast_embeddings()."""

    def test_raises_value_error_when_ast_module_missing(self):
        """capture_ast_embeddings raises ValueError when AST module is absent."""
        from generate_embeddings import capture_ast_embeddings

        mock_model = MagicMock()
        # model.model exists but has no audio_spectrogram_transformer.
        mock_model.model = MagicMock(spec=[])

        with pytest.raises(ValueError, match="Unable to locate AST module"):
            capture_ast_embeddings(mock_model)

    def test_raises_value_error_when_model_attribute_missing(self):
        """capture_ast_embeddings raises ValueError when model has no .model attr."""
        from generate_embeddings import capture_ast_embeddings

        mock_model = MagicMock(spec=[])  # No 'model' attribute.

        with pytest.raises(ValueError, match="Unable to locate AST module"):
            capture_ast_embeddings(mock_model)

    def test_returns_list_and_handle(self):
        """capture_ast_embeddings returns (embeddings_list, handle)."""
        from generate_embeddings import capture_ast_embeddings

        mock_handle = MagicMock()
        mock_ast = MagicMock()
        mock_ast.register_forward_hook.return_value = mock_handle

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast

        embeddings, handle = capture_ast_embeddings(mock_model)

        assert isinstance(embeddings, list)
        assert handle is mock_handle
        mock_ast.register_forward_hook.assert_called_once()

    def test_hook_collects_2d_tensor(self):
        """The registered hook collects 2-D tensors into embeddings."""
        import torch
        from generate_embeddings import capture_ast_embeddings

        captured_hook = None

        def fake_register(hook_fn):
            nonlocal captured_hook
            captured_hook = hook_fn
            return MagicMock()

        mock_ast = MagicMock()
        mock_ast.register_forward_hook.side_effect = fake_register

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast

        embeddings, _handle = capture_ast_embeddings(mock_model)

        # Simulate a 2-D tensor output (batch=1, embedding_dim=4).
        tensor_2d = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        captured_hook(None, None, tensor_2d)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 4

    def test_hook_squeezes_3d_tensor_to_cls_token(self):
        """The hook extracts the CLS token (index 0) from 3-D tensors."""
        import torch
        from generate_embeddings import capture_ast_embeddings

        captured_hook = None

        def fake_register(hook_fn):
            nonlocal captured_hook
            captured_hook = hook_fn
            return MagicMock()

        mock_ast = MagicMock()
        mock_ast.register_forward_hook.side_effect = fake_register

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast

        embeddings, _handle = capture_ast_embeddings(mock_model)

        # Shape (batch=1, seq=3, dim=4); CLS token is seq index 0.
        tensor_3d = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0]]])
        captured_hook(None, None, tensor_3d)

        assert len(embeddings) == 1
        # CLS token is [:, 0, :] → [1.0, 2.0, 3.0, 4.0]
        assert embeddings[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_hook_ignores_non_2d_non_3d_tensor(self):
        """The hook does nothing for tensors that are neither 2-D nor 3-D."""
        import torch
        from generate_embeddings import capture_ast_embeddings

        captured_hook = None

        def fake_register(hook_fn):
            nonlocal captured_hook
            captured_hook = hook_fn
            return MagicMock()

        mock_ast = MagicMock()
        mock_ast.register_forward_hook.side_effect = fake_register

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast

        embeddings, _handle = capture_ast_embeddings(mock_model)

        # 1-D tensor should be ignored.
        tensor_1d = torch.tensor([0.1, 0.2, 0.3])
        captured_hook(None, None, tensor_1d)

        assert embeddings == []


# ---------------------------------------------------------------------------
# Tests for write_embedding_rows()
# ---------------------------------------------------------------------------

class TestWriteEmbeddingRows:
    """Tests for write_embedding_rows()."""

    def test_returns_zero_for_empty_embeddings(self, tmp_path):
        """write_embedding_rows returns 0 when inference_result has no embeddings."""
        from generate_embeddings import write_embedding_rows

        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_2023_08_18_00_59_53_PST.wav"
        result = _make_inference_result(embeddings=[])

        count = write_embedding_rows(
            tmp_path / "out.csv", sample, wav_path, result
        )
        assert count == 0

    def test_creates_csv_with_header_on_first_write(self, tmp_path):
        """write_embedding_rows creates a new CSV with the correct header."""
        from generate_embeddings import write_embedding_rows, EMBEDDING_CSV_BASE_FIELDS

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(embeddings=[[0.1, 0.2, 0.3]])
        write_embedding_rows(out_csv, sample, wav_path, result)

        assert out_csv.exists()
        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        for base_field in EMBEDDING_CSV_BASE_FIELDS:
            assert base_field in fieldnames

        # Embedding columns.
        assert "embedding_0" in fieldnames
        assert "embedding_1" in fieldnames
        assert "embedding_2" in fieldnames

    def test_writes_correct_base_field_values(self, tmp_path):
        """write_embedding_rows writes correct metadata in each row."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample(
            row_index=7,
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="https://example.com/1",
            description="J pod calls",
            notes="tp_human_only",
        )
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(
            embeddings=[[0.5, 0.6]],
            local_predictions=[1],
            local_prediction_labels=["resident"],
            local_confidences=[0.8],
            hop_duration=2.0,
            segment_duration=3.0,
            global_prediction_label="resident",
            global_confidence=0.75,
        )

        write_embedding_rows(out_csv, sample, wav_path, result)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert row["manifest_row_index"] == "7"
        assert row["category"] == "resident"
        assert row["node_name"] == "rpi_orcasound_lab"
        assert row["timestamp"] == "2023_08_18_00_59_53_PST"
        assert row["uri"] == "https://example.com/1"
        assert row["description"] == "J pod calls"
        assert row["notes"] == "tp_human_only"
        assert row["model_type"] == "podsai"
        assert row["segment_index"] == "0"
        assert float(row["start_time_seconds"]) == pytest.approx(0.0)
        assert float(row["duration_seconds"]) == pytest.approx(3.0)
        assert row["predicted_label"] == "resident"
        assert row["predicted_class_id"] == "1"
        assert float(row["local_confidence"]) == pytest.approx(0.8)
        assert row["global_prediction_label"] == "resident"
        assert float(row["global_confidence"]) == pytest.approx(0.75)

    def test_writes_embedding_values_as_floats(self, tmp_path):
        """write_embedding_rows stores each embedding dimension as a float column."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(embeddings=[[1.1, 2.2, 3.3]])
        write_embedding_rows(out_csv, sample, wav_path, result)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["embedding_0"]) == pytest.approx(1.1)
        assert float(rows[0]["embedding_1"]) == pytest.approx(2.2)
        assert float(rows[0]["embedding_2"]) == pytest.approx(3.3)

    def test_multiple_embeddings_produce_multiple_rows(self, tmp_path):
        """write_embedding_rows writes one row per embedding."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = _make_inference_result(
            embeddings=embeddings,
            local_predictions=[1, 0, 1],
            local_prediction_labels=["resident", "water", "resident"],
            local_confidences=[0.9, 0.1, 0.8],
            hop_duration=2.0,
        )
        count = write_embedding_rows(out_csv, sample, wav_path, result)

        assert count == 3
        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["segment_index"] == "0"
        assert rows[1]["segment_index"] == "1"
        assert rows[2]["segment_index"] == "2"

    def test_start_time_computed_from_hop_duration(self, tmp_path):
        """write_embedding_rows computes start_time_seconds = index * hop_duration."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(
            embeddings=[[0.1], [0.2], [0.3]],
            local_predictions=[0, 0, 0],
            local_prediction_labels=["water", "water", "water"],
            local_confidences=[0.5, 0.5, 0.5],
            hop_duration=2.5,
        )
        write_embedding_rows(out_csv, sample, wav_path, result)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["start_time_seconds"]) == pytest.approx(0.0)
        assert float(rows[1]["start_time_seconds"]) == pytest.approx(2.5)
        assert float(rows[2]["start_time_seconds"]) == pytest.approx(5.0)

    def test_appends_to_existing_csv_without_duplicate_header(self, tmp_path):
        """write_embedding_rows appends rows to an existing CSV (no second header)."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result1 = _make_inference_result(embeddings=[[0.1, 0.2]])
        result2 = _make_inference_result(embeddings=[[0.3, 0.4]])

        write_embedding_rows(out_csv, sample, wav_path, result1)
        write_embedding_rows(out_csv, sample, wav_path, result2)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Two data rows (not three, which would happen if the header was counted).
        assert len(rows) == 2

    def test_ground_truth_label_taken_from_wav_parent_dir(self, tmp_path):
        """write_embedding_rows sets ground_truth_label to wav_path.parent.name."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample(category="resident")
        wav_path = tmp_path / "humpback" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(embeddings=[[0.1]])
        write_embedding_rows(out_csv, sample, wav_path, result)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["ground_truth_label"] == "humpback"

    def test_missing_local_predictions_uses_empty_string(self, tmp_path):
        """write_embedding_rows uses '' for predicted_class_id when local_predictions is short."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(
            embeddings=[[0.1], [0.2]],
            local_predictions=[],
            local_prediction_labels=[],
            local_confidences=[],
        )
        write_embedding_rows(out_csv, sample, wav_path, result)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["predicted_class_id"] == ""
        assert rows[1]["predicted_class_id"] == ""

    def test_returns_embedding_count(self, tmp_path):
        """write_embedding_rows returns the number of embeddings written."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(
            embeddings=[[0.1], [0.2], [0.3], [0.4]]
        )
        count = write_embedding_rows(out_csv, sample, wav_path, result)

        assert count == 4

    def test_creates_parent_directory_if_missing(self, tmp_path):
        """write_embedding_rows creates missing parent directories for the CSV."""
        from generate_embeddings import write_embedding_rows

        out_csv = tmp_path / "deep" / "nested" / "dir" / "embeddings.csv"
        sample = _make_sample()
        wav_path = tmp_path / "resident" / "rpi-orcasound-lab_ts.wav"

        result = _make_inference_result(embeddings=[[0.1]])
        write_embedding_rows(out_csv, sample, wav_path, result)

        assert out_csv.exists()


# ---------------------------------------------------------------------------
# Tests for run_ast_inference()
# ---------------------------------------------------------------------------

class TestRunAstInference:
    """Tests for run_ast_inference()."""

    def _make_model_with_ast(self, predict_return=None):
        """Build a mock model that has a valid AST module for hook registration."""
        mock_handle = MagicMock()
        mock_ast = MagicMock()
        mock_ast.register_forward_hook.return_value = mock_handle

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast

        if predict_return is None:
            predict_return = {
                "local_predictions": [1],
                "local_confidences": [0.8],
                "global_prediction": 1,
                "global_prediction_label": "resident",
                "global_confidence": 0.8,
                "hop_duration": 2.0,
                "segment_duration": 3.0,
            }

        mock_model.predict.return_value = predict_return
        mock_model.id2label = {0: "water", 1: "resident"}
        return mock_model

    def test_returns_expected_keys(self):
        """run_ast_inference returns a dict with all expected keys."""
        from generate_embeddings import run_ast_inference

        model = self._make_model_with_ast()
        result = run_ast_inference(model, "/fake/path.wav")

        expected_keys = {
            "ast_embeddings",
            "local_predictions",
            "local_prediction_labels",
            "local_confidences",
            "hop_duration",
            "segment_duration",
            "global_prediction_label",
            "global_confidence",
        }
        assert expected_keys.issubset(result.keys())

    def test_resolves_integer_predictions_via_id2label(self):
        """run_ast_inference converts integer local_predictions to label strings."""
        from generate_embeddings import run_ast_inference

        model = self._make_model_with_ast(
            predict_return={
                "local_predictions": [0, 1, 0],
                "local_confidences": [0.9, 0.8, 0.7],
                "global_prediction": 0,
                "global_prediction_label": "water",
                "global_confidence": 0.9,
                "hop_duration": 2.0,
                "segment_duration": 3.0,
            }
        )
        model.id2label = {0: "water", 1: "resident"}

        result = run_ast_inference(model, "/fake/path.wav")

        assert result["local_prediction_labels"] == ["water", "resident", "water"]

    def test_string_predictions_passed_through_unchanged(self):
        """run_ast_inference passes through string local_predictions directly."""
        from generate_embeddings import run_ast_inference

        model = self._make_model_with_ast(
            predict_return={
                "local_predictions": ["resident", "water"],
                "local_confidences": [0.7, 0.3],
                "global_prediction_label": "resident",
                "global_confidence": 0.7,
                "hop_duration": 2.0,
                "segment_duration": 3.0,
            }
        )

        result = run_ast_inference(model, "/fake/path.wav")

        assert result["local_prediction_labels"] == ["resident", "water"]

    def test_unknown_integer_prediction_uses_str_fallback(self):
        """run_ast_inference falls back to str(id) for unknown prediction integers."""
        from generate_embeddings import run_ast_inference

        model = self._make_model_with_ast(
            predict_return={
                "local_predictions": [99],
                "local_confidences": [0.5],
                "global_prediction_label": "water",
                "global_confidence": 0.5,
                "hop_duration": 2.0,
                "segment_duration": 3.0,
            }
        )
        model.id2label = {0: "water", 1: "resident"}

        result = run_ast_inference(model, "/fake/path.wav")

        assert result["local_prediction_labels"] == ["99"]

    def test_handle_removed_after_predict(self):
        """run_ast_inference removes the forward hook even after predict succeeds."""
        from generate_embeddings import run_ast_inference

        mock_handle = MagicMock()
        mock_ast = MagicMock()
        mock_ast.register_forward_hook.return_value = mock_handle

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast
        mock_model.predict.return_value = {
            "local_predictions": [],
            "local_confidences": [],
            "global_prediction_label": "water",
            "global_confidence": 0.0,
        }
        mock_model.id2label = {}

        run_ast_inference(mock_model, "/fake/path.wav")

        mock_handle.remove.assert_called_once()

    def test_handle_removed_even_if_predict_raises(self):
        """run_ast_inference removes the hook in a finally block on predict error."""
        from generate_embeddings import run_ast_inference

        mock_handle = MagicMock()
        mock_ast = MagicMock()
        mock_ast.register_forward_hook.return_value = mock_handle

        mock_model = MagicMock()
        mock_model.model.audio_spectrogram_transformer = mock_ast
        mock_model.predict.side_effect = RuntimeError("predict failed")

        with pytest.raises(RuntimeError, match="predict failed"):
            run_ast_inference(mock_model, "/fake/path.wav")

        mock_handle.remove.assert_called_once()

    def test_uses_default_hop_and_segment_duration_when_absent(self):
        """run_ast_inference uses 2.0s hop and 3.0s segment when predict omits them."""
        from generate_embeddings import run_ast_inference

        model = self._make_model_with_ast(
            predict_return={
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction_label": "water",
                "global_confidence": 0.0,
                # hop_duration and segment_duration intentionally absent.
            }
        )

        result = run_ast_inference(model, "/fake/path.wav")

        assert result["hop_duration"] == pytest.approx(2.0)
        assert result["segment_duration"] == pytest.approx(3.0)

    def test_ast_embeddings_list_returned(self):
        """run_ast_inference returns ast_embeddings as a list."""
        from generate_embeddings import run_ast_inference

        model = self._make_model_with_ast()
        result = run_ast_inference(model, "/fake/path.wav")

        assert isinstance(result["ast_embeddings"], list)
