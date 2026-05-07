# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for add_samples.py."""

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from add_samples import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    HOP_DURATION,
    SEGMENT_DURATION,
    add_samples,
    format_timestamp_pst,
    get_segment_prediction,
    parse_node_and_timestamp_from_filename,
    parse_timestamp_pst,
    split_wav_into_segments,
)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


class TestParseTimestampPst:
    """Tests for parse_timestamp_pst."""

    def test_parses_timestamp_with_pst_suffix(self):
        """parse_timestamp_pst should strip _PST and return a localized datetime."""
        dt = parse_timestamp_pst("2025_01_15_12_30_00_PST")
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 0

    def test_parses_timestamp_without_pst_suffix(self):
        """parse_timestamp_pst should also accept strings without _PST."""
        dt = parse_timestamp_pst("2025_01_15_12_30_00")
        assert dt.year == 2025
        assert dt.hour == 12


class TestFormatTimestampPst:
    """Tests for format_timestamp_pst."""

    def test_roundtrip(self):
        """Formatting a parsed timestamp should reproduce the original string."""
        original = "2025_06_01_08_15_30_PST"
        dt = parse_timestamp_pst(original)
        assert format_timestamp_pst(dt) == original

    def test_adds_pst_suffix(self):
        """format_timestamp_pst should always end with _PST."""
        dt = parse_timestamp_pst("2025_01_01_00_00_00_PST")
        assert format_timestamp_pst(dt).endswith("_PST")


# ---------------------------------------------------------------------------
# parse_node_and_timestamp_from_filename
# ---------------------------------------------------------------------------


class TestParseNodeAndTimestampFromFilename:
    """Tests for parse_node_and_timestamp_from_filename."""

    def test_parses_standard_filename(self):
        """Should extract node name (with underscores) and timestamp from a well-formed filename."""
        node, ts = parse_node_and_timestamp_from_filename(
            "rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav"
        )
        assert node == "rpi_orcasound_lab"
        assert ts == "2025_12_17_22_34_03_PST"

    def test_parses_path_with_directory(self):
        """Should ignore leading directory components and parse only the basename."""
        node, ts = parse_node_and_timestamp_from_filename(
            "/some/path/rpi-sunset-bay_2026_01_01_00_00_00_PST.wav"
        )
        assert node == "rpi_sunset_bay"
        assert ts == "2026_01_01_00_00_00_PST"

    def test_node_name_with_single_segment(self):
        """Node names that contain no hyphens should still be parsed correctly."""
        node, ts = parse_node_and_timestamp_from_filename(
            "orcasound-lab_2025_06_01_08_15_30_PST.wav"
        )
        assert node == "orcasound_lab"
        assert ts == "2025_06_01_08_15_30_PST"

    def test_raises_for_filename_without_timestamp(self):
        """Should raise ValueError when the filename has no recognizable timestamp."""
        with pytest.raises(ValueError, match="Cannot infer"):
            parse_node_and_timestamp_from_filename("recording.wav")

    def test_raises_for_filename_missing_pst_suffix(self):
        """Should raise ValueError when _PST is absent from the timestamp portion."""
        with pytest.raises(ValueError, match="Cannot infer"):
            parse_node_and_timestamp_from_filename(
                "rpi-orcasound-lab_2025_12_17_22_34_03.wav"
            )

    def test_raises_for_non_wav_extension(self):
        """Should raise ValueError for files that do not end in .wav."""
        with pytest.raises(ValueError, match="Cannot infer"):
            parse_node_and_timestamp_from_filename(
                "rpi-orcasound-lab_2025_12_17_22_34_03_PST.mp3"
            )


# ---------------------------------------------------------------------------
# split_wav_into_segments
# ---------------------------------------------------------------------------


class TestSplitWavIntoSegments:
    """Tests for split_wav_into_segments."""

    def test_saves_correct_number_of_segments_for_60s_audio(self, tmp_path):
        """A 60-second file should produce 29 segments with 3s/2s settings."""
        # floor((60 - 3) / 2) + 1 = 29
        fake_probe = {"format": {"duration": "60.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run") as mock_run:

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert len(segments) == 29
        assert mock_run.call_count == 29

    def test_single_segment_for_short_audio(self, tmp_path):
        """Audio shorter than segment_duration should still produce one segment."""
        fake_probe = {"format": {"duration": "2.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert len(segments) == 1

    def test_filename_uses_node_name_with_hyphens(self, tmp_path):
        """Output filenames should replace underscores in the node name with hyphens."""
        fake_probe = {"format": {"duration": "3.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert len(segments) == 1
        name = segments[0][0].name
        assert name.startswith("rpi-orcasound-lab_")

    def test_first_segment_timestamp_matches_base(self, tmp_path):
        """The first segment's filename should encode the base timestamp."""
        fake_probe = {"format": {"duration": "3.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert "2025_01_15_12_30_00_PST" in segments[0][0].name

    def test_second_segment_timestamp_incremented_by_hop(self, tmp_path):
        """Each subsequent segment should be offset by hop_duration seconds."""
        fake_probe = {"format": {"duration": "10.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        # Second segment should start 2 seconds later.
        assert "2025_01_15_12_30_02_PST" in segments[1][0].name

    def test_returns_empty_on_probe_failure(self, tmp_path, capsys):
        """split_wav_into_segments should return [] and print an error if probing fails."""
        with patch("add_samples.ffmpeg.probe", side_effect=Exception("probe error")):
            segments = split_wav_into_segments(
                wav_file="missing.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert segments == []
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_skips_existing_segments(self, tmp_path, capsys):
        """split_wav_into_segments should skip segments that already exist."""
        fake_probe = {"format": {"duration": "3.0"}}
        # Pre-create the expected output file.
        existing = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
        existing.write_bytes(b"existing")

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.run") as mock_run:

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        # ffmpeg.run should not be called for an already-existing segment.
        mock_run.assert_not_called()
        assert len(segments) == 1
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    def test_output_dir_is_created(self, tmp_path):
        """split_wav_into_segments should create the output directory if needed."""
        fake_probe = {"format": {"duration": "3.0"}}
        new_dir = tmp_path / "new_subdir"
        assert not new_dir.exists()

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=new_dir,
            )

        assert new_dir.exists()


# ---------------------------------------------------------------------------
# get_segment_prediction
# ---------------------------------------------------------------------------


class TestGetSegmentPrediction:
    """Tests for get_segment_prediction."""

    def test_podsai_returns_label_and_confidence(self, tmp_path):
        """Prediction returns both label and confidence from model output."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "humpback",
            "global_confidence": 0.85
        }

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "humpback"
        assert confidence == 0.85

    def test_returns_unknown_and_zero_confidence_when_label_missing(self, tmp_path):
        """get_segment_prediction should return ('unknown', 0.0) when key is absent."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {}

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "unknown"
        assert confidence == 0.0

    def test_returns_label_with_default_confidence_when_confidence_missing(self, tmp_path):
        """get_segment_prediction should return label with 0.0 confidence if confidence key is absent."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction_label": "resident"}

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "resident"
        assert confidence == 0.0

    def test_returns_unknown_and_zero_on_inference_failure(self, tmp_path, capsys):
        """get_segment_prediction should return ('unknown', 0.0) if inference raises."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "unknown"
        assert confidence == 0.0
        captured = capsys.readouterr()
        assert "Warning" in captured.err


# ---------------------------------------------------------------------------
# add_samples (integration-style)
# ---------------------------------------------------------------------------


class TestAddSamples:
    """Integration-style tests for add_samples."""

    def _fake_split(self, tmp_path):
        """Return a fake segments list with two pre-created files."""
        seg1 = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
        seg2 = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav"
        seg1.write_bytes(b"")
        seg2.write_bytes(b"")
        return [
            (seg1, "2025_01_15_12_30_00_PST"),
            (seg2, "2025_01_15_12_30_02_PST"),
        ]

    def test_returns_list_of_dicts_with_manual_samples_format(self, tmp_path):
        """add_samples should return list of dicts matching manual_samples.csv format."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.92
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            results = add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        assert len(results) == 2
        for row in results:
            assert isinstance(row, dict)
            assert row["Category"] == "water"
            assert row["NodeName"] == "rpi_orcasound_lab"
            assert row["Timestamp"] in ["2025_01_15_12_30_00_PST", "2025_01_15_12_30_02_PST"]
            assert row["URI"] == "https://example.com/test"
            assert row["Description"] == ""
            assert row["Notes"] == "manual"
            assert row["Confidence"] == "92.0"  # Confidence is percentage string (0.92 * 100 = 92.0)

    def test_default_model_path_is_used(self, tmp_path):
        """add_samples should use DEFAULT_MODEL_PATH when model_path is not provided."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.75
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
            )

        mock_get_model.assert_called_once_with(
            model_type="podsai", model_path=DEFAULT_MODEL_PATH
        )

    def test_returns_empty_when_no_segments(self, tmp_path):
        """add_samples should return [] if split_wav_into_segments yields nothing."""
        with patch("add_samples.split_wav_into_segments", return_value=[]):
            results = add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
            )

        assert results == []

    def test_model_loaded_once_for_all_segments(self, tmp_path):
        """The model should be loaded exactly once regardless of the number of segments."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.80
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        assert mock_get_model.call_count == 1
        assert mock_model.predict.call_count == 2

    def test_always_uses_podsai_model_type(self, tmp_path):
        """add_samples should always call get_model_inference with model_type='podsai'."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.70
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        call_args = mock_get_model.call_args
        assert call_args[1]["model_type"] == "podsai"

    def test_infers_node_and_timestamp_from_filename(self, tmp_path):
        """add_samples should parse node_name and base_timestamp from a well-formed filename."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.65
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments) as mock_split, \
             patch("add_samples.get_model_inference", return_value=mock_model), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_samples(
                wav_file="rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        # Verify the inferred values were passed to split_wav_into_segments.
        call_kwargs = mock_split.call_args
        assert call_kwargs[0][1] == "rpi_orcasound_lab"         # node_name
        assert call_kwargs[0][2] == "2025_01_15_12_30_00_PST"   # base_timestamp

    def test_explicit_args_override_filename_inference(self, tmp_path):
        """Explicit node_name and base_timestamp should take precedence over filename."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.88
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments) as mock_split, \
             patch("add_samples.get_model_inference", return_value=mock_model), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_samples(
                wav_file="rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav",
                node_name="rpi_sunset_bay",
                base_timestamp="2026_06_01_00_00_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        call_kwargs = mock_split.call_args
        assert call_kwargs[0][1] == "rpi_sunset_bay"
        assert call_kwargs[0][2] == "2026_06_01_00_00_00_PST"

    def test_raises_for_bad_filename_when_no_explicit_args(self):
        """add_samples should raise ValueError when the filename cannot be parsed and no args given."""
        with pytest.raises(ValueError, match="Cannot infer"):
            add_samples(wav_file="recording.wav")
