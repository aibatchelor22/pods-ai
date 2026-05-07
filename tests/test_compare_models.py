#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for compare_models.py.

Tests cover:
- load_test_samples() CSV parsing
- find_wav_file() path construction
- is_resident_prediction() label mapping
- evaluate_model() with mocked run_inference
- print_summary() output
- ModelResult property calculations
- main() CLI error handling
"""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, rows, fieldnames=None):
    """Write a CSV file with the given rows."""
    if not fieldnames:
        fieldnames = ["Category", "NodeName", "Timestamp", "URI", "Description", "Notes", "Confidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_testing_rows():
    """Return a list of testing sample rows."""
    return [
        {
            "Category": "resident",
            "NodeName": "rpi_orcasound_lab",
            "Timestamp": "2023_08_18_00_59_53_PST",
            "URI": "https://example.com/1",
            "Description": "J pod calls",
            "Notes": "tp_human_only",
            "Confidence": "",
        },
        {
            "Category": "human",
            "NodeName": "rpi_sunset_bay",
            "Timestamp": "2024_08_07_11_23_23_PST",
            "URI": "https://example.com/2",
            "Description": "Human voices",
            "Notes": "fp_machine_only",
            "Confidence": "62.3839",
        },
        {
            "Category": "humpback",
            "NodeName": "rpi_orcasound_lab",
            "Timestamp": "2023_10_28_07_33_52_PST",
            "URI": "https://example.com/3",
            "Description": "Humpback",
            "Notes": "tp_human_only",
            "Confidence": "",
        },
    ]


# ---------------------------------------------------------------------------
# Tests for load_test_samples()
# ---------------------------------------------------------------------------

class TestLoadTestSamples:
    """Tests for load_test_samples()."""

    def test_loads_all_samples(self, tmp_path):
        """load_test_samples loads all rows from testing_samples.csv."""
        from compare_models import load_test_samples
        testing_rows = _make_testing_rows()

        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, testing_rows)

        samples = load_test_samples(testing_csv)
        assert len(samples) == 3
        uris = {s.uri for s in samples}
        assert "https://example.com/1" in uris
        assert "https://example.com/2" in uris
        assert "https://example.com/3" in uris

    def test_parses_fields_correctly(self, tmp_path):
        """load_test_samples correctly maps CSV columns to TestSample fields."""
        from compare_models import load_test_samples
        testing_rows = _make_testing_rows()

        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, testing_rows)

        samples = load_test_samples(testing_csv)
        first = samples[0]
        assert first.category == "resident"
        assert first.node_name == "rpi_orcasound_lab"
        assert first.timestamp == "2023_08_18_00_59_53_PST"
        assert first.uri == "https://example.com/1"
        assert first.notes == "tp_human_only"

    def test_returns_empty_list_for_missing_file(self, tmp_path):
        """load_test_samples returns [] when testing_samples.csv is missing."""
        from compare_models import load_test_samples

        samples = load_test_samples(Path("/nonexistent/testing_samples.csv"))
        assert samples == []

    def test_respects_max_samples_limit(self, tmp_path):
        """load_test_samples respects max_samples parameter."""
        from compare_models import load_test_samples
        testing_rows = _make_testing_rows()

        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, testing_rows)

        samples = load_test_samples(testing_csv, max_samples=2)
        assert len(samples) == 2

    def test_category_filter_returns_only_matching_samples(self, tmp_path):
        """load_test_samples returns only samples matching the category filter."""
        from compare_models import load_test_samples
        testing_rows = _make_testing_rows()

        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, testing_rows)

        samples = load_test_samples(testing_csv, category_filter="resident")
        assert len(samples) == 1
        assert all(s.category == "resident" for s in samples)

    def test_category_filter_returns_empty_for_no_match(self, tmp_path):
        """load_test_samples returns [] when category filter matches no rows."""
        from compare_models import load_test_samples
        testing_rows = _make_testing_rows()

        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, testing_rows)

        samples = load_test_samples(testing_csv, category_filter="transient")
        assert samples == []

    def test_category_filter_combined_with_max_samples(self, tmp_path):
        """load_test_samples applies both category_filter and max_samples."""
        from compare_models import load_test_samples

        rows = [
            {"Category": "humpback", "NodeName": "rpi_a", "Timestamp": "2024_01_01_00_00_00_PST",
             "URI": "https://example.com/1", "Description": "", "Notes": "tp_machine_only", "Confidence": ""},
            {"Category": "humpback", "NodeName": "rpi_b", "Timestamp": "2024_01_01_00_01_00_PST",
             "URI": "https://example.com/2", "Description": "", "Notes": "tp_machine_only", "Confidence": ""},
            {"Category": "humpback", "NodeName": "rpi_c", "Timestamp": "2024_01_01_00_02_00_PST",
             "URI": "https://example.com/3", "Description": "", "Notes": "tp_machine_only", "Confidence": ""},
        ]
        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, rows)

        samples = load_test_samples(testing_csv, max_samples=2, category_filter="humpback")
        assert len(samples) == 2
        assert all(s.category == "humpback" for s in samples)

    def test_handles_csv_error(self, tmp_path):
        """load_test_samples returns [] on csv.Error."""
        import csv
        from unittest.mock import patch
        from compare_models import load_test_samples

        testing_csv = tmp_path / "testing_samples.csv"
        testing_csv.write_text("Category,NodeName,Timestamp,URI,Description,Notes\n")

        # Patch DictReader iteration to raise csv.Error mid-read.
        with patch("compare_models.csv.DictReader") as mock_reader_cls:
            mock_reader_cls.return_value.__iter__ = lambda self: iter([])
            mock_reader_cls.return_value.__enter__ = lambda self: self
            mock_reader_cls.side_effect = csv.Error("simulated CSV parse error")
            samples = load_test_samples(testing_csv)

        assert samples == []

    def test_handles_unicode_decode_error(self, tmp_path):
        """load_test_samples returns [] for files with encoding issues."""
        from compare_models import load_test_samples
        
        testing_csv = tmp_path / "testing_samples.csv"
        # Write file with invalid UTF-8
        with open(testing_csv, "wb") as f:
            f.write(b"Category,NodeName,Timestamp,URI,Description,Notes\n")
            f.write(b"resident,rpi_lab,2023_01_01_00_00_00_PST,http://example.com,test\x8f,notes\n")
        
        samples = load_test_samples(testing_csv)
        assert samples == []


# ---------------------------------------------------------------------------
# Tests for find_wav_file()
# ---------------------------------------------------------------------------

class TestFindWavFile:
    """Tests for find_wav_file()."""

    def test_returns_path_when_wav_exists(self, tmp_path):
        """find_wav_file returns the correct path when the WAV file is present."""
        from compare_models import TestSample, find_wav_file

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = tmp_path / "testing-wav"
        expected = wav_dir / "resident" / "rpi-orcasound-lab_2023_08_18_00_59_53_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)
        assert result == expected

    def test_returns_none_when_wav_missing(self, tmp_path):
        """find_wav_file returns None when the WAV file does not exist."""
        from compare_models import TestSample, find_wav_file

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        result = find_wav_file(sample, wav_dir)
        assert result is None

    def test_replaces_underscores_with_dashes_in_node_name(self, tmp_path):
        """find_wav_file converts underscores to dashes in the node name portion of the filename."""
        from compare_models import TestSample, find_wav_file

        sample = TestSample(
            category="human",
            node_name="rpi_sunset_bay",
            timestamp="2024_08_07_11_23_23_PST",
            uri="",
            description="",
            notes="fp_machine_only",
        )
        wav_dir = tmp_path / "testing-wav"
        expected = wav_dir / "human" / "rpi-sunset-bay_2024_08_07_11_23_23_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)
        assert result == expected


# ---------------------------------------------------------------------------
# Tests for is_resident_prediction()
# ---------------------------------------------------------------------------

class TestIsResidentPrediction:
    """Tests for is_resident_prediction()."""

    def test_fastai_resident_is_resident(self):
        """FastAI "resident" prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "fastai") is True

    def test_fastai_other_is_not_resident(self):
        """FastAI "other" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("other", "fastai") is False

    def test_orcahello_resident_is_resident(self):
        """OrcaHello "resident" prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "orcahello") is True

    def test_orcahello_other_is_not_resident(self):
        """OrcaHello "other" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("other", "orcahello") is False

    def test_podsai_resident_is_resident(self):
        """PODS-AI "resident" prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "podsai") is True

    def test_podsai_water_is_not_resident(self):
        """PODS-AI "water" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("water", "podsai") is False

    def test_podsai_transient_is_not_resident(self):
        """PODS-AI "transient" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("transient", "podsai") is False

    def test_podsai_humpback_is_not_resident(self):
        """PODS-AI "humpback" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("humpback", "podsai") is False

    def test_podsai_human_is_not_resident(self):
        """PODS-AI "human" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("human", "podsai") is False

    def test_podsai_vessel_is_not_resident(self):
        """PODS-AI "vessel" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("vessel", "podsai") is False


# ---------------------------------------------------------------------------
# Tests for ModelResult properties
# ---------------------------------------------------------------------------

class TestModelResultProperties:
    """Tests for ModelResult computed properties."""

    def test_evaluated_excludes_skipped(self):
        """evaluated = total - skipped."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, skipped=3)
        assert r.evaluated == 7

    def test_accuracy_none_when_no_evaluated(self):
        """accuracy is None when evaluated == 0."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=5, skipped=5)
        assert r.accuracy is None

    def test_accuracy_correct_fraction(self):
        """accuracy is correct/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, correct=8, skipped=0)
        assert abs(r.accuracy - 0.8) < 1e-9

    def test_false_positive_rate_none_when_no_evaluated(self):
        """false_positive_rate is None when evaluated == 0."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=3, skipped=3)
        assert r.false_positive_rate is None

    def test_false_positive_rate_correct_fraction(self):
        """false_positive_rate is false_positives/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, false_positives=2, skipped=0)
        assert abs(r.false_positive_rate - 0.2) < 1e-9

    def test_false_negative_rate_correct_fraction(self):
        """false_negative_rate is false_negatives/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, false_negatives=1, skipped=0)
        assert abs(r.false_negative_rate - 0.1) < 1e-9

    def test_avg_predict_time_none_when_no_times(self):
        """avg_predict_time is None when predict_times is empty."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=5, skipped=5)
        assert r.avg_predict_time is None

    def test_avg_predict_time_calculates_mean(self):
        """avg_predict_time is the mean of predict_times."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=3, predict_times=[1.0, 2.0, 3.0])
        assert abs(r.avg_predict_time - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests for evaluate_model()
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    """Tests for evaluate_model() with mocked run_inference."""

    def _make_wav_files(self, tmp_path, samples):
        """Create dummy WAV files for the given samples under tmp_path/testing-wav."""
        wav_dir = tmp_path / "testing-wav"
        for sample in samples:
            node_name_in_filename = sample.node_name.replace("_", "-")
            wav_filename = f"{node_name_in_filename}_{sample.timestamp}.wav"
            wav_file = wav_dir / sample.category / wav_filename
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            wav_file.touch()
        return wav_dir

    def test_correct_resident_prediction_counted(self, tmp_path):
        """A resident sample predicted as "resident" (fastai) counts as correct."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.skipped == 0

    def test_false_positive_counted(self, tmp_path):
        """A non-resident sample predicted as "resident" (fastai) counts as false positive."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="human",
            node_name="rpi_sunset_bay",
            timestamp="2024_08_07_11_23_23_PST",
            uri="",
            description="",
            notes="fp_machine_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.7, "predict_time": 1.2}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 0
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_false_negative_counted(self, tmp_path):
        """A resident sample predicted as "other" (fastai) counts as false negative."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.1, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 0
        assert result.false_positives == 0
        assert result.false_negatives == 1

    def test_skips_sample_when_wav_missing(self, tmp_path):
        """Samples whose WAV file is missing are counted as skipped."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        with patch("compare_models.run_inference") as mock_infer:
            result = evaluate_model("fastai", "./model", [sample], wav_dir)
            mock_infer.assert_not_called()

        assert result.skipped == 1
        assert result.correct == 0

    def test_skips_sample_on_inference_error(self, tmp_path):
        """Samples that raise an exception during inference are counted as skipped."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        with patch("compare_models.run_inference", side_effect=RuntimeError("model error")):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.skipped == 1
        assert result.correct == 0

    def test_podsai_resident_prediction_correct(self, tmp_path):
        """PODS-AI "resident" prediction for a resident sample counts as correct."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 2.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_podsai_water_prediction_correct_for_non_resident(self, tmp_path):
        """PODS-AI "water" prediction for a non-resident sample counts as correct."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="human",
            node_name="rpi_sunset_bay",
            timestamp="2024_08_07_11_23_23_PST",
            uri="",
            description="",
            notes="fp_machine_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "water", "global_confidence": 0.8, "predict_time": 1.8}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_total_equals_sample_count(self, tmp_path):
        """ModelResult.total always equals the number of samples passed."""
        from compare_models import TestSample, evaluate_model

        samples = [
            TestSample("resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST", "", "", "tp_human_only"),
            TestSample("human", "rpi_sunset_bay", "2024_08_07_11_23_23_PST", "", "", "fp_machine_only"),
        ]
        wav_dir = self._make_wav_files(tmp_path, samples)

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.1, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", samples, wav_dir)

        assert result.total == 2

    def test_records_predict_times(self, tmp_path):
        """evaluate_model records predict_time from inference results."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 2.5}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert len(result.predict_times) == 1
        assert abs(result.predict_times[0] - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# Tests for print_summary()
# ---------------------------------------------------------------------------

class TestPrintSummary:
    """Tests for print_summary() output formatting."""

    def test_prints_header_and_separator(self, capsys):
        """print_summary prints the "Model Comparison Summary" header."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Model Comparison Summary" in captured

    def test_prints_model_name(self, capsys):
        """print_summary includes the model type in the output."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="orcahello", total=5, correct=3, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "orcahello" in captured

    def test_prints_accuracy_percentage(self, capsys):
        """print_summary shows accuracy as a percentage."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=10, correct=8, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "80.0%" in captured

    def test_prints_na_when_no_evaluated_samples(self, capsys):
        """print_summary shows N/A for accuracy when all samples are skipped."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, skipped=5)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "N/A" in captured

    def test_prints_skipped_count_when_nonzero(self, capsys):
        """print_summary shows the skipped count when some samples were skipped."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=10, correct=5, skipped=3)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "3 skipped" in captured

    def test_prints_definitions(self, capsys):
        """print_summary includes the definitions block."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Definitions:" in captured
        assert "false+" in captured or "FP" in captured

    def test_prints_avg_time(self, capsys):
        """print_summary includes average time column."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0, predict_times=[1.5, 2.0, 1.8, 2.2, 1.7])]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Avg Time" in captured


# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Tests for the main() entry point."""

    def _write_testing_csv(self, tmp_path, testing_rows):
        """Write testing_samples.csv to tmp_path."""
        testing_csv = tmp_path / "testing_samples.csv"
        _write_csv(testing_csv, testing_rows)
        return testing_csv

    def test_returns_1_for_missing_testing_csv(self, tmp_path):
        """main() returns 1 when testing_samples.csv does not exist."""
        from compare_models import main
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", str(tmp_path / "nonexistent.csv"),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_missing_wav_dir(self, tmp_path):
        """main() returns 1 when the WAV directory does not exist."""
        from compare_models import main

        testing_csv = self._write_testing_csv(tmp_path, _make_testing_rows())
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(tmp_path / "nonexistent-wav-dir"),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_unknown_model(self, tmp_path):
        """main() returns 1 when an unrecognised model type is specified."""
        from compare_models import main

        testing_csv = self._write_testing_csv(tmp_path, _make_testing_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "unknown_model",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_0_on_success_with_fastai(self, tmp_path):
        """main() returns 0 when it successfully evaluates fastai on test samples."""
        from compare_models import main

        rows = _make_testing_rows()
        testing_csv = self._write_testing_csv(tmp_path, rows)

        wav_dir = tmp_path / "testing-wav"
        for row in rows:
            node = row["NodeName"].replace("_", "-")
            wav = wav_dir / row["Category"] / f"{node}_{row['Timestamp']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--fastai-model-path", "./model",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.run_inference", return_value=mock_result):
                result = main()
        assert result == 0

    def test_returns_1_when_no_test_samples(self, tmp_path):
        """main() returns 1 when testing_samples.csv is empty."""
        from compare_models import main

        testing_csv = self._write_testing_csv(tmp_path, [])
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_respects_max_samples_argument(self, tmp_path):
        """main() respects --max-samples argument."""
        from compare_models import main

        rows = _make_testing_rows()
        testing_csv = self._write_testing_csv(tmp_path, rows)

        wav_dir = tmp_path / "testing-wav"
        for row in rows[:2]:  # Only create WAVs for first 2
            node = row["NodeName"].replace("_", "-")
            wav = wav_dir / row["Category"] / f"{node}_{row['Timestamp']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--max-samples", "2",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.run_inference", return_value=mock_result) as mock_infer:
                result = main()
                # Should only call inference twice (max 2 samples)
                assert mock_infer.call_count == 2
        assert result == 0

    def test_returns_1_for_invalid_max_samples(self, tmp_path):
        """main() returns 1 when --max-samples is zero or negative."""
        from compare_models import main

        testing_csv = self._write_testing_csv(tmp_path, _make_testing_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--max-samples", "0",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_category_filter_returns_only_matching_samples(self, tmp_path):
        """main() --category filters samples to the specified category."""
        from compare_models import main

        rows = _make_testing_rows()
        testing_csv = self._write_testing_csv(tmp_path, rows)

        wav_dir = tmp_path / "testing-wav"
        # Only create WAV for the resident sample.
        resident_row = rows[0]
        node = resident_row["NodeName"].replace("_", "-")
        wav = wav_dir / resident_row["Category"] / f"{node}_{resident_row['Timestamp']}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.touch()

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 1.0}
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--fastai-model-path", "./model",
            "--category", "resident",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.run_inference", return_value=mock_result) as mock_infer:
                result = main()
                # Only the one resident sample should be evaluated.
                assert mock_infer.call_count == 1
        assert result == 0

    def test_returns_1_for_category_with_no_samples(self, tmp_path):
        """main() returns 1 when --category matches no rows in testing_samples.csv."""
        from compare_models import main

        testing_csv = self._write_testing_csv(tmp_path, _make_testing_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", str(testing_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--category", "transient",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1


# ---------------------------------------------------------------------------
# Tests for ModelResult confusion_matrix tracking
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    """Tests for per-class confusion matrix tracking in ModelResult and evaluate_model()."""

    def _make_sample(self, category, node_name="rpi_orcasound_lab", timestamp="2023_08_18_00_59_53_PST"):
        """Return a TestSample with the given category."""
        from compare_models import TestSample
        return TestSample(
            category=category,
            node_name=node_name,
            timestamp=timestamp,
            uri="",
            description="",
            notes="",
        )

    def _make_wav(self, tmp_path, sample):
        """Create a dummy WAV file for the sample and return the wav_dir."""
        from compare_models import TestSample
        wav_dir = tmp_path / "testing-wav"
        node_name_in_filename = sample.node_name.replace("_", "-")
        wav_file = wav_dir / sample.category / f"{node_name_in_filename}_{sample.timestamp}.wav"
        wav_file.parent.mkdir(parents=True, exist_ok=True)
        wav_file.touch()
        return wav_dir

    def test_confusion_matrix_populated_on_correct_prediction(self, tmp_path):
        """evaluate_model records actual→predicted in confusion_matrix for correct predictions."""
        from compare_models import evaluate_model

        sample = self._make_sample("resident")
        wav_dir = self._make_wav(tmp_path, sample)

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/model", [sample], wav_dir)

        assert result.confusion_matrix == {"resident": {"resident": 1}}

    def test_confusion_matrix_populated_on_false_positive(self, tmp_path):
        """evaluate_model records actual→predicted in confusion_matrix for false positives."""
        from compare_models import evaluate_model

        sample = self._make_sample("humpback")
        wav_dir = self._make_wav(tmp_path, sample)

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/model", [sample], wav_dir)

        assert result.confusion_matrix == {"humpback": {"resident": 1}}

    def test_confusion_matrix_not_updated_for_skipped_samples(self, tmp_path):
        """Skipped samples (missing WAV) do not appear in the confusion matrix."""
        from compare_models import evaluate_model

        sample = self._make_sample("resident")
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()  # WAV file does not exist.

        with patch("compare_models.run_inference") as mock_infer:
            result = evaluate_model("fastai", "./model", [sample], wav_dir)
            mock_infer.assert_not_called()

        assert result.confusion_matrix == {}

    def test_confusion_matrix_accumulates_multiple_samples(self, tmp_path):
        """evaluate_model accumulates counts across multiple samples."""
        from compare_models import evaluate_model, TestSample

        samples = [
            TestSample("resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST", "", "", ""),
            TestSample("resident", "rpi_orcasound_lab", "2023_08_19_00_00_00_PST", "", "", ""),
            TestSample("humpback", "rpi_sunset_bay", "2023_08_20_00_00_00_PST", "", "", ""),
        ]

        wav_dir = tmp_path / "testing-wav"
        for s in samples:
            node = s.node_name.replace("_", "-")
            wav = wav_dir / s.category / f"{node}_{s.timestamp}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()

        def fake_infer(wav_path, model_type, model_path):
            if "humpback" in str(wav_path):
                return {"global_prediction_label": "water", "global_confidence": 0.7, "predict_time": 1.0}
            return {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 1.0}

        with patch("compare_models.run_inference", side_effect=fake_infer):
            result = evaluate_model("podsai", "/model", samples, wav_dir)

        assert result.confusion_matrix["resident"]["resident"] == 2
        assert result.confusion_matrix["humpback"]["water"] == 1

    def test_confusion_matrix_empty_when_no_matrix(self, capsys):
        """print_confusion_matrix does nothing when confusion_matrix is empty."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(model_type="fastai")
        print_confusion_matrix(result)
        captured = capsys.readouterr().out
        assert captured == ""

    def test_print_confusion_matrix_contains_labels(self, capsys):
        """print_confusion_matrix includes all seen labels in header and rows."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="podsai",
            confusion_matrix={
                "resident": {"resident": 5, "water": 1},
                "humpback": {"water": 3, "humpback": 2},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        assert "resident" in captured
        assert "humpback" in captured
        assert "water" in captured

    def test_print_confusion_matrix_shows_correct_counts(self, capsys):
        """print_confusion_matrix displays the right numeric values."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "resident": {"resident": 7, "other": 2},
                "other": {"resident": 1, "other": 9},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        assert "7" in captured
        assert "2" in captured
        assert "1" in captured
        assert "9" in captured

    def test_print_confusion_matrix_zero_for_unseen_pairs(self, capsys):
        """print_confusion_matrix shows 0 for class pairs that never occurred."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="podsai",
            confusion_matrix={
                "resident": {"resident": 3},
                "humpback": {"humpback": 4},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        # "resident" predicted as "humpback" should be 0, shown somewhere.
        lines = [line for line in captured.splitlines() if "resident" in line and not line.strip().startswith("Confusion")]
        # The resident row should contain a zero for the humpback column.
        assert any("0" in line for line in lines)

    def test_print_confusion_matrix_omits_all_zero_rows(self, capsys):
        """print_confusion_matrix omits rows where every predicted count is zero."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "resident": {"other": 9, "resident": 1},
                "other": {},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        # The "other" actual row is all-zero and must not appear as a row label.
        lines = captured.splitlines()
        row_lines = [line for line in lines if not line.strip().startswith("Confusion") and line.strip()]
        row_labels = [line.split()[0] for line in row_lines[1:]]  # skip header line
        assert "other" not in row_labels

    def test_print_confusion_matrix_omits_all_zero_columns(self, capsys):
        """print_confusion_matrix omits columns where every count across all rows is zero."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "human": {"other": 1, "resident": 1},
                "resident": {"other": 9, "resident": 1},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        # Only "other" and "resident" columns were ever predicted; "human" must not appear.
        header_line = [line for line in captured.splitlines() if "other" in line and "resident" in line][0]
        assert "human" not in header_line

    def test_print_summary_includes_confusion_matrices(self, capsys):
        """print_summary prints confusion matrices for each model after the table."""
        from compare_models import ModelResult, print_summary

        results = [
            ModelResult(
                model_type="fastai",
                total=2,
                correct=2,
                confusion_matrix={"resident": {"resident": 2}},
            ),
        ]
        print_summary(results)
        captured = capsys.readouterr().out

        assert "Confusion Matrix for fastai" in captured
        assert "resident" in captured
