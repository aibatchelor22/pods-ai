# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for the write_training_samples pipeline in extract_training_samples.py.

Tests focus on the process_sample helper, which contains the core logic for
adjusting timestamps, URIs, and confidence values for each training sample.
"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from extract_training_samples import (
    process_sample,
    remove_zero_confidence_detections,
    select_testing_samples,
    sort_by_preference,
    write_testing_samples,
    write_training_samples,
)


# ---------------------------------------------------------------------------
# Test fixtures: raw input rows (as they appear in detections.csv)
# ---------------------------------------------------------------------------

TP_HUMAN_ONLY_INPUT = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_12_12_01_14_55_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-12-12T09%3A14%3A55.000Z',
    'Description': 'Humpback!',
    'Notes': 'tp_human_only',
    'Confidence': '',
}

TP_HUMAN_ONLY_EXPECTED = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_12_12_01_14_17_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-12-12T09%3A14%3A17.000Z',
    'Description': 'Humpback!',
    'Notes': 'tp_human_only',
    'Confidence': '43.2',
}

TP_MACHINE_ONLY_INPUT = {
    'Category': 'humpback',
    'NodeName': 'rpi_orcasound_lab',
    'Timestamp': '2025_12_17_22_33_50_PST',
    'URI': 'https://live.orcasound.net/bouts/new/orcasound-lab?time=2025-12-18T06%3A33%3A50.000Z',
    'Description': 'Humpback whale song',
    'Notes': 'tp_machine_only',
    'Confidence': '58.3503',
}

TP_MACHINE_ONLY_EXPECTED = {
    'Category': 'humpback',
    'NodeName': 'rpi_orcasound_lab',
    'Timestamp': '2025_12_17_22_33_47_PST',
    'URI': 'https://live.orcasound.net/bouts/new/orcasound-lab?time=2025-12-18T06%3A33%3A47.000Z',
    'Description': 'Humpback whale song',
    'Notes': 'tp_machine_only',
    'Confidence': '58.4',
}

# Manual timestamp correction: tp_human_only with a manual override in manual_timestamps.csv.
MANUAL_TIMESTAMP_INPUT = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_11_24_20_13_43_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-11-25T04%3A13%3A43.000Z',
    'Description': 'Distant humpback calls at 20:14.',
    'Notes': 'tp_human_only',
    'Confidence': '',
}

MANUAL_TIMESTAMP_EXPECTED = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_11_24_20_14_00_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-11-25T04%3A14%3A00.000Z',
    'Description': 'Distant humpback calls at 20:14.',
    'Notes': 'tp_human_only',
    'Confidence': '100.0',
}

# ubuntu/windows discrepancy: resident tp_human_only from rpi_bush_point.
RESIDENT_TP_HUMAN_ONLY_INPUT = {
    'Category': 'resident',
    'NodeName': 'rpi_bush_point',
    'Timestamp': '2023_11_28_14_12_51_PST',
    'URI': 'https://live.orcasound.net/bouts/new/bush-point?time=2023-11-28T22%3A12%3A51.000Z',
    'Description': 'J pod, getting louder now',
    'Notes': 'tp_human_only',
    'Confidence': '',
}

RESIDENT_TP_HUMAN_ONLY_EXPECTED = {
    'Category': 'resident',
    'NodeName': 'rpi_bush_point',
    'Timestamp': '2023_11_28_14_12_43_PST',
    'URI': 'https://live.orcasound.net/bouts/new/bush-point?time=2023-11-28T22%3A12%3A43.000Z',
    'Description': 'J pod, getting louder now',
    'Notes': 'tp_human_only',
    'Confidence': '86.6',
}

# Zero-confidence detection: manual_confidences marks this as 0.0, so it must be filtered out.
ZERO_CONFIDENCE_INPUT = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_12_13_18_33_00_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-12-14T02%3A33%3A00.000Z',
    'Description': 'Humpback',
    'Notes': 'tp_human_only',
    'Confidence': '',
}

# ---------------------------------------------------------------------------
# process_sample tests
# ---------------------------------------------------------------------------

class TestProcessSampleTpMachineOnly:
    """Tests for tp_machine_only samples (fixed-offset timestamp correction)."""

    def test_timestamp_subtracted_by_segment_duration(self):
        """Timestamp should be moved back by the default segment duration (3 s)."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result['Timestamp'] == TP_MACHINE_ONLY_EXPECTED['Timestamp']

    def test_uri_updated_to_match_new_timestamp(self):
        """URI time parameter should reflect the corrected timestamp."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result['URI'] == TP_MACHINE_ONLY_EXPECTED['URI']

    def test_confidence_formatted_to_one_decimal(self):
        """Confidence value should be rounded and formatted to one decimal place."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result['Confidence'] == TP_MACHINE_ONLY_EXPECTED['Confidence']

    def test_full_output_row_matches_expected(self):
        """Complete output row should match the expected training_samples.csv row."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result == TP_MACHINE_ONLY_EXPECTED


class TestProcessSampleTpHumanOnly:
    """Tests for tp_human_only samples (model-based timestamp correction)."""

    def _make_result(self, mock_return=('2025_12_12_01_14_17_PST', 43.2)):
        """Call process_sample with a mocked compute_correct_timestamp_for_tp_human_only."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=mock_return,
        ):
            return process_sample(
                TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=mock_model, tmp_dir='/tmp'
            )

    def test_timestamp_from_model_inference(self):
        """Timestamp should come from the model-based correction."""
        result = self._make_result()
        assert result['Timestamp'] == TP_HUMAN_ONLY_EXPECTED['Timestamp']

    def test_uri_updated_to_match_new_timestamp(self):
        """URI time parameter should reflect the model-corrected timestamp."""
        result = self._make_result()
        assert result['URI'] == TP_HUMAN_ONLY_EXPECTED['URI']

    def test_confidence_formatted_to_one_decimal(self):
        """Model confidence should be formatted to one decimal place."""
        result = self._make_result()
        assert result['Confidence'] == TP_HUMAN_ONLY_EXPECTED['Confidence']

    def test_full_output_row_matches_expected(self):
        """Complete output row should match the expected training_samples.csv row."""
        result = self._make_result()
        assert result == TP_HUMAN_ONLY_EXPECTED

    def test_model_inference_called_with_correct_args(self):
        """compute_correct_timestamp_for_tp_human_only should receive the right arguments."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=('2025_12_12_01_14_17_PST', 43.2),
        ) as mock_compute:
            process_sample(
                TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=mock_model, tmp_dir='/tmp'
            )
        mock_compute.assert_called_once_with(TP_HUMAN_ONLY_INPUT, mock_model, '/tmp', 3)

    def test_falls_back_to_fixed_offset_when_no_model(self):
        """Without model_inference, tp_human_only should use the fixed offset instead."""
        result = process_sample(TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=None)
        # 2025_12_12_01_14_55 - 3 seconds = 2025_12_12_01_14_52
        assert result['Timestamp'] == '2025_12_12_01_14_52_PST'

    def test_resident_bush_point_full_output_row_matches_expected(self):
        """resident tp_human_only from rpi_bush_point should produce the correct output row (ubuntu/windows discrepancy)."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=('2023_11_28_14_12_43_PST', 86.6),
        ):
            result = process_sample(
                RESIDENT_TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=mock_model, tmp_dir='/tmp'
            )
        assert result == RESIDENT_TP_HUMAN_ONLY_EXPECTED


# ---------------------------------------------------------------------------
# manual_timestamps tests
# ---------------------------------------------------------------------------

class TestManualTimestamps:
    """Tests for samples that have manual timestamp/confidence corrections."""

    def test_manual_timestamp_applied(self):
        """process_sample should use the manually-corrected timestamp and URI."""
        manual_timestamps = {MANUAL_TIMESTAMP_INPUT['URI']: '2025_11_24_20_14_00_PST'}
        manual_confidences = {MANUAL_TIMESTAMP_INPUT['URI']: '100.0'}
        result = process_sample(MANUAL_TIMESTAMP_INPUT, manual_timestamps, manual_confidences)
        assert result == MANUAL_TIMESTAMP_EXPECTED

    def test_zero_confidence_detection_filtered_out(self):
        """remove_zero_confidence_detections should remove a sample with 0.0 confidence."""
        manual_confidences = {ZERO_CONFIDENCE_INPUT['URI']: '0.0'}
        filtered = remove_zero_confidence_detections([ZERO_CONFIDENCE_INPUT], manual_confidences)
        assert filtered == []


# ---------------------------------------------------------------------------
# sort_by_preference tests
# ---------------------------------------------------------------------------

# Shared detection factory for sort tests.
def _make_det(uri: str, notes: str = 'tp_human_only', timestamp: str = '2025_01_01_00_00_00_PST',
              description: str = '') -> dict:
    return {
        'Category': 'humpback',
        'NodeName': 'rpi_test',
        'Timestamp': timestamp,
        'URI': uri,
        'Description': description,
        'Notes': notes,
        'Confidence': '',
    }


class TestSortByPreference:
    """Tests for sort_by_preference manual-timestamp ordering."""

    def test_hundred_confidence_sorted_before_no_entry(self):
        """A detection with 100.0 confidence should come before one with no manual entry."""
        det_100 = _make_det('http://example.com/100')
        det_none = _make_det('http://example.com/none')
        manual_confidences = {'http://example.com/100': '100.0'}
        result = sort_by_preference([det_none, det_100], manual_confidences)
        assert result[0]['URI'] == 'http://example.com/100'
        assert result[1]['URI'] == 'http://example.com/none'

    def test_preferred_notes_still_sort_first(self):
        """Preferred notes (tp_machine_only) should still rank above 100.0 confidence."""
        det_preferred = _make_det('http://example.com/preferred', notes='tp_machine_only')
        det_100 = _make_det('http://example.com/100', notes='tp_human_only')
        manual_confidences = {'http://example.com/100': '100.0'}
        result = sort_by_preference([det_100, det_preferred], manual_confidences)
        assert result[0]['URI'] == 'http://example.com/preferred'

    def test_quality_issue_deprioritized_within_tier(self):
        """Within the same tier, quality issues should be deprioritized."""
        det_clean = _make_det('http://example.com/clean')
        det_faint = _make_det('http://example.com/faint', description='faint signal')
        result = sort_by_preference([det_faint, det_clean], {})
        assert result[0]['URI'] == 'http://example.com/clean'
        assert result[1]['URI'] == 'http://example.com/faint'


# ---------------------------------------------------------------------------
# write_training_samples integration test
# ---------------------------------------------------------------------------

class TestWriteTrainingSamples:
    """Integration tests for write_training_samples (CSV output)."""

    def test_tp_machine_only_row_written_correctly(self):
        """write_training_samples should produce the correct CSV row for tp_machine_only."""
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / 'training_samples.csv'
            write_training_samples(
                samples=[TP_MACHINE_ONLY_INPUT],
                output_path=output_path,
                manual_timestamps={},
                manual_confidences={},
            )
            rows = _read_csv(output_path)
        assert len(rows) == 1
        assert rows[0] == TP_MACHINE_ONLY_EXPECTED

    def test_tp_human_only_row_written_correctly(self):
        """write_training_samples should produce the correct CSV row for tp_human_only."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=('2025_12_12_01_14_17_PST', 43.2),
        ):
            with TemporaryDirectory() as tmp:
                output_path = Path(tmp) / 'training_samples.csv'
                write_training_samples(
                    samples=[TP_HUMAN_ONLY_INPUT],
                    output_path=output_path,
                    manual_timestamps={},
                    manual_confidences={},
                    model_inference=mock_model,
                )
                rows = _read_csv(output_path)
        assert len(rows) == 1
        assert rows[0] == TP_HUMAN_ONLY_EXPECTED


# ---------------------------------------------------------------------------
# Testing samples selection and output tests.
# ---------------------------------------------------------------------------

class TestTestingSamples:
    """Tests for testing sample selection and CSV writing."""

    def test_select_testing_samples_applies_eligibility_rules(self):
        """select_testing_samples should filter out ineligible rows and training rows."""
        detections = [
            # Eligible resident tp_human_only (standard eligible).
            {
                'Category': 'resident',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_00_PST',
                'URI': 'https://example.org/resident-tp-human',
                'Description': 'resident human detection',
                'Notes': 'tp_human_only',
                'Confidence': '',
            },
            # Eligible: tp_machine_only + resident (added as bonus group).
            {
                'Category': 'resident',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_01_PST',
                'URI': 'https://example.org/resident-tp-machine',
                'Description': 'resident machine detection',
                'Notes': 'tp_machine_only',
                'Confidence': '',
            },
            # Eligible: tp_human_only + negative category (added as bonus group).
            {
                'Category': 'water',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_02_PST',
                'URI': 'https://example.org/water-tp-human',
                'Description': 'water human detection',
                'Notes': 'tp_human_only',
                'Confidence': '',
            },
            # Ineligible: confidence 0.0.
            {
                'Category': 'transient',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_03_PST',
                'URI': 'https://example.org/transient-zero-confidence',
                'Description': 'transient detection',
                'Notes': 'tp_human_only',
                'Confidence': '',
            },
            # Ineligible: already selected for training.
            {
                'Category': 'humpback',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_04_PST',
                'URI': 'https://example.org/training-selected',
                'Description': 'humpback detection',
                'Notes': 'tp_machine_only',
                'Confidence': '',
            },
            # Eligible non-resident machine detection (standard eligible).
            {
                'Category': 'humpback',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_05_PST',
                'URI': 'https://example.org/humpback-tp-machine',
                'Description': 'humpback machine detection',
                'Notes': 'tp_machine_only',
                'Confidence': '',
            },
        ]
        training_samples = [
            {
                'Category': 'humpback',
                'NodeName': 'rpi_test',
                'Timestamp': '2025_01_01_00_00_04_PST',
                'URI': 'https://example.org/training-selected',
                'Description': 'humpback detection',
                'Notes': 'tp_machine_only',
                'Confidence': '',
            }
        ]
        # This row is excluded because manual_confidences maps its URI to 0.0.
        manual_confidences = {'https://example.org/transient-zero-confidence': '0.0'}

        selected = select_testing_samples(detections, training_samples, manual_confidences)
        selected_uris = {row['URI'] for row in selected}

        assert 'https://example.org/resident-tp-human' in selected_uris
        assert 'https://example.org/humpback-tp-machine' in selected_uris
        assert 'https://example.org/resident-tp-machine' in selected_uris
        assert 'https://example.org/water-tp-human' in selected_uris
        assert 'https://example.org/transient-zero-confidence' not in selected_uris
        assert 'https://example.org/training-selected' not in selected_uris

    def test_select_testing_samples_limits_each_category_to_ten(self):
        """select_testing_samples should cap each category at 10 rows."""
        detections = []
        for i in range(12):
            detections.append({
                'Category': 'humpback',
                'NodeName': 'rpi_test',
                'Timestamp': f'2025_01_01_00_00_{i:02d}_PST',
                'URI': f'https://example.org/humpback-{i}',
                'Description': f'humpback {i}',
                'Notes': 'tp_machine_only',
                'Confidence': '',
            })

        selected = select_testing_samples(detections, [], {})
        assert len(selected) == 10

    def test_write_testing_samples_uses_detection_csv_schema(self):
        """write_testing_samples should emit rows with detections.csv columns."""
        sample = {
            'Category': 'resident',
            'NodeName': 'rpi_test',
            'Timestamp': '2025_01_01_00_00_00_PST',
            'URI': 'https://example.org/sample',
            'Description': 'sample row',
            'Notes': 'tp_human_only',
            'Confidence': '100.0',
            'IgnoredColumn': 'ignored',
        }
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / 'testing_samples.csv'
            write_testing_samples([sample], output_path)
            rows = _read_csv(output_path)

        assert len(rows) == 1
        assert rows[0] == {
            'Category': 'resident',
            'NodeName': 'rpi_test',
            'Timestamp': '2025_01_01_00_00_00_PST',
            'URI': 'https://example.org/sample',
            'Description': 'sample row',
            'Notes': 'tp_human_only',
            'Confidence': '100.0',
        }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return its rows as a list of dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))
