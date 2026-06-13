# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for testing sample download logic in download_wavs.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, patch
import os

import pytest

from download_wavs import (
    CSVRow,
    add_seconds_to_timestamp_pst,
    download_testing_sample,
    process_csv,
    process_testing_csv,
    run_download_wavs,
    validate_no_overlaps,
)


class TestDownloadTestingSample:
    """Tests for download_testing_sample routing behavior."""

    def test_tp_human_only_downloads_60s_audio_to_testing_directory(self):
        """tp_human_only rows should use download_60s_audio and save output file."""
        row = CSVRow(
            category="resident",
            node_name="rpi_andrews_bay",
            timestamp_pst="2025_01_01_00_00_00_PST",
            uri="https://example.org/sample",
            description="sample",
            notes="tp_human_only",
        )

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "testing-wav"

            def _fake_download_60s_audio(node_name: str, timestamp_str: str, tmp_dir: str):
                """Create and return a temporary fake 60-second WAV path."""
                wav_path = Path(tmp_dir) / "temp_60s.wav"
                wav_path.write_bytes(b"fake wav content")
                return str(wav_path)

            with patch("download_wavs.download_60s_audio", side_effect=_fake_download_60s_audio):
                download_testing_sample(row, output_root)

            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert expected.exists()

    def test_tp_machine_only_downloads_centered_60s_clip(self):
        """tp_machine_only rows should use download_60s_audio with a +30s timestamp."""
        row = CSVRow(
            category="humpback",
            node_name="rpi_orcasound_lab",
            timestamp_pst="2025_01_01_00_00_03_PST",
            uri="https://example.org/sample",
            description="sample",
            notes="tp_machine_only",
        )

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "testing-wav"
            def _fake_download_60s_audio(node_name: str, timestamp_str: str, tmp_dir: str):
                """Create and return a temporary fake 60-second WAV path."""
                wav_path = Path(tmp_dir) / "temp_60s.wav"
                wav_path.write_bytes(b"fake wav content")
                return str(wav_path)

            with patch("download_wavs.download_60s_audio", side_effect=_fake_download_60s_audio) as mock_download_60s:
                download_testing_sample(row, output_root)

            mock_download_60s.assert_called_once_with(
                "rpi_orcasound_lab", "2025_01_01_00_00_33_PST", ANY
            )
            expected = output_root / "humpback" / "rpi-orcasound-lab_2025_01_01_00_00_03_PST.wav"
            assert expected.exists()


class TestTimestampHelpers:
    """Tests for timestamp conversion helpers."""

    def test_add_seconds_to_timestamp_pst_adds_30_seconds(self):
        """add_seconds_to_timestamp_pst should add requested seconds in PST format."""
        assert add_seconds_to_timestamp_pst("2025_01_01_00_00_03_PST", 30) == "2025_01_01_00_00_33_PST"


class TestOverlapValidation:
    def test_validate_no_overlaps_allows_non_overlapping_rows(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_00_00_PST", "", "", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_00_03_PST", "", "", ""),
        ]
        testing_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_59_00_PST", "", "", "tp_human_only"),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_01_06_PST", "", "", "tp_human_only"),
        ]
        validate_no_overlaps(training_rows, testing_rows)

    def test_validate_no_overlaps_rejects_training_overlap(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_00_PST", "", "", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_02_PST", "", "", ""),
        ]
        with pytest.raises(ValueError, match="training overlap"):
            validate_no_overlaps(training_rows, [])

    def test_validate_no_overlaps_rejects_cross_file_overlap(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_00_PST", "", "", ""),
        ]
        testing_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_02_PST", "", "", "tp_machine_only"),
        ]
        with pytest.raises(ValueError, match="cross-file overlap"):
            validate_no_overlaps(training_rows, testing_rows)


class TestCacheAndCleanup:
    def test_process_csv_copies_from_cache_without_downloading(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            cache_root = tmp_path / "cache-wav"
            cached_file = cache_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"cached")

            with patch("download_wavs.get_cached_folders", side_effect=AssertionError("should not download")):
                process_csv(csv_path, output_root, cache_root=cache_root)

            downloaded_file = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert downloaded_file.exists()
            assert downloaded_file.read_bytes() == b"cached"

    def test_process_csv_deletes_stale_wavs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            stale = output_root / "resident" / "old.wav"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"keep")
            stale.write_bytes(b"remove")

            process_csv(csv_path, output_root)

            assert expected.exists()
            assert not stale.exists()

    def test_process_testing_csv_copies_from_cache_and_deletes_stale(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "testing_60s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,tp_human_only\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-testing-wav"
            stale = output_root / "resident" / "old.wav"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"remove")

            cache_root = tmp_path / "cache-testing-wav"
            cached_file = cache_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"cached")

            with patch("download_wavs.download_60s_audio", side_effect=AssertionError("should not download")):
                process_testing_csv(csv_path, output_root, cache_root=cache_root)

            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert expected.exists()
            assert expected.read_bytes() == b"cached"
            assert not stale.exists()


class TestValidateOnly:
    def test_run_download_wavs_validate_only_skips_download_processing(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_dir = tmp_path / "output" / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            (csv_dir / "training_3s_samples.csv").write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_01_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )
            (csv_dir / "testing_60s_samples.csv").write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_01_01_06_PST,uri,desc,tp_human_only\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                with patch("download_wavs.process_csv") as mock_process_csv, patch("download_wavs.process_testing_csv") as mock_process_testing_csv:
                    run_download_wavs(validate_only=True)
                mock_process_csv.assert_not_called()
                mock_process_testing_csv.assert_not_called()
            finally:
                os.chdir(original_cwd)
