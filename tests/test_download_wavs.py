# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for testing sample download logic in download_wavs.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, patch

from download_wavs import CSVRow, add_seconds_to_timestamp_pst, download_testing_sample


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
