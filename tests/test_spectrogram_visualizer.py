# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for spectrogram_visualizer helpers."""

import numpy as np
import pytest
from pathlib import Path

import spectrogram_visualizer


class TestFreqLabel:
    """Tests for _freq_label()."""

    def test_formats_hz_below_1000(self) -> None:
        """Values below 1 kHz are formatted as plain Hz integers."""
        assert spectrogram_visualizer._freq_label(20.0) == "20"
        assert spectrogram_visualizer._freq_label(500.0) == "500"
        assert spectrogram_visualizer._freq_label(999.9) == "1000"

    def test_formats_khz_without_decimals_when_whole(self) -> None:
        """Exact kHz values are formatted without a decimal point."""
        assert spectrogram_visualizer._freq_label(1000.0) == "1k"
        assert spectrogram_visualizer._freq_label(5000.0) == "5k"
        assert spectrogram_visualizer._freq_label(20000.0) == "20k"

    def test_formats_khz_with_one_decimal_when_fractional(self) -> None:
        """Fractional kHz values are formatted with one decimal place."""
        assert spectrogram_visualizer._freq_label(1500.0) == "1.5k"
        assert spectrogram_visualizer._freq_label(7500.0) == "7.5k"


class TestPickFreqTicks:
    """Tests for _pick_freq_ticks()."""

    def test_returns_candidates_within_range(self) -> None:
        """Only candidate frequencies within [f_min, f_max] are returned."""
        ticks = spectrogram_visualizer._pick_freq_ticks(200, 5000)
        assert all(200 <= t <= 5000 for t in ticks)
        assert len(ticks) > 0

    def test_falls_back_to_geomspace_when_no_candidates(self) -> None:
        """Falls back to 6 geomspace ticks when no candidates are in range."""
        ticks = spectrogram_visualizer._pick_freq_ticks(50000, 100000)
        assert len(ticks) == 6
        assert ticks[0] >= 50000
        assert ticks[-1] <= 100000

    def test_full_range_includes_common_ocean_freqs(self) -> None:
        """Typical 20 Hz – 22 kHz range includes common candidate frequencies."""
        ticks = spectrogram_visualizer._pick_freq_ticks(20, 22000)
        assert 1000 in ticks
        assert 5000 in ticks


class TestWriteSpectrogram:
    """Integration tests for write_spectrogram()."""

    def _make_wav(self, tmp_path, duration_s=2.0, sample_rate=16000):
        """Create a minimal WAV file for testing."""
        import soundfile as sf

        wav_path = tmp_path / "test.wav"
        samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
        sf.write(str(wav_path), samples, sample_rate)
        return wav_path

    def test_returns_path_next_to_wav_by_default(self, tmp_path) -> None:
        """Without output_path, PNG is saved alongside the WAV file."""
        wav_path = self._make_wav(tmp_path)
        result = spectrogram_visualizer.write_spectrogram(str(wav_path))
        assert result == str(tmp_path / "test.png")

    def test_respects_explicit_output_path(self, tmp_path) -> None:
        """When output_path is given, PNG is saved at that location."""
        wav_path = self._make_wav(tmp_path)
        custom_png = tmp_path / "custom_output.png"
        result = spectrogram_visualizer.write_spectrogram(str(wav_path), output_path=str(custom_png))
        assert result == str(custom_png)
        assert custom_png.exists()

    def test_output_file_is_created(self, tmp_path) -> None:
        """write_spectrogram creates a non-empty PNG file."""
        wav_path = self._make_wav(tmp_path)
        result = spectrogram_visualizer.write_spectrogram(str(wav_path))
        png_path = tmp_path / "test.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_accepts_pathlib_path(self, tmp_path) -> None:
        """write_spectrogram works with pathlib.Path as wav_file_path."""
        wav_path = self._make_wav(tmp_path)
        result = spectrogram_visualizer.write_spectrogram(Path(wav_path))
        assert result.endswith(".png")
        assert (tmp_path / "test.png").exists()
