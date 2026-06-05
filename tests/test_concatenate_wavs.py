# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import soundfile as sf

from concatenate_wavs import concatenate_wavs_with_beeps, generate_beep


def _write_wav(path: Path, sample_rate: int, value: float, num_samples: int) -> np.ndarray:
    audio = np.full(num_samples, value, dtype=np.float32)
    sf.write(path, audio, sample_rate)
    return audio


def test_concatenate_wavs_with_beeps_orders_files_and_no_trailing_beep(tmp_path: Path) -> None:
    sample_rate = 8000
    num_samples = 400

    first = _write_wav(tmp_path / "a.wav", sample_rate, 0.1, num_samples)
    second = _write_wav(tmp_path / "b.wav", sample_rate, 0.2, num_samples)
    third = _write_wav(tmp_path / "c.wav", sample_rate, 0.3, num_samples)

    concatenate_wavs_with_beeps(tmp_path)

    output_audio, output_sr = sf.read(tmp_path / "concatenated.wav", dtype="float32")
    assert output_sr == sample_rate

    beep = generate_beep(volume=0.25, duration_seconds=0.1, sample_rate=sample_rate)
    expected = np.concatenate([first, beep, second, beep, third])

    assert len(output_audio) == len(expected)
    np.testing.assert_allclose(output_audio, expected, rtol=0, atol=1e-4)
