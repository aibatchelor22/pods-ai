# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Concatenate all WAV files in a directory with beep sounds between them.

Usage:
    python src/concatenate_wavs.py <directory>
    python src/concatenate_wavs.py output/wav/resident
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_beep(volume: float = 0.25, duration_seconds: float = 0.1, frequency_hz: int = 1000, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a simple sine wave beep sound.
    
    Args:
        volume: Volume of the beep (0.0 to 1.0).
        duration_seconds: Duration of the beep in seconds.
        frequency_hz: Frequency of the beep tone in Hz.
        sample_rate: Sample rate in Hz.
    
    Returns:
        Numpy array of audio samples.
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    beep = np.sin(2 * np.pi * frequency_hz * t) * volume
    
    # Apply fade in/out to avoid clicks
    fade_samples = min(int(0.01 * sample_rate), num_samples // 2)  # 10ms fade (capped)
    if fade_samples:
        fade_in, fade_out = np.linspace(0, 1, fade_samples, endpoint=False), np.linspace(1, 0, fade_samples, endpoint=False)
        beep[:fade_samples] *= fade_in
        beep[-fade_samples:] *= fade_out
    
    return beep.astype(np.float32)


def concatenate_wavs_with_beeps(directory: Path, output_filename: str = "concatenated.wav") -> None:
    """
    Concatenate all WAV files in a directory with beeps between them.
    
    Args:
        directory: Directory containing WAV files.
        output_filename: Name of the output concatenated file.
    """
    # Find all WAV files in the directory (excluding the output file)
    output_path = (directory / output_filename).resolve()
    wav_files = sorted([f for f in directory.glob("*.wav") if f.resolve() != output_path])
    if not wav_files:
        print(f"Error: No WAV files found in {directory}", file=sys.stderr)
        return
    
    print(f"Found {len(wav_files)} WAV files in {directory}")
    
    # Read the first file to get sample rate and number of channels
    try:
        first_audio, sample_rate = sf.read(wav_files[0], dtype="float32")
    except Exception as e:
        print(f"Error reading {wav_files[0].name}: {e}", file=sys.stderr)
        return
    num_channels = first_audio.shape[1] if first_audio.ndim > 1 else 1
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {num_channels}")
    
    # Generate beep (mono)
    beep = generate_beep(volume=0.25, duration_seconds=0.1, sample_rate=sample_rate)
    
    # If audio is stereo, duplicate beep to both channels
    if num_channels == 2:
        beep = np.stack([beep, beep], axis=1)
    
    # Concatenate all audio files with beeps
    concatenated = []
    
    for i, wav_file in enumerate(wav_files):
        print(f"Processing {i+1}/{len(wav_files)}: {wav_file.name}")
        
        try:
            audio, sr = sf.read(wav_file, dtype="float32")
            
            # Verify sample rate matches
            if sr != sample_rate:
                print(f"  Warning: {wav_file.name} has different sample rate ({sr} Hz), skipping")
                continue
            
            # Ensure correct number of channels
            if audio.ndim == 1 and num_channels == 2:
                # Convert mono to stereo
                audio = np.stack([audio, audio], axis=1)
            elif audio.ndim == 2 and num_channels == 1:
                # Convert stereo to mono
                audio = np.mean(audio, axis=1)
            
            if concatenated:
                concatenated.append(beep)
            concatenated.append(audio)
        
        except Exception as e:
            print(f"  Error reading {wav_file.name}: {e}", file=sys.stderr)
            continue
    
    if not concatenated:
        print("Error: No audio data to concatenate", file=sys.stderr)
        return
    
    # Concatenate all segments
    print("Concatenating audio...")
    final_audio = np.concatenate(concatenated, axis=0)
    
    # Write output file
    output_path = directory / output_filename
    print(f"Writing output to {output_path}")
    sf.write(output_path, final_audio, sample_rate)
    
    duration = len(final_audio) / sample_rate
    print(f"Done! Output duration: {duration:.2f} seconds")


def main() -> int:
    """Entry point for the concatenate_wavs_with_beeps CLI."""
    parser = argparse.ArgumentParser(
        description="Concatenate all WAV files in a directory with beep sounds between them."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing WAV files to concatenate.",
    )
    parser.add_argument(
        "--output",
        default="concatenated.wav",
        help="Name of the output file (default: concatenated.wav).",
    )
    # Beep characteristics are currently fixed (see generate_beep()).
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        return 1
    
    if not args.directory.is_dir():
        print(f"Error: Not a directory: {args.directory}", file=sys.stderr)
        return 1
    
    try:
        concatenate_wavs_with_beeps(args.directory, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())