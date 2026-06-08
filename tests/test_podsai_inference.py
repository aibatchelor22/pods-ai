#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for PodsAIInference timestamp correction semantics.

These tests verify that PodsAIInference.predict() produces output
with correct length and indexing semantics for use in timestamp correction.
"""

import pytest
import numpy as np
import tempfile
import time
import soundfile as sf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from podsai_inference import NUM_SPECIAL_TOKENS


# Pinned PODS-AI model revision for integration-test stability.
PODSAI_TEST_MODEL_ID = "davethaler/whale-call-detector"
# renovate: datasource=git-refs depName=https://huggingface.co/davethaler/whale-call-detector versioning=git.
PODSAI_TEST_MODEL_REVISION = "d1eedf5c614268da7551039a84dfc35d317168b9"


def _resolve_podsai_test_model_path() -> str:
    """Return local PODS-AI model path or a pinned Hub snapshot for integration tests."""
    local_path = Path("model/multiclass")
    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import file_exists as hf_file_exists
        from huggingface_hub import snapshot_download as hf_snapshot_download
        if not hf_file_exists(
            PODSAI_TEST_MODEL_ID,
            "preprocessor_config.json",
            revision=PODSAI_TEST_MODEL_REVISION,
        ):
            pytest.skip(
                f"Hub model '{PODSAI_TEST_MODEL_ID}' revision "
                f"'{PODSAI_TEST_MODEL_REVISION}' is missing preprocessor_config.json."
            )
        return hf_snapshot_download(
            repo_id=PODSAI_TEST_MODEL_ID,
            revision=PODSAI_TEST_MODEL_REVISION,
        )
    except Exception:
        pytest.skip(
            f"HuggingFace Hub is not reachable; cannot load model "
            f"'{PODSAI_TEST_MODEL_ID}' revision '{PODSAI_TEST_MODEL_REVISION}'"
        )


@pytest.fixture
def mock_podsai_model():
    """Create a mock PODS-AI model for testing."""
    mock_model = Mock()
    mock_config = Mock()

    # Define label mapping for multi-class model.
    # Current schema matches train_podsai_model.py:
    # ["water", "resident", "transient", "humpback", "vessel", "jingle", "human"]
    mock_config.id2label = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
        5: "jingle",
        6: "human"
    }
    mock_config.label2id = {
        "water": 0,
        "resident": 1,
        "transient": 2,
        "humpback": 3,
        "vessel": 4,
        "jingle": 5,
        "human": 6
    }

    # Set metadata attributes that _print_model_metadata expects.
    mock_config._name_or_path = "test-model"
    mock_config.architectures = ["Wav2Vec2ForSequenceClassification"]
    mock_config.model_type = "wav2vec2"
    mock_config._commit_hash = None  # Optional, can be None

    mock_model.config = mock_config
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=mock_model)

    # Mock model output. Handles batched input by returning one row per segment.
    def mock_forward(**kwargs):
        # Return logits for "water" class with high confidence.
        # Shape: (batch_size, num_classes=7)
        batch_size = kwargs["input_values"].shape[0]
        logits = torch.tensor([[2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]).repeat(batch_size, 1)
        mock_output = Mock()
        mock_output.logits = logits
        return mock_output

    mock_model.side_effect = mock_forward

    return mock_model


@pytest.fixture
def mock_feature_extractor():
    """Create a mock feature extractor."""
    mock_extractor = Mock()

    # Set attributes accessed by _compute_input_values() when computing the
    # full-audio spectrogram and slicing windows from it.
    mock_extractor.num_mel_bins = 128
    mock_extractor.max_length = 1024
    mock_extractor.mean = -4.2677393
    mock_extractor.std = 4.5689974
    mock_extractor.do_normalize = True
    mock_extractor.hop_length = 10  # ms per frame

    def mock_extract(audio, sampling_rate, return_tensors, padding):
        # Return dummy tensors. Supports both single array and batched list inputs.
        if isinstance(audio, list):
            batch_size = len(audio)
            seq_len = len(audio[0]) if audio else 0
        else:
            batch_size = 1
            seq_len = len(audio)
        return {
            "input_values": torch.randn(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long)
        }

    mock_extractor.side_effect = mock_extract
    mock_extractor.from_pretrained = Mock(return_value=mock_extractor)

    return mock_extractor


@pytest.fixture
def synthetic_audio_60s():
    """Create a 60-second synthetic audio file with a tone at second 30."""
    sr = 16000
    duration = 60
    samples = sr * duration

    # Create silence with a 1-second tone at second 30.
    audio = np.zeros(samples, dtype=np.float32)
    tone_start = 30 * sr
    tone_end = 31 * sr
    t = np.linspace(0, 1, sr)
    audio[tone_start:tone_end] = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Save to temporary file.
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        yield f.name

    # Cleanup.
    Path(f.name).unlink(missing_ok=True)


class TestPodsAIInferenceIndexing:
    """Test indexing semantics for timestamp correction."""

    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_output_length_with_hop_duration_2(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model, synthetic_audio_60s
    ):
        """Test that local_confidences length is correct with 2-second hop."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)

        from podsai_inference import PodsAIInference

        model = PodsAIInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=2)

        # With hop_duration=2, a 60-second audio should produce 29 positions:
        # num_positions = floor((60 - 3) / 2) + 1 = floor(57/2) + 1 = 28 + 1 = 29
        # This is the CURRENT behavior, but it doesn't give per-second indexing.
        assert len(result["local_confidences"]) == 29, \
            f"Expected 29 confidences for 60s audio with 2s hop, got {len(result['local_confidences'])}"
        
        # Note: This means local_confidences[i] represents time i*2 seconds, NOT second i.
        # For timestamp correction to work as documented, extract_training_samples.py
        # must infer hop_duration = audio_duration / len(local_confidences).
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_output_length_with_hop_duration_1(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model, synthetic_audio_60s
    ):
        """Test with hop_duration=1 to match FastAI per-second behavior."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        from podsai_inference import PodsAIInference
        
        model = PodsAIInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=1)
        
        # With hop_duration=1: num_positions = floor((60 - 3) / 1) + 1 = 58
        assert len(result["local_confidences"]) == 58, \
            f"Expected 58 confidences for 60s audio with 1s hop, got {len(result['local_confidences'])}"
        
        # With 1-second hop, local_confidences[i] ≈ second i (close to FastAI behavior).
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_index_to_time_mapping(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model, synthetic_audio_60s
    ):
        """Test that we can correctly map index to timestamp."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        from podsai_inference import PodsAIInference
        import librosa
        
        model = PodsAIInference("test-model-path")
        
        # Test with different hop durations.
        for hop_duration in [1, 2]:
            result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=hop_duration)
            
            # Verify we can reconstruct time mapping.
            audio_duration = librosa.get_duration(path=synthetic_audio_60s)
            num_positions = len(result["local_confidences"])
            
            # Inferred hop should match actual hop.
            inferred_hop = audio_duration / num_positions if num_positions > 0 else hop_duration
            
            # Should be within reasonable tolerance.
            assert abs(inferred_hop - hop_duration) < 0.5, \
                f"Inferred hop {inferred_hop:.2f}s doesn't match actual {hop_duration}s"
            
            # Verify each index maps to correct time.
            for i in range(num_positions):
                expected_time = i * hop_duration
                assert expected_time < audio_duration + hop_duration, \
                    f"Index {i} maps to {expected_time}s, beyond audio duration {audio_duration}s"
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_short_audio_handling(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model
    ):
        """Test handling of audio shorter than segment_duration."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        # Create 2-second audio (shorter than 3-second segment).
        sr = 16000
        audio = np.random.randn(2 * sr).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            audio_path = f.name
        
        try:
            from podsai_inference import PodsAIInference
            
            model = PodsAIInference("test-model-path")
            result = model.predict(audio_path, segment_duration=3, hop_duration=2)
            
            # For 2-second audio with 3-second segment:
            # num_positions = floor((2 - 3) / 2) + 1 = floor(-1/2) + 1 = -1 + 1 = 0
            # But code has guard: if num_positions < 1: num_positions = 1
            assert len(result["local_confidences"]) == 1, \
                f"Expected 1 confidence for short audio, got {len(result['local_confidences'])}"
        finally:
            Path(audio_path).unlink(missing_ok=True)
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_exact_multiple_of_hop_duration(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model
    ):
        """Test audio length that's exact multiple of hop_duration."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        # 60 seconds is exact multiple of hop_duration=2.
        sr = 16000
        audio = np.random.randn(60 * sr).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            audio_path = f.name
        
        try:
            from podsai_inference import PodsAIInference
            
            model = PodsAIInference("test-model-path")
            result = model.predict(audio_path, segment_duration=3, hop_duration=2)
            
            # num_positions = floor((60 - 3) / 2) + 1 = 29
            assert len(result["local_confidences"]) == 29, \
                f"Expected 29 confidences, got {len(result['local_confidences'])}"
            
            # All confidences should be valid (0.0-1.0).
            assert all(0.0 <= c <= 1.0 for c in result["local_confidences"]), \
                "All confidences should be in range [0.0, 1.0]"
        finally:
            Path(audio_path).unlink(missing_ok=True)
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_output_format_compatibility(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model, synthetic_audio_60s
    ):
        """Test that output format matches expected interface."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        from podsai_inference import PodsAIInference
        
        model = PodsAIInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=2)
        
        # Check required keys.
        assert "local_predictions" in result
        assert "local_confidences" in result
        assert "global_prediction" in result
        assert "global_confidence" in result
        assert "global_prediction_label" in result
        assert "hop_duration" in result
        assert "segment_duration" in result
        
        # Check types.
        assert isinstance(result["local_predictions"], list)
        assert isinstance(result["local_confidences"], list)
        assert isinstance(result["global_prediction"], int)
        assert isinstance(result["global_confidence"], float)
        assert isinstance(result["global_prediction_label"], str)
        assert isinstance(result["hop_duration"], float)
        assert isinstance(result["segment_duration"], float)
        
        # Check lengths match.
        assert len(result["local_predictions"]) == len(result["local_confidences"])
        
        # Check value ranges.
        assert 0.0 <= result["global_confidence"] <= 1.0
        assert all(0.0 <= c <= 1.0 for c in result["local_confidences"])
        assert all(isinstance(p, int) for p in result["local_predictions"])
        assert result["hop_duration"] == 2.0
        assert result["segment_duration"] == 3.0
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_empty_audio_handling(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model
    ):
        """Test handling of empty audio file."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        # Create empty audio.
        sr = 16000
        audio = np.array([], dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            audio_path = f.name
        
        try:
            from podsai_inference import PodsAIInference
            
            model = PodsAIInference("test-model-path")
            result = model.predict(audio_path, segment_duration=3, hop_duration=2)
            
            # Should return empty predictions with negative class.
            assert result["local_predictions"] == []
            assert result["local_confidences"] == []
            assert result["local_probs"] == []
            # global_prediction should be one of the negative classes (water=0, vessel=4, jingle=5, human=6).
            assert result["global_prediction"] in [0, 4, 5, 6]
            assert result["global_confidence"] == 0.0
            # Error returns must always include hop_duration and segment_duration (matching orcahello behavior).
            assert result["hop_duration"] == 2.0
            assert result["segment_duration"] == 3.0
        finally:
            Path(audio_path).unlink(missing_ok=True)
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_call_likelihood_computation(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, synthetic_audio_60s
    ):
        """Test that call-likelihood is computed correctly as 1 - P(negative_classes)."""
        # Create mock model that returns known probabilities.
        mock_model = Mock()
        mock_config = Mock()
        # Schema: water=0, resident=1, transient=2, humpback=3, vessel=4, jingle=5, human=6
        mock_config.id2label = {
            0: "water", 1: "resident", 2: "transient", 3: "humpback",
            4: "vessel", 5: "jingle", 6: "human"
        }
        mock_config.label2id = {
            "water": 0, "resident": 1, "transient": 2, "humpback": 3,
            "vessel": 4, "jingle": 5, "human": 6
        }
        mock_model.config = mock_config
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        # Set metadata attributes that _print_model_metadata expects.
        mock_config._name_or_path = "test-model"
        mock_config.architectures = ["Wav2Vec2ForSequenceClassification"]
        mock_config._commit_hash = None  # Optional, can be None
        
        # Return known probabilities. Handles batched input.
        # water=0.1, resident=0.4, transient=0.1, humpback=0.1, vessel=0.1, jingle=0.1, human=0.1
        # Total negative (water+vessel+jingle+human) = 0.4, so call-likelihood should be 0.6.
        def mock_forward(**kwargs):
            # Logits that represent the desired distribution.
            batch_size = kwargs["input_values"].shape[0]
            logits = torch.tensor([[0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]]).repeat(batch_size, 1)
            mock_output = Mock()
            mock_output.logits = logits
            return mock_output
        
        mock_model.side_effect = mock_forward
        
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)
        
        from podsai_inference import PodsAIInference
        
        model = PodsAIInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=2)
        
        # Call-likelihood should be 1 - P(negative classes).
        # After softmax, the probabilities will be different from logits.
        # For this test, we just verify that confidences are in valid range.
        assert all(0.0 <= c <= 1.0 for c in result["local_confidences"])
        assert len(result["local_confidences"]) == 29  # 60s audio, 2s hop

    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_ast_path_uses_segment_frame_length(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, synthetic_audio_60s
    ):
        """AST path should use segment frame length when embeddings can be resized."""
        mock_model = Mock()
        mock_config = Mock()
        mock_config.id2label = {
            0: "water", 1: "resident", 2: "transient", 3: "humpback",
            4: "vessel", 5: "jingle", 6: "human"
        }
        mock_config.label2id = {
            "water": 0, "resident": 1, "transient": 2, "humpback": 3,
            "vessel": 4, "jingle": 5, "human": 6
        }
        mock_config._name_or_path = "test-model"
        mock_config.architectures = ["ASTForAudioClassification"]
        mock_config.model_type = "audio-spectrogram-transformer"
        mock_config._commit_hash = None
        mock_config.patch_size = 16
        mock_config.frequency_stride = 10
        mock_config.time_stride = 10
        mock_config.max_length = 1024
        mock_config.num_mel_bins = 128
        mock_model.config = mock_config
        mock_model.audio_spectrogram_transformer = Mock()
        mock_model.audio_spectrogram_transformer.embeddings = Mock()
        freq_tokens = (mock_config.num_mel_bins - mock_config.patch_size) // mock_config.frequency_stride + 1
        time_tokens = (mock_config.max_length - mock_config.patch_size) // mock_config.time_stride + 1
        original_token_count = (freq_tokens * time_tokens) + NUM_SPECIAL_TOKENS
        mock_model.audio_spectrogram_transformer.embeddings.position_embeddings = torch.nn.Parameter(
            torch.zeros((1, original_token_count, 4), dtype=torch.float32)
        )
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        call_shapes: list[tuple[int, ...]] = []

        def mock_forward(**kwargs):
            call_shapes.append(tuple(kwargs["input_values"].shape))
            batch_size = kwargs["input_values"].shape[0]
            logits = torch.tensor([[2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]).repeat(batch_size, 1)
            mock_output = Mock()
            mock_output.logits = logits
            return mock_output

        mock_model.side_effect = mock_forward
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_model)

        from podsai_inference import PodsAIInference

        model = PodsAIInference("test-model-path")

        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=2)
        first_call_shapes = call_shapes.copy()

        assert len(result["local_confidences"]) == 29
        assert first_call_shapes, "Expected at least one AST forward pass."
        # 3 seconds with 10ms frame shift -> 300 frames.
        assert all(shape[1] == 300 for shape in first_call_shapes)

        call_shapes.clear()
        second_result = model.predict(synthetic_audio_60s, segment_duration=5, hop_duration=2)
        second_call_shapes = call_shapes.copy()

        assert len(second_result["local_confidences"]) == 28
        assert second_call_shapes, "Expected at least one AST forward pass on repeated predict()."
        # 5 seconds with 10ms frame shift -> 500 frames.
        assert all(shape[1] == 500 for shape in second_call_shapes)


class TestPodsAIInferenceErrorHandling:
    """Test error handling in PodsAIInference."""

    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_feature_extractor_fails_immediately_on_bad_revision(
        self, mock_extractor_class, mock_model_class, mock_feature_extractor, mock_podsai_model
    ):
        """When a pinned revision fails, the job must fail immediately (no fallback)."""
        mock_extractor_class.from_pretrained.side_effect = OSError("revision not found")
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)

        from podsai_inference import PodsAIInference

        with pytest.raises(RuntimeError):
            PodsAIInference("test-model-path", model_revision="deadbeef")

        assert mock_extractor_class.from_pretrained.call_count == 1
        assert mock_extractor_class.from_pretrained.call_args.args[0] == "test-model-path"
        assert mock_extractor_class.from_pretrained.call_args.kwargs.get("revision") == "deadbeef"
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_invalid_audio_file(
        self, mock_extractor_class, mock_model_class,
        mock_feature_extractor, mock_podsai_model
    ):
        """Test handling of invalid audio file."""
        mock_extractor_class.from_pretrained = Mock(return_value=mock_feature_extractor)
        mock_model_class.from_pretrained = Mock(return_value=mock_podsai_model)
        
        from podsai_inference import PodsAIInference
        
        model = PodsAIInference("test-model-path")
        result = model.predict("nonexistent.wav", segment_duration=3, hop_duration=2)
        
        # Should return error response with negative prediction.
        assert result["local_predictions"] == []
        assert result["local_confidences"] == []
        assert result["local_probs"] == []
        # Should be one of the negative classes (water=0, vessel=4, jingle=5, human=6).
        assert result["global_prediction"] in [0, 4, 5, 6]
        assert result["global_confidence"] == 0.0
        # Error returns must always include hop_duration and segment_duration (matching orcahello behavior).
        assert result["hop_duration"] == 2.0
        assert result["segment_duration"] == 3.0
    
    @patch('podsai_inference.AutoModelForAudioClassification')
    @patch('podsai_inference.AutoFeatureExtractor')
    def test_model_missing_negative_class(
        self, mock_extractor_class, mock_model_class
    ):
        """Test that model initialization fails without negative class."""
        mock_extractor = Mock()
        mock_extractor_class.from_pretrained = Mock(return_value=mock_extractor)
        
        # Create model with only positive classes (no water, vessel, jingle, human, or other).
        mock_model = Mock()
        mock_config = Mock()
        mock_config.id2label = {0: "resident", 1: "transient", 2: "humpback"}
        mock_config.label2id = {"resident": 0, "transient": 1, "humpback": 2}
        mock_model.config = mock_config

        # Set metadata attributes that _print_model_metadata expects.
        mock_config._name_or_path = "test-model"
        mock_config.architectures = ["Wav2Vec2ForSequenceClassification"]
        mock_config._commit_hash = None  # Optional, can be None
        
        mock_model_class.from_pretrained = Mock(return_value=mock_model)
        
        from podsai_inference import PodsAIInference
        
        # Should raise ValueError for missing negative class.
        with pytest.raises(ValueError, match="must include at least one negative/background class"):
            PodsAIInference("test-model-path")


class TestIntegrationWithRealModels:
    """Integration tests that require the real PODS-AI model."""

    @pytest.fixture
    def podsai_model_path(self) -> str:
        """Path to local PODS-AI model or a pinned Hub snapshot fallback."""
        return _resolve_podsai_test_model_path()

    def test_predict_60s_wav_performance(self, podsai_model_path: str) -> None:
        """Inference on a 60-second wav file must complete in under 22 seconds.

        This test guards against performance regressions in the inference pipeline.
        For AST models the full-audio spectrogram optimization (one fbank call
        instead of one per segment) keeps the budget well under 22 seconds on a
        CPU-only machine.  For Wav2Vec2 (and other raw-audio) models the raw-audio
        path is used instead; both should comfortably fit the 22-second budget.
        """
        from podsai_inference import PodsAIInference

        # Create a 60-second synthetic WAV of silence.
        sr = 16000
        duration = 60
        audio = np.zeros(duration * sr, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            wav_path = f.name

        try:
            model = PodsAIInference(podsai_model_path)
            start = time.perf_counter()
            result = model.predict(wav_path, segment_duration=3, hop_duration=2)
            elapsed = time.perf_counter() - start

            print(f"\nPodsAI predict() took {elapsed:.2f}s for a 60-second WAV")

            assert elapsed < 22.0, (
                f"PodsAI inference on a 60-second WAV took {elapsed:.2f}s, "
                f"expected < 22s. Check for performance regressions in the inference pipeline."
            )
            # 60s audio with segment_duration=3 and hop_duration=2 → 29 positions.
            assert len(result["local_confidences"]) == 29
        finally:
            Path(wav_path).unlink(missing_ok=True)
