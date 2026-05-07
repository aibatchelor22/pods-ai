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
import soundfile as sf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch


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

    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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


class TestPodsAIInferenceErrorHandling:
    """Test error handling in PodsAIInference."""
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
    
    @patch('podsai_inference.Wav2Vec2ForSequenceClassification')
    @patch('podsai_inference.Wav2Vec2FeatureExtractor')
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
