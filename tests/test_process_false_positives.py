# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for process_false_positives.py."""

import csv
from datetime import datetime, timezone
from unittest.mock import patch

from make_csv import OrcaHelloDetection
from orcasite_feeds import OrcasiteFeed
from process_false_positives import (
    append_manual_samples,
    get_corrected_class,
    process_false_positives,
)


def _make_feed() -> OrcasiteFeed:
    """Build a minimal test feed."""
    return OrcasiteFeed(
        id="feed_1",
        name="Test Feed",
        node_name="rpi_test",
        slug="test-feed",
        bucket="audio-orcasound-net",
        bucket_region="us-west-2",
        visible=True,
        location=(47.0, -122.0),
    )


class TestGetCorrectedClass:
    """Tests for get_corrected_class."""

    def test_detects_transient_keywords(self):
        """Transient-related comments should map to transient."""
        assert get_corrected_class("Likely transient calls from Bigg's whales.") == "transient"

    def test_detects_vessel_keywords(self):
        """Boat-like comments should map to vessel."""
        assert get_corrected_class("This is boat noise, not whales.") == "vessel"

    def test_returns_none_for_unsure_comments(self):
        """Ambiguous comments should be skipped."""
        assert get_corrected_class("Not sure what this is.") is None


class TestAppendManualSamples:
    """Tests for append_manual_samples."""

    def test_skips_duplicate_uris(self, tmp_path):
        """Rows with URIs already in the file should not be appended again."""
        manual_samples_path = tmp_path / "manual_samples.csv"
        manual_samples_path.write_text(
            "Category,NodeName,Timestamp,URI,Description,Notes,Confidence\n"
            "vessel,rpi_test,2025_01_01_00_00_00_PST,https://example.com/existing,desc,notes,90.0\n",
            encoding="utf-8",
        )
        existing_uris = {"https://example.com/existing"}
        rows = [
            {
                "Category": "vessel",
                "NodeName": "rpi_test",
                "Timestamp": "2025_01_01_00_00_00_PST",
                "URI": "https://example.com/existing",
                "Description": "desc",
                "Notes": "notes",
                "Confidence": "90.0",
            },
            {
                "Category": "vessel",
                "NodeName": "rpi_test",
                "Timestamp": "2025_01_01_00_00_02_PST",
                "URI": "https://example.com/new",
                "Description": "desc",
                "Notes": "notes",
                "Confidence": "91.0",
            },
        ]

        appended, duplicates = append_manual_samples(manual_samples_path, rows, existing_uris)

        assert appended == 1
        assert duplicates == 1
        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            written_rows = list(csv.DictReader(handle))
        assert len(written_rows) == 2
        assert written_rows[-1]["URI"] == "https://example.com/new"


class TestProcessFalsePositives:
    """Integration-style tests for process_false_positives."""

    def test_uses_retrying_feed_fetcher(self, tmp_path):
        """Processing should use the shared retrying feed-fetch helper."""
        feed = _make_feed()
        with patch(
            "process_false_positives.get_orcasite_feeds_with_retry",
            return_value=[feed],
        ) as mock_get_feeds, patch(
            "process_false_positives.get_model_inference"
        ), patch("process_false_positives.get_orcahello_detections", return_value=[]):
            summary = process_false_positives(
                manual_samples_path=tmp_path / "manual_samples.csv",
                output_dir=tmp_path / "segments",
            )

        assert summary["rejected"] == 0
        mock_get_feeds.assert_called_once_with()

    def test_appends_only_mismatched_whale_segments_with_corrected_class(self, tmp_path):
        """Whale-class segments should be rewritten unless they already match the corrected class."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"
        manual_samples_path.write_text(
            "Category,NodeName,Timestamp,URI,Description,Notes,Confidence\n"
            "vessel,rpi_test,2025_01_01_04_00_00_PST,https://example.com/existing,desc,manual,90.0\n",
            encoding="utf-8",
        )

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_orcahello_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_samples",
                 return_value=[
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_00_PST",
                         "URI": "https://example.com/existing",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "90.0",
                     },
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_02_PST",
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                     {
                         "Category": "water",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_04_PST",
                         "URI": "https://example.com/water",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "20.0",
                      },
                  ],
              ) as mock_add_samples:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
                start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            )

        assert summary["rejected"] == 1
        assert summary["whale_mismatch_segments"] == 2
        assert summary["appended"] == 1
        assert summary["duplicates"] == 1
        assert mock_model.predict.call_count == 1
        assert mock_add_samples.call_args.kwargs["model"] is mock_model
        assert mock_add_samples.call_args.kwargs["corrected_class"] == "vessel"
        assert mock_add_samples.call_args.kwargs["fallback_description"] == detection.comments
        assert mock_add_samples.call_args.kwargs["fallback_notes"] == "fp_machine"

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2
        assert rows[-1]["Category"] == "vessel"
        assert rows[-1]["URI"] == "https://example.com/new"

    def test_continues_when_global_prediction_is_not_resident(self, tmp_path):
        """A non-resident 60-second prediction should still process mismatched whale segments."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_orcahello_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_samples",
                 return_value=[
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_02_PST",
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                     {
                         "Category": "water",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_04_PST",
                         "URI": "https://example.com/water",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "20.0",
                     },
                 ],
             ) as mock_add_samples:
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "water",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
            )

        assert summary["not_false_positive"] == 1
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1
        mock_add_samples.assert_called_once()

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["Category"] == "vessel"
        assert rows[0]["URI"] == "https://example.com/new"

    def test_skips_segments_that_already_match_corrected_whale_class(self, tmp_path):
        """Segments already predicted as the corrected whale class should not be appended."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Likely transient calls from Bigg's whales.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_orcahello_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_samples",
                 return_value=[
                     {
                         "Category": "transient",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_00_PST",
                         "URI": "https://example.com/correct",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "92.0",
                     },
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_02_PST",
                         "URI": "https://example.com/resident",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                     {
                         "Category": "humpback",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_04_PST",
                         "URI": "https://example.com/humpback",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "89.0",
                     },
                     {
                         "Category": "water",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_00_06_PST",
                         "URI": "https://example.com/water",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "20.0",
                     },
                 ],
             ):
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
            )

        assert summary["whale_mismatch_segments"] == 2
        assert summary["appended"] == 2
        assert summary["duplicates"] == 0

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert [row["URI"] for row in rows] == [
            "https://example.com/resident",
            "https://example.com/humpback",
        ]
        assert all(row["Category"] == "transient" for row in rows)

    def test_continues_after_processing_failure(self, tmp_path):
        """A processing error for one detection should be counted and not stop later detections."""
        feed = _make_feed()
        failed_detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        next_detection = OrcaHelloDetection(
            id="det_2",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch(
                 "process_false_positives.get_orcahello_detections",
                 return_value=[next_detection, failed_detection],
             ), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_samples",
                 side_effect=[
                     RuntimeError("decode error"),
                     [
                         {
                             "Category": "resident",
                             "NodeName": "rpi_test",
                             "Timestamp": "2025_01_01_04_05_02_PST",
                             "URI": "https://example.com/new",
                             "Description": "desc",
                             "Notes": "manual",
                             "Confidence": "91.0",
                         },
                     ],
                 ],
             ):
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
            )

        assert summary["rejected"] == 2
        assert summary["processing_failed"] == 1
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["Category"] == "vessel"
        assert rows[0]["URI"] == "https://example.com/new"

    def test_filters_by_actual_category_before_model_inference(self, tmp_path):
        """A category filter should skip non-matching detections before inference."""
        feed = _make_feed()
        matching_detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        non_matching_detection = OrcaHelloDetection(
            id="det_2",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Likely transient calls from Bigg's whales.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch(
                 "process_false_positives.get_orcahello_detections",
                 return_value=[matching_detection, non_matching_detection],
             ), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_samples",
                 return_value=[
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "Timestamp": "2025_01_01_04_05_02_PST",
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                 ],
             ) as mock_add_samples:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
                actual_category_filter="VESSEL",
            )

        assert summary["rejected"] == 2
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1
        assert mock_model.predict.call_count == 1
        assert mock_add_samples.call_count == 1
        assert mock_add_samples.call_args.kwargs["corrected_class"] == "vessel"
