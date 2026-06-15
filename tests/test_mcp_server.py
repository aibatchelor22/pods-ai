# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
import sys
import unittest
from unittest.mock import MagicMock, patch

# Try to import mcp to see if it is installed.
try:
    import mcp
    import mcp.server.fastmcp
except (ImportError, ModuleNotFoundError):
    # Mock dependencies if they cannot be genuinely imported.
    class MockFastMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func
            return decorator

        def run(self):
            pass

    mock_mcp_module = MagicMock()
    mock_mcp_module.server.fastmcp.FastMCP = MockFastMCP
    sys.modules["mcp"] = mock_mcp_module
    sys.modules["mcp.server"] = mock_mcp_module.server
    sys.modules["mcp.server.fastmcp"] = mock_mcp_module.server.fastmcp

# Mock other optional dependencies only if they cannot be genuinely imported.
for dep in [
    "boto3", "botocore", "botocore.config", "structlog", "pytz",
    "torch", "pandas", "pydub", "librosa", "torchaudio", "numpy", "fastai",
    "fastai.basic_train", "audio", "audio.data"
]:
    try:
        __import__(dep)
    except Exception:
        sys.modules[dep] = MagicMock()

# Now import the code to be tested.
from mcp_server import (
    _validate_node_name,
    list_hydrophones,
    get_recent_detections,
    list_s3_recordings,
    get_sample_stats,
    find_unlabeled_detections,
)
from orcasite_feeds import OrcasiteFeed

class TestMCPServer(unittest.TestCase):

    def test_validate_node_name_valid(self):
        _validate_node_name("rpi_sunset_bay")
        _validate_node_name("sunset-bay")

    def test_validate_node_name_invalid(self):
        with self.assertRaises(ValueError):
            _validate_node_name("invalid name!")
        with self.assertRaises(ValueError):
            _validate_node_name("")

    @patch("mcp_server.get_orcasite_feeds_with_retry")
    def test_list_hydrophones(self, mock_feeds):
        mock_feeds.return_value = [
            OrcasiteFeed(
                id="1",
                name="Station 1",
                node_name="node_1",
                slug="slug_1",
                bucket="bucket_1",
                bucket_region="region_1",
                visible=True,
                location=(45.0, -123.0),
                cloudfront_url="http://cf.url"
            )
        ]
        
        result = list_hydrophones()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Station 1")
        self.assertEqual(result[0]["node_name"], "node_1")

    @patch("mcp_server.get_orcasite_feeds_with_retry")
    @patch("requests.get")
    def test_get_recent_detections(self, mock_get, mock_feeds):
        mock_feeds.return_value = [
            OrcasiteFeed(
                id="feed_123",
                name="Station 1",
                node_name="node_1",
                slug="slug_1",
                bucket="bucket_1",
                bucket_region="region_1",
                visible=True,
                location=(45.0, -123.0)
            )
        ]
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "det_1",
                    "attributes": {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "category": "whale",
                        "source": "human"
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_recent_detections("node_1", limit=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "det_1")
        self.assertEqual(result[0]["category"], "whale")
        
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertIn("feed_123", kwargs["params"]["filter[feed_id]"])

    @patch("boto3.client")
    def test_list_s3_recordings(self, mock_boto):
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "node_1/hls/1700000000/"},
                    {"Prefix": "node_1/hls/1700000060/"}
                ]
            }
        ]
        
        result = list_s3_recordings("node_1")
        self.assertEqual(result["node_name"], "node_1")
        self.assertEqual(result["total_found"], 2)
        self.assertIn("1700000000", result["timestamps"])

    @patch("mcp_server._read_csv")
    @patch("mcp_server.Path.exists")
    def test_get_sample_stats(self, mock_exists, mock_read):
        mock_exists.return_value = True
        mock_read.return_value = [
            {"Category": "whale", "NodeName": "node_1"},
            {"Category": "whale", "NodeName": "node_2"},
            {"Category": "vessel", "NodeName": "node_1"},
        ]
        
        result = get_sample_stats("training")
        self.assertEqual(result["total_samples"], 3)
        self.assertEqual(result["by_category"]["whale"], 2)
        self.assertEqual(result["by_category"]["vessel"], 1)
        self.assertEqual(result["by_station"]["node_1"]["whale"], 1)
        self.assertEqual(result["by_station"]["node_1"]["vessel"], 1)

    @patch("mcp_server._read_csv")
    @patch("mcp_server.Path.exists")
    @patch("mcp_server.get_recent_detections")
    def test_find_unlabeled_detections(self, mock_get_det, mock_exists, mock_read):
        mock_exists.return_value = True
        mock_read.return_value = [
            {"URI": "s3://bucket/node/hls/1700000000/live000.ts"}
        ]
        
        mock_get_det.return_value = [
            {
                "id": "det_new",
                "playlist_timestamp": "1700000060",
                "category": "whale"
            },
            {
                "id": "det_old",
                "playlist_timestamp": "1700000000",
                "category": "whale"
            }
        ]
        
        result = find_unlabeled_detections("node_1")
        self.assertEqual(result["unlabeled_count"], 1)
        self.assertEqual(result["unlabeled"][0]["id"], "det_new")

    @patch("model_inference.get_model_inference")
    @patch("mcp_server.Path.exists")
    @patch("mcp_server.Path.is_absolute")
    def test_compare_models_on_clip(self, mock_abs, mock_exists, mock_get_model):
        mock_abs.return_value = True
        mock_exists.return_value = True
        
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "resident",
            "global_confidence": 0.9,
            "local_confidences": [0.8, 0.9, 1.0],
            "hop_duration": 1.0,
            "segment_duration": 3.0
        }
        mock_get_model.return_value = mock_model
        
        from mcp_server import compare_models_on_clip
        # Need to mock the Path suffix as well
        with patch("mcp_server.Path.suffix", new_callable=unittest.mock.PropertyMock) as mock_suffix:
            mock_suffix.return_value = ".wav"
            result = compare_models_on_clip("/abs/path/test.wav")
        
        self.assertEqual(result["orcahello_label"], "resident")
        self.assertEqual(result["podsai_label"], "resident")
        self.assertTrue(result["models_agree"])

    @patch("mcp_server.find_unlabeled_detections")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("mcp_server.csv.DictWriter")
    def test_export_unlabeled_to_csv(self, mock_writer, mock_open, mock_find):
        mock_find.return_value = {
            "unlabeled": [
                {"id": "det_1", "category": "whale"}
            ]
        }
        
        from mcp_server import export_unlabeled_to_csv
        result = export_unlabeled_to_csv("node_1", "test.csv")
        
        self.assertIn("Successfully created dataset", result)
        mock_open.assert_called_once()
        mock_writer.assert_called_once()

    def test_export_unlabeled_to_csv_invalid_inputs(self):
        from mcp_server import export_unlabeled_to_csv

        # Invalid node name.
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("invalid station!", "test.csv")

        # Invalid limit.
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("node_1", "test.csv", limit=0)
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("node_1", "test.csv", limit=300)

        # Path traversal and invalid filenames.
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("node_1", "../test.csv")
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("node_1", "subdir/test.csv")
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("node_1", "test.txt")
        with self.assertRaises(ValueError):
            export_unlabeled_to_csv("node_1", "")

if __name__ == "__main__":
    unittest.main()
