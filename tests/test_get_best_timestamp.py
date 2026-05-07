# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for get_best_timestamp.py.

Tests focus on node_slug_to_name() and build_sample(), which contain the
helper logic specific to this script.
"""
import pytest

from get_best_timestamp import build_sample, node_slug_to_name


# ---------------------------------------------------------------------------
# node_slug_to_name
# ---------------------------------------------------------------------------

class TestNodeSlugToName:
    """Tests for converting a URL slug to a node_name."""

    def test_prepends_rpi_prefix(self):
        """'rpi_' prefix should be added."""
        assert node_slug_to_name('orcasound-lab') == 'rpi_orcasound_lab'

    def test_replaces_hyphens_with_underscores(self):
        """Hyphens should become underscores."""
        assert node_slug_to_name('andrews-bay') == 'rpi_andrews_bay'

    def test_two_word_slug(self):
        """A two-word slug should convert correctly."""
        assert node_slug_to_name('bush-point') == 'rpi_bush_point'

    def test_multi_word_slug(self):
        """Multi-word slugs should convert all hyphens to underscores."""
        assert node_slug_to_name('sunset-bay') == 'rpi_sunset_bay'

    def test_single_word_slug(self):
        """A single-word slug (no hyphens) should just get the prefix."""
        assert node_slug_to_name('lab') == 'rpi_lab'


# ---------------------------------------------------------------------------
# build_sample
# ---------------------------------------------------------------------------

class TestBuildSample:
    """Tests for constructing a sample dict from node_slug and timestamp_str."""

    SLUG = 'orcasound-lab'
    TIMESTAMP = '2023_08_18_00_59_53_PST'

    def _sample(self):
        return build_sample(self.SLUG, self.TIMESTAMP)

    def test_node_name_derived_from_slug(self):
        """NodeName should be the node_name derived from the slug."""
        assert self._sample()['NodeName'] == 'rpi_orcasound_lab'

    def test_timestamp_preserved(self):
        """Timestamp should be the original timestamp_str argument."""
        assert self._sample()['Timestamp'] == self.TIMESTAMP

    def test_notes_is_tp_human_only(self):
        """Notes should be 'tp_human_only' so process_sample uses model correction."""
        assert self._sample()['Notes'] == 'tp_human_only'

    def test_uri_contains_slug(self):
        """URI should contain the original slug."""
        assert 'orcasound-lab' in self._sample()['URI']

    def test_uri_starts_with_bouts_base(self):
        """URI should start with the Orcasound bouts interface base URL."""
        assert self._sample()['URI'].startswith(
            'https://live.orcasound.net/bouts/new/orcasound-lab'
        )

    def test_uri_contains_encoded_timestamp(self):
        """URI should contain the UTC-encoded version of the given timestamp."""
        # 2023-08-18 is in summer (PDT = UTC-7), so 00:59:53 PDT = 07:59:53 UTC.
        assert '2023-08-18T07%3A59%3A53' in self._sample()['URI']
