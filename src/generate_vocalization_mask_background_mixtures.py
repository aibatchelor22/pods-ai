#!/usr/bin/env python3
"""Generate paper-style PCA vocalization-mask/background mixtures.

This is the convenient entry point for the PCA-percentile foreground mode in
``generate_controlled_background_mixtures.py``. It creates three-second WAV
files, a generated-only manifest, a combined trainer-ready manifest, audit and
summary files, and Kaggle dataset metadata. Explicit command-line arguments
override the defaults supplied here.
"""

from __future__ import annotations

import sys

import generate_controlled_background_mixtures as generator


DEFAULTS = [
    "--foreground-mask-method",
    "pca_percentile",
    "--mask-percentile",
    "95",
    "--pca-components",
    "1",
    "--annotation-levels",
    "Call,Detection,File",
    "--output-dir",
    "/kaggle/working/dclde_vocalization_mask_mixtures",
    "--kaggle-dataset-id",
    "leonisviridis/dclde-vocalization-mask-mixtures",
    "--kaggle-title",
    "DCLDE Vocalization Mask Background Mixtures",
]


def main() -> int:
    # Defaults are inserted first so a value supplied by the user later on the
    # command line wins under argparse's normal last-value behavior.
    sys.argv[1:1] = DEFAULTS
    return generator.main()


if __name__ == "__main__":
    raise SystemExit(main())
