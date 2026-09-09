#!/usr/bin/env python3
"""Generate balanced V2 annotation-rectangle controlled-background mixtures.

This is the V2-compatible preset for the mixing method used by the earlier
successful controlled-background experiment. It uses annotated Abiotic, KW,
and HW training clips as foreground donors, preserves each complete softened
annotation time/frequency rectangle, and adds it at controlled SNR to an
ambient training background from another provider/dataset domain.

All general data discovery, leak safeguards, balancing, checkpointing, and
Kaggle publishing are implemented by
``generate_multispecies_cetacean_pca_mixtures.py``. Command-line arguments
supplied by the user occur after these preset values and therefore override
them.
"""

from __future__ import annotations

import sys

from generate_multispecies_cetacean_pca_mixtures import main


PRESET_ARGUMENTS = [
    "--foreground-mask-method",
    "annotation_rectangle",
    "--donor-labels",
    "Abiotic,KW,HW",
    "--num-mixtures",
    "30000",
    "--label-counts",
    "Abiotic=10000,KW=10000,HW=10000",
    "--sampling-mode",
    "hierarchical_balanced",
    "--domain-columns",
    "Provider,Dataset",
    "--require-different-domain",
    "--output-dir",
    "/kaggle/working/multispecies_cetacean_controlled_background_mixtures",
    "--kaggle-dataset-id",
    "leonisviridis/multispecies-cetacean-v2-controlled-background-mixtures",
    "--kaggle-title",
    "Multispecies Cetacean V2 Controlled Background Mixtures",
]


if __name__ == "__main__":
    sys.argv[1:1] = PRESET_ARGUMENTS
    raise SystemExit(main())
