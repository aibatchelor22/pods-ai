#!/usr/bin/env python3
"""Checkpoint-enabled entry point for the multispecies dataset builder.

The implementation lives in ``build_multispecies_cetacean_dataset.py`` so the
ordinary and checkpointed workflows share exactly the same extraction, audio
normalization, QC, manifest, and Kaggle upload behavior. Use
``--checkpoint-every-remote-clips`` with ``--upload`` and
``--adopt-existing-dataset`` to enable durable remote-source checkpoints.
"""

from build_multispecies_cetacean_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
