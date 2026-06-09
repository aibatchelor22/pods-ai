# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Pytest configuration for pods-ai unit tests.

Adds the src directory to sys.path so that modules under src can
be imported directly, and mocks heavy dependencies (ML, audio) that are not
needed for unit tests so the suite can run without a full GPU/fastai environment.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src/ and bootstrap/src are on the path before any test module is imported.
sys.path.insert(0, str(Path(__file__).parent.parent / 'bootstrap' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Packages that are either heavy (torch/fastai) or optional (numpy/pandas) in
# this environment.  Each is stubbed out only when it cannot be genuinely
# imported; CI (which runs pip install -r requirements.txt) will use the real
# packages.
_OPTIONAL_DEPS = [
    'azure',
    'azure.cosmos',
    'numpy',
    'pandas',
    'torch',
    'torchvision',
    'torchaudio',
    'torchaudio.transforms',
    'fastai',
    'fastai.basic_train',
    'pydub',
    'pydub.audio_segment',
    'librosa',
    'soundfile',
    'audio',
    'audio.data',
    'scipy',
    'scipy.signal',
    'huggingface_hub',
]
for _dep in _OPTIONAL_DEPS:
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except ImportError:
            sys.modules[_dep] = MagicMock()
