#!/bin/bash
# Script to patch fastai_audio for Python 3.11+ compatibility
# Based on https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/patch_fastai_audio.sh

echo "Applying Python 3.11 compatibility patch to fastai_audio..."

# Find the fastai_audio installation directory using pip show
AUDIO_PACKAGE_DIR=$(python3 -c "import sys; import site; print([p for p in sys.path if 'site-packages' in p][0])" 2>/dev/null)

if [ -z "$AUDIO_PACKAGE_DIR" ]; then
    echo "Error: Could not find site-packages directory"
    exit 1
fi

# Look for the audio package
AUDIO_DATA_FILE="${AUDIO_PACKAGE_DIR}/audio/data.py"

if [ ! -f "$AUDIO_DATA_FILE" ]; then
    echo "Error: Could not find audio/data.py at $AUDIO_DATA_FILE"
    echo "Checking if audio package is installed..."
    python3 -c "import audio" 2>&1 || echo "audio package not found"
    exit 1
fi

echo "Found audio/data.py at: $AUDIO_DATA_FILE"

# Create a backup
cp "$AUDIO_DATA_FILE" "${AUDIO_DATA_FILE}.bak"

# Apply the fix using sed
# Change: sg_cfg: SpectrogramConfig = SpectrogramConfig()
# To: sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)

# First, add 'field' to the dataclasses import
sed -i 's/from dataclasses import dataclass, asdict/from dataclasses import dataclass, asdict, field/' "$AUDIO_DATA_FILE"

# Then, fix the sg_cfg line
sed -i 's/sg_cfg: SpectrogramConfig = SpectrogramConfig()/sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)/' "$AUDIO_DATA_FILE"

echo "Patch applied successfully!"
echo "Backup saved to: ${AUDIO_DATA_FILE}.bak"

# Verify the patch was applied
if grep -q "field(default_factory=SpectrogramConfig)" "$AUDIO_DATA_FILE"; then
    echo "✓ Patch verification successful: field(default_factory=SpectrogramConfig) found"
else
    echo "✗ Warning: Patch may not have been applied correctly"
    exit 1
fi
