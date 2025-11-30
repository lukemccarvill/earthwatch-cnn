#!/usr/bin/env bash
set -euo pipefail

echo "Downloading UNet_best_bs8.keras from Google Drive..."
mkdir -p models

FILE_ID="14eMxThxImXsjYkA_PMuFWVQf2caPSq6r"
OUTPUT="models/UNet_best_bs8.keras"

# Install gdown if missing
python3 -m pip install --user gdown >/dev/null 2>&1 || true

# Try direct gdown binary first, fallback to module form if needed
if command -v gdown >/dev/null 2>&1; then
    gdown --id "$FILE_ID" -O "$OUTPUT"
else
    python3 -m gdown.cli --id "$FILE_ID" -O "$OUTPUT"
fi

echo "Download complete:"
ls -lh "$OUTPUT"

echo "Generating SHA256 checksum..."
sha256sum "$OUTPUT" > "$OUTPUT.sha256"

echo "✔ Model saved to $OUTPUT"
echo "✔ Checksum saved to $OUTPUT.sha256"