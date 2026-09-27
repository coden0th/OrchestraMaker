#!/usr/bin/env bash
# Download the MuseScore General SoundFont (MIT, based on FluidR3) into data/soundfonts/.
set -euo pipefail
dir="$(dirname "$0")/../data/soundfonts"
base="https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General"
mkdir -p "$dir"
if [ ! -f "$dir/MuseScore_General.sf2" ]; then
    curl -fL --progress-bar -C - -o "$dir/MuseScore_General.sf2.part" "$base/MuseScore_General.sf2"
    mv "$dir/MuseScore_General.sf2.part" "$dir/MuseScore_General.sf2"
fi
curl -fsSL -o "$dir/MuseScore_General_License.md" "$base/MuseScore_General_License.md"
echo "SoundFont ready in $dir"
