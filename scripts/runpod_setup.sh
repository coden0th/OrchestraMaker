#!/usr/bin/env bash
# One-shot setup on a fresh RunPod PyTorch pod: code, venv (reusing the image's torch), data, tokens.
# Usage on the pod:  bash <(curl -fsSL https://raw.githubusercontent.com/coden0th/OrchestraMaker/main/scripts/runpod_setup.sh)
set -euxo pipefail
cd /root
[ -d OrchestraMaker ] || git clone -q https://github.com/coden0th/OrchestraMaker.git
cd OrchestraMaker && git pull -q
python -m venv --system-site-packages .venv
.venv/bin/pip install -q pyarrow symusic numpy "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

# PianoCoRe (classical, CC BY-NC-SA 4.0), PiJAMA (jazz, CC BY-NC 4.0), Aria-MIDI deduped (CC BY-NC-SA 4.0)
.venv/bin/hf download SyMuPe/PianoCoRe --repo-type dataset --local-dir data/raw/pianocore_hf
.venv/bin/hf download loubb/aria-midi aria-midi-v1-deduped-ext.tar.gz --repo-type dataset --local-dir data/raw/aria_midi
mkdir -p data/raw/pijama && cd data/raw/pijama
[ -d midi_kong ] || { curl -fsSL -o midi_kong.zip https://zenodo.org/api/records/8354955/files/midi_kong.zip/content && unzip -q midi_kong.zip; }
cd /root/OrchestraMaker

# Quota-limited pods report more CPUs than they may use; take the cgroup quota when there is one.
quota=$(awk '{ if ($1 > 0) print int($1 / 100000) }' /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true)
workers=${quota:-$(nproc)}
.venv/bin/python scripts/prepare_v1.py --compact --out data/tokens_v15 --workers "$((workers - 1))" \
    --aria-midi data/raw/aria_midi/aria-midi-v1-deduped-ext.tar.gz
echo SETUP_DONE
