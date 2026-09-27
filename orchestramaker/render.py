"""Turn notes into audio with FluidSynth and a General MIDI SoundFont."""

from pathlib import Path

import fluidsynth
import numpy as np
import soundfile as sf

from .instruments import Instrument, Note

SAMPLE_RATE = 44100
DEFAULT_SOUNDFONT = Path(__file__).resolve().parent.parent / "data/soundfonts/MuseScore_General.sf2"

Track = tuple[Instrument, list[Note]]


def render(tracks: list[Track], out_path: str | Path, soundfont: Path = DEFAULT_SOUNDFONT,
           tail_sec: float = 1.5) -> Path:
    """Render each (instrument, notes) track on its own MIDI channel and write a WAV file."""
    synth = fluidsynth.Synth(gain=0.4, samplerate=SAMPLE_RATE)
    sfid = synth.sfload(str(soundfont))
    if sfid == -1:
        raise FileNotFoundError(f"could not load SoundFont {soundfont} (run scripts/get_soundfont.sh)")

    channels = [c for c in range(16) if c != 9][:len(tracks)]  # channel 9 is GM drums
    events = []
    for channel, (instrument, notes) in zip(channels, tracks):
        synth.program_select(channel, sfid, 0, instrument.gm_program)
        for n in notes:
            events.append((n.start, 1, channel, n.pitch, n.velocity))
            events.append((n.end, 0, channel, n.pitch, 0))
    events.sort()  # note-offs (0) before note-ons (1) at the same time

    chunks, cursor = [], 0
    for time, is_on, channel, pitch, velocity in events:
        target = round(time * SAMPLE_RATE)
        if target > cursor:
            chunks.append(synth.get_samples(target - cursor))
            cursor = target
        if is_on:
            synth.noteon(channel, pitch, velocity)
        else:
            synth.noteoff(channel, pitch)
    chunks.append(synth.get_samples(round(tail_sec * SAMPLE_RATE)))
    synth.delete()

    audio = np.concatenate(chunks).reshape(-1, 2).astype(np.float32) / 32768
    peak = np.abs(audio).max()
    if peak > 0:
        audio *= 0.9 / peak

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE)
    return out_path
