"""Stage 1 sanity check: each instrument plays a scale, then a small jazz trio plays a ii-V-I.

Everything here is hand-written — no model yet. It verifies the instrument rules and the audio engine.
Run: .venv/bin/python scripts/demo_instruments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestramaker.instruments import ALTO_SAX, BASS, GUITAR, INSTRUMENTS, Note
from orchestramaker.render import render

OUT = Path(__file__).resolve().parent.parent / "outputs/stage1"
BPM = 132
BEAT = 60 / BPM
MAJOR = (0, 2, 4, 5, 7, 9, 11)


def scale(instrument, step=0.25):
    tonic = next(p for p in range(instrument.low, instrument.high) if p % 12 == 0)  # first C in range
    up = [p for p in range(tonic, min(tonic + 24, instrument.high) + 1) if (p - tonic) % 12 in MAJOR]
    pitches = up + up[-2::-1]
    return [Note(p, i * step, step * 0.95, 85) for i, p in enumerate(pitches)]


def swing(beat):
    """Straight beat position -> swung position: each beat splits long-short (2:1) instead of 1:1."""
    whole, frac = divmod(beat, 1)
    return whole + (frac * 4 / 3 if frac <= 0.5 else 2 / 3 + (frac - 0.5) * 2 / 3)


def swing_line(pitches_and_beats):
    """(pitch, beats) pairs -> notes with swung eighths."""
    notes, beat = [], 0
    for pitch, beats in pitches_and_beats:
        start, end = swing(beat), swing(beat + beats)
        if pitch is not None:
            notes.append(Note(pitch, start * BEAT, (end - start) * BEAT * 0.9, 95 if beat % 1 == 0 else 80))
        beat += beats
    return notes


def trio():
    dm7, g7, cmaj7 = (50, 57, 60, 65, 69), (43, 53, 59, 62), (48, 55, 59, 64, 67)
    changes = [dm7, g7, cmaj7, cmaj7]
    guitar = [Note(p, (bar * 4 + b) * BEAT, BEAT * 0.45, 70 if b % 2 == 0 else 88)
              for bar, chord in enumerate(changes) for b in range(4) for p in chord]

    walking = [38, 40, 41, 42, 43, 47, 50, 49, 48, 40, 43, 45, 48, 43, 40, 39]
    bass = [Note(p, i * BEAT, BEAT * 0.9, 95) for i, p in enumerate(walking)]

    e, q, h = 0.5, 1, 2
    melody = [
        (None, e), (65, e), (64, e), (62, e), (60, e), (57, e), (60, e), (62, e),   # Dm7
        (65, e), (62, e), (59, e), (55, e), (56, e), (59, e), (62, e), (65, e),     # G7
        (64, q + e), (62, e), (60, e), (59, e), (60, q),                            # Cmaj7
        (67, h), (64, q), (None, q),
    ]
    sax = swing_line(melody)

    parts = [(GUITAR, guitar), (BASS, bass), (ALTO_SAX, sax)]
    for instrument, notes in parts:
        problems = instrument.check(notes)
        print(f"  {instrument.name:14s} {'playable' if not problems else problems}")
    # Play the 4 bars twice.
    loop = 16 * BEAT
    return [(i, notes + [Note(n.pitch, n.start + loop, n.duration, n.velocity) for n in notes])
            for i, notes in parts]


def main():
    print("Scales:")
    for instrument in INSTRUMENTS.values():
        notes = scale(instrument)
        assert not instrument.check(notes), instrument.check(notes)
        print(f"  {render([(instrument, notes)], OUT / f'scale_{instrument.name}.wav').name}")

    print("Trio (ii-V-I in C):")
    print(f"  {render(trio(), OUT / 'trio_ii_V_I.wav').name}")

    print("Rule checks on deliberately unplayable input:")
    print("  sax chord:     ", ALTO_SAX.check([Note(60, 0, 1), Note(64, 0, 1)]))
    print("  guitar E-F-F#: ", GUITAR.check([Note(40, 0, 1), Note(41, 0, 1), Note(42, 0, 1)]))
    print("  sax too low:   ", ALTO_SAX.check([Note(40, 0, 1)]))
    print("  sax no breath: ", ALTO_SAX.check([Note(60 + i % 5, i * 0.25, 0.25) for i in range(60)]))
    print(f"\nWAV files in {OUT}")


if __name__ == "__main__":
    main()
