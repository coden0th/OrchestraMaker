"""Instrument definitions: what each instrument can physically play.

Pitches are sounding MIDI numbers (middle C = 60), times are in seconds.
"""

from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    duration: float
    velocity: int = 90

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Instrument:
    name: str
    gm_program: int                        # General MIDI program, 0-indexed
    low: int                               # lowest sounding pitch
    high: int                              # highest sounding pitch
    max_polyphony: int                     # notes sounding at the same time
    strings: tuple[int, ...] = ()          # open-string pitches, low to high
    frets: int = 0
    max_fret_span: int = 4                 # comfortable hand stretch
    max_phrase_sec: float | None = None    # wind players have to breathe

    def check(self, notes: list[Note]) -> list[str]:
        """Return human-readable reasons why `notes` can't be played as written."""
        problems = [
            f"{n.pitch} at {n.start:.2f}s is outside range {self.low}-{self.high}"
            for n in notes if not self.low <= n.pitch <= self.high
        ]
        problems += self._check_polyphony(notes)
        if self.strings:
            problems += self._check_fretboard(notes)
        if self.max_phrase_sec:
            problems += self._check_breath(notes)
        return problems

    def _check_polyphony(self, notes: list[Note]) -> list[str]:
        # Note-offs sort before note-ons at the same time, so legato lines don't count as overlap.
        events = sorted([(n.start, 1) for n in notes] + [(n.end, -1) for n in notes])
        sounding, problems = 0, []
        for time, delta in events:
            sounding += delta
            if sounding > self.max_polyphony:
                problems.append(f"{sounding} notes at {time:.2f}s, max is {self.max_polyphony}")
        return problems

    def _check_fretboard(self, notes: list[Note]) -> list[str]:
        problems = []
        for start, chord in groupby(sorted(notes, key=lambda n: n.start), key=lambda n: n.start):
            pitches = sorted(n.pitch for n in chord)
            if len(pitches) > 1 and self.fingering(pitches) is None:
                problems.append(f"chord {pitches} at {start:.2f}s has no playable fingering")
        return problems

    def fingering(self, pitches: list[int]) -> list[tuple[int, int]] | None:
        """Assign each pitch a (string index, fret), one note per string, within the hand span."""
        def place(i: int, used: set[int], placed: list[tuple[int, int]]):
            if i == len(pitches):
                return placed
            for s, open_pitch in enumerate(self.strings):
                fret = pitches[i] - open_pitch
                if s in used or not 0 <= fret <= self.frets:
                    continue
                fretted = [f for _, f in placed if f > 0] + ([fret] if fret > 0 else [])
                if fretted and max(fretted) - min(fretted) > self.max_fret_span:
                    continue
                if found := place(i + 1, used | {s}, placed + [(s, fret)]):
                    return found
            return None
        return place(0, set(), [])

    def _check_breath(self, notes: list[Note], min_breath_gap: float = 0.15) -> list[str]:
        problems, phrase_start, last_end = [], None, None
        for n in sorted(notes, key=lambda n: n.start):
            if last_end is None or n.start - last_end >= min_breath_gap:
                phrase_start = n.start
            if n.end - phrase_start > self.max_phrase_sec:
                problems.append(f"phrase from {phrase_start:.2f}s runs past {self.max_phrase_sec}s without a breath")
                phrase_start = n.end  # report once per phrase
            last_end = n.end if last_end is None else max(last_end, n.end)
        return problems


GUITAR = Instrument("jazz_guitar", 26, low=40, high=86, max_polyphony=6,
                    strings=(40, 45, 50, 55, 59, 64), frets=22)
BASS = Instrument("electric_bass", 33, low=28, high=67, max_polyphony=4,
                  strings=(28, 33, 38, 43), frets=24)
ALTO_SAX = Instrument("alto_sax", 65, low=49, high=80, max_polyphony=1, max_phrase_sec=10)
TENOR_SAX = Instrument("tenor_sax", 66, low=44, high=76, max_polyphony=1, max_phrase_sec=12)
PIANO = Instrument("piano", 0, low=21, high=108, max_polyphony=10)

INSTRUMENTS = {i.name: i for i in (GUITAR, BASS, ALTO_SAX, TENOR_SAX, PIANO)}
