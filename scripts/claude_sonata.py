"""Sonata movement in D minor, "Aufklärung" (Enlightenment) - written note by note by Claude.

The story is the era's: from darkness to light. D minor, Mozart's stormy key, in sonata form:
  Exposition   1-8   first theme, D minor: a rising arpeggio "rocket", then sighing appoggiaturas
               9-12  transition, sixteenth-note runs, to F major
               13-20 second theme, F major, cantabile over an Alberti bass, cadential trill
               21-24 closing
  Development  25-36 the main motif through G minor, F minor, F major, the Neapolitan (E flat), a diminished
                     seventh, a long dominant pedal - then a general pause
  Recapitulation 37-58 the first theme returns; the second theme returns in D MAJOR - the light
  Coda         59-62 a last memory of the minor subdominant, then D major
A harmony checker flags every melody note that is neither a chord tone nor a stepwise non-chord tone.

Run: .venv/bin/python scripts/claude_sonata.py   -> outputs/claude/sonata_in_d_minor.{wav,json}
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.instruments import PIANO, Note  # noqa: E402
from orchestramaker.takes import save_take  # noqa: E402

BPM = 126
NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def p(name: str) -> int:
    step, rest = name[0], name[1:]
    return 12 * (int(rest.strip("#b")) + 1) + NAMES[step] + rest.count("#") - rest.count("b")


def trill(upper, main, beats, end=()):
    """Alternating 32nds starting on the upper note (classical practice), then an optional turn."""
    n = int(round((beats - sum(d for _, d in end)) / 0.125))
    return [(upper if i % 2 == 0 else main, 0.125) for i in range(n)] + list(end)


def run(names, dur=0.25):
    return [(n, dur) for n in names.split()]


# ---- right hand: (pitch, beats) per bar; None = rest -------------------------------------------------------
THEME1 = [
    [("D4", .5), ("F4", .5), ("A4", .5), ("D5", .5), ("F5", 1), ("A5", 1)],                   # 1  i
    [("Bb5", 1), ("A5", 1), ("G5", .5), ("E5", .5), ("C#5", 1)],                              # 2  V7  sigh
    [("D5", .5), ("F5", .5), ("A5", 1), ("D6", 1), ("C#6", .5), ("D6", .5)],                  # 3  i
    [("Bb5", .5), ("A5", .5), ("G5", .5), ("F5", .5), ("E5", 2)],                             # 4  iv6 | V  (half cadence)
    [("D4", .5), ("F4", .5), ("A4", .5), ("D5", .5), ("F5", 1), ("A5", 1)],                   # 5  i
    [("G5", .5), ("Bb5", .5), ("D6", 1), ("B5", 1), ("G#5", 1)],                              # 6  iv | vii°7/V
    [("A5", 1), ("F5", .5), ("D5", .5)] + trill("F5", "E5", 2, end=[("D5", .25), ("E5", .25)]),  # 7  i6/4 | V7 trill
    [("D5", 1), ("A4", 1), ("D4", 2)],                                                        # 8  i  (perfect cadence)
]
TRANSITION = [
    run("D5 E5 F5 G5 A5 Bb5 A5 G5 F5 E5 D5 E5 F5 G5 A5 F5"),                                  # 9  i
    run("D6 C6 Bb5 A5 G5 F5 E5 D5 C5 E5 G5 Bb5 C6 Bb5 G5 E5"),                                # 10 VI (Bb) | V7 of F
    [("A5", .5), ("C6", .5), ("F6", 1), ("E6", .5), ("D6", .5), ("C6", .5), ("Bb5", .5)],     # 11 F | C7
    [("A5", .5), ("G5", .5), ("F5", .5), ("E5", .5), ("G5", 1.5), (None, .5)],                # 12 C  (half cadence in F)
]
THEME2 = [  # in F major
    [("A5", 1.5), ("Bb5", .5), ("A5", .5), ("G5", .5), ("F5", 1)],                           # 13 I
    [("E5", 1), ("G5", .5), ("Bb5", .5), ("A5", 1), ("G5", 1)],                               # 14 V7  (A: appoggiatura)
    [("Bb5", 1), ("D6", .5), ("Bb5", .5), ("A5", .5), ("G5", .5), ("E5", .5), ("G5", .5)],    # 15 ii6 | V
    [("G5", .5), ("F5", 1.5), ("C5", .5), ("D5", .5), ("E5", .5), ("F5", .5)],                # 16 I   (G: appoggiatura)
    [("A5", 1), ("C6", .5), ("Bb5", .5), ("A5", .5), ("G5", .5), ("F5", 1)],                  # 17 I
    [("D6", 1), ("Bb5", .5), ("G5", .5), ("F5", .5), ("D5", .5), ("B4", 1)],                  # 18 IV | V7/V (B natural)
    [("C6", 1), ("A5", .5), ("F5", .5)] + trill("A5", "G5", 2, end=[("F5", .25), ("G5", .25)]),  # 19 I6/4 | V7 trill
    [("F5", 1), ("C5", 1), ("A4", 1), ("F4", 1)],                                             # 20 I
]
CLOSING = [  # in F major
    run("F5 A5 C6 F6 C6 A5 F5 C5", .5),                                                      # 21 I
    run("Bb5 D6 F6 D6 C6 Bb5 G5 E5", .5),                                                    # 22 IV | V7
    [("F5", 1), ("A5", 1), ("C6", 1), ("A5", 1)],                                             # 23 I
    [("F5", 2), (None, 2)],                                                                   # 24 I
]
DEVELOPMENT = [
    [("G4", .5), ("Bb4", .5), ("D5", .5), ("G5", .5), ("Bb5", 1), ("D6", 1)],                 # 25 g: i
    [("Eb6", 1), ("D6", 1), ("C6", .5), ("A5", .5), ("F#5", 1)],                              # 26 g: V7  sigh
    [("F4", .5), ("Ab4", .5), ("C5", .5), ("F5", .5), ("Ab5", 1), ("C6", 1)],                 # 27 f: i
    [("Db6", 1), ("C6", 1), ("Bb5", .5), ("G5", .5), ("E5", 1)],                              # 28 f: V7  sigh
    [("A5", .5), ("C6", .5), ("F6", 1), ("D6", 1), ("F6", 1)],                                # 29 F | Bb  (a glimpse of light)
    [("G5", .5), ("Bb5", .5), ("Eb6", 1), ("Eb6", .5), ("D6", .5), ("C6", .5), ("Bb5", .5)],  # 30 Neapolitan (Eb/G)
    [("B5", .5), ("D6", .5), ("F6", 1), ("G#5", 1), ("B5", 1)],                               # 31 vii°7/V
    [("A5", 1), ("D6", 1), ("C#6", 1), ("E6", 1)],                                            # 32 i6/4 | V7
    run("E5 F5 E5 D5 C#5 D5 E5 F5 G5 F5 E5 D5 C#5 D5 E5 C#5"),                                # 33 V7 pedal
    run("A5 G5 F5 E5 D5 C#5 D5 E5 F5 E5 D5 C#5 Bb4 A4 G4 E4"),                                # 34 V7 pedal
    [("C#5", .5), ("E5", .5), ("G5", .5), ("Bb5", .5), ("A5", 2)],                            # 35 V7(b9)
    [(None, 4)],                                                                              # 36 general pause
]
RETRANSITION = [
    TRANSITION[0],                                                                            # 45 i
    [("C#5", .5), ("E5", .5), ("A5", .5), ("C#6", .5), ("E6", 1.5), (None, .5)],              # 46 V  (half cadence)
]
CODA = [
    run("D5 F#5 A5 D6 A5 F#5 D5 A4", .5),                                                    # 59 I
    [("G5", 1), ("B5", 1), ("Bb5", 1), ("A5", 1)],                                            # 60 IV | iv (minor: a memory)
    [("F#5", .5), ("A5", .5), ("D6", 1), ("E6", .5), ("C#6", .5), ("A5", 1)],                 # 61 I6/4 | V7
    [("D6", 2), (None, 2)],                                                                   # 62 I
]


def transpose(bars, semitones):
    rev = {v: k for k, v in NAMES.items()}
    def name(n):
        m = p(n) + semitones
        pc = m % 12
        return (rev[pc] if pc in rev else rev[pc - 1] + "#") + str(m // 12 - 1)
    return [[(name(n) if n else None, d) for n, d in bar] for bar in bars]


RIGHT = (THEME1 + TRANSITION + THEME2 + CLOSING + DEVELOPMENT + THEME1 + RETRANSITION
         + transpose(THEME2, -3) + transpose(CLOSING, -3) + CODA)

# ---- harmony per half bar, and the left-hand texture of each bar ----------------------------------------
H1 = [("Dm", "Dm"), ("A7", "A7"), ("Dm", "Dm"), ("Gm6", "A"), ("Dm", "Dm"), ("Gm", "G#o7"), ("Dm64", "A7"),
      ("Dm", "Dm")]
HT = [("Dm", "Dm"), ("Bb", "C7"), ("F", "C7"), ("C", "C")]
H2F = [("F", "F"), ("C7", "C7"), ("Gm6", "C"), ("F", "F"), ("F", "F"), ("Bb", "G7"), ("F64", "C7"), ("F", "F")]
HCF = [("F", "F"), ("Bb", "C7"), ("F", "F"), ("F", "END")]
HD = [("Gm", "Gm"), ("D7", "D7"), ("Fm", "Fm"), ("C7", "C7"), ("F", "Bb"), ("Eb6", "Eb6"), ("G#o7", "G#o7"),
      ("Dm64", "A7"), ("A7", "A7"), ("A7", "A7"), ("A7", "A7"), ("A7", "REST")]
HR = [("Dm", "Dm"), ("A", "A")]
H2D = [("D", "D"), ("A7", "A7"), ("Em6", "A"), ("D", "D"), ("D", "D"), ("G", "E7"), ("D64", "A7"), ("D", "D")]
HCD = [("D", "D"), ("G", "A7"), ("D", "D"), ("D", "D")]
HCODA = [("D", "D"), ("G", "Gm"), ("D64", "A7"), ("D", "END")]
HARMONY = H1 + HT + H2F + HCF + HD + H1 + HR + H2D + HCD + HCODA

TEXTURE = (["octaves"] + ["pulse"] * 7 + ["broken"] * 4 + ["alberti"] * 8 + ["pulse"] * 4
           + ["broken"] * 8 + ["pedal"] * 3 + ["chord"]
           + ["octaves"] + ["pulse"] * 7 + ["broken"] * 2 + ["alberti"] * 8 + ["pulse"] * 4
           + ["arpeggio", "chord", "chord", "chord"])

# chord: (bass, upper voices); ALB: Alberti (low, high, middle)
CHORDS = {
    "Dm": ("D2", "F3 A3 D4"), "A": ("A2", "C#3 E3 A3"), "A7": ("A2", "C#3 E3 G3"), "Gm": ("G2", "G3 Bb3 D4"),
    "Gm6": ("Bb2", "D3 G3 Bb3"), "G#o7": ("G#2", "B2 D3 F3"), "Dm64": ("A2", "D3 F3 A3"),
    "F": ("F2", "A3 C4 F4"), "C7": ("C3", "E3 G3 Bb3"), "C": ("C3", "E3 G3 C4"), "Bb": ("Bb2", "D3 F3 Bb3"),
    "G7": ("G2", "B2 D3 F3"), "F64": ("C3", "F3 A3 C4"), "D7": ("D2", "F#3 A3 C4"), "Fm": ("F2", "Ab3 C4 F4"),
    "Eb6": ("G2", "G3 Bb3 Eb4"), "D": ("D2", "F#3 A3 D4"), "G": ("G2", "G3 B3 D4"), "Em6": ("G2", "B2 E3 G3"),
    "E7": ("E2", "G#2 B2 D3"), "D64": ("A2", "D3 F#3 A3"),
}
ALB = {"F": ("F2", "C3", "A2"), "C7": ("E2", "Bb2", "G2"), "Gm6": ("Bb2", "G3", "D3"), "C": ("E2", "C3", "G2"),
       "Bb": ("Bb2", "F3", "D3"), "G7": ("B2", "F3", "D3"), "F64": ("C3", "A3", "F3"),
       "D": ("D2", "A2", "F#2"), "A7": ("C#2", "G2", "E2"), "Em6": ("G2", "E3", "B2"), "A": ("C#2", "A2", "E2"),
       "G": ("G2", "D3", "B2"), "E7": ("G#2", "D3", "B2"), "D64": ("A2", "F#3", "D3")}

# dynamics (right-hand velocity) and tempo stretch per bar
LOUD = ([88] * 4 + [70] * 4 + [76, 82, 88, 80] + [62] * 8 + [90] * 4 + [92, 86, 88, 82, 76, 84, 90, 94]
        + [80, 90, 100, 96] + [88] * 4 + [70] * 4 + [80, 86] + [72] * 8 + [90] * 4 + [88, 80, 90, 96])
STRETCH = {12: 1.08, 20: 1.05, 24: 1.1, 35: 1.12, 36: 1.6, 46: 1.08, 54: 1.05, 61: 1.12, 62: 1.35}


def check_harmony():
    """Every melody note should be a chord tone, or a non-chord tone reached or left by step."""
    problems = []
    for b, (bar, (h1, h2)) in enumerate(zip(RIGHT, HARMONY)):
        pos, notes = 0.0, [(n, d) for n, d in bar]
        for i, (name, dur) in enumerate(notes):
            chord = h1 if pos < 2 else h2
            if name and chord not in ("END", "REST"):
                bass, upper = CHORDS[chord]
                tones = {p(x) % 12 for x in [bass] + upper.split()}
                if p(name) % 12 not in tones:
                    prev = next((notes[j][0] for j in range(i - 1, -1, -1) if notes[j][0]), None)
                    nxt = next((notes[j][0] for j in range(i + 1, len(notes)) if notes[j][0]), None)
                    step = lambda a: a is not None and abs(p(a) - p(name)) <= 2
                    if not (step(prev) or step(nxt)):
                        problems.append(f"bar {b + 1} beat {pos + 1:g}: {name} over {chord} by leap on both sides")
            pos += dur
    return problems


def main():
    assert len(RIGHT) == len(HARMONY) == len(TEXTURE) == len(LOUD) == 62, (len(RIGHT), len(HARMONY), len(TEXTURE), len(LOUD))
    for b, bar in enumerate(RIGHT):
        assert abs(sum(d for _, d in bar) - 4) < 1e-6, f"bar {b + 1}: {sum(d for _, d in bar)} beats"
    for problem in check_harmony():
        print("harmony:", problem)

    rng = random.Random(0)
    beat = 60 / BPM
    starts = [0.0]
    for b in range(62):
        starts.append(starts[-1] + 4 * beat * STRETCH.get(b + 1, 1.0))
    at = lambda b, pos: starts[b] + (starts[b + 1] - starts[b]) * pos / 4
    notes = []

    for b, bar in enumerate(RIGHT):
        pos = 0.0
        legato = 1.0 if TEXTURE[b] == "alberti" else 0.9
        for name, dur in bar:
            if name:
                accent = 6 if pos % 2 == 0 else 0
                sigh = -10 if dur >= 1 and pos % 1 == 0 and pos > 0 and bar[0][1] >= 1 else 0
                notes.append(Note(p(name), at(b, pos) + rng.uniform(-.006, .006), (at(b, pos + dur) - at(b, pos)) * legato,
                                  max(30, min(115, LOUD[b] + accent + sigh + rng.randint(-3, 3)))))
            pos += dur

    for b, (h1, h2) in enumerate(HARMONY):
        lv = int(LOUD[b] * 0.68)
        for half, chord in enumerate((h1, h2)):
            t = 2 * half
            if chord == "REST":
                continue
            if chord == "END":
                last = h1
                bass, upper = CHORDS[last]
                for name in [bass, bass[:-1] + str(int(bass[-1]) - 1)] + upper.split():
                    notes.append(Note(p(name), at(b, t), at(b, 3.9) - at(b, t), lv))
                continue
            bass, upper = CHORDS[chord]
            kind = TEXTURE[b]
            add = lambda name, pos, dur, v=lv: notes.append(
                Note(p(name), at(b, pos) + rng.uniform(-.005, .005), (at(b, pos + dur) - at(b, pos)) * .88, v))
            if kind == "pulse":  # agitated repeated chords, bass on the half bar
                add(bass, t, 1.0, lv + 6)
                for k in range(4):
                    for name in upper.split():
                        add(name, t + .5 * k, .45, lv - 8 + (4 if k == 0 else 0))
            elif kind == "octaves":
                for k in range(2):
                    add(bass, t + k, .9, lv + 8)
                    add(bass[:-1] + str(int(bass[-1]) + 1), t + k, .9, lv + 4)
            elif kind == "broken":
                lo, hi = bass, bass[:-1] + str(int(bass[-1]) + 1)
                for k, name in enumerate((lo, hi, lo, hi)):
                    add(name, t + .5 * k, .5, lv + (6 if k == 0 else 0))
            elif kind == "alberti":
                lo, top, mid = ALB[chord]
                for k, name in enumerate((lo, top, mid, top)):
                    add(name, t + .5 * k, .5, lv - 6 + (6 if k == 0 else 0))
            elif kind == "pedal":  # dominant pedal on A, growing
                for k, name in enumerate(("A1", "A2", "A1", "A2")):
                    add(name, t + .5 * k, .5, lv + 2 * b % 7)
            elif kind in ("chord", "arpeggio"):
                if kind == "arpeggio":
                    for k, name in enumerate([bass] + upper.split()):
                        add(name, t + .5 * k, 2 - .5 * k, lv)
                else:
                    add(bass, t, 1.9, lv + 6)
                    for name in upper.split():
                        add(name, t, 1.9, lv)

    meta = {"title": "Claude: Sonata movement in D minor, \"Aufklärung\" (from darkness to light)",
            "style": "mozart", "instrument": "piano", "source": "claude",
            "composer": "Claude (written note by note, not generated by the model)"}
    path = save_take(ROOT / "outputs/claude", "sonata_in_d_minor", [("as written", [(PIANO, notes)])], meta)
    print(f"{len(notes)} notes, {max(n.end for n in notes):.0f} s -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
