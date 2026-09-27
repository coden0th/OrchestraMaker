"""v1 data: PianoCoRe (classical piano) + PiJAMA (jazz piano), tokenized in parallel into a memory map.

Writes <out>/vocab.json, <out>/tokens.bin (uint16) and <out>/index.json (one entry per performance).
Styles: composers with >= --min-hours of tier-B piano get their own STYLE token (others share
classical_other); jazz pianists with >= --jazz-min-hours get jazz_<name> (others share jazz_piano).

Run:  .venv/bin/python scripts/prepare_v1.py --pianocore data/raw/pianocore_hf --pijama data/raw/pijama
Reuse a vocabulary (e.g. the one made on the full data): --vocab path/to/vocab.json
"""

import argparse
import json
import os
import sys
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import symusic  # noqa: E402

from orchestramaker.datasets import (aria_midi_files, pianocore_row_groups, pianocore_rows,  # noqa: E402
                                     pijama_files, pijama_pianist, score_notes, slugify)
from orchestramaker.tokenizer import Tokenizer  # noqa: E402

TOKENIZER: Tokenizer | None = None
STYLE_OF: dict[str, str] = {}


def composer_styles(hours: Counter, min_hours: float, known: set[str] | None = None) -> dict[str, str]:
    """Composer name -> style; surname only unless two kept composers share it (bach_johann_sebastian).
    With a known vocabulary, a surname style goes to the composer with the most hours under that surname."""
    if known is not None:
        styles = {}
        for c in sorted(hours, key=hours.get, reverse=True):
            surname, full = slugify(c.split(",")[0]), slugify(c)
            styles[c] = full if full in known else surname if surname in known and surname not in styles.values() \
                else "classical_other"
        return styles
    kept = [c for c, h in hours.items() if h >= min_hours]
    surname = {c: slugify(c.split(",")[0]) for c in kept}
    clashes = Counter(surname.values())
    styles = {c: surname[c] if clashes[surname[c]] == 1 else slugify(c) for c in kept}
    return {c: styles.get(c, "classical_other") for c in hours}


def metadata(args):
    hours = Counter()
    groups = list(pianocore_row_groups(args.pianocore))
    for path, group in groups:
        for row in pianocore_rows(path, group, ["composer", "performance_duration"]):
            hours[row["composer"]] += row["performance_duration"] / 3600
    known = set(Tokenizer.load(args.vocab).styles) if args.vocab else None
    styles = composer_styles(hours, args.min_hours, known)

    files = pijama_files(args.pijama)
    jazz_hours = Counter()
    for f in files:
        jazz_hours[pijama_pianist(f)] += symusic.Score(str(f), ttype="second").end() / 3600
    for pianist, h in jazz_hours.items():
        styles[f"pijama:{pianist}"] = f"jazz_{slugify(pianist)}" if h >= args.jazz_min_hours else "jazz_piano"
    print(f"PianoCoRe tier B: {sum(hours.values()):,.0f} h, {len(hours)} composers in {len(groups)} row groups; "
          f"PiJAMA: {sum(jazz_hours.values()):,.0f} h, {len(jazz_hours)} pianists")

    aria = []
    if args.aria_midi:
        # Aria-MIDI performers are surnames only; reuse a PiJAMA pianist style when the surname is unambiguous.
        by_surname = {}
        for pianist, style in ((k[7:], v) for k, v in styles.items() if k.startswith("pijama:") and v != "jazz_piano"):
            by_surname.setdefault(slugify(pianist).split("_")[-1], set()).add(style)
        genres = tuple(args.aria_genres.split(","))
        for path, md in aria_midi_files(args.aria_midi, genres, args.aria_min_score):
            if md["genre"] != "jazz":
                style = md["genre"]                                   # ragtime, blues
            else:
                match = by_surname.get(slugify(md.get("performer") or ""), set())
                style = match.pop() if len(match) == 1 else "jazz_piano"
            aria.append((path, style, md))
        print(f"Aria-MIDI {'/'.join(genres)}: {len(aria)} files (audio score >= {args.aria_min_score})")
    return styles, groups, files, aria


def init_worker(tokenizer, style_of):
    global TOKENIZER, STYLE_OF
    TOKENIZER, STYLE_OF = tokenizer, style_of


def encode(notes, style):
    if style not in TOKENIZER.styles:
        style = "jazz_piano" if style.startswith("jazz_") else "classical_other"
    return np.array(TOKENIZER.encode(notes, style, "piano"), dtype=np.uint16), style


def tokenize_row_group(task):
    path, group = task
    out = []
    cols = ["composer", "composition", "movement", "split", "performance_midi_bytes"]
    for row in pianocore_rows(path, group, cols):
        try:
            notes = score_notes(symusic.Score.from_midi(row["performance_midi_bytes"], ttype="second"))
        except Exception:
            continue  # a handful of files don't parse
        if len(notes) < 32:
            continue
        ids, style = encode(notes, STYLE_OF[row["composer"]])
        title = f"{row['composer'].replace('_', ' ')} - {row['composition']}" + \
            (f" - {row['movement']}" if row["movement"] else "")
        split = "train" if row["split"] == "train" else "validation"
        out.append(("pianocore", style, split, title, ids))
    return out


def tokenize_pijama(path):
    notes = score_notes(symusic.Score(str(path), ttype="second"))
    if len(notes) < 32:
        return []
    ids, style = encode(notes, STYLE_OF[f"pijama:{pijama_pianist(path)}"])
    split = "validation" if zlib.crc32(str(path.relative_to(path.parents[3])).encode()) % 20 == 0 else "train"
    return [("pijama", style, split, f"{pijama_pianist(path)} - {path.stem}", ids)]


def tokenize_aria(item):
    path, style, md = item
    notes = score_notes(symusic.Score(str(path), ttype="second"))
    if len(notes) < 32:
        return []
    ids, style = encode(notes, style)
    split = "validation" if zlib.crc32(path.name.encode()) % 20 == 0 else "train"
    who = md.get("performer") or md.get("composer") or ""
    return [("aria_jazz", style, split, f"Aria-MIDI {md['genre']} {who} {path.stem}".replace("  ", " "), ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pianocore", type=Path, default=ROOT / "data/raw/pianocore_hf")
    ap.add_argument("--pijama", type=Path, default=ROOT / "data/raw/pijama")
    ap.add_argument("--out", type=Path, default=ROOT / "data/tokens_v1")
    ap.add_argument("--vocab", type=Path)
    ap.add_argument("--compact", action="store_true", help="NOTE_p_v tokens: ~2.9 instead of ~3.9 tokens per note")
    ap.add_argument("--min-hours", type=float, default=20)
    ap.add_argument("--jazz-min-hours", type=float, default=3)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--aria-midi", type=Path, help="aria-midi-v1-deduped-ext.tar.gz, for more jazz")
    ap.add_argument("--aria-genres", default="jazz,ragtime,blues")
    ap.add_argument("--aria-min-score", type=float, default=0.95)
    args = ap.parse_args()
    if args.vocab:  # the vocabulary decides the styles: map every pianist, encode() folds unknown ones
        args.jazz_min_hours = 0

    style_of, groups, files, aria = metadata(args)
    if args.vocab:
        tokenizer = Tokenizer.load(args.vocab)
    else:
        styles = set(style_of.values()) | {s for _, s, _ in aria} | {"classical_other", "jazz_piano"}
        tokenizer = Tokenizer(sorted(styles), ["piano"],
                              compact=args.compact)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out / "vocab.json")
    print(f"{len(tokenizer.styles)} styles, vocab {len(tokenizer)}; tokenizing with {args.workers} workers")

    index = {k: [] for k in ("source", "style", "split", "title", "offset", "length")}
    offset, done = 0, 0
    with open(args.out / "tokens.bin", "wb") as f, \
            Pool(args.workers, initializer=init_worker, initargs=(tokenizer, style_of)) as pool:
        jobs = ([(tokenize_row_group, g) for g in groups] + [(tokenize_pijama, p) for p in files]
                + [(tokenize_aria, a) for a in aria])
        for results in pool.imap_unordered(run_job, jobs, chunksize=1):
            for source, style, split, title, ids in results:
                f.write(ids.tobytes())
                for key, value in zip(index, (source, style, split, title, offset, len(ids))):
                    index[key].append(value)
                offset += len(ids)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} jobs, {offset / 1e9:.2f}B tokens", flush=True)
    index["instrument"] = "piano"
    (args.out / "index.json").write_text(json.dumps(index))

    stats = Counter()
    for source, style, split, n in zip(index["source"], index["style"], index["split"], index["length"]):
        stats[(source, split)] += n
    for (source, split), n in sorted(stats.items()):
        print(f"{source:10s} {split:10s} {n / 1e6:9.1f}M tokens")
    by_style = Counter()
    for style, n in zip(index["style"], index["length"]):
        by_style[style] += n
    print("styles:", ", ".join(f"{s} {n / 1e6:.0f}M" for s, n in by_style.most_common()))


def run_job(job):
    fn, arg = job
    return fn(arg)


if __name__ == "__main__":
    main()
