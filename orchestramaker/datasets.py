"""Openly licensed training data, converted to `Piece`s of performed notes.

- Weimar Jazz Database (ODbL 1.0): transcribed jazz solos, plus the walking bass line under them.
- MAESTRO v3 (CC BY-NC-SA 4.0): classical piano performances, labelled by composer.
"""

import csv
import sqlite3
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import symusic

from .instruments import Note

RAW = Path(__file__).resolve().parent.parent / "data/raw"
WJAZZD_URL = "https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"
MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

WJAZZD_INSTRUMENTS = {
    "ts": "tenor_sax", "ts-c": "tenor_sax", "as": "alto_sax", "ss": "soprano_sax", "bs": "baritone_sax",
    "tp": "trumpet", "cor": "trumpet", "tb": "trombone", "cl": "clarinet",
    "vib": "vibraphone", "p": "piano", "g": "jazz_guitar",
}
MIN_COMPOSER_PIECES = 20  # rarer MAESTRO composers share the "classical_other" style


@dataclass
class Piece:
    source: str
    title: str
    style: str
    instrument: str
    split: str          # train / validation / test
    notes: list[Note]


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        part = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.rename(dest)
    return dest


def load_wjazzd() -> list[Piece]:
    db = sqlite3.connect(download(WJAZZD_URL, RAW / "wjazzd.db"))
    pieces = []
    solos = db.execute("select melid, performer, title, instrument, style from solo_info").fetchall()
    for melid, performer, title, code, style in solos:
        if code not in WJAZZD_INSTRUMENTS:
            continue
        split = {0: "validation", 1: "test"}.get(melid % 20, "train")
        style = f"jazz_{style.lower()}"
        name = f"{performer} - {title}"

        rows = db.execute("select onset, pitch, duration, loud_med from melody where melid = ? order by onset",
                          (melid,)).fetchall()
        t0 = rows[0][0]
        # Loudness is in dB and recording levels differ, so normalize per solo before mapping to velocity.
        loud = np.array([r[3] if r[3] is not None and r[3] > 0 else np.nan for r in rows])
        z = (loud - np.nanmean(loud)) / (np.nanstd(loud) + 1e-6)
        velocity = np.nan_to_num(np.clip(80 + 15 * z, 35, 120), nan=80).astype(int)
        notes = [Note(int(p), on - t0, d, int(v)) for (on, p, d, _), v in zip(rows, velocity)]
        pieces.append(Piece("wjazzd", name, style, WJAZZD_INSTRUMENTS[code], split, notes))

        beats = db.execute("select onset, bass_pitch from beats where melid = ? order by onset", (melid,)).fetchall()
        bass = [Note(int(p), on - t0, min(nxt - on, 2.0) * 0.9, 80)
                for (on, p), (nxt, _) in zip(beats, beats[1:]) if p and p > 0 and on >= t0]
        if len(bass) >= 16:
            pieces.append(Piece("wjazzd", f"{name} (bass)", style, "acoustic_bass", split, bass))
    return pieces


def composer_style(name: str) -> str:
    last = name.split("/")[0].split()[-1]
    return unicodedata.normalize("NFKD", last).encode("ascii", "ignore").decode().lower()


def load_maestro() -> list[Piece]:
    root = RAW / "maestro-v3.0.0"
    if not root.exists():
        with zipfile.ZipFile(download(MAESTRO_URL, RAW / "maestro-v3.0.0-midi.zip")) as z:
            z.extractall(RAW)
    rows = list(csv.DictReader(open(root / "maestro-v3.0.0.csv", encoding="utf-8")))
    counts = Counter(r["canonical_composer"] for r in rows)
    pieces = []
    for r in rows:
        composer = r["canonical_composer"]
        style = composer_style(composer) if counts[composer] >= MIN_COMPOSER_PIECES and "/" not in composer \
            else "classical_other"
        track = symusic.Score(str(root / r["midi_filename"]), ttype="second").tracks[0]
        notes = apply_sustain_pedal(track)
        pieces.append(Piece("maestro", f"{composer} - {r['canonical_title']}", style, "piano", r["split"], notes))
    return pieces


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return "_".join("".join(c if c.isalnum() else " " for c in ascii_text).split())


def score_notes(score: symusic.Score) -> list[Note]:
    """All notes of a (piano) MIDI score in seconds, with the sustain pedal applied per track."""
    return [n for track in score.tracks for n in apply_sustain_pedal(track)]


def pianocore_row_groups(root: Path):
    """(parquet file, row group) pairs of PianoCoRe (CC BY-NC-SA 4.0), for parallel reading."""
    import pyarrow.parquet as pq
    for path in sorted(root.rglob("*.parquet")):
        for group in range(pq.ParquetFile(path).num_row_groups):
            yield path, group


def pianocore_rows(path: Path, group: int, columns: list[str]):
    """Tier B rows (deduplicated, quality-filtered) of one row group."""
    import pyarrow.parquet as pq
    for row in pq.ParquetFile(path).read_row_group(group, columns=["tier_b", *columns]).to_pylist():
        if row["tier_b"]:
            yield row


def pijama_files(root: Path) -> list[Path]:
    """PiJAMA (CC BY-NC 4.0) Kong transcriptions: midi_kong/<studio|live>/<pianist>/<album>/<title>.midi"""
    return sorted(root.rglob("*.midi"))


def pijama_pianist(path: Path) -> str:
    return path.parent.parent.name


def aria_midi_files(archive: Path, genres: tuple[str, ...], min_score: float) -> list[tuple[Path, dict]]:
    """Aria-MIDI (CC BY-NC-SA 4.0) files of the given genres whose audio-quality score is >= min_score,
    extracted once from the deduped tarball next to it. Returns (path, metadata) pairs."""
    import json
    import tarfile
    out = archive.parent / f"extracted_{'_'.join(genres)}"
    meta_path = out / "metadata.json"
    if not meta_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            wanted = None
            for member in tar:  # streaming: metadata.json comes before data/
                name = Path(member.name).name
                if name == "metadata.json":
                    meta = json.load(tar.extractfile(member))
                    wanted = {k for k, v in meta.items() if v["metadata"].get("genre") in genres
                              and np.mean(list(v.get("audio_scores", {"0": 0}).values())) >= min_score}
                elif name.endswith(".mid") and wanted is not None and str(int(name.split("_")[0])) in wanted:
                    (out / name).write_bytes(tar.extractfile(member).read())
        meta_path.write_text(json.dumps({k: meta[k] for k in wanted}))
    meta = json.loads(meta_path.read_text())
    return [(p, meta[str(int(p.name.split("_")[0]))]["metadata"]) for p in sorted(out.glob("*.mid"))]


def apply_sustain_pedal(track) -> list[Note]:
    """Hold notes released under the sustain pedal until the pedal lifts (or the key is struck again)."""
    n = track.notes.numpy()
    start, pitch, velocity = n["time"], n["pitch"], n["velocity"]
    end = start + n["duration"]
    if len(track.pedals):
        p = track.pedals.numpy()
        p_start, p_end = p["time"], p["time"] + p["duration"]
        idx = np.searchsorted(p_start, end, side="right") - 1
        held = (idx >= 0) & (end < p_end[np.maximum(idx, 0)])
        end = np.where(held, np.maximum(end, p_end[np.maximum(idx, 0)]), end)
    order = np.lexsort((start, pitch))  # by pitch, then time
    same_pitch_next = np.full(len(start), np.inf)
    same_pitch_next[order[:-1]] = np.where(pitch[order[1:]] == pitch[order[:-1]], start[order[1:]], np.inf)
    end = np.minimum(end, same_pitch_next)
    return [Note(int(pi), float(s), float(max(e - s, 0.01)), int(v))
            for s, e, pi, v in zip(start, end, pitch, velocity)]
