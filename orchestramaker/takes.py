"""A "take" = one or more rendered versions of a performance, saved as WAV + JSON for the web UI."""

import json
from pathlib import Path

from .render import Track, render

Version = tuple[str, list[Track]]  # (label, tracks), e.g. ("as generated", [(ALTO_SAX, notes)])


def save_take(out_dir: str | Path, name: str, versions: list[Version], meta: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    take = {"name": name, "meta": meta or {}, "versions": []}
    for i, (label, tracks) in enumerate(versions):
        wav = render(tracks, out_dir / f"{name}{'' if i == 0 else f'_{i}'}.wav")
        take["versions"].append({
            "label": label,
            "audio": wav.name,
            "tracks": [{"instrument": inst.name, "low": inst.low, "high": inst.high,
                        "max_polyphony": inst.max_polyphony, "problems": inst.check(notes),
                        "notes": [[n.pitch, round(n.start, 4), round(n.duration, 4), n.velocity] for n in notes]}
                       for inst, notes in tracks],
        })
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(take))
    return path
