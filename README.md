# OrchestraMaker

**An experiment: instead of generating raw audio, teach a small model to *play instruments*.**

Most music-generating AI models (MusicGen, Stable Audio, Suno-like systems) produce the
audio signal itself — they predict compressed waveform/spectrogram tokens directly.
OrchestraMaker takes the other road, the way a human musician does it:

1. **Know the instruments** — every instrument has a range, a polyphony limit and a set of
   techniques (a sax plays one note at a time and has to breathe; a guitar has 6 strings and
   a limited hand stretch; a bass lives in the low register).
2. **Learn the music** — a small transformer studies scores/performances of jazz and classical
   music (Mozart, Bach, jazz solos…) as *note events*, not as sound.
3. **Play** — the model writes a performance (notes, timing, dynamics, articulation) for a
   chosen instrument and style, and an instrument engine turns it into sound.

The question we want to answer: *can a model that fits on a laptop GPU learn to play
convincingly this way?*

## Why this exists

This started as a question I was curious about: music AIs mostly synthesize sound directly —
what happens if a model learns to play instruments instead? OrchestraMaker is my attempt to
find out without it becoming a big time investment.

It is built with **heavy AI-assisted coding**: most of the code is written together with an
AI coding assistant, while I steer the ideas, experiments and decisions. Expect a curiosity
project rather than a polished library.

## Architecture

```
 style + instrument ──► ┌──────────────────────┐   note events    ┌──────────────────┐
     (e.g. "jazz",      │  Performer model     │ ───────────────► │ Instrument engine │ ──► audio
      "alto sax")       │  (small transformer) │  pitch, onset,   │  (SoundFont now,  │
                        └──────────────────────┘  duration, vel., │   DDSP later)     │
                                  ▲               articulation    └──────────────────┘
                                  │                        ▲
                        instrument rules: range, polyphony, playability
```

- **Performer model** — GPT-style decoder over note tokens, conditioned on style and
  instrument tokens. Target size: ~10–50M parameters.
- **Tokens** — `BOS STYLE_mozart INST_piano`, then per note `[TIME_k] PITCH_p VEL_v DUR_d`.
  Time is *performance* time in 10 ms steps rather than a score grid, so swing feel and
  rubato are kept. 345 tokens in total.
- **Instrument definitions** — range, max polyphony, breathing/phrase limits, idiomatic
  techniques. Used both as training-time conditioning and as generation-time constraints.
- **Instrument engine** — starts with [FluidSynth](https://www.fluidsynth.org/) + a free
  General MIDI SoundFont. Later stage: [DDSP](https://github.com/magenta/ddsp)-style
  synthesis, where a network learns to *control* an instrument's pitch and loudness curves
  from real recordings — the closest thing to actually learning to play it.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA build
.venv/bin/pip install -r requirements.txt
scripts/get_soundfont.sh                        # ~215 MB, MuseScore General (MIT)
.venv/bin/python scripts/demo_instruments.py    # writes WAVs to outputs/stage1/
.venv/bin/python scripts/prepare_data.py        # downloads datasets (~100 MB), tokenizes into data/tokens/
.venv/bin/python scripts/train.py configs/base.json   # ~35 min on an RTX 5060 Laptop GPU
.venv/bin/python scripts/generate.py --all            # takes into outputs/stage4/
```

`pyfluidsynth` needs the system FluidSynth library (`sudo apt install libfluidsynth3` on Ubuntu).

## Listening UI

```bash
.venv/bin/python scripts/serve.py                                  # http://127.0.0.1:8000
.venv/bin/python scripts/watch_progress.py --run checkpoints/base  # optional: takes from each new checkpoint
```

Training on a rented GPU? Mirror it to the same UI (loss curve, speed, time left, and takes from
checkpoints as they arrive):

```bash
.venv/bin/python scripts/sync_pod.py --host root@<pod-ip> --port <ssh-port> --run v1_piano
.venv/bin/python scripts/watch_progress.py --run checkpoints/v1_piano --every 1
```

Every take (hand-written demos, dataset round trips, model generations) shows up with its audio and a
piano roll: the keys light up as notes sound, notes outside the instrument's range are red, and model
takes can be switched between *as generated* and *made playable*.

![Web UI: piano roll of the hand-written ii-V-I trio](docs/webui.png)

## Roadmap

The model grows in three versions: first one instrument at a time, then one instrument well,
then instruments together.

**v0 — single instrument, baseline** (done)
- [x] Setup: venv, PyTorch with CUDA on an RTX 5060 (Blackwell needs CUDA 12.8+ builds), FluidSynth.
- [x] Instruments: ranges, polyphony, breathing, guitar/bass fingering; scales and a hand-written trio.
- [x] Data: Weimar Jazz Database + MAESTRO, performance-time tokens with style/instrument conditioning.
- [x] Train: 25M-parameter GPT, ~35 min on the laptop GPU.
- [x] Play: generation, playability report, web UI with a synced piano roll.

The pipeline works end to end, but the music is not good yet: phrases wander and there is little
long-range structure.

**v1 — one instrument, played well (piano first)**
- [x] Data: [PianoCoRe](https://huggingface.co/datasets/SyMuPe/PianoCoRe) tier B (18,758 h of
      classical piano, 478 composers, deduplicated and quality-filtered) +
      [PiJAMA](https://almostimplemented.github.io/PiJAMA/) (221 h of jazz piano, 120 pianists);
      60 style tokens (composers, jazz pianists), 1.97B training tokens.
- [x] Model: 85M parameters (12 × 768), 2048-token context, KV-cache generation.
- [x] Trained on a rented A100 (RunPod): 26k steps, 1.7B tokens, 123 min, ~4 USD; followed live
      from the local web UI. Validation loss 1.56 (classical), 2.10 (jazz).
- [x] Measured takes against real music (`scripts/evaluate_samples.py`) and fixed sampling.
- [ ] Long context: `configs/v1_long.json` fine-tunes v1 at 8192 tokens (a whole 4-minute piece).
      Ready to run: ~40 min on an A100, ~4 h on the laptop.

What we learned from v1:
- **Sampling mattered as much as training.** With temperature 1.0 / top-p 0.95, Chopin takes had
  half the note density of real Chopin and 6 of 16 looped; 0 of 16 looked typical. Temperature 1.1
  and no top-p: 9 of 16 (Mozart: 7 → 15 of 16). min-p made takes go silent. Picking the most
  coherent of 8 typical candidates (model log-probability) helps further.
- **Mozart comes out best, jazz worst.** Per-style validation loss: Grieg 1.36, Debussy 1.44,
  Chopin 1.51, Mozart 1.61, jazz piano 2.03, Art Tatum 2.23. Chopin is predicted well but
  generated worse than Mozart: one wrong note derails chromatic harmony more easily.
- **Takes are original.** A Chopin take shares 5.5% of its 8-note sequences with 44k Chopin
  training performances (max 2.1% with any single piece) — `scripts/check_memorization.py`.
- **No long-form structure yet.** Four-minute takes never bring the opening theme back (real
  Mozart: 25% of opening shapes return in the last third), even with the opening kept in context;
  without it the texture drifts (20 notes/s mid-piece vs ~8 in real Mozart).

**v1.5 — our own model, aiming at Aria's quality** (done)

[EleutherAI's Aria](https://github.com/EleutherAI/aria) (0.7B parameters, 60k hours) continued our prompts
convincingly (`scripts/compare_aria.py`), so it became the quality bar for a model of our own.

- [x] Laptop ablations first (`configs/ablation.json`, 19M params, equal tokens), in bits per note:
      GPT-2 13.57 → LLaMA-style 11.60 → LLaMA + compact tokens 11.78 (35% more music per context).
- [x] Data: PianoCoRe + PiJAMA + 18,377 jazz/ragtime/blues files from Aria-MIDI (~1,060 h, audio score ≥ 0.95).
- [x] 153M params, LLaMA-style (RoPE, RMSNorm, SwiGLU), compact tokens (NOTE_p_v), 4096 context;
      25k steps, 3.3B tokens, 7.9 h on an A100 (~13.5 USD). The pod removed itself after the laptop had
      the final checkpoint (`scripts/runpod_finish.sh`).

Results:

| | v1 | v1.5 |
|---|---|---|
| bits/note on the same held-out notes, classical | 8.60 | 8.56 |
| bits/note on the same held-out notes, jazz | 11.02 | **10.06** |
| typical takes of 16 (Mozart / Chopin / jazz piano / Art Tatum) | 15 / 9 / 13 / 7 | 14 / **11** / **16** / **10** |

- Classical prediction barely improved; jazz clearly did. Generated takes look more like real music,
  especially jazz. Best temperature is style-dependent: Mozart 1.0, Chopin and jazz 1.1.
- **Coherence-based picking can select memorized passages**: the most coherent of 8 Mozart candidates was
  26% Sonata K.545. `scripts/build_fingerprint.py` + `pick_take` now skip takes that quote popular repertoire.
- Continuing real prompts, v1.5 copies nothing (Aria: 5–15% of the original) but runs denser than the
  real music, and truncated sampling (min-p, as Aria uses) makes it collapse into loops. Aria's
  distribution is still much cleaner — the gap is mostly scale.

**v1.6 — an ear and a critic** (in progress)

The listener rated ~160 takes blind (1-5) in the web UI (`serve.py --pool`, keys 1-5), without knowing
model, settings or whether a take was a real performance (real: 3.72 on average, v1.5: 2.52).
- `ear.py`: MERT-v1-95M listens to each rendered take. `train_critic.py` learns the listener's taste:
  Spearman 0.51 with their ratings under nested cross-validation (note statistics alone: 0.45).
  `generate.py --critic` keeps the candidate the critic predicts they'd like most; in a blind A/B test
  its picks won 6 of 9 pairs (2.22 vs 1.56 stars; not yet significant).
- What the ratings say: the worst stretch decides (worst-window dissonance: Spearman -0.43, peak density
  -0.17). Takes often start well and fall apart - v1.5's Liszt was twice as dissonant as real Liszt.
  Adaptive brakes on runaway density and dissonance bring it back to real levels (on by default), but
  ratings of 40-second braked takes (1.88) show the real problem: **quality decays with length**.
- A popular-repertoire fingerprint over all 195k training performances catches quotes (the take heard
  as a meme song was Chopin's Nocturne Op.9 No.2, 81% of it in the training data).
- For comparison, Claude wrote two pieces note by note from classical-style rules (`claude_allegretto.py`,
  `claude_sonata.py`: a sonata movement from D minor to D major). They keep form over one and two minutes -
  what the models can't yet. Next idea: let a plan (form, harmony, cadences) give the structure and the
  model fill it in its learned style.

**v2 — instruments together**
- [ ] Multi-track tokens: every note carries its instrument; the header lists the ensemble.
- [ ] Jazz combo from the Weimar Jazz Database: solo + walking bass + chord-based comping, time-aligned.
- [ ] Real arrangements from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (strings, winds, guitar, drums).

**Later**
- Expressive performance (vibrato, bends, ghost notes), DDSP timbre, audio → notes transcription
  so the model can "listen" to recordings.

## Data (openly licensed only)

| Source | Used for | Size | License |
|---|---|---|---|
| [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html) | 456 transcribed jazz solos (sax, trumpet, trombone, …) and the walking bass line under them | 200k solo notes, 1.3M tokens | ODbL 1.0 |
| [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) | 1,276 classical piano performances, labelled by composer (Mozart: 38 pieces, 5.6 h) | 199 h, 27M tokens | CC BY-NC-SA 4.0 (non-commercial) |
| [PianoCoRe](https://huggingface.co/datasets/SyMuPe/PianoCoRe) | *v1*: classical piano performances with composer labels | ~18,700 h (tier B) | CC BY-NC-SA 4.0 (non-commercial) |
| [PiJAMA](https://almostimplemented.github.io/PiJAMA/) | *v1*: solo jazz piano | 200+ h | CC BY-NC 4.0 (non-commercial) |
| [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) | *v2*: multi-instrument arrangements | ~170k files | CC BY 4.0 |

Jazz is ~5% of the tokens, so training will sample by source rather than by token count.

Datasets are **not** committed to this repository; scripts will download them into `data/`.
Each dataset keeps its own license — check it before reuse.

## Hardware target

- **Development and small runs:** a laptop with an RTX 5060 Laptop GPU (8 GB VRAM) and 32 GB RAM.
- **Larger training runs:** a rented cloud GPU (e.g. on [RunPod](https://www.runpod.io/)).

Training is config-driven and checkpointed, so the same run can move between the laptop and a
cloud machine. The goal is training in hours, not weeks.

## License

Code: [MIT](LICENSE). Datasets and SoundFonts: their respective licenses.
