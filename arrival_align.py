"""Place a gpt-live run's model track on the user timeline by arrival time, as every other driver does.

live_align.py places GPT-Live on the server's session clock, which removes the network delay. The other
hosted systems can only be placed by arrival, because their APIs return no server timestamps for audio,
so a session-clock GPT-Live and arrival-placed GPT-Realtime / Gemini Live are measured on two clocks.
This module rebuilds GPT-Live's arrival placement from what the relay recorded during the run:

  clock.jsonl            one row per output audio delta: model samples received and user samples sent,
                         both counted from the first delta
  B_model_raw.wav        every output sample, concatenated in arrival order
  transcript_clock.json  the server's transcript deltas, each with the user-track position at which it
                         arrived (absolute, not relative)

Each audio delta is played the way a client plays it: at the moment it arrives, or right after the audio
already queued if that has not finished yet. The same rule places GPT-Realtime and Gemini Live, whose
drivers anchor each response at the arrival of its first chunk and play the rest contiguously.

Runs recorded before 2026-09-23 store user positions relative to the first delta only, so the first
delta's absolute position is estimated from the transcript deltas. For every model word found in both the
raw stream's ASR and the server transcript, (arrival of the transcript delta) - (relative arrival of the
audio delta holding the word) estimates that position; the run's anchor is the median plus CALIBRATED_LEAD.
Later runs record the absolute position (`user_abs_s`) and are placed exactly.

CALIBRATED_LEAD: on 2026-09-23 three turn-taking items were re-run with the absolute position recorded.
The exact anchor was 0.32, 0.36, and 0.40 s later than the transcript estimate, because the audio of a
word arrives after its transcript delta. Archived runs use the mean, 0.36 s. The three runs are kept in
tts_review/_gptlive_arrival_calibration/turn_taking/{10,11,18}/; compare exact_anchor() with anchor() there.
"""
import json
from pathlib import Path
from statistics import median, pstdev

import numpy as np
import soundfile as sf

import live_align

SR = 24000
CALIBRATED_LEAD = 0.36


def deltas(out_dir: Path):
    """[(raw_start_s, raw_end_s, user_rel_s)], one per output audio delta, in arrival order."""
    rows = [json.loads(line) for line in (out_dir / "clock.jsonl").read_text().splitlines() if line.strip()]
    raw_len = sf.info(str(out_dir / "B_model_raw.wav")).frames / SR
    first_len = raw_len - rows[-1]["model_s"]          # the first delta is already counted in row 0
    out, prev_end = [], 0.0
    for r in rows:
        end = first_len + r["model_s"]
        out.append((prev_end, end, r["user_s"]))
        prev_end = end
    return out


def _server_words_with_arrival(tr):
    out = []
    for e in tr:
        if e.get("who") != "model" or e.get("start_ms") is None or e.get("user_s") is None:
            continue
        for w in e.get("text", "").split():
            if live_align._norm(w):
                out.append((live_align._norm(w), e["start_ms"] / 1000, e["user_s"]))
    return out


def anchor(out_dir: Path, dl):
    """Absolute user-track position of the first audio delta, with the spread of the per-word estimates."""
    tr = json.loads((out_dir / "transcript_clock.json").read_text())
    server = _server_words_with_arrival(tr)
    asr = live_align._asr_words(out_dir / "B_model_raw.parakeet.json")
    ends = [d[1] for d in dl]

    def rel_arrival(t):
        i = int(np.searchsorted(ends, t, side="right"))
        return dl[min(i, len(dl) - 1)][2]

    est, j = [], 0
    for word, _, arrived in server:
        for k in range(j, min(j + live_align.MAX_MATCH_SKIP, len(asr))):
            if asr[k][0] == word:
                est.append(arrived - rel_arrival(asr[k][1]))
                j = k + 1
                break
    if not est:
        return None, 0, None
    return median(est), len(est), pstdev(est)


def exact_anchor(out_dir: Path):
    """The first delta's absolute position, when the relay recorded it (runs after 2026-09-23); else None."""
    first = (out_dir / "clock.jsonl").read_text().splitlines()[0]
    return json.loads(first).get("user_abs_s")


def place(out_dir: Path, dest: Path, lead: float = CALIBRATED_LEAD, fallback_start=None):
    """Write dest/B_model.wav, dest/A_user.wav, and dest/combined.wav, placed by arrival. Returns a summary.

    `fallback_start`: for runs in which no word can be matched (whispered replies the ASR cannot read), the
    anchor is taken as (wall time of the first delta) - fallback_start, the median of that difference over
    the runs that were placed from their words. It is about 0.24 s less precise, so only placement-
    independent scenarios use it.
    """
    dl = deltas(out_dir)
    exact = exact_anchor(out_dir)
    source = "recorded"
    if exact is not None:
        u0, n_words, sd, lead = exact, 0, 0.0, 0.0
    else:
        u0, n_words, sd = anchor(out_dir, dl)
        source = "transcript"
    if u0 is None and fallback_start is not None:
        first = json.loads((out_dir / "clock.jsonl").read_text().splitlines()[0])
        u0, n_words, sd, lead, source = first["t"] - fallback_start, 0, 0.0, 0.0, "wall clock"
    if u0 is None:
        return {"placed": False, "why": "no model word matched between the raw stream and the transcript"}
    u0 += lead
    raw, _ = sf.read(str(out_dir / "B_model_raw.wav"), dtype="int16")
    user, _ = sf.read(str(out_dir / "A_user.wav"), dtype="int16")

    spans, play_end = [], 0.0
    for r0, r1, u_rel in dl:
        start = max(u0 + u_rel, play_end, 0.0)
        spans.append((start, r0, r1))
        play_end = start + (r1 - r0)

    total = max(len(user), int(np.ceil(play_end * SR)) + 1)
    out = np.zeros(total, dtype=np.int16)
    for start, r0, r1 in spans:
        i0, i1 = int(round(r0 * SR)), min(int(round(r1 * SR)), len(raw))
        pos = int(round(start * SR))
        seg = raw[i0:i1][: total - pos]
        out[pos:pos + len(seg)] = seg
    ux = np.zeros(total, dtype=np.int16)
    ux[:len(user)] = user

    dest.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest / "B_model.wav"), out, SR, subtype="PCM_16")
    sf.write(str(dest / "A_user.wav"), ux, SR, subtype="PCM_16")
    sf.write(str(dest / "combined.wav"),
             np.clip(ux.astype(np.int32) + out.astype(np.int32), -32768, 32767).astype(np.int16),
             SR, subtype="PCM_16")
    return {"placed": True, "anchor_s": round(u0, 3), "anchor_from": source, "lead_s": lead,
            "anchor_words": n_words, "anchor_sd_s": round(sd, 3), "deltas": len(dl)}
