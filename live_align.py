"""Place a gpt-live run's model track on the user timeline, using the server's own session clock.

Why this exists
---------------
On the Live API's primary WebSocket, `session.output_audio.delta` carries no timestamps and the server
drops silent output frames, so the delta stream is not a timeline. Two obvious placements both fail:

  · concatenate from one anchor  → every dropped frame pulls the rest earlier (-1.6s over a 43s item)
  · anchor each chunk on arrival → the error becomes the network latency (+1.2s through a proxy),
                                   so the same item scores differently depending on the connection

The transcript deltas, though, are stamped on the server's session clock (`start_ms`), which is locked
to the user audio stream -- measured constant to within 80ms across a whole item. That clock is
latency-independent, and it is what we place against:

    target  = model word's start_ms                      (where that word belongs on the user timeline)

Only the MODEL's words are used. The input transcripts look like they would pin the session clock to
the user track, but they are stamped by a streaming recogniser and run ~0.6s late (measured: a user
word at true position 0.00 is reported at 600ms), so folding them in subtracts that recogniser latency
from the model track a second time -- which placed one item's answer before the question that prompted
it. The output transcripts carry no such lag: they are the model's own alignment for audio it is about
to emit, and on a short item they match the raw stream to within 20ms.

The one assumption left is that the session clock and the user track share an origin, i.e. session
time S is user-track position S. The user-side stamps bound any residual offset to at most ~0.4s, and
it applies equally to every item, so cross-item comparisons hold either way.

Each speech burst is shifted by the median offset of the words inside it, so a dropped frame can never
propagate past the burst it happened in. Bursts the ASR and the server disagree about inherit the
nearest resolved burst's shift.
"""
import json
from pathlib import Path
from statistics import median

import numpy as np
import soundfile as sf

SR = 24000
MAX_MATCH_SKIP = 6        # ASR and the server rarely disagree by more than this many words in a row


def _norm(w):
    return w.strip(" .,?!'\"“”‘’").lower()


def _server_words(tr, who):
    out = []
    for e in tr:
        if e.get("who") != who or e.get("start_ms") is None:
            continue
        for w in e.get("text", "").split():
            if _norm(w):
                out.append((_norm(w), e["start_ms"] / 1000))
    return out


def _asr_words(path: Path):
    d = json.loads(path.read_text())
    return [(_norm(w["word"]), w["t0"]) for w in d["words"] if _norm(w["word"])]


def _pairs(server, asr):
    """Walk both word streams in order, matching on the word itself → [(asr_t, server_t)]."""
    out, j = [], 0
    for sw, st in server:
        for k in range(j, min(j + MAX_MATCH_SKIP, len(asr))):
            if asr[k][0] == sw:
                out.append((asr[k][1], st))
                j = k + 1
                break
    return out


def place(out_dir: Path, transcribe, vad_segments, verbose=True):
    """Rewrite B_model.wav (and combined.wav) from B_model_raw.wav, placed on the session clock.

    `transcribe(path) -> (words, text)` and `vad_segments(path) -> [(start, end)]` are injected so this
    module does not pull in the ASR stack on import. Returns a dict describing what it did, or None
    when the run lacks what it needs (then the relay's own fallback placement is left alone).
    """
    raw_p, tr_p = out_dir / "B_model_raw.wav", out_dir / "transcript_clock.json"
    if not raw_p.exists() or not tr_p.exists():
        return None
    tr = json.loads(tr_p.read_text())

    # ---- the model's words, in the raw stream's own coordinates ----
    mw_p = out_dir / "B_model_raw.parakeet.json"
    words, text = transcribe(str(raw_p))
    mw_p.write_text(json.dumps({"words": words, "text": text}, ensure_ascii=False))
    m_pairs = _pairs(_server_words(tr, "model"), _asr_words(mw_p))

    raw, _ = sf.read(str(raw_p), dtype="int16")
    bursts = vad_segments(str(raw_p))
    if not bursts:
        return {"placed": False, "why": "no speech in the model stream"}

    # ---- per-burst shift: median over the words that fall inside it ----
    shifts = []
    for b0, b1 in bursts:
        inside = [st - at for at, st in m_pairs if b0 - 0.2 <= at <= b1 + 0.2]
        shifts.append(median(inside) if inside else None)
    known = [i for i, s in enumerate(shifts) if s is not None]
    if not known:
        return {"placed": False, "why": f"no model word matched ({len(m_pairs)} pairs)"}
    for i, s in enumerate(shifts):                       # bursts with no matched word follow the nearest one
        if s is None:
            shifts[i] = shifts[min(known, key=lambda k: abs(k - i))]

    # ---- render ----
    user, _ = sf.read(str(out_dir / "A_user.wav"), dtype="int16")
    total = len(user)
    for (b0, b1), sh in zip(bursts, shifts):
        total = max(total, int(round((b1 + sh) * SR)) + 1)
    out = np.zeros(total, dtype=np.int16)
    for (b0, b1), sh in zip(bursts, shifts):
        i0, i1 = int(round(b0 * SR)), min(int(round(b1 * SR)), len(raw))
        pos = int(round((b0 + sh) * SR))
        if i1 <= i0:
            continue
        seg = raw[i0:i1]
        pos = max(pos, 0)
        end = min(pos + len(seg), total)
        if end > pos:
            out[pos:end] = seg[:end - pos]
    sf.write(str(out_dir / "B_model.wav"), out, SR, subtype="PCM_16")
    ux = np.zeros(total, dtype=np.int16); ux[:len(user)] = user
    sf.write(str(out_dir / "A_user.wav"), ux, SR, subtype="PCM_16")
    sf.write(str(out_dir / "combined.wav"),
             np.clip(ux.astype(np.int32) + out.astype(np.int32), -32768, 32767).astype(np.int16),
             SR, subtype="PCM_16")

    res = {"placed": True, "bursts": len(bursts), "model_pairs": len(m_pairs),
           "shift_ms": [round(s * 1000) for s in shifts]}
    if verbose:
        # the first burst should need almost no shift -- dropped frames have not piled up yet.
        # A large one there means the stream and the session clock disagree from the start.
        print(f"  placed on the session clock: {res['bursts']} bursts, "
              f"shifts {res['shift_ms']} ms ({res['model_pairs']} words matched)", flush=True)
    return res
