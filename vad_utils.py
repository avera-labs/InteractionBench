"""VAD boundary utilities — ASR handles "which word / what was said", VAD handles "when there's actually voice".

Shared by grade_live / grade_model (kept here to avoid a circular import). Follows a standard get_timing approach
(it uses Silero-VAD to get speech spans, overlap = intersection of the two tracks); no torch locally, so use webrtcvad.

Why VAD: in ASR's word-level timestamps, the **last word of an utterance** often has its t1 stretched into trailing silence (e.g. 'here.' 34.16→41.20),
and using it as the "user finished speaking" moment computes false overlap. VAD measures phonation directly, so its boundaries are accurate. src may be a wav path or wav bytes.
"""
import io
import math

VAD_MERGE_GAP = 0.6      # speech-segment merge gap
TURN_GAP = 0.8           # max pause allowed within a single turn
MAX_WORD_S = 1.2         # no-VAD fallback: cap per-word duration to block inflated last words


def vad_segments(src):
    """webrtcvad measures real phonation spans [(start,end)…] in seconds. src = wav path or wav bytes."""
    import numpy as np
    import scipy.signal as ss
    import soundfile as sf
    import webrtcvad
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    x, sr = sf.read(src)
    x = x if getattr(x, "ndim", 1) == 1 else x.mean(1)
    if sr != 16000:
        g = math.gcd(int(sr), 16000)
        x = ss.resample_poly(x, 16000 // g, int(sr) // g); sr = 16000
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    vad = webrtcvad.Vad(2); n = int(sr * 0.03) * 2          # 30ms frames
    segs, st, last = [], None, 0.0
    for i in range(0, len(pcm) - n, n):
        t = (i // 2) / sr
        sp = vad.is_speech(pcm[i:i + n], sr)
        if sp and st is None:
            st = t
        elif not sp and st is not None:
            segs.append((st, t)); st = None
        last = t + 0.03
    if st is not None:
        segs.append((st, last))
    if not segs:
        return []
    m = [list(segs[0])]                                    # merge by gap
    for s, e in segs[1:]:
        if s - m[-1][1] <= VAD_MERGE_GAP:
            m[-1][1] = e
        else:
            m.append([s, e])
    return [(s, e) for s, e in m if e - s > 0.15]


def snap_end(t0, t1, segs):
    """Snap an ASR word's end time to the end of its VAD segment — mid-utterance words unchanged, an inflated last word (t1 past segment end) is clamped back."""
    if not segs:
        return t1
    for s, e in segs:
        if s - 0.15 <= t0 <= e + 0.15:
            return min(t1, e)
    return t1                                              # fell into no segment → keep the ASR value


def turn_end_vad(segs, trig_word):
    """Starting from the VAD segment containing the trigger word (or the nearest one after it), extend across gaps ≤TURN_GAP; return this turn's real finish moment."""
    if not segs:
        return None
    tt = trig_word["t0"]
    idx = next((i for i, (s, e) in enumerate(segs) if e >= tt - 0.1), None)
    if idx is None:
        return None
    end = segs[idx][1]
    for s, e in segs[idx + 1:]:
        if s - end > TURN_GAP:
            break
        end = e
    return end


def turn_end_asr(words, trig_word):
    """No-VAD fallback: extend ASR words + cap duration to block inflated last words."""
    def wend(w):
        return min(w["t1"], w["t0"] + MAX_WORD_S)
    end = wend(trig_word)
    for w in words:
        if w["t0"] < trig_word["t0"]:
            continue
        if w["t0"] - end > TURN_GAP:
            break
        end = wend(w)
    return end
