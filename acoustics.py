"""Acoustic analyzer — shared by the volume_understanding / whisper_production tests.

Given a clip of audio, measure four paralinguistic features:
  rms            loudness (energy) — the strongest signal for loud/quiet
  voiced_ratio   voiced-frame fraction (librosa.pyin) — real whisper drops it sharply (breathy = no pitch), but TTS whisper often stays fairly voiced
  mean_f0        mean pitch — raised when loud (Lombard), lowered when quiet
  spec_centroid  spectral centroid — usually higher for breathy/whispered speech

★ Judging "quieted down / whispering" requires looking at the **change relative to the same speaker's normal baseline (delta)**, not an absolute threshold
  (models differ a lot in timbre; MiMo's whisper voiced≈0.65 is not truly voiceless). See whisper_delta().
"""
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def _load(src):
    if isinstance(src, (bytes, bytearray)):
        a, sr = sf.read(io.BytesIO(src), dtype="float32", always_2d=False)
    else:
        a, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = a[:, 0]
    return a.astype("float32"), sr


def analyze(src, words=None) -> dict:
    """src = wav path / bytes / np audio. Returns {dur, rms, voiced_ratio, mean_f0, spec_centroid}. Safe on empty/silence.

    If words is given (parakeet [{t0,t1}]) → only analyze the [first word, last word] speech span, dropping trailing-silence dilution (so RMS/voiced are accurate).
    """
    a, sr = _load(src)
    if a.size == 0:
        return dict(dur=0.0, rms=0.0, voiced_ratio=None, mean_f0=None, spec_centroid=None)
    if words:
        lo = max(0, int(words[0]["t0"] * sr))
        hi = min(len(a), int(words[-1]["t1"] * sr))
        if hi - lo > int(0.1 * sr):                         # only trim if at least 0.1s, otherwise use the whole clip
            a = a[lo:hi]
    if a.size == 0:
        return dict(dur=0.0, rms=0.0, voiced_ratio=None, mean_f0=None, spec_centroid=None)
    rms = float(np.sqrt(np.mean(a ** 2)))
    try:
        f0, vflag, _ = librosa.pyin(a, fmin=65, fmax=400, sr=sr)
        voiced_ratio = float(np.mean(vflag)) if vflag is not None and len(vflag) else None
        f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        mean_f0 = float(np.mean(f0v)) if len(f0v) else None
    except Exception:  # noqa: BLE001
        voiced_ratio = mean_f0 = None
    sc = float(np.mean(librosa.feature.spectral_centroid(y=a, sr=sr)))
    # spectral flatness: whisper = noise-excited → near 1; voiced = harmonic-rich → near 0
    flat = float(np.mean(librosa.feature.spectral_flatness(y=a)))
    # HNR harmonics-to-noise ratio (praat To Harmonicity): voiced 10-20dB, whisper near 0/negative. Take the mean of defined frames.
    try:
        import parselmouth
        snd = parselmouth.Sound(a.astype("float64"), sampling_frequency=sr)
        vals = snd.to_harmonicity_cc().values
        vals = vals[vals > -200]                        # -200 = praat's "no harmonicity" placeholder
        hnr = float(np.mean(vals)) if vals.size else None
    except Exception:  # noqa: BLE001
        hnr = None
    return dict(dur=round(len(a) / sr, 2), rms=round(rms, 4),
                voiced_ratio=round(voiced_ratio, 3) if voiced_ratio is not None else None,
                mean_f0=round(mean_f0, 1) if mean_f0 is not None else None,
                spec_centroid=round(sc), spectral_flatness=round(flat, 4),
                hnr=round(hnr, 1) if hnr is not None else None)


def whisper_delta(target: dict, baseline: dict) -> dict:
    """Decide whether target is a whisper. Returns per-feature deltas + a combined whisperness score + is_whisper.

    ★ Primary test = phonation mechanism (consistent with Eve research / acoustic physics): a whisper is breathy = vocal folds don't vibrate = aperiodic,
      so **HNR low (<6dB)** and **voiced low (<0.4)**. This is an absolute criterion, needs no baseline, and is robust to volume normalization
      — it exactly fixes the earlier trap where "RMS/Gemini perception was fooled by freezeomni's delivery" (it had HNR 13-17dB, fully harmonic = not whispering).
    The rest (rms_ratio / centroid / flatness) only serve as supporting evidence and ranking.
    """
    def g(d, k):
        return d.get(k)
    br, tr = g(baseline, "rms"), g(target, "rms")
    rms_ratio = (tr / br) if (br and tr is not None) else None
    bv, tv = g(baseline, "voiced_ratio"), g(target, "voiced_ratio")
    voiced_drop = (bv - tv) if (bv is not None and tv is not None) else None
    bh, th = g(baseline, "hnr"), g(target, "hnr")
    hnr_drop = (bh - th) if (bh is not None and th is not None) else None
    bfl, tfl = g(baseline, "spectral_flatness"), g(target, "spectral_flatness")
    flat_up = (tfl - bfl) if (bfl is not None and tfl is not None) else None
    bc, tc = g(baseline, "spec_centroid"), g(target, "spec_centroid")
    centroid_up = (tc - bc) if (bc is not None and tc is not None) else None
    # primary test: low HNR + low voiced (breathy/aperiodic, absolute threshold, physically grounded)
    is_whisper = (th is not None and th < 6.0) and (tv is not None and tv < 0.40)
    if th is None:                                       # HNR not computed → fallback: very low voiced + big HNR drop
        is_whisper = (tv is not None and tv < 0.30) and (hnr_drop is not None and hnr_drop >= 8)
    # combined score (report/ranking): low HNR & low voiced dominate, flatness/rms assist
    score = 0.0
    if th is not None:
        score += max(0.0, min(1.0, (12.0 - th) / 10.0))               # HNR 12→0, 2→1
    if tv is not None:
        score += max(0.0, min(1.0, (0.45 - tv) / 0.35))               # voiced 0.45→0, 0.1→1
    if flat_up is not None:
        score += 0.2 * max(0.0, min(1.0, flat_up / 0.15))
    if rms_ratio is not None:
        score += 0.2 * max(0.0, min(1.0, (0.85 - rms_ratio) / 0.6))
    return dict(hnr=th, hnr_drop=round(hnr_drop, 1) if hnr_drop is not None else None,
                voiced_ratio=tv, voiced_drop=round(voiced_drop, 2) if voiced_drop is not None else None,
                rms_ratio=round(rms_ratio, 2) if rms_ratio is not None else None,
                flatness_up=round(flat_up, 3) if flat_up is not None else None,
                centroid_up=round(centroid_up) if centroid_up is not None else None,
                whisperness=round(score, 2), is_whisper=is_whisper)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(f"{Path(p).name}: {analyze(p)}")
