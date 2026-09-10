"""Compress the big "both-tracks-silent" gaps in a dual-track recording down to a natural length (keep ~0.7s),
trimming A_user/B_model in lockstep to stay aligned, rebuild combined.wav, then re-run parakeet so the word-level
timestamps line up. For gemini-style recordings with trailing turn boundaries and long dead air between turns.

Usage: uv run python compact_gaps.py --model gemini            # process every scenario for this model
       uv run python compact_gaps.py --model gemini --items 01 05
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import parakeet_local

CAT = "test_set/interaction_groundedness"
GAP_MIN = 1.3       # only compress silence longer than this
GAP_KEEP = 0.7      # compress it down to this natural pause
HOP = 0.02          # energy frame 20ms
THR = 0.01          # silence RMS threshold (dead air only when both tracks are below it)


def _env(x, hop):
    return np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) if len(x[i:i + hop]) else 0.0
                     for i in range(0, len(x), hop)])


def compact_dir(d: Path):
    au, sr = sf.read(str(d / "A_user.wav"))
    bm, sr2 = sf.read(str(d / "B_model.wav"))
    if au.ndim > 1:
        au = au[:, 0]
    if bm.ndim > 1:
        bm = bm[:, 0]
    n = min(len(au), len(bm)); au, bm = au[:n], bm[:n]
    hop = int(HOP * sr)
    silent = (np.maximum(_env(au, hop), _env(bm, hop)) < THR)   # both tracks silent = dead air
    keep_hi = int(GAP_KEEP / HOP)
    segs, i, nh = [], 0, len(silent)                            # sample ranges to keep
    while i < nh:
        if silent[i]:
            j = i
            while j < nh and silent[j]:
                j += 1
            if (j - i) * HOP > GAP_MIN:                         # big dead air → keep only the first keep_hi frames
                segs.append((i * hop, (i + keep_hi) * hop))
            else:
                segs.append((i * hop, j * hop))
            i = j
        else:
            j = i
            while j < nh and not silent[j]:
                j += 1
            segs.append((i * hop, j * hop))                    # keep all speech segments
            i = j
    idx = np.concatenate([np.arange(a, min(b, n)) for a, b in segs]) if segs else np.arange(n)
    au2, bm2 = au[idx], bm[idx]
    sf.write(str(d / "A_user.wav"), au2, sr, subtype="PCM_16")
    sf.write(str(d / "B_model.wav"), bm2, sr, subtype="PCM_16")
    sf.write(str(d / "combined.wav"), np.clip(au2 + bm2, -1, 1), sr, subtype="PCM_16")
    for trk in ("A_user", "B_model"):                          # re-run parakeet
        w, _ = parakeet_local.transcribe_wav(str(d / f"{trk}.wav"))
        json.dump({"words": w}, open(d / f"{trk}.parakeet.json", "w"))
    return round(len(au) / sr, 1), round(len(au2) / sr, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", nargs="*", default=[f"{i:02d}" for i in range(1, 11)])
    a = ap.parse_args()
    for it in a.items:
        ds = glob.glob(f"{CAT}/{it}/live_*_{a.model}")
        if not ds:
            print(f"  {it}: no {a.model}"); continue
        before, after = compact_dir(Path(ds[0]))
        print(f"  {a.model}/{it}: {before}s → {after}s  (removed {round(before-after,1)}s dead air)")


if __name__ == "__main__":
    main()
