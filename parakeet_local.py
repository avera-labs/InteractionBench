"""Local parakeet ASR (Apple Silicon / MLX) — no NeMo/CUDA/torch.

Runs parakeet-tdt-0.6b-v3 on an M-series GPU, producing **word-level absolute timestamps** for per-track wavs,
with JSON matching the audio_*.parakeet.json the server's asr_tracks.py writes (word/t0/t1/confidence).
parakeet-mlx loads + resamples audio internally, so just feed it 24k wav.

Install: uv pip install --python .venv/bin/python parakeet-mlx

Usage:
  # a dashboard run directory (containing A_user.wav + B_model.wav)
  .venv/bin/python parakeet_local.py dashboard/runs/<id>/<model>/
  # or pass some wavs directly
  .venv/bin/python parakeet_local.py a.wav b.wav
Writes <name>.parakeet.json next to each wav; can also import transcribe_wav / transcribe_files to reuse.
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"     # same weights as the server's nvidia/parakeet-tdt-0.6b-v3
ASR_SR = 16000
CHUNK_S = 60.0                                        # long audio must be chunked, otherwise >~45s gets truncated
OVERLAP_S = 15.0


@lru_cache(maxsize=1)
def _model():
    import parakeet_mlx
    return parakeet_mlx.from_pretrained(MODEL_NAME)


def _tokens_to_words(tokens):
    """parakeet-mlx tokens are subwords (SentencePiece, '▁' marks a word start) → merge into words + word-level timing/confidence."""
    words = []
    for tk in tokens:
        raw = tk.text or ""
        boundary = raw.startswith("▁") or raw.startswith(" ") or not words
        piece = raw.replace("▁", " ")
        conf = getattr(tk, "confidence", 1.0)
        if boundary:
            words.append({"w": piece.strip(), "t0": tk.start, "t1": tk.end, "c": [conf]})
        else:
            words[-1]["w"] += piece
            words[-1]["t1"] = tk.end
            words[-1]["c"].append(conf)
    out = []
    for w in words:
        wt = w["w"].strip()
        if not wt:
            continue
        c = sum(w["c"]) / len(w["c"]) if w["c"] else 1.0
        out.append({"word": wt, "t0": round(w["t0"], 3), "t1": round(w["t1"], 3),
                    "confidence": round(float(c), 3)})
    return out


def transcribe_wav(path, chunk_duration=CHUNK_S):
    """Single wav → ([{word,t0,t1,confidence}...], full_text). Times are absolute seconds.
    chunk_duration controls chunking (default 60s, overlap 15s) — long audio without chunking truncates at ~45s."""
    r = _model().transcribe(str(path), chunk_duration=chunk_duration, overlap_duration=OVERLAP_S)
    words = []
    for s in r.sentences:
        words.extend(_tokens_to_words(s.tokens))
    return words, (r.text or "").strip()


def transcribe_wav_segmented(path, pad=0.15):
    """First VAD-split into speech segments, transcribe each, then reassemble by offsetting each segment's start (absolute timestamps).
    Fixes parakeet dropping utterances after long silence in a **whole track with long silence** (e.g. the whole track yields only the first half,
    and the "what time is it" after 5s of silence is swallowed) — feeding each segment separately transcribes it. Falls back to the whole track if there are no VAD segments."""
    import os
    import tempfile

    import soundfile as sf
    try:
        import vad_utils
    except Exception:  # noqa: BLE001
        return transcribe_wav(path)
    d, sr = sf.read(str(path))
    if getattr(d, "ndim", 1) > 1:
        d = d.mean(axis=1)
    segs = vad_utils.vad_segments(str(path))
    if not segs:
        return transcribe_wav(path)
    all_words = []
    for s, e in segs:
        i0 = max(0, int((s - pad) * sr)); i1 = min(len(d), int((e + pad) * sr))
        if i1 <= i0:
            continue
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); f.close()
        try:
            sf.write(f.name, d[i0:i1], sr, subtype="PCM_16")
            words, _t = transcribe_wav(f.name)
        finally:
            try:
                os.unlink(f.name)
            except Exception:  # noqa: BLE001
                pass
        off = i0 / sr                                    # segment-relative time → absolute conversation time
        for w in words:
            w["t0"] = round(w["t0"] + off, 3); w["t1"] = round(w["t1"] + off, 3)
            all_words.append(w)
    return all_words, " ".join(w["word"] for w in all_words)


def transcribe_files(files: dict) -> dict:
    """{key: wav_path} → {key: [word...]}, matching the second return value of the server's transcribe_speakers."""
    return {k: transcribe_wav(p)[0] for k, p in files.items()}


def _speaker_of(name: str) -> str:
    n = name.lower()
    if "user" in n or n.endswith("0") or n.endswith("_a"):
        return "A"
    if "model" in n or n.endswith("1") or n.endswith("_b"):
        return "B"
    return "?"


def _doc_for(wav: Path, words, text):
    return {
        "track": wav.stem,
        "speaker": _speaker_of(wav.stem),
        "sr": ASR_SR,
        "text": text or " ".join(w["word"] for w in words),
        "words": words,
        "note": "local MLX parakeet-tdt-0.6b-v3; t0/t1 are ABSOLUTE (conversation clock, seconds).",
    }


def _expand(paths):
    """Directory → the A_user.wav/B_model.wav inside it; .wav → itself. combined.wav is skipped (it's a mixdown)."""
    wavs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # combined.wav is a mixdown; model_raw.wav is the pre-assembly raw track (same content as B_model.wav) → skip both
            skip = {"combined.wav", "model_raw.wav"}
            for w in sorted(p.glob("*.wav")):
                if w.name not in skip:
                    wavs.append(w)
        elif p.suffix.lower() == ".wav":
            wavs.append(p)
        else:
            print(f"  skip (not wav / not a directory): {p}")
    return wavs


def main():
    ap = argparse.ArgumentParser(description="Local MLX parakeet word-level ASR over per-track wavs")
    ap.add_argument("paths", nargs="+", help="wav files, or a directory containing per-track wavs")
    ap.add_argument("--overwrite", action="store_true", help="rerun even if .parakeet.json already exists")
    args = ap.parse_args()

    wavs = _expand(args.paths)
    if not wavs:
        print("no wav found")
        return
    print(f"Local parakeet (MLX, {MODEL_NAME}): {len(wavs)} tracks")
    for wav in wavs:
        out = wav.parent / (wav.stem + ".parakeet.json")
        if out.exists() and not args.overwrite:
            print(f"  skip (already exists): {out.name} (--overwrite to rerun)")
            continue
        t = time.perf_counter()
        words, text = transcribe_wav(wav)
        doc = _doc_for(wav, words, text)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✓ {wav.name} → {out.name}  {len(words)} words  {time.perf_counter()-t:.1f}s")
        print(f"      \"{doc['text'][:90]}\"")


if __name__ == "__main__":
    main()
