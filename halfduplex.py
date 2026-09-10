"""Half-duplex agent baseline (cascaded pipeline, strict turn-taking) — for comparing against full-duplex models on the same benchmark.

Architecture = the equivalent of LiveKit+Silero-style half-duplex, except it doesn't start WebRTC; it simulates the timing offline right inside the harness:
  user audio →[VAD decides 'done speaking' (endpoint silence)]→ parakeet STT → MiMo LLM → MiMo TTS → placed on the B_model track.
Deliberately turn-based: doesn't listen while speaking, relies on the VAD endpoint to detect turn-end → a mid-sentence pause is misread as "done speaking" and it barges in,
never backchannels while listening, won't yield when interrupted. Produces A_user/B_model dual tracks for the existing grader.
"""
import io
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _retry(fn, tries=4):
    """MiMo gateway calls flake through the congested proxy → back off and retry."""
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.2 * (k + 1))
    raise last

SR = 24000
DEFAULT_SYS = ("You are a friendly voice assistant in a spoken conversation. "
               "Answer directly and briefly, in one or two short spoken sentences. Always reply in English.")


def _turns_from_vad(user_f32, endpoint_s):
    """VAD slices the user audio into "turns": gap between speech segments < endpoint_s → no endpoint yet, merge into one turn;
    gap >= endpoint_s → the agent decides "the user is done", ending the previous turn. Returns [(start, end)] in seconds."""
    import vad_utils
    buf = io.BytesIO()
    sf.write(buf, np.clip(user_f32, -1, 1), SR, format="wav", subtype="PCM_16")
    segs = vad_utils.vad_segments(buf.getvalue())
    if not segs:
        return []
    turns, (cs, ce) = [], segs[0]
    for s, e in segs[1:]:
        if s - ce < endpoint_s:
            ce = e
        else:
            turns.append((cs, ce))
            cs, ce = s, e
    turns.append((cs, ce))
    return turns


def _stt_full(user_f32):
    """Run parakeet once over the whole track to get word-level timestamps for per-turn slicing (avoids transcribing each turn separately)."""
    import parakeet_local
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, np.clip(user_f32, -1, 1), SR, subtype="PCM_16")
        p = tf.name
    try:
        words, _ = parakeet_local.transcribe_wav(p)
    finally:
        Path(p).unlink(missing_ok=True)
    return words


def _stt_text(clip):
    return " ".join(w["word"] for w in _stt_full(clip)).strip()


def _gpt_chat(history, model="gpt-4o-mini"):
    """The halfduplex brain: switched to OpenAI (the MiMo gateway is unstable). Goes through the HTTPS_PROXY env proxy. Returns (text, None)."""
    import os
    import requests
    key = os.environ["OPENAI_API_KEY"]
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": history, "max_tokens": 200}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], None


_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
              "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}


def _norm_toks(s):
    """Lowercase, strip punctuation, number-words→digits, return a token list (digits kept whole)."""
    import re
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return [str(_NUM_WORDS[t]) if t in _NUM_WORDS else t for t in s.split()]


def _tts_ok(intended, heard):
    """Post-synthesis self-check: did ASR read it back faithfully? Focus on number words (isolated monosyllabic digits are the most likely to garble in TTS→ASR: two→Sue, eight→Aked, ten→Tuke).
    If there are digit targets → every digit must be heard; otherwise require >=60% overlap of keywords (digits/long words)."""
    it = _norm_toks(intended)
    if not it:
        return True
    hd = set(_norm_toks(heard))
    nums = [t for t in it if t.isdigit()]
    if nums and not all(n in hd for n in nums):        # any target digit not read back → count as garbled, re-synthesize
        return False
    keys = [t for t in it if t.isdigit() or len(t) >= 3]
    if not keys:
        return True
    hit = sum(1 for k in keys if k in hd)
    return hit / len(keys) >= 0.6


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n):
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _int_to_words(n % 100))
    if n < 1_000_000:
        return _int_to_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _int_to_words(n % 1000))
    return str(n)


def _speakable(text):
    """Convert bare Arabic numerals into English words before feeding TTS — MiMo is very unstable reading isolated digits ('2.') and much steadier reading words ('two')."""
    import re
    return re.sub(r"\d+", lambda m: _int_to_words(int(m.group())), text or "")


def _decode_f32(wb):
    a, srr = sf.read(io.BytesIO(wb), dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = a[:, 0]
    if srr != SR:
        import soxr
        a = soxr.resample(a, srr, SR).astype("float32")
    return a


def _tts_wav(text, tts_voice, tts_model, *, verify=True, tries=4):
    """TTS goes direct to xiaomimimo (voice is still MiMo Milo). After synthesis, use parakeet to self-check whether it was read back faithfully; re-synthesize if garbled
    — MiMo TTS is stochastic, isolated number words occasionally get garbled/misheard by ASR, and another take usually reads correctly. Returns a correct take, falling back to the last take if all are garbled."""
    import os
    import generate_tts_mimo as gt
    tok = os.environ["MIMO_TTS_TOKEN"].strip()
    say = _speakable(text)                                 # digits→words, MiMo reads it steadily
    best = None
    for i in range(max(1, tries)):
        wb, _ = _retry(lambda: gt.synthesize(api_key=tok, ref_data_uri=tts_voice, text=say, model=tts_model, proxy=None))
        best = _decode_f32(wb)
        if not verify:
            return best
        heard = _stt_text(best)
        if _tts_ok(text, heard):
            return best
        print(f"    ↻ TTS self-check failed (meant {text!r} → heard {heard!r}), re-synthesizing {i + 1}/{tries}", flush=True)
    return best


def simulate_turns(clips, out_dir, *, endpoint_s=0.5, latency_s=0.7, inter_turn_pause=0.8,
                   sys_prompt=DEFAULT_SYS, llm_model="gpt-4o-mini",
                   tts_voice="Milo", tts_model="mimo-v2.5-tts"):
    """Multi-turn benchmark (A1→reply→A2→reply): each user utterance is placed **right after the model finishes its previous reply + a natural pause**,
    rather than at a fixed long gap (otherwise a fast-answering model leaves a big dead stretch of idle waiting). Produces A_user/B_model/combined.wav."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = [{"role": "system", "content": sys_prompt}]
    user_segs, responses, log, n_user_turns = [], [], [], 0
    cursor = 0.0
    for clip in clips:
        ts = cursor
        te = ts + len(clip) / SR
        user_segs.append((ts, clip))
        txt = _stt_text(clip)
        rend = te
        if txt:
            n_user_turns += 1
            history.append({"role": "user", "content": txt})
            resp, _ = _retry(lambda: _gpt_chat(history, model=llm_model), tries=7)
            resp = (resp or "").strip()
            history.append({"role": "assistant", "content": resp})
            commit = te + endpoint_s + latency_s              # speak up after endpoint silence + STT/LLM/TTS processing latency
            if resp:
                wav = _tts_wav(resp, tts_voice, tts_model)
                responses.append((commit, wav))
                rend = commit + len(wav) / SR
            log.append({"turn": [round(ts, 2), round(te, 2)], "heard": txt[:70], "said": resp[:70], "at": round(commit, 2)})
        cursor = max(te, rend) + inter_turn_pause             # next user utterance right after model finishes + natural pause (removes dead air)
    if n_user_turns and not responses:
        raise RuntimeError("halfduplex: user spoke but LLM/TTS all failed (0 responses), counted as failure (network)")
    total = cursor
    for st, wav in responses:
        total = max(total, st + len(wav) / SR)
    n = int(np.ceil(max(total, 0.1) * SR)) + 1
    A = np.zeros(n, dtype=np.float32)
    for st, clip in user_segs:
        i = int(st * SR); j = min(n, i + len(clip)); A[i:j] += clip[:j - i]
    B = np.zeros(n, dtype=np.float32)
    for st, wav in responses:
        i = int(st * SR); j = min(n, i + len(wav)); B[i:j] += wav[:j - i]
    sf.write(str(out_dir / "A_user.wav"), np.clip(A, -1, 1), SR, subtype="PCM_16")
    sf.write(str(out_dir / "B_model.wav"), np.clip(B, -1, 1), SR, subtype="PCM_16")
    sf.write(str(out_dir / "combined.wav"), np.clip(A + B, -1, 1), SR, subtype="PCM_16")
    return {"secs": round(total, 1), "turns": log}


def simulate(user_f32, out_dir, *, endpoint_s=0.5, latency_s=0.7, sys_prompt=DEFAULT_SYS,
             llm_model="gpt-4o-mini", tts_voice="Milo", tts_model="mimo-v2.5-tts"):
    """Offline simulation of a half-duplex agent (single-track input, for behavior tasks), produces A_user/B_model/combined.wav. Returns {secs, turns:[...]}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    turns = _turns_from_vad(user_f32, endpoint_s)
    words = _stt_full(user_f32) if turns else []

    def turn_text(ts, te):
        return " ".join(w["word"] for w in words if ts - 0.25 <= w["t0"] <= te + 0.35).strip()

    history = [{"role": "system", "content": sys_prompt}]
    total = len(user_f32) / SR
    responses, log, n_user_turns = [], [], 0
    agent_free_at = 0.0                                # the moment the agent finishes its current reply and starts listening again. The core half-duplex constraint of "one mouth + deaf while speaking"
    for ts, te in turns:
        if te <= agent_free_at + 1e-3:                # whole turn falls within the agent's speaking window → it was deaf and didn't hear it, so drop it (won't reply to it)
            log.append({"turn": [round(ts, 2), round(te, 2)], "skip": "deaf_while_speaking"})
            continue
        txt = turn_text(max(ts, agent_free_at), te)   # only recognize the part heard after the agent became free
        if not txt:
            continue
        n_user_turns += 1
        history.append({"role": "user", "content": txt})
        resp, _ = _retry(lambda: _gpt_chat(history, model=llm_model), tries=7)   # LLM (OpenAI; retry on failure; if still failing after retries → rerun the whole thing)
        resp = (resp or "").strip()
        history.append({"role": "assistant", "content": resp})
        commit = max(te + endpoint_s + latency_s, agent_free_at)  # speak up after endpoint silence + processing latency, never before the previous reply finished (replies are serial, don't overlap itself)
        wav = None
        if resp:
            wav = _tts_wav(resp, tts_voice, tts_model)
        log.append({"turn": [round(ts, 2), round(te, 2)], "heard": txt[:70], "said": resp[:70],
                    "at": round(commit, 2)})
        if wav is not None:
            responses.append((commit, wav))
            agent_free_at = commit + len(wav) / SR    # speaks until this moment, deaf throughout
            total = max(total, agent_free_at)
    if n_user_turns and not responses:                # user turns exist but not one response produced = all network failures → raise so the caller reruns, don't save a silent junk result
        raise RuntimeError("halfduplex: detected the user speaking but LLM/TTS all failed (0 responses), counted as failure (network)")

    n = int(np.ceil(max(total, 0.1) * SR)) + 1
    A = np.zeros(n, dtype=np.float32)
    A[:len(user_f32)] = user_f32
    B = np.zeros(n, dtype=np.float32)
    for st, wav in responses:
        i = int(st * SR)
        j = min(n, i + len(wav))
        B[i:j] += wav[:j - i]
    sf.write(str(out_dir / "A_user.wav"), np.clip(A, -1, 1), SR, subtype="PCM_16")
    sf.write(str(out_dir / "B_model.wav"), np.clip(B, -1, 1), SR, subtype="PCM_16")
    sf.write(str(out_dir / "combined.wav"), np.clip(A + B, -1, 1), SR, subtype="PCM_16")
    return {"secs": round(total, 1), "turns": log}
