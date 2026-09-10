"""Grade volume_understanding / whisper_production —— acoustics (relative to the model's own normal baseline) + content.

Baseline: for each model, take the median speaking-segment acoustics from its own normal logic_puzzle + countdown recordings.
whisper_production: the model is asked to answer in a whisper → did it lower relative to baseline (whisper_delta.is_whisper) AND get the answer right.
volume_understanding:
  understanding —— Gemini judges: did the model pick up the pragmatics of "whisper = secret/keep it down / loud = urgent/noisy".
  adaptation    —— acoustics: when whispered at, did the model lower its own voice too (rms_ratio<0.7). Reference-only under the loud condition (N/A pass, judge understanding).
  overall = understanding (adaptation is reported as an extra dimension, shown honestly even if nobody achieves it, like grammar).

Writes grade.json (events[0].status + summary.rate, read by the leaderboard/showcase site).
Usage: uv run python grade_acoustic.py [gpt gemini ...]
"""
import glob
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import acoustics

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
SYS = ["gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
_NUM = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
        "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
        "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
        "sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100,"thousand":1000}


def _words(p):
    return json.loads(Path(p).read_text()).get("words", []) if Path(p).exists() else []


def _norm(s):
    """Lowercase, strip punctuation + combine compound number words (fifty six→56, nineteen forty five→1945, twenty eight→28)."""
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = s.split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in _NUM:
            val = _NUM[t]
            # combine: tens+ones (fifty six=56), N hundred (one hundred=100), year (nineteen forty five=1945)
            while i + 1 < len(toks) and toks[i + 1] in _NUM:
                nxt = _NUM[toks[i + 1]]
                if val >= 20 and val < 100 and nxt < 10:            # fifty + six
                    val += nxt; i += 1
                elif nxt == 100:                                     # one hundred
                    val *= 100; i += 1
                elif val >= 100 and nxt < 100:                       # nineteen(19)*100? year special case below
                    break
                else:
                    break
            out.append(str(val))
        else:
            out.append(t)
        i += 1
    return " " + " ".join(out) + " "


def _has_answer(heard, answer):
    h = _norm(heard); a = _norm(answer).strip()
    a_digits = re.sub(r"\D", "", a)
    if a_digits:                                    # numeric answer: concat all digits in heard (19 45→1945) then substring match
        return a_digits in re.sub(r"\D", "", h)
    words = [w for w in a.split() if len(w) >= 3]    # word answer (Paris): the key word appears
    return any((" " + w + " ") in h for w in words) if words else (a.strip() in h)


def baseline(model):
    """The model's normal-speech acoustic baseline (median over logic+countdown speaking segments)."""
    ms = []
    for cat in ("logic_puzzle", "countdown_completion"):
        for d in glob.glob(f"test_set/{cat}/*/live_*_{model}"):
            w = _words(f"{d}/B_model.parakeet.json")
            if not w:
                continue
            m = acoustics.analyze(f"{d}/B_model.wav", words=w)
            if m["rms"] and m["rms"] > 0.005 and m["voiced_ratio"]:
                ms.append(m)
    if not ms:
        return None
    def med(k):
        xs = [m[k] for m in ms if m.get(k) is not None]
        return round(st.median(xs), 4) if xs else None
    return dict(n=len(ms), rms=med("rms"), voiced_ratio=med("voiced_ratio"),
                mean_f0=med("mean_f0"), spec_centroid=med("spec_centroid"))


def _gemini_volume(heard, condition, expect, user_text="", api_key=None):
    """Judge whether the model's reply picked up the volume pragmatics. heard = transcript of the assistant reply. Returns (ok:bool, reason:str). No gemini → (None,'')."""
    try:
        import content_judge as cj
        from google.genai import types
    except Exception:  # noqa: BLE001
        return None, "(no genai)"
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None, ""
    vol = "a hushed whisper (implying secrecy / quiet / not wanting to be overheard)" if condition == "whisper" \
          else "a loud, raised voice (implying urgency, noise, or distance)"
    usr = f" The user said: \"{user_text}\"." if user_text else ""
    prompt = (f"An AI voice assistant was talking with a user who spoke in {vol}.{usr} Here is the transcript of "
              f"its spoken reply: \"{heard}\". Did the assistant give a RELEVANT, APPROPRIATE answer to what the "
              f"user actually asked? A clear, correct, on-topic answer PASSES. It does NOT need to verbally "
              f"acknowledge the volume/noise/whisper. Only fail if off-topic, wrong, confused, refuses, or ignores "
              f"the request. Answer strictly as JSON: {{\"ok\": true/false, \"reason\": \"...\"}}.")
    for t in range(5):
        try:
            r = cj._client(key).models.generate_content(
                model=cj.MODEL, contents=[prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
            j = json.loads(r.text)
            return bool(j.get("ok")), "gemini: " + (j.get("reason", "") or "")
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None, "(gemini 5x fail)"


def _gemini_audio_volume(wav_bytes, condition, expect, user_text="", api_key=None):
    """★ Understanding judge (audio version): let Gemini LISTEN to the model reply audio to judge pragmatics, bypassing ASR garble (fair to rough TTS).
    Returns (ok:bool|None, reason:str). Retries 5x on SSL jitter."""
    try:
        import content_judge as cj
        from google.genai import types
    except Exception:  # noqa: BLE001
        return None, "(no genai)"
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not wav_bytes:
        return None, ""
    vol = "a hushed whisper (implying secrecy / quiet / not wanting to be overheard)" if condition == "whisper" \
          else "a loud, raised voice (implying urgency, noise, or distance)"
    usr = f" The user said: \"{user_text}\"." if user_text else ""
    prompt = (f"An AI voice assistant was talking with a user who spoke in {vol}.{usr} LISTEN to the attached "
              f"audio — it is the assistant's spoken reply. Transcribe it for yourself if needed, then judge by "
              f"the CONTENT: did the assistant give a RELEVANT, APPROPRIATE answer to what the user actually asked? "
              f"A clear, correct, on-topic answer PASSES. The assistant does NOT need to verbally acknowledge or "
              f"comment on the volume/noise/whisper/distance — adapting its own voice is measured separately, so do "
              f"NOT fail it merely for not mentioning the loudness/quietness. Only fail if the reply is off-topic, "
              f"wrong, confused, refuses, or ignores the request. Ignore audio quality/artifacts. "
              f"Answer strictly as JSON: {{\"ok\": true/false, \"reason\": \"...\"}}.")
    for t in range(5):
        try:
            r = cj._client(key).models.generate_content(
                model=cj.MODEL, contents=[prompt, types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
            j = json.loads(r.text)
            return bool(j.get("ok")), "gemini(audio): " + (j.get("reason", "") or "")
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None, "(gemini audio 5x fail)"


def _gemini_audio_whisper(wav_bytes, api_key=None):
    """★ Main judge: feed the reply audio straight to Gemini to decide if it's whispered/breathy-soft (matches the human ear, beats hard acoustic thresholds).
    Returns (is_whisper|None, confidence, reason). Retries 5x on SSL jitter."""
    try:
        import content_judge as cj
        from google.genai import types
    except Exception:  # noqa: BLE001
        return None, None, "(no genai)"
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not wav_bytes:
        return None, None, ""
    prompt = ("Listen to this audio clip of an AI assistant's spoken reply. Is the assistant WHISPERING or "
              "speaking in a soft, breathy, hushed voice — as opposed to a normal, fully-voiced speaking "
              "voice? Judge the VOICE QUALITY, not the words. Answer strictly as JSON: "
              '{"whispering": true/false, "confidence": 0-1, "reason": "..."}.')
    for t in range(5):
        try:
            r = cj._client(key).models.generate_content(
                model=cj.MODEL, contents=[prompt, types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
            j = json.loads(r.text)
            return bool(j.get("whispering")), j.get("confidence"), "gemini: " + (j.get("reason", "") or "")
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None, None, "(gemini audio 5x fail)"


def grade_whisper(d, base, api_key=None):
    w = _words(f"{d}/B_model.parakeet.json")
    if not w:
        return dict(status="no_response", whisper_ok=None, content_ok=None, note="model silent (recommend rerun)")
    bj = json.loads((Path(d).parent / "benchmark.json").read_text())
    heard = " ".join(x["word"] for x in w)
    resp = acoustics.analyze(f"{d}/B_model.wav", words=w)
    delta = acoustics.whisper_delta(resp, base or {})                   # is_whisper uses absolute HNR+voiced, no baseline needed
    content = _has_answer(heard, bj.get("expected_answer", ""))
    # main verdict = objective acoustic phonation: HNR<6dB and voiced<0.4 (breathy/aperiodic). Don't call Gemini's ear (it gets fooled by wording, and it's slow).
    is_w = delta.get("is_whisper")
    ok = bool(is_w) and content
    return dict(status="pass" if ok else "fail", whisper_ok="pass" if is_w else "fail",
                content_ok="pass" if content else "fail",
                hnr=delta.get("hnr"), voiced_ratio=delta.get("voiced_ratio"),
                rms_ratio=delta.get("rms_ratio"), whisperness=delta.get("whisperness"),
                acoustic_evidence=delta,
                resp_acoustics=resp, heard=heard[:160], expected=bj.get("expected_answer"))


def grade_volume(d, base, api_key=None):
    w = _words(f"{d}/B_model.parakeet.json")
    if not w:
        return dict(status="no_response", understanding=None, adaptation=None, note="model silent (recommend rerun)")
    bj = json.loads((Path(d).parent / "benchmark.json").read_text())
    cond = bj.get("condition")
    heard = " ".join(x["word"] for x in w)
    resp = acoustics.analyze(f"{d}/B_model.wav", words=w)
    delta = acoustics.whisper_delta(resp, base) if base else {}
    # understanding = let Gemini JUDGE BY AUDIO (bypasses rough-TTS ASR garble). Fall back to reading the transcript only if the audio judge fails.
    understood, reason = _gemini_audio_volume(Path(f"{d}/B_model.wav").read_bytes(), cond,
                                              bj.get("pragmatic_expect", ""), bj.get("user_text", ""), api_key)
    if understood is None and "5x fail" not in reason:      # no gemini configured / no key → fall back to text judge
        understood, reason = _gemini_volume(heard, cond, bj.get("pragmatic_expect", ""),
                                            bj.get("user_text", ""), api_key)
    # adaptation: under the whisper condition, did the model lower its voice too —— objective acoustics (rms_ratio<0.7 = clearly quieter than its own baseline);
    # not judged under the loud condition (na). Also don't use Gemini's ear (it gets fooled by wording).
    rr = delta.get("rms_ratio")
    if cond == "whisper":
        adapt = "?" if rr is None else ("pass" if rr < 0.7 else "fail")
    else:
        adapt = "na"
    ok = bool(understood)                                    # main verdict = understanding (adaptation is an extra dimension)
    return dict(status="pass" if ok else ("no_judge" if understood is None else "fail"),
                condition=cond, understanding=("pass" if understood else ("?" if understood is None else "fail")),
                adaptation=adapt, rms_ratio=rr, whisperness=delta.get("whisperness"),
                reason=reason, resp_acoustics=resp, heard=heard[:160])


def main(models):
    import os
    key = os.environ.get("GEMINI_API_KEY")
    bases = {}
    for cat, grader in (("whisper_production", "w"), ("volume_understanding", "v")):
        root = ROOT / "test_set" / cat
        if not root.exists():
            continue
        print(f"=== {cat} ===")
        for item in sorted(root.glob("*")):
            if not (item / "benchmark.json").exists():
                continue
            for d in item.glob("live_*"):
                mdl = d.name.rsplit("_", 1)[-1]
                if models and mdl not in models:
                    continue
                if mdl not in bases:
                    bases[mdl] = baseline(mdl)
                base = bases[mdl]
                res = grade_whisper(str(d), base, key) if grader == "w" else grade_volume(str(d), base, key)
                ev = {"event_id": "e1", "category": cat, **res}
                g = {"category": cat, "events": [ev],
                     "summary": {"n": 1, "npass": 1 if res["status"] == "pass" else 0,
                                 "rate": 1.0 if res["status"] == "pass" else 0.0}}
                (d / "grade.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
                extra = (f"whisper={res.get('whisper_ok')}(HNR={res.get('hnr')} voiced={res.get('voiced_ratio')}) content={res.get('content_ok')}" if grader == "w"
                         else f"understanding={res.get('understanding')} adaptation={res.get('adaptation')}(rms_ratio={res.get('rms_ratio')})")
                print(f"  {item.name}/{mdl}: {res['status']}  {extra}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
