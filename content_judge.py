"""Content judging (the content-scoring dimension): Gemini LISTENS to the AI response audio and judges **whether the produced content is correct**.

Division-of-labor rule: timing/overlap/silence are millisecond-level objective facts → always computed in code, never handed to Gemini;
only "content" — short words/phrases/numbers/semantics — goes to Gemini to judge **by listening to the audio** (not reading ASR text, which would inherit ASR's errors).

The reference answer comes from benchmark.json (expected_completion / expected_value / target_response, all derived from the cards),
and Gemini only answers "does what the model said == the reference answer", returning {heard, content_ok, reason}.

Uses raw/ut{NN}_{spk}.wav per-utterance clips: a single-utterance response takes one clip; a countdown SELF_COMPLETE (the AI counts down + says the completion word)
concatenates the AI's continuous run ending at the response line.
"""
import io
import json
import os
import wave

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

_SCHEMA = {
    "type": "object",
    "properties": {
        "heard": {"type": "string"},
        "content_ok": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "reason": {"type": "string"},
    },
    "required": ["heard", "content_ok", "reason"],
}


_CLIENT = None


def _client(api_key=None):
    global _CLIENT
    if _CLIENT is None:                     # singleton: a fresh one each time gets GC-closed → "client has been closed"
        _CLIENT = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    return _CLIENT


def _fetch_clip(reader, tl, i):
    """Fetch the raw clip bytes of the i-th utterance from reader (reader.raw_wav works with S3/local)."""
    return reader.raw_wav(tl, i)


def _concat_wavs(wavs):
    """Concatenate multiple same-format WAVs with stdlib wave → bytes."""
    out = io.BytesIO()
    w = None
    for b in wavs:
        r = wave.open(io.BytesIO(b), "rb")
        if w is None:
            w = wave.open(out, "wb"); w.setparams(r.getparams())
        w.writeframes(r.readframes(r.getnframes()))
        r.close()
    if w:
        w.close()
    return out.getvalue()


def _response_audio(reader, tl, event):
    """Response audio under evaluation: for SELF_COMPLETE (the AI counts down + says the completion word), concatenate the AI's continuous run ending at the response line; otherwise a single utterance."""
    i = event["response_line_index"]
    if event.get("expected_action") == "SELF_COMPLETE":
        spk = tl[i]["speaker"]
        start = i
        while start - 1 >= 0 and tl[start - 1].get("speaker") == spk:
            start -= 1
        return _concat_wavs([_fetch_clip(reader, tl, k) for k in range(start, i + 1)])
    return _fetch_clip(reader, tl, i)


def _rubric(event, category):
    exp = event.get("expected_completion") or event.get("target_response") or ""
    if category == "countdown_completion":
        return (f"An AI voice assistant was doing a countdown drill. It should count DOWN in order to 'one', "
                f"then say the completion cue: \"{exp}\". Judge CONTENT only (ignore timing): did it "
                f"(a) count down in order to one, and (b) say the completion \"{exp}\" — the RIGHT cue word, "
                f"not a different one (e.g. saying 'stop' when it should say 'go' is a FAIL). "
                f"Minor TTS artifacts are fine; a WRONG or missing cue is a fail.")
    if category == "predictive_number_continuation":
        val = event.get("expected_value")
        tail = f" (which is \"{exp}\")" if exp else ""
        return (f"An AI voice assistant was continuing a number sequence and should say the number "
                f"{val}{tail}. Judge CONTENT only: did it clearly say the value {val}? A DIFFERENT number "
                f"is a FAIL; a minor stutter/artifact that still clearly conveys {val} is a pass.")
    if category == "phrase_completion":
        return (f"An AI voice assistant was completing a well-known phrase. The correct completion is \"{exp}\". "
                f"Judge CONTENT only: did it give the correct completion (any natural phrasing of it counts)? "
                f"A wrong or unrelated completion is a FAIL.")
    if category == "logic_puzzle":
        q = (event.get("question") or "").strip()
        qline = f'The user asked: "{q}". ' if q else ""
        return (f'{qline}The correct answer is "{exp}". The audio is the AI\'s spoken reply. '
                f'Judge ONLY whether the reply conveys the answer "{exp}". Do NOT re-solve the puzzle and do '
                f'NOT reinterpret the AI\'s reply as a new question — just compare what the AI actually said '
                f'against the given correct answer "{exp}". Treat number words and digits as equal '
                f'(e.g. "two"=="2"), ignore filler/phrasing, and accept any wording that states "{exp}". '
                f'content_ok=pass if the reply gives "{exp}", otherwise FAIL.')
    # forbidden / deliberate / keyword / grammar / wrong_word
    return (f"An AI voice assistant was supposed to respond with: \"{exp}\". Judge CONTENT only (ignore timing): "
            f"did it say that — the right words / meaning? A different or wrong response is a FAIL; "
            f"minor TTS artifacts are fine. For grammar/word corrections, any correct phrasing of the fix counts.")


def judge_content(reader, tl, event, category, api_key=None):
    """→ {content_ok: pass|fail|uncertain, heard, reason}. Audio fetch / API call failure → uncertain (no hard verdict).

    reader must provide raw_wav(tl, i) → raw WAV bytes of the i-th utterance (S3 or local dir both fine)."""
    try:
        audio = _response_audio(reader, tl, event)
    except Exception as e:  # noqa: BLE001
        return {"content_ok": "uncertain", "heard": "", "reason": f"audio fetch failed: {type(e).__name__}: {e}"}
    prompt = (_rubric(event, category)
              + ' Return JSON {"heard":"<exact transcript of what you hear>",'
                '"content_ok":"pass"|"fail"|"uncertain","reason":"<one short sentence>"}.')
    try:
        r = _client(api_key).models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=_SCHEMA))
        return json.loads(r.text)
    except Exception as e:  # noqa: BLE001
        return {"content_ok": "uncertain", "heard": "", "reason": f"Gemini call/parse failed: {type(e).__name__}: {e}"}
