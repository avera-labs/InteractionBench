"""grade_live.py —— grade one recording of a **live human conversation** with benchmark.json (the reference answer).

Unlike grade_model: a live recording has no matching timeline.json (the user track is the human's own speech, whose timing doesn't line up with the reference track),
so this is **timeline-free**: find the trigger word's occurrence directly in the user track's whole-track ASR, then find the graded response in the model track,
reusing grade_model's four-dimension logic (①timing ②overlap ③content ④silence).

All inputs are **local** data (dashboard live recording + local parakeet):
  benchmark        reference-answer dict (benchmark.json pulled from S3)
  user_words       user track whole-track ASR words (A_user.parakeet.json["words"], absolute seconds)
  model_words      model track whole-track ASR words (B_model.parakeet.json["words"])
  model_wav_path   model track audio (B_model.wav) → ③content handed to Gemini to listen
"""
import io
import json
import time

import grade_model as gm     # reuse _norm / _response_onset_abs / TOL / SILENCE_CATS
import vad_utils             # VAD boundary utils (same logic shared with grade_model)


def _occurrences(user_words, trig):
    tn = gm._norm(trig)
    return [w for w in user_words if gm._norm(w["word"]) == tn]


def _gemini_local(wav_path, onset, end, event, category, api_key, asr=""):
    """Cut the local model track audio [onset-0.2, end+0.4] and feed it to Gemini to judge content.
    asr = parakeet full-word transcript of this clip: the cue/completion word often appears only after a long preamble or repeated counting, so attaching the transcript
    keeps the judge from missing the ending (the audio is still authoritative)."""
    try:
        import content_judge
        from pydub import AudioSegment
        from google.genai import types
        seg = AudioSegment.from_file(wav_path)
        clip = seg[max(0, int((onset - 0.2) * 1000)): int((end + 0.4) * 1000)]
        buf = io.BytesIO(); clip.export(buf, format="wav")
        rub = content_judge._rubric(event, category)
        hint = (f' A word-level ASR transcript of this exact clip is: "{asr}". The key/completion word may '
                f'appear LATE — after a long preamble or a repeated count — so use this transcript to avoid '
                f'missing it; the audio remains authoritative for pronunciation.') if asr else ''
        parts = [types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                 rub + hint + ' Return JSON {"heard":"...","content_ok":"pass"|"fail"|"uncertain","reason":"..."}.']
        cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                          response_schema=content_judge._SCHEMA)
        last = None
        for attempt in range(3):                       # transient errors like proxy SSL EOF → backoff retry
            try:
                r = content_judge._client(api_key).models.generate_content(
                    model=content_judge.MODEL, contents=parts, config=cfg)
                v = json.loads(r.text)
                return {"status": v.get("content_ok", "uncertain"), "heard": v.get("heard", ""),
                        "reason": "gemini: " + (v.get("reason") or "")}
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        return {"status": "uncertain", "heard": "", "reason": f"gemini_err:{type(last).__name__}:{last}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "uncertain", "heard": "", "reason": f"gemini_err:{type(e).__name__}:{e}"}


def _gemini_text(transcript, event, category, api_key):
    """Text-only content judging: hand the parakeet whole-track transcript (more accurate than Gemini's ear) to Gemini for a semantic verdict.
    Used for content-type IQ (countdown/logic) —— the cue/answer is often buried in a verbose reply, which the audio judge misses but the transcript judge doesn't."""
    try:
        import content_judge
        from google.genai import types
        rub = content_judge._rubric(event, category)
        prompt = (rub + f' The assistant\'s spoken reply, transcribed word-for-word by a dedicated ASR '
                  f'(more reliable than listening), was: "{transcript}". Judge from this transcript — the '
                  f'key/answer word may appear late, after a long preamble or a repeated count. '
                  'Return JSON {"heard":"...","content_ok":"pass"|"fail"|"uncertain","reason":"..."}.')
        last = None
        for attempt in range(3):
            try:
                r = content_judge._client(api_key).models.generate_content(
                    model=content_judge.MODEL, contents=[prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                       response_schema=content_judge._SCHEMA))
                v = json.loads(r.text)
                return {"status": v.get("content_ok", "uncertain"),
                        "heard": v.get("heard", transcript[:140]),
                        "reason": "gemini(text): " + (v.get("reason") or "")}
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        return {"status": "uncertain", "heard": "", "reason": f"gemini_err:{type(last).__name__}:{last}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "uncertain", "heard": "", "reason": f"gemini_err:{type(e).__name__}:{e}"}


def grade_live(benchmark, user_words, model_words, model_wav_path=None,
               user_wav_path=None, gemini=False, api_key=None):
    """Run all graded events on one live recording; each event returns 4 dimensions + an overall verdict.
    For ②overlap, the "user finished-speaking moment" prefers user_wav's VAD (aligned with the user track); without a wav, falls back to the ASR cap."""
    category = benchmark.get("category")
    user_segs = None
    if user_wav_path:
        try:
            user_segs = vad_utils.vad_segments(user_wav_path)
        except Exception:  # noqa: BLE001
            user_segs = None
    out = []
    for e in benchmark.get("expected_events", []):
        if not e.get("graded"):
            continue
        rec = {"event": e["event_id"], "category": category, "action": e.get("expected_action"),
               "grade_dimension": e.get("grade_dimension")}
        trg = e.get("trigger"); win = e.get("onset_window_ms")

        if trg and win is not None:
            occ = _occurrences(user_words, trg["anchor_word"])
            if not occ:
                rec["status"] = "no_trigger"
                rec["note"] = f"trigger word {trg.get('anchor_word')!r} not found in the user track"
                out.append(rec); continue
            # pick the occurrence "followed by a model response" (avoid the same word from the rule-setup turn); if none, use the last one
            trig_w = onset = end = heard = None
            for w in occ:
                o, en, hd = gm._response_onset_abs(model_words, w["t1"], e.get("target_response"))
                if o is not None:
                    trig_w, onset, end, heard = w, o, en, hd; break
            if trig_w is None:
                trig_w = occ[-1]
                onset, end, heard = gm._response_onset_abs(model_words, trig_w["t1"], e.get("target_response"))
            trig_end = vad_utils.snap_end(trig_w["t0"], trig_w["t1"], user_segs)   # snap anchor end to VAD
            if onset is None:
                rec["timing"] = {"status": "no_response"}; rec["tor"] = 0
                rec["status"] = "fail"; out.append(rec); continue
            rec["tor"] = 1

            lat = (onset - trig_end) * 1000.0
            t_status = "pass" if win[0] <= lat <= win[1] else ("early" if lat < win[0] else "late")
            rec["window_ms"] = win
            rec["timing"] = {"status": t_status, "latency_ms": round(lat, 1),
                             "trig_end": round(trig_end, 3), "resp_onset": round(onset, 3)}

            a_end = vad_utils.turn_end_vad(user_segs, trig_w) if user_segs else None  # VAD: true finished-speaking moment
            if a_end is None:
                a_end = vad_utils.turn_end_asr(user_words, trig_w)                   # fallback: ASR cap
            ov = a_end - onset; exp_ov = e.get("expected_overlap")
            o_status = ("pass" if ov > -gm.TOL else "fail") if exp_ov == "required" else \
                       ("pass" if ov < gm.TOL else "fail") if exp_ov == "forbidden" else "n/a"
            rec["overlap"] = {"status": o_status, "overlap_s": round(ov, 3), "exp": exp_ov}

            if category in gm.SILENCE_CATS:
                rec["silence"] = {"status": "pass" if onset >= trig_end - gm.TOL else "fail"}

            if gemini and model_wav_path:
                rec["content"] = _gemini_local(model_wav_path, onset, end, e, category, api_key,
                                               asr=" ".join(w["word"] for w in model_words))
            else:
                tgt = gm._norm(e.get("target_response"))
                rec["content"] = {"status": "pass" if tgt and tgt in gm._norm(heard) else "uncertain",
                                  "heard": heard, "target": e.get("target_response")}

            ok = (t_status == "pass" and o_status in ("pass", "n/a")
                  and rec["content"]["status"] != "fail"
                  and rec.get("silence", {}).get("status", "pass") == "pass")
            rec["status"] = "pass" if ok else "fail"
        else:
            # content event (SELF_COMPLETE / countdown, no trigger): judge only the model track content
            heard = " ".join(w["word"] for w in model_words[:40])
            rec["tor"] = 1 if model_words else 0
            if gemini and model_words:
                # judge from the parakeet whole-track transcript (Gemini's ear misses the cue/answer buried in a verbose reply)
                rec["content"] = _gemini_text(" ".join(w["word"] for w in model_words), e, category, api_key)
            else:
                tgt = gm._norm(e.get("target_response") or e.get("expected_completion"))
                rec["content"] = {"status": "pass" if tgt and tgt in gm._norm(heard) else "uncertain",
                                  "heard": heard}
            rec["status"] = rec["content"]["status"]   # follow the judge directly: pass/fail/uncertain (judge failure = uncertain, not a pass)
        out.append(rec)
    return out
