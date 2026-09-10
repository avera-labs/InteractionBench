"""Importable, path-based graders for the v4 behavioral tests.

Single source of truth shared by the dashboard (`dashboard/server.py` web handlers) and the
standalone CLI (`grade.py`). Each grader takes a *run directory* — a folder holding the two-track
word-level ASR (`A_user.parakeet.json` / `B_model.parakeet.json`) and, where content judging is
used, the model wav (`B_model.wav`) — plus a Gemini `api_key` (None → skip content judging) and the
`use_gemini` flag (whether the caller asked for it, so the reported `gemini.key_found` stays honest).

Returns a plain dict; `{"error": "..."}` on any precondition failure. No web/env dependencies —
callers resolve the run dir and the API key. Covers:
  backchannel · user_interruption · smooth_turn_taking · pause_handling · user_backchannel
"""
import io
import json
import re
from pathlib import Path

from pydub import AudioSegment

# —— backchannel candidate/heuristic thresholds (matching the dashboard) ——
BC_CAND_MAX_DUR = 3.0    # ≤ this duration → send to Gemini to judge (longer → counted as a floor-take outright)
BC_CAND_MAX_WORDS = 10   # ≤ this many words → send to Gemini to judge (more → counted as a floor-take outright)
BC_HEUR_DUR = 1.5        # heuristic without Gemini: ≤1.5s and ≤3 words → treat as a backchannel
BC_HEUR_WORDS = 3


# ============================ shared helpers ============================
def load_words(p):
    """Read *.parakeet.json → word list [{word,t0,t1}] (missing file / no words → [])."""
    p = Path(p)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("words") or []


def group_utts(words, gap=1.5):
    """Word list → utterances split on pauses [{t0,t1,text}] (pause > gap seconds → new utterance)."""
    utts = []
    for w in words:
        if utts and (w["t0"] - utts[-1]["t1"]) <= gap:
            utts[-1]["t1"] = w["t1"]; utts[-1]["text"] += " " + w["word"]
        else:
            utts.append({"t0": w["t0"], "t1": w["t1"], "text": w["word"]})
    return utts


def find_interrupt_idx(user_words, interrupt_text):
    """Locate the interruption utterance's start in the user word stream (the interruption is at the tail; align its first few words back-to-front). Returns the start index or None."""
    nrm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    itoks = [t for t in (nrm(w) for w in interrupt_text.split()) if t]
    uwn = [nrm(w["word"]) for w in user_words]
    if not itoks or not uwn:
        return None
    key = itoks[:min(3, len(itoks))]; k = len(key)
    for i in range(len(uwn) - k, -1, -1):
        if uwn[i:i + k] == key:
            return i
    for i in range(len(uwn) - 1, -1, -1):
        if uwn[i] == key[0]:
            return i
    return None


def _gemini_meta(use_gemini, api_key):
    return {"asked": bool(use_gemini), "key_found": bool(api_key) if use_gemini else True}


def _gc_retry(fn, tries=4):
    """Judge calls through a congested proxy fail transiently (ConnectError/SSL EOF) → backoff retry; only raise to each caller's except once exhausted."""
    import time
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


# ============================ Gemini judges ============================
_BC_PROMPT = (
    "This is a short utterance an AI voice model produced WHILE THE USER WAS STILL TELLING A STORY. Decide if it "
    "is a BACKCHANNEL — a brief listener response that shows attention/empathy but does NOT take over the turn. "
    "Backchannels include acknowledgments and continuers ('mm-hm', 'yeah', 'right', 'I see', 'uh-huh', 'sure', "
    "'got it') AND brief empathetic REACTIONS to what the user just said ('oh no', \"oh no, that's terrible\", "
    "\"that's awful\", 'wow', \"wow, that's crazy\", 'no way', 'oh my god', 'that sucks'). It is NOT a backchannel "
    "if it TAKES THE FLOOR: introducing a new topic, asking a question that redirects, giving substantive "
    "information/advice, correcting the user, or otherwise trying to grab and hold the turn. Judge by "
    'CONTENT/INTENT, not by length. Return JSON {"heard":"<what was said>", "is_backchannel": true|false, '
    '"reason":"<short>"}.')


def bc_judge(wav_bytes, api_key):
    """Judge one short speech segment: backchannel, or an attempt to speak/take over. On failure → conservatively treat as backchannel."""
    try:
        import content_judge
        from google.genai import types
        r = _gc_retry(lambda: content_judge._client(api_key).models.generate_content(
            model=content_judge.MODEL,
            contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), _BC_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json")))
        v = json.loads(r.text)
        return {"is_backchannel": bool(v.get("is_backchannel", True)),
                "heard": v.get("heard", ""), "reason": v.get("reason", "")}
    except Exception as ex:  # noqa: BLE001
        return {"is_backchannel": True, "heard": "", "reason": f"judge_err:{type(ex).__name__}"}


def interrupt_relevance(wav_path, s, e, interrupt_text, api_key):
    """Judge: after being interrupted, does the model's reply stop the old line and address the interruption content."""
    try:
        import content_judge
        from google.genai import types
        seg = AudioSegment.from_file(wav_path)[max(0, int((s - 0.1) * 1000)): int((e + 0.3) * 1000)]
        buf = io.BytesIO(); seg.export(buf, format="wav")
        prompt = ('A user INTERRUPTED an AI mid-response with: "%s". The audio is what the AI said right after. '
                  "Judge whether the AI's reply ADDRESSES or RELATES TO the interruption — i.e. it answers the "
                  "interrupting question or reacts to its topic. A reply on the SAME TOPIC counts as \"yes\" "
                  "(e.g. the interruption asks about the weather and the AI talks about the weather = yes), EVEN "
                  "IF the AI also finished its earlier sentence, hedged, or was slightly off on a detail "
                  "(today vs tomorrow). Mark \"partial\" if it only briefly acknowledges without really "
                  "answering. Mark \"no\" ONLY if the AI ignores the interruption and keeps talking about the "
                  "UNRELATED previous topic. Return JSON {\"heard\":\"<what the AI said>\", "
                  '"addressed":"yes"|"partial"|"no", "reason":"<short>"}.' % (interrupt_text or ""))
        r = _gc_retry(lambda: content_judge._client(api_key).models.generate_content(
            model=content_judge.MODEL,
            contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")))
        v = json.loads(r.text)
        return {"addressed": v.get("addressed", "partial"), "heard": v.get("heard", ""),
                "reason": v.get("reason", "")}
    except Exception as ex:  # noqa: BLE001
        return {"addressed": "uncertain", "heard": "", "reason": f"judge_err:{type(ex).__name__}"}


_PAUSE_PROMPT = (
    "A user was speaking and PAUSED MID-SENTENCE — they are NOT finished (their thought trails off, more is "
    "coming). During that pause the AI produced this audio. Decide whether the AI TOOK THE FLOOR — grabbed the "
    "turn with something that now expects the user to respond to the AI or redirects the conversation (a "
    "question, a statement, advice, a new topic) — OR merely gave a brief SUPPORTIVE CONTINUER / acknowledgment "
    "that YIELDS the floor back so the user keeps talking (e.g. 'mm-hm', 'yeah?', 'go on', 'oh no', 'right', "
    "'take your time'). Judge by intent, not length. "
    'Return JSON {"heard":"<what the AI said>", "took_floor": true|false, "reason":"<short>"}.')


def pause_judge(wav_bytes, api_key):
    """Judge: during the pause, is the model's segment "taking the floor" or a "supportive continuer (yielding the floor)"."""
    try:
        import content_judge
        from google.genai import types
        r = _gc_retry(lambda: content_judge._client(api_key).models.generate_content(
            model=content_judge.MODEL,
            contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), _PAUSE_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json")))
        v = json.loads(r.text)
        return {"took_floor": bool(v.get("took_floor", True)), "heard": v.get("heard", ""),
                "reason": v.get("reason", "")}
    except Exception as ex:  # noqa: BLE001
        return {"took_floor": True, "heard": "", "reason": f"judge_err:{type(ex).__name__}"}


# last word landing on these function/connective words with no sentence-ending punctuation → judged a mid-sentence cutoff (unfinished)
_TAIL_STOP = {"the", "a", "an", "and", "or", "but", "so", "to", "of", "with", "in", "on", "at", "for",
              "is", "are", "was", "were", "that", "this", "it", "its", "my", "your", "his", "her",
              "their", "our", "as", "if", "when", "then", "than", "into", "from", "by", "about",
              "because", "which", "who", "how", "what", "some", "these", "those", "such", "very"}

_UB_PROMPT = (
    "An assistant was giving a spoken answer to the user's request. While the assistant was speaking, the user "
    "only inserted brief BACKCHANNELS — listener acknowledgments like 'mm-hm', 'ok', 'yeah', 'right', 'uh-huh'. "
    "These are NOT questions, requests, or new turns; a good assistant should keep going and finish its answer.\n"
    "SCOPE: this test measures ONLY whether the backchannels threw the assistant OFF — nothing else. You are NOT "
    "grading answer quality, factual accuracy, level of detail, or how fully it satisfied the request. An answer "
    "that stayed on the requested topic and was not knocked off course is CORRECT here even if it was brief, "
    "skipped some requested detail (e.g. wasn't as 'step by step' as asked), or wasn't the best possible answer.\n"
    "Assign exactly one tag:\n"
    "- C_RESUME: gave a coherent, on-topic turn and did NOT react to the acknowledgments (GOOD). A backchannel "
    "landing on the assistant's final words does NOT make this a derailment — it simply finished. This is the "
    "default whenever the assistant stayed on topic and wasn't visibly disrupted.\n"
    "- C_RESPOND: treated an acknowledgment as a turn — paused to address it, or answered a non-existent question.\n"
    "- C_UNCERTAIN: asked the user to repeat / showed confusion ('sorry, what?', 'did you say something?').\n"
    "- C_UNKNOWN: the acknowledgment visibly BROKE it — speech went incoherent/garbled, switched to an unrelated "
    "topic, OR got cut off MID-SENTENCE right when a mid-answer backchannel hit and never resumed. Do NOT tag "
    "C_UNKNOWN merely because the answer lacked detail or the assistant went quiet after a backchannel that "
    "landed at/after the answer was already a complete turn.\n"
    'Return JSON {"behaviour":"C_*", "derailed": true|false, "reason":"<short>"}. derailed=false ONLY for C_RESUME.')


def ub_judge(prompt_text, bc_words, reply_text, timing_fact, api_key):
    """Judge the C axis: when backchanneled, does the model keep going (C_RESUME) or get derailed (C_RESPOND/UNCERTAIN/UNKNOWN).
    timing_fact is precomputed by the caller from "backchannel landing point vs answer end", a key timing hint for Gemini."""
    try:
        import content_judge
        from google.genai import types
        msg = (f'{_UB_PROMPT}\n\nUser request: "{prompt_text}"\nBackchannels inserted: {bc_words}\n'
               f'Timing fact: {timing_fact}\n'
               f'Assistant transcript: "{reply_text}"')
        r = _gc_retry(lambda: content_judge._client(api_key).models.generate_content(
            model=content_judge.MODEL, contents=[msg],
            config=types.GenerateContentConfig(response_mime_type="application/json")))
        v = json.loads(r.text)
        return {"behaviour": v.get("behaviour", "C_UNKNOWN"), "derailed": bool(v.get("derailed", True)),
                "reason": v.get("reason", "")}
    except Exception as ex:  # noqa: BLE001
        return {"behaviour": "C_UNKNOWN", "derailed": None, "reason": f"judge_err:{type(ex).__name__}"}


# ============================ graders (directory-level) ============================
def grade_backchannel(d, use_gemini=False, api_key=None):
    """Test backchannel ability: VAD-segment the model track → for each short speech segment, first judge "backchannel or an attempt to speak" → count only true backchannels.
    Needs B_model.wav + B_model.parakeet.json (optional A_user.wav/parakeet to set the end of the user's turn)."""
    d = Path(d)
    mwav, model_j, uwav = d / "B_model.wav", d / "B_model.parakeet.json", d / "A_user.wav"
    if not mwav.exists():
        return {"error": "no model track recording B_model.wav"}
    if not model_j.exists():
        return {"error": "no model track ASR B_model.parakeet.json (run parakeet first)"}
    import vad_utils
    model_words = json.loads(model_j.read_text())["words"]
    segs = vad_utils.vad_segments(str(mwav))
    track = AudioSegment.from_file(mwav)
    user_secs = round(len(AudioSegment.from_file(uwav)) / 1000.0, 1) if uwav.exists() else None
    user_end = None
    uj = d / "A_user.parakeet.json"
    if uj.exists():
        uw = json.loads(uj.read_text()).get("words") or []
        if uw:
            user_end = max(w["t1"] for w in uw)
    if user_end is None and uwav.exists():
        usegs = vad_utils.vad_segments(str(uwav))
        if usegs:
            user_end = usegs[-1][1]
    out, tor, n_bc, n_att, n_resp = [], 0, 0, 0, 0
    for s, e in segs:
        dur = e - s
        seg_ws = [w for w in model_words if s - 0.15 <= w["t0"] <= e + 0.15]
        words = [w["word"] for w in seg_ws]
        nw = len(words)
        rec = {"start": round(s, 2), "end": round(e, 2), "dur": round(dur, 2), "words": " ".join(words)}
        if nw == 0:
            rec["kind"] = "noise"
        elif user_end is not None and s > user_end + 0.4:
            rec["kind"] = "response"; n_resp += 1
        elif dur > BC_CAND_MAX_DUR or nw > BC_CAND_MAX_WORDS:
            rec["kind"] = "floor_take"; tor = 1
        elif api_key:
            c0 = min([s] + [w["t0"] for w in seg_ws]) - 0.15
            c1 = max([e] + [w["t1"] for w in seg_ws]) + 0.35
            clip = track[max(0, int(c0 * 1000)): int(c1 * 1000)]
            buf = io.BytesIO(); clip.export(buf, format="wav")
            j = bc_judge(buf.getvalue(), api_key)
            rec["heard"], rec["reason"] = j["heard"], j["reason"]
            if j["is_backchannel"]:
                rec["kind"] = "backchannel"; n_bc += 1
            else:
                rec["kind"] = "attempt"; n_att += 1
        elif dur <= BC_HEUR_DUR and nw <= BC_HEUR_WORDS:
            rec["kind"] = "backchannel"; n_bc += 1
        else:
            rec["kind"] = "floor_take"; tor = 1
        out.append(rec)
    freq = round(n_bc / user_secs, 3) if user_secs else None
    return {"n_backchannel": n_bc, "n_attempt": n_att, "n_response": n_resp, "tor": tor,
            "freq_per_s": freq, "user_secs": user_secs, "user_end": user_end,
            "n_segments": len(segs), "gemini": _gemini_meta(use_gemini, api_key), "segments": out}


def grade_interruption(d, use_gemini=False, api_key=None, interrupt_text=""):
    """Test "user interrupts the model": the user track's last utterance = the interruption → ①was the model talking when interrupted ②TOR ③how fast ④does it address the interruption content.
    Needs A_user/B_model.parakeet.json (+ B_model.wav for relevance). If interrupt_text is given, align the interruption utterance by text."""
    d = Path(d)
    user_j, model_j, mwav = d / "A_user.parakeet.json", d / "B_model.parakeet.json", d / "B_model.wav"
    if not user_j.exists() or not model_j.exists():
        return {"error": "missing split-track ASR (A_user/B_model.parakeet.json, run parakeet first)"}
    interrupt_text = (interrupt_text or "").strip()
    uw, mw = load_words(user_j), load_words(model_j)
    idx = find_interrupt_idx(uw, interrupt_text) if interrupt_text else None
    if idx is not None:
        i_s, i_e = uw[idx]["t0"], uw[-1]["t1"]
        itext = interrupt_text
    else:
        u_utts = group_utts(uw, 1.5)
        if len(u_utts) < 2:
            return {"error": "can't locate the interruption utterance —— no interruption matched in the user track and no two segments detected. "
                             "Pass --interrupt-text, or make sure the user track has both the first utterance + the interruption."}
        interrupt = u_utts[-1]
        i_s, i_e, itext = interrupt["t0"], interrupt["t1"], (interrupt_text or interrupt["text"])
    was_talking = any(w["t0"] < i_s and w["t1"] > i_s - 1.0 for w in mw)
    model_after = [w for w in mw if w["t0"] >= i_e - 0.2]
    heard_after = " ".join(w["word"] for w in model_after[:50])
    after_dur = (model_after[-1]["t1"] - model_after[0]["t0"]) if model_after else 0
    tor = 1 if (len(model_after) > 3 or after_dur > 1.0) else 0
    latency_ms = round((model_after[0]["t0"] - i_e) * 1000, 1) if (tor and model_after) else None
    relevance = None
    if tor and api_key and mwav.exists() and model_after:
        relevance = interrupt_relevance(str(mwav), model_after[0]["t0"], model_after[-1]["t1"], itext, api_key)
    return {"was_talking": was_talking, "tor": tor, "latency_ms": latency_ms,
            "interrupt_text": itext, "interrupt_at": [round(i_s, 2), round(i_e, 2)],
            "heard_after": heard_after, "relevance": relevance,
            "gemini": _gemini_meta(use_gemini, api_key)}


def grade_turn_taking(d):
    """Smooth turn-taking: you finish an utterance → the model should reply promptly. Uses the last user segment to compute TOR↑/latency↓.
    Needs A_user/B_model.parakeet.json."""
    d = Path(d)
    user_j, model_j = d / "A_user.parakeet.json", d / "B_model.parakeet.json"
    if not user_j.exists() or not model_j.exists():
        return {"error": "missing split-track ASR (A_user/B_model.parakeet.json, run parakeet first)"}
    uw, mw = load_words(user_j), load_words(model_j)
    u_utts = group_utts(uw, 1.5)
    if not u_utts:
        return {"error": "no speech detected in the user track —— say one complete utterance normally for the model to pick up."}
    turn_end = u_utts[-1]["t1"]
    model_after = [w for w in mw if w["t0"] >= turn_end - 0.2]
    tor = 1 if len(model_after) >= 1 else 0
    latency_ms = round((model_after[0]["t0"] - turn_end) * 1000, 1) if (tor and model_after) else None
    return {"tor": tor, "latency_ms": latency_ms, "turn_end": round(turn_end, 2),
            "heard_after": " ".join(w["word"] for w in model_after[:40])}


def grade_pause(d, use_gemini=False, api_key=None):
    """Pause handling: a mid-utterance pause (first-half / second-half, two segments) → the model shouldn't jump in. Model speaking within the pause window = bad.
    If gemini is given, judge by "took floor vs continuer", otherwise use the v1 word-count threshold. Needs A_user/B_model.parakeet.json (+ B_model.wav for intent)."""
    d = Path(d)
    user_j, model_j, mwav = d / "A_user.parakeet.json", d / "B_model.parakeet.json", d / "B_model.wav"
    if not user_j.exists() or not model_j.exists():
        return {"error": "missing split-track ASR (A_user/B_model.parakeet.json, run parakeet first)"}
    uw, mw = load_words(user_j), load_words(model_j)
    u_utts = group_utts(uw, 0.8)   # lower the pause threshold to 0.8s: even ~1-2s pauses split into two segments (1.5s would miss a pause of exactly 1.4s)
    if len(u_utts) < 2:
        return {"error": "only 1 segment detected in the user track —— the pause test needs two segments: 'first half' + (clear pause ≥0.8s) + 'second half'."}
    p_s, p_e = u_utts[0]["t1"], u_utts[1]["t0"]
    pause_dur = round(p_e - p_s, 2)
    in_pause = [w for w in mw if p_s - 0.1 <= w["t0"] < p_e]
    dur = (in_pause[-1]["t1"] - in_pause[0]["t0"]) if in_pause else 0
    judged = None
    if in_pause and api_key and mwav.exists():
        c0, c1 = in_pause[0]["t0"] - 0.1, in_pause[-1]["t1"] + 0.3
        clip = AudioSegment.from_file(mwav)[max(0, int(c0 * 1000)): int(c1 * 1000)]
        buf = io.BytesIO(); clip.export(buf, format="wav")
        judged = pause_judge(buf.getvalue(), api_key)
        if str(judged["reason"]).startswith("judge_err"):
            tor = 1 if (len(in_pause) > 3 or dur > 1.0) else 0
        else:
            tor = 1 if judged["took_floor"] else 0
    else:
        tor = 1 if (len(in_pause) > 3 or dur > 1.0) else 0
    return {"pause_window": [round(p_s, 2), round(p_e, 2)], "pause_dur": pause_dur,
            "jumped_in": bool(in_pause), "tor": tor, "judged": judged,
            "heard_in_pause": " ".join(w["word"] for w in in_pause[:30]),
            "gemini": _gemini_meta(use_gemini, api_key)}


def grade_user_backchannel(d, use_gemini=False, api_key=None):
    """user_backchannel: while the model speaks a long stretch, the user drops a few backchannels (ok/mm-hm) → it should finish its answer and not get derailed.
    The user track's first segment = the prompt utterance; the short segments after (≤4 words) = backchannels. gemini judges the C axis; without gemini, fall back to timing. Needs A_user/B_model.parakeet.json."""
    d = Path(d)
    user_j, model_j = d / "A_user.parakeet.json", d / "B_model.parakeet.json"
    if not user_j.exists() or not model_j.exists():
        return {"error": "missing split-track ASR (A_user/B_model.parakeet.json, run parakeet first)"}
    uw, mw = load_words(user_j), load_words(model_j)
    u_utts = group_utts(uw, 0.8)   # split threshold 0.8s: when the model replies fast, backchannels packed within <1.5s still split apart (aligned with grade_pause)
    if len(u_utts) < 2:
        return {"error": "only 1 segment detected in the user track —— send the prompt utterance first, then after the model starts speaking add at least one backchannel (ok/mm-hm)."}
    prompt_utt = u_utts[0]
    bc_utts = [u for u in u_utts[1:] if len((u.get("text") or "").split()) <= 4]
    if not bc_utts:
        return {"error": "no backchannel detected —— after the prompt utterance, add short backchannels (ok/mm-hm, ≤4 words)."}
    bc_times = [round(u["t0"], 2) for u in bc_utts]
    prompt_text = prompt_utt.get("text", "")
    reply = [w for w in mw if w["t0"] >= prompt_utt["t1"] - 0.5]
    if not reply:
        return {"error": "the model didn't speak after the prompt utterance —— let the model say a stretch first, then add backchannels."}
    reply_text = " ".join(w["word"] for w in reply)
    reply_end = reply[-1]["t1"]
    # whether the answer is "cut off mid-sentence": last word has no sentence-ending punctuation and lands on a function/connective word → clearly unfinished (e.g. "...catches the")
    last_tok = (reply[-1]["word"] or "").strip()
    ends_clean = last_tok[-1:] in ".!?…"
    tail_word = last_tok.rstrip(".,!?;:—…").lower()
    looks_truncated = (not ends_clean) and (tail_word in _TAIL_STOP)
    # only backchannels "landing mid-answer" are used to test derailment-resistance; ones on the final 0.5s / after the answer finished don't count
    mid_bc = [t for t in bc_times if t < reply_end - 0.5]
    tail_bc = [t for t in bc_times if t >= reply_end - 0.5]
    if mid_bc:
        continued_after = any(w["t0"] > mid_bc[-1] + 0.3 for w in reply)
        if continued_after and not looks_truncated:
            timing_fact = ("YES — the assistant kept speaking well after the last mid-answer backchannel "
                           "and its answer reads as a finished turn.")
        elif looks_truncated:
            timing_fact = (f"The assistant went SILENT right around the last mid-answer backchannel and its answer "
                           f"is CUT OFF mid-sentence — it ends on '{last_tok}' with no sentence-ending punctuation, "
                           "then silence. This mid-sentence cutoff coincides with the backchannel: strong evidence "
                           "the backchannel derailed it → C_UNKNOWN.")
        else:
            timing_fact = ("The assistant produced no speech after the last mid-answer backchannel. "
                           "Only call this a derailment (C_UNKNOWN) if the transcript is cut off mid-thought; "
                           "if the transcript is a complete answer, it simply finished (C_RESUME).")
    else:
        # all backchannels land at the answer's end or after it finished —— usually no mid-point that could derail it
        continued_after = True
        if looks_truncated:
            timing_fact = (f"The assistant's answer is CUT OFF mid-sentence — it ends on '{last_tok}' with no "
                           "sentence-ending punctuation, then silence. If it broke off rather than finishing, C_UNKNOWN.")
        else:
            timing_fact = ("All backchannels landed at or after the assistant had finished its answer "
                           "(the last one overlapped its final words), so there was nothing left to say afterwards. "
                           "Judge purely from whether the transcript is a COMPLETE, coherent answer.")
    judged = ub_judge(prompt_text, [u.get("text", "") for u in bc_utts], reply_text, timing_fact, api_key) if api_key else None
    if judged and judged["derailed"] is not None:
        derailed, behaviour, reason = judged["derailed"], judged["behaviour"], judged["reason"]
    else:
        derailed = (not continued_after) or looks_truncated
        behaviour = "C_RESUME" if not derailed else "C_UNKNOWN"
        reason = ("The assistant kept speaking after the backchannel and the answer is complete." if not derailed
                  else (f"The answer is cut off mid-sentence at the backchannel (…{last_tok}); likely derailed." if looks_truncated
                        else "The assistant stopped after a mid-utterance backchannel; possibly derailed."))
    return {"n_backchannel": len(bc_utts), "bc_times": bc_times, "prompt_text": prompt_text,
            "reply_text": reply_text[:400], "reply_dur": round(reply_end - reply[0]["t0"], 2),
            "continued_after_last_bc": continued_after, "looks_truncated": looks_truncated,
            "mid_bc": [round(t, 2) for t in mid_bc], "tail_bc": [round(t, 2) for t in tail_bc],
            "behaviour": behaviour, "derailed": derailed, "reason": reason, "judged": judged,
            "gemini": _gemini_meta(use_gemini, api_key)}


def grade_benchmark(d, use_gemini=False, api_key=None):
    """benchmark.json reference-answer eval (logic_puzzle / countdown / grammar, etc.): runs grade_live's 4 dimensions
    (timing/overlap/content/silence) + content judging. Needs benchmark.json + A_user/B_model.parakeet.json (+ wav).
    → {category, events, summary:{n,npass,rate}, gemini}."""
    d = Path(d)
    bj = d / "benchmark.json"
    if not bj.exists():
        return {"error": "no benchmark.json (this isn't a reference-answer test)"}
    user_j, model_j = d / "A_user.parakeet.json", d / "B_model.parakeet.json"
    if not user_j.exists() or not model_j.exists():
        return {"error": "missing split-track ASR (A_user/B_model.parakeet.json, run parakeet first)"}
    import grade_live as gl
    bench = json.loads(bj.read_text())
    uw, mw = load_words(user_j), load_words(model_j)
    mwav, uwav = d / "B_model.wav", d / "A_user.wav"
    recs = gl.grade_live(bench, uw, mw,
                         model_wav_path=str(mwav) if mwav.exists() else None,
                         user_wav_path=str(uwav) if uwav.exists() else None,
                         gemini=use_gemini, api_key=api_key)
    graded = [r for r in recs if "content" in r or "timing" in r]
    npass = sum(r.get("status") == "pass" for r in graded)
    return {"category": bench.get("category"), "events": recs,
            "summary": {"n": len(graded), "npass": npass,
                        "rate": round(npass / len(graded), 3) if graded else 0},
            "gemini": _gemini_meta(use_gemini, api_key)}


# task name → grader —— shared by the CLI and the dashboard
TASKS = {
    "backchannel": grade_backchannel,
    "interruption": grade_interruption,
    "turn_taking": grade_turn_taking,
    "pause": grade_pause,
    "user_backchannel": grade_user_backchannel,
    "benchmark": grade_benchmark,
}
