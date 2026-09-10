"""grade_model.py —— grade one "model response track" with benchmark.json (Track-A real-model eval entry point).

This consumes the **whole-track ASR** (absolute time), structurally identical to
"grade a real model: run one ASR over its output track".

Inputs (one scenario's S3 root + variant):
  benchmark.json           reference answer
  audio_0.parakeet.json    user track (A) whole-track ASR (absolute seconds) → locate the graded trigger word
  audio_1.parakeet.json    ★ "model response track" whole-track ASR (absolute seconds) → locate onset / content
  audio_1.wav              "model response track" audio → Gemini judges content (③)
  timeline.json            only used to decide **which occurrence of the trigger word is the graded one** (the graded trigger's position on the fixed user track is deterministic)

Validation mode (current): the model response track = the reference's own audio_1 —— confirms the grader computes correctly and that whole-track ASR removes the per-utterance stitching offsets.
Real eval: replace audio_1.parakeet.json / audio_1.wav with the **real model output track**'s ASR + audio; everything else unchanged.

Usage:
  python grade_model.py s3://.../<sid>/ --variant A --gemini
  python grade_model.py s3://.../<sid>/ --variant A --model-asr s3://.../model_out.parakeet.json \
                        --model-wav s3://.../model_out.wav          # grade a real model: pass the model track explicitly
"""
import argparse
import io
import json
import os
import re
from urllib.parse import urlparse

from dotenv import load_dotenv; load_dotenv(".env")           # noqa: E402
from pydub import AudioSegment                                 # noqa: E402

import vad_utils                                               # noqa: E402  VAD boundaries (shared with grade_live)

TOL = 0.05
SILENCE_CATS = {"countdown_completion", "keyword_wait",
                "forbidden_phrase_interruption", "deliberate_overlap"}
_M = {"pass": "✅", "fail": "❌", "early": "❌early", "late": "❌late", "uncertain": "❓", "n/a": "—"}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _s3():
    from task import _get_s3_client
    return _get_s3_client()


def _get_json(s3, bucket, key):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def _trigger_word_abs(user_words, trig, win_lo, win_hi):
    """In the user track's whole-track ASR, take the trigger-word occurrence that falls within [win_lo,win_hi] and matches (returns the **word**; caller VAD-snaps t1).
    win is the absolute time range of the "user utterance holding the graded trigger" from the reference — avoids hitting the same word from the rule-setup turn."""
    tn = _norm(trig)
    cands = [w for w in user_words if _norm(w["word"]) == tn and win_lo - 0.3 <= w["t0"] <= win_hi + 0.3]
    if cands:
        return cands[-1], "exact"
    # no match (ASR misspelling) → fall back to the last word in the window
    inwin = [w for w in user_words if win_lo - 0.3 <= w["t0"] <= win_hi + 0.3]
    if inwin:
        return inwin[-1], "last_in_window"
    return None, "not_found"


def _response_onset_abs(model_words, after_t, target):
    """In the model track's whole-track ASR, find the graded response onset (absolute seconds).
    Graded response = the model's speech after the trigger. Allow a little early (INTERRUPT jumps in before the trigger word ends).
    Returns (onset_t0, end_t1, heard_text)."""
    lo = after_t - 0.5                       # allow a slightly early start (before the trigger word finishes); too wide catches pre-trigger speech (e.g. the tail of the previous turn's ack)
    seg = [w for w in model_words if w["t0"] >= lo]
    if not seg:
        return None, None, ""
    # take the "first continuous run of words after the trigger": from the first word ≥lo, break on a >0.8s pause
    onset = seg[0]["t0"]; end = seg[0]["t1"]; words = [seg[0]]
    for w in seg[1:]:
        if w["t0"] - end > 0.8:
            break
        end = w["t1"]; words.append(w)
    return onset, end, " ".join(w["word"] for w in words)


def _gemini_content(bucket, base, variant, model_wav_key, onset, end, event, category, api_key):
    """Cut the model track audio [onset-0.2, end+0.4] and feed it to Gemini to judge content."""
    try:
        import content_judge
        s3 = _s3()
        raw = s3.get_object(Bucket=bucket, Key=model_wav_key)["Body"].read()
        seg = AudioSegment.from_file(io.BytesIO(raw))
        clip = seg[max(0, int((onset - 0.2) * 1000)): int((end + 0.4) * 1000)]
        buf = io.BytesIO(); clip.export(buf, format="wav")
        rub = content_judge._rubric(event, category)
        from google.genai import types
        r = content_judge._client(api_key).models.generate_content(
            model=content_judge.MODEL,
            contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                      rub + ' Return JSON {"heard":"...","content_ok":"pass"|"fail"|"uncertain","reason":"..."}.'],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=content_judge._SCHEMA))
        v = json.loads(r.text)
        return {"status": v.get("content_ok", "uncertain"), "heard": v.get("heard", ""),
                "reason": "gemini: " + (v.get("reason") or "")}
    except Exception as e:  # noqa: BLE001
        return {"status": "uncertain", "heard": "", "reason": f"gemini_err:{type(e).__name__}:{e}"}


def grade(s3, bucket, root, variant, gemini, api_key, model_asr_key=None, model_wav_key=None):
    base = f"{root}/{variant}"
    bj = _get_json(s3, bucket, f"{base}/benchmark.json")
    tl = _get_json(s3, bucket, f"{base}/timeline.json")
    user_asr = _get_json(s3, bucket, f"{base}/audio_0.parakeet.json")["words"]
    model_asr_key = model_asr_key or f"{base}/audio_1.parakeet.json"
    model_wav_key = model_wav_key or f"{base}/audio_1.wav"
    model_asr = _get_json(s3, bucket, model_asr_key)["words"]
    # user track VAD segments: set turn boundaries (ASR last-word timestamps run long and unreliable) —— same logic as grade_live
    try:
        user_segs = vad_utils.vad_segments(
            s3.get_object(Bucket=bucket, Key=f"{base}/audio_0.wav")["Body"].read())
    except Exception:  # noqa: BLE001
        user_segs = None
    category = bj.get("category")
    out = []
    for e in bj.get("expected_events", []):
        if not e.get("graded"):
            continue
        rec = {"event": e["event_id"], "category": category, "action": e.get("expected_action"),
               "grade_dimension": e.get("grade_dimension")}
        # content event (SELF_COMPLETE, no trigger/window): judge content only
        trg = e.get("trigger"); win = e.get("onset_window_ms")
        # absolute time range of the user utterance holding the graded trigger (locate that graded same-word occurrence from the reference timeline)
        ri = e.get("response_line_index")
        if trg and win is not None and ri is not None:
            ti = next((i for i in range(ri - 1, -1, -1)
                       if tl[i]["speaker"] in ("A", "0")), None)
            if ti is None:
                rec["status"] = "no_trigger_line"; out.append(rec); continue
            t_lo, t_hi = tl[ti]["start"], tl[ti]["start"] + tl[ti]["dur"]
            trig_w, tsrc = _trigger_word_abs(user_asr, trg["anchor_word"], t_lo, t_hi)
            if trig_w is None:
                rec["status"] = "no_trigger"; out.append(rec); continue
            trig_end = vad_utils.snap_end(trig_w["t0"], trig_w["t1"], user_segs)   # snap anchor end to VAD
            onset, end, heard = _response_onset_abs(model_asr, trig_end, e.get("target_response"))
            if onset is None:
                rec["timing"] = {"status": "no_response"}; rec["tor"] = 0
                rec["status"] = "fail"; out.append(rec); continue
            rec["tor"] = 1
            # ① timing
            lat = (onset - trig_end) * 1000.0
            t_status = "pass" if win[0] <= lat <= win[1] else ("early" if lat < win[0] else "late")
            rec["window_ms"] = win
            rec["timing"] = {"status": t_status, "latency_ms": round(lat, 1),
                             "trig_end": round(trig_end, 3), "resp_onset": round(onset, 3), "trig_src": tsrc}
            # ② overlap: user's "finished-speaking moment" via VAD (fall back to ASR cap without VAD) vs onset
            a_end = vad_utils.turn_end_vad(user_segs, trig_w) if user_segs else None
            if a_end is None:
                a_end = vad_utils.turn_end_asr(user_asr, trig_w)
            ov = a_end - onset
            exp_ov = e.get("expected_overlap")
            o_status = ("pass" if ov > -TOL else "fail") if exp_ov == "required" else \
                       ("pass" if ov < TOL else "fail") if exp_ov == "forbidden" else "n/a"
            rec["overlap"] = {"status": o_status, "overlap_s": round(ov, 3), "exp": exp_ov}
            # ④ silence: before the graded trigger, the model track shouldn't have this response's onset (approximated here by onset≥trig_end−tol)
            if category in SILENCE_CATS:
                rec["silence"] = {"status": "pass" if onset >= trig_end - TOL else "fail"}
            # ③ content
            if gemini:
                rec["content"] = _gemini_content(bucket, base, variant, model_wav_key, onset, end, e, category, api_key)
            else:
                tgt = _norm(e.get("target_response"))
                rec["content"] = {"status": "pass" if tgt and tgt in _norm(heard) else "uncertain",
                                  "heard": heard, "target": e.get("target_response")}
            ok = (t_status == "pass" and o_status in ("pass", "n/a")
                  and rec["content"]["status"] != "fail"
                  and rec.get("silence", {}).get("status", "pass") == "pass")
            rec["status"] = "pass" if ok else "fail"
        else:
            # content event (SELF_COMPLETE)
            onset = tl[ri]["start"] if ri is not None and ri < len(tl) else 0
            seg = [w for w in model_asr if w["t0"] >= onset - 0.5]
            heard = " ".join(w["word"] for w in seg[:20])
            rec["tor"] = 1 if seg else 0
            if gemini and seg:
                end = seg[min(len(seg), 12) - 1]["t1"]
                rec["content"] = _gemini_content(bucket, base, variant, model_wav_key, seg[0]["t0"], end, e, category, api_key)
            else:
                tgt = _norm(e.get("target_response"))
                rec["content"] = {"status": "pass" if tgt and tgt in _norm(heard) else "uncertain", "heard": heard}
            rec["status"] = "fail" if rec["content"]["status"] == "fail" else "pass"
        out.append(rec)
    return out


def _print_recs(tag, recs):
    print(f"\n── {tag} ──")
    for r in recs:
        if "timing" not in r:
            c = r.get("content", {})
            print(f"  {r['event']} [{r['category']}/{r['action']}] content event → {'PASS' if r['status']=='pass' else 'FAIL'}")
            print(f"      ③ content {_M.get(c.get('status'),'?')} heard={c.get('heard')!r}"
                  + (f"  ({c['reason']})" if c.get('reason') else "")); continue
        t = r["timing"]
        if "latency_ms" not in t:
            print(f"  {r['event']} ⚠️ {t.get('status')}"); continue
        o = r.get("overlap", {}); c = r.get("content", {}); sil = r.get("silence", {}).get("status")
        print(f"  {r['event']} [{r['category']}/{r['action']}] → {'PASS' if r['status']=='pass' else 'FAIL'}")
        print(f"      ① timing {_M.get(t['status'],'?')} {t['latency_ms']}ms  win={r.get('window_ms')}  "
              f"(trig_end={t['trig_end']}s resp={t['resp_onset']}s src={t['trig_src']})")
        print(f"      ② overlap {_M.get(o.get('status'),'?')} overlap={o.get('overlap_s')}s  expected={o.get('exp')}")
        print(f"      ③ content {_M.get(c.get('status'),'?')} heard={c.get('heard')!r}"
              + (f"  ({c['reason']})" if c.get('reason') else ""))
        if sil is not None:
            print(f"      ④ silence {_M.get(sil,'?')}")


def _summary(all_recs):
    import collections
    by = collections.defaultdict(list)
    for r in all_recs:
        by[r["category"]].append(r)
    print("\n=== batch summary (whole-track eval) ===")
    for cat in sorted(by):
        rs = by[cat]
        npass = sum(r.get("status") == "pass" for r in rs)
        def dim(d):
            c = collections.Counter(r[d]["status"] for r in rs if d in r and "status" in r[d])
            return " ".join(f"{_M.get(k,k)}{v}" for k, v in c.most_common()) or "—"
        lats = [r["timing"]["latency_ms"] for r in rs if "timing" in r and "latency_ms" in r["timing"]]
        tor = [r["tor"] for r in rs if "tor" in r]
        med = sorted(lats)[len(lats) // 2] if lats else None
        print(f"  {cat:32s} PASS {npass}/{len(rs)}  ①{dim('timing')} ②{dim('overlap')} ③{dim('content')} ④{dim('silence')}"
              f"  | TOR {sum(tor)}/{len(tor)}" + (f"  lat med {med:.0f}ms" if med is not None else ""))
    tp = sum(r.get("status") == "pass" for r in all_recs)
    print(f"\n  total event PASS {tp}/{len(all_recs)} = {tp/max(len(all_recs),1):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s3_uri", nargs="?", help="single scenario root s3://bucket/.../<sid>/")
    ap.add_argument("--variant", default="A")
    ap.add_argument("--prompt-key", dest="prompt_key", default=None, help="batch: sample N per category by prompt_key")
    ap.add_argument("--per-cat", dest="per_cat", type=int, default=2)
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--model-asr", dest="model_asr", default=None, help="real model output track whole-track ASR (default = reference audio_1.parakeet.json)")
    ap.add_argument("--model-wav", dest="model_wav", default=None, help="real model output track audio (default = reference audio_1.wav)")
    args = ap.parse_args()

    api_key = None
    if args.gemini:
        api_key = os.environ.get("GEMINI_API_KEY")  # loaded from .env at import time
    s3 = _s3()

    if args.prompt_key:
        from asr_tracks import _roots_from_prompt_key
        roots = _roots_from_prompt_key(args.prompt_key, args.per_cat, [args.variant])
        all_recs = []
        for bucket, root in roots:
            try:
                recs = grade(s3, bucket, root, args.variant, args.gemini, api_key, args.model_asr, args.model_wav)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {root.split('/')[-1][:8]}: {type(e).__name__}: {e}"); continue
            all_recs.extend(recs)
        _summary(all_recs)
    elif args.s3_uri:
        u = urlparse(args.s3_uri); bucket, root = u.netloc, u.path.strip("/")
        recs = grade(s3, bucket, root, args.variant, args.gemini, api_key, args.model_asr, args.model_wav)
        _print_recs(f"{root}  variant {args.variant}", recs)
    else:
        ap.error("provide an s3_uri, or --prompt-key for batch")


if __name__ == "__main__":
    main()
