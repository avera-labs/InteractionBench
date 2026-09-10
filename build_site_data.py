"""Prepare data for the GitHub Pages showcase site.

Produces:
  docs/audio/<task>/<item>/<system>.mp3   —— combined track of the featured example (converted to mp3)
  docs/data/manifest.json                 —— leaderboards + featured examples (verdicts/transcripts/dual-track timeline)

Example selection: per task, pick the item with the biggest good/bad split ACROSS systems, so the
winner/loser is obvious on a single listen.
Leaderboards: aggregated over every item per task (not just the featured examples).
"""
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"                    # docs/ at the repo root (GitHub Pages only serves /docs from the root)
AUD = DOCS / "audio"
DATA = DOCS / "data"

SYSTEMS = ["gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
SYS_LABEL = {"gpt": "GPT-Realtime", "gemini": "Gemini-Live", "moshi": "Moshi",
             "personaplex": "PersonaPlex", "freezeomni": "FreezeOmni",
             "halfduplex": "Turn-based cascade"}
BEHAV = ["backchannel", "pause", "turn_taking", "user_backchannel", "interruption"]
# IQ = the combined "spoken-task accuracy" table (per task, reads grade.json's summary.rate).
# grammar_correction = the non-native-speaker version with assorted accents (the original native-English
# version is archived as grammar_correction_old, not on the board); keyword_wait/stay_quiet are restraint types.
IQ = ["logic_puzzle", "countdown_completion", "grammar_correction", "keyword_wait", "stay_quiet_until_help"]
ALT = "alternating_count"                        # single turn-discipline variant, usually shown under the logic_puzzle group (not on the IQ table)
PARA = ["whisper_production", "volume_understanding"]   # paralinguistic dimensions (breathy whisper / volume understanding), their own board + examples
IG = "interaction_groundedness"                  # long multi-turn logic grounding (probes: nonsense/state/unknown/constraint...), its own board + examples
GRAMMAR = ("grammar_correction",)                # grammar proactive-correction task (internal category name grammar_correction)
# pin specific items to display for a task (these were re-run with correct timing; don't let auto-selection drop them when a verdict changes)
PINNED = {"user_backchannel": ["17", "29"], "volume_understanding": ["03", "01"]}   # in 01, gemini softens to a whisper too, which demos well
REASON_CONTENT = ("logic_puzzle", "countdown_completion", "grammar_correction", "keyword_wait")
K_PER_TASK = 2                                   # how many examples to feature per task

# ---------------------------------------------------------------- per-item verdict
def _num(x):
    return x if isinstance(x, (int, float)) else None


def eval_item(task, j):
    """Read one result json → {good, badge, cls, metric, heard}. badge is English (matches the page). cls ∈ good/bad/warn."""
    if task == "backchannel":
        nb, tor, nr = j.get("n_backchannel", 0), j.get("tor", 0), j.get("n_response", 0)
        if nr == 0 and nb == 0:
            return dict(good=False, badge="no reply", cls="warn", metric="—", heard="")
        good = nb >= 1 and not tor
        badge = (f"{nb} backchannel" + ("s" if nb != 1 else "") if nb else "no backchannel") + (" · took floor" if tor else "")
        heard = " ".join(s.get("words", "") for s in j.get("segments", [])[:2])
        return dict(good=good, badge=badge, cls="good" if good else "bad", metric=f"bc={nb}", heard=heard)
    if task == "pause":
        ji = j.get("jumped_in")
        good = ji is False
        badge = "barged in on pause" if ji else "waited patiently"
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=f"pause={j.get('pause_dur')}s", heard=j.get("heard_in_pause", "") or "")
    if task == "turn_taking":
        # Clean turn-taking = wait for the user to finish, then reply promptly. A negative latency
        # means the model was already talking when the user stopped (talked over / never yielded) —
        # the opposite of taking a clean turn, so it does NOT count as good/fast.
        tor, lat = j.get("tor", 0), _num(j.get("latency_ms"))
        heard = j.get("heard_after", "") or ""
        if not tor or lat is None:
            return dict(good=False, badge="no reply", cls="bad", metric="—", heard=heard)
        if lat < -50:                                    # started before the user finished → overlap
            return dict(good=False, badge=f"talked over ({int(lat)}ms)", cls="warn",
                        metric=f"{int(lat)}ms", heard=heard)
        good = lat <= 2500
        badge = f"replied in {int(lat)}ms" if good else f"slow reply ({int(lat)}ms)"
        return dict(good=good, badge=badge, cls="good" if good else "warn",
                    metric=f"{int(lat)}ms", heard=heard)
    if task == "user_backchannel":
        der = j.get("derailed")
        good = der is False
        badge = "derailed / cut off" if der else "stayed on track"
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=("derailed" if der else "ok"), heard=j.get("reply_text", "") or "")
    if task == "interruption":
        wt, rel = j.get("was_talking"), (j.get("relevance") or {}).get("addressed")
        good = bool(wt) and rel == "yes"
        if not wt and rel == "yes":
            badge = "answered (not mid-speech)"
            cls = "warn"
        elif good:
            badge = "yielded & answered"
            cls = "good"
        else:
            badge = "ignored interrupt"
            cls = "bad"
        return dict(good=good, badge=badge, cls=cls, metric=f"addr={rel}", heard=j.get("heard_after", "") or "")
    if task in GRAMMAR:
        # proactive interrupt-correction = a timing task, not a pure content one: report timing failures honestly (don't label them wrong/off-topic)
        ev = (j.get("events") or [{}])[0]
        tim = ev.get("timing") or {}
        lat = _num(tim.get("latency_ms"))
        good = ev.get("status") == "pass"
        heard = ((ev.get("content") or {}).get("heard")) or ""
        if good:
            badge = "corrected in time"
        elif not ev.get("tor"):
            badge = "never corrected"
        elif lat is not None and lat > 300:            # only report seconds when clearly late
            badge = f"waited {lat/1000:.1f}s — no interrupt"
        else:
            badge = "no proactive correction"           # abnormal timing (negative / near-zero)
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=(f"{int(lat)}ms" if lat is not None else "—"), heard=heard)
    if task == "keyword_wait":
        # composite dimension: carried out the interrupt (took the floor, responded after the keyword) + answered correctly (said the target word)
        ev = (j.get("events") or [{}])[0]
        st = ev.get("status")
        heard = (ev.get("content") or {}).get("heard", "") if isinstance(ev.get("content"), dict) else ""
        if st == "no_response":
            return dict(good=False, badge="no reply", cls="warn", metric="—", heard="")
        good = st == "pass"
        if good:
            badge = "interjected & correct"
        elif ev.get("content_ok") != "pass":
            badge = "wrong / off-topic word"
        elif ev.get("interrupt") != "pass":
            badge = "never interjected"
        else:
            badge = "missed"
        lat = _num(ev.get("latency_ms"))
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=(f"{int(lat)}ms" if lat is not None else "—"), heard=heard)
    if task == "stay_quiet_until_help":
        # restraint (holds back until asked) + content (answers correctly once asked)
        ev = (j.get("events") or [{}])[0]
        st = ev.get("status")
        if st == "no_response":
            return dict(good=False, badge="no reply", cls="warn", metric="—", heard="")
        good = st == "pass"
        if good:
            badge = "stayed quiet, then answered"
        elif ev.get("restraint") != "pass":
            badge = "barged in before asked"
        elif (ev.get("content_ok") or ev.get("content")) != "pass":
            badge = "wrong answer"
        else:
            badge = "missed"
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=f"restraint={ev.get('restraint')}", heard=ev.get("post_heard", "") or "")
    if task == ALT:
        # sequence (right order) + discipline (one per turn, no reciting the count in a row)
        ev = (j.get("events") or [{}])[0]
        st = ev.get("status")
        if st == "no_response":
            return dict(good=False, badge="no reply", cls="warn", metric="—", heard="")
        good = st == "pass"
        df = ev.get("discipline_fail")
        if good:
            badge = "took turns cleanly"
        elif ev.get("sequence") != "pass":
            badge = "wrong / missing count"
        elif df == "reciting":
            badge = "recited, didn't alternate"
        elif df == "blurt":
            badge = "blurted all at once"
        else:
            badge = "off-pattern"
        said = ev.get("said") or []
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=ev.get("n_correct", "—"), heard=" ".join(str(x) for x in said))
    if task == "whisper_production":
        # whether it breathily whispers on command (HNR<6 & voiced<0.4) + answers correctly
        ev = (j.get("events") or [{}])[0]
        st = ev.get("status")
        if st == "no_response":
            return dict(good=False, badge="no reply", cls="warn", metric="—", heard="")
        wok = ev.get("whisper_ok") == "pass"
        cok = ev.get("content_ok") == "pass"
        good = st == "pass"
        if good:
            badge = "whispered & answered"
        elif cok and not wok:
            badge = "answered aloud — didn't whisper"
        elif wok and not cok:
            badge = "whispered but wrong"
        else:
            badge = "no whisper, wrong"
        hnr = ev.get("hnr")
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=(f"HNR {hnr}dB" if hnr is not None else "—"), heard=ev.get("heard", "") or "")
    if task == "volume_understanding":
        # understands the volume pragmatics (judged from audio) + adapts (softens when whispered to)
        ev = (j.get("events") or [{}])[0]
        st = ev.get("status")
        if st == "no_response":
            return dict(good=False, badge="no reply", cls="warn", metric="—", heard="")
        understood = ev.get("understanding") == "pass"
        cond = ev.get("condition")
        adapt = ev.get("adaptation")
        good = understood
        if good:
            badge = "understood" + (" & softened voice" if adapt == "pass" else "")
        else:
            badge = "missed the cue / off-topic"
        return dict(good=good, badge=badge, cls="good" if good else "bad",
                    metric=("🔊 loud" if cond == "loud" else "🔈 whisper"), heard=ev.get("heard", "") or "")
    # IQ (logic / countdown: pure content)
    rate = (j.get("summary") or {}).get("rate", 0)
    good = rate >= 0.999
    ev = (j.get("events") or [{}])[0]
    heard = ((ev.get("content") or {}).get("heard")) or ""
    return dict(good=good, badge="correct" if good else "wrong / off-topic", cls="good" if good else "bad",
                metric=f"rate={rate:.0%}", heard=heard)


# ---------------------------------------------------------------- dirs/files
def is_graded(task):
    """Tasks that use grade.json + the test_set root (all of IQ + alternating + paralinguistic). Behavior tasks use <task>.json + tts_review."""
    return task in IQ or task == ALT or task in PARA


def result_name(task):
    return "grade.json" if is_graded(task) else f"{task}.json"


def judge_reason(task, j):
    """The judge's reason for this item (IQ = content judge; interruption = relevance; user_backchannel = derailment reason)."""
    if task in REASON_CONTENT:
        c = (j.get("events") or [{}])[0].get("content")
        r = c.get("reason", "") if isinstance(c, dict) else ""   # stay_quiet's content is a string / alternating has no content
    elif task == "interruption":
        r = (j.get("relevance") or {}).get("reason", "")
    elif task == "user_backchannel":
        r = j.get("reason", "") or ""
    elif task == "volume_understanding":                    # reason from the audio-understanding judge
        r = (j.get("events") or [{}])[0].get("reason", "") or ""
    else:
        r = ""
    r = (r or "").strip()
    for pre in ("gemini(audio): ", "gemini(audio):", "gemini(text): ", "gemini(text):", "gemini: ", "gemini:"):   # strip the judge prefix
        if r.startswith(pre):
            r = r[len(pre):].strip()
            break
    return r


def item_dirs(task):
    """{item_id: {system: folder}}, keeping only items where all 6 systems are present (folder, not zip) and have combined.wav."""
    root = "test_set" if is_graded(task) else "tts_review"
    out = {}
    for sysn in SYSTEMS:
        for d in glob.glob(f"{root}/{task}/*/live_*_{sysn}"):
            item = Path(d).parent.name
            if os.path.exists(f"{d}/combined.wav") and os.path.exists(f"{d}/{result_name(task)}"):
                out.setdefault(item, {})[sysn] = d
    return {it: m for it, m in out.items() if all(s in m for s in SYSTEMS)}


# ---------------------------------------------------------------- timeline
def speech_segs(path, th=0.012, hop=0.05, merge=0.4):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a[:, 0]
    w = int(hop * sr)
    env = np.array([np.sqrt(np.mean(a[i:i + w] ** 2)) for i in range(0, max(1, len(a) - w), w)])
    on = env > th
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and (on[j] or (j + 4 < len(on) and on[j:j + 5].any())):
                j += 1
            segs.append([round(i * hop, 2), round(j * hop, 2)])
            i = j
        else:
            i += 1
    m = []
    for s, e in segs:
        if m and s - m[-1][1] < merge:
            m[-1][1] = e
        else:
            m.append([s, e])
    return m, round(len(a) / sr, 2)


def overlaps(u, b):
    ov = []
    for bs, be in b:
        for us, ue in u:
            lo, hi = max(bs, us), min(be, ue)
            if hi - lo > 0.15:
                ov.append([round(lo, 2), round(hi, 2)])
    return ov


def timeline(folder):
    u, dur = speech_segs(f"{folder}/A_user.wav")
    b, durb = speech_segs(f"{folder}/B_model.wav")
    ov = overlaps(u, b)
    ov_s = round(sum(e - s for s, e in ov), 1)
    return dict(dur=max(dur, durb), user=u, model=b, overlap=ov, overlap_s=ov_s)


def track_words(folder, fname, cap=90):
    """Word-level ASR (parakeet) for one track → [[word, t0, t1], ...], for the page's word-by-word highlight sync."""
    p = Path(folder) / fname
    if not p.exists():
        return []
    ws = json.loads(p.read_text()).get("words", [])[:cap]
    return [[w["word"], round(w["t0"], 2), round(w["t1"], 2)] for w in ws]


def _norm(w):
    return re.sub(r"[^a-z0-9]", "", (w or "").lower())


def ig_dialogue(folder, user_texts):
    """Clean multi-turn dialogue structure for long conversations (rendered directly by the site), strictly USER→MODEL→USER→MODEL:
    ① user turns: the user speaks FIXED SCRIPT TEXT (TTS), so align the ASR words back to the script by content
       → cut exactly on each turn boundary (overlap/compression eats the sentence-level gaps, timestamps alone
       can't separate turns, content alignment is what holds up).
    ② model turns: EVERYTHING the model says AFTER a user turn and BEFORE the next user turn = that turn's reply
       (bucketed by time window, not split on the model's own pauses — otherwise it misaligns / bleeds across
       turns). If the model was silent in a turn, that turn has no MODEL line."""
    u = track_words(folder, "A_user.parakeet.json", cap=600)
    m = track_words(folder, "B_model.parakeet.json", cap=600)
    if not u:
        return []
    # ① content-align to cut user turns (order-preserving, with empty-turn placeholders to keep time windows aligned)
    if user_texts:
        stoks = [(ti, _norm(w)) for ti, t in enumerate(user_texts) for w in (t or "").split()]
        buckets = [[] for _ in user_texts]
        si = 0
        for aw in u:
            an = _norm(aw[0]); ti = None
            for k in range(si, min(si + 5, len(stoks))):
                if stoks[k][1] and stoks[k][1] == an:
                    ti = stoks[k][0]; si = k + 1; break
            if ti is None:
                ti = stoks[min(si, len(stoks) - 1)][0] if stoks else 0
            buckets[ti].append(aw)
        uturns = [b for b in buckets if b]
    else:
        uturns = [u]
    # ② bucket the model by time window: model words fall by start time into "before the nearest user turn that starts after them" → that user turn's reply
    starts = [ut[0][1] for ut in uturns]
    out = []
    for i, ut in enumerate(uturns):
        out.append(dict(spk="u", t0=ut[0][1], t1=ut[-1][2],
                        words=[[w[0], w[1], w[2]] for w in ut]))
        lo = starts[i]
        hi = starts[i + 1] if i + 1 < len(starts) else float("inf")
        mws = [w for w in m if lo <= w[1] < hi]
        if mws:
            out.append(dict(spk="m", t0=mws[0][1], t1=mws[-1][2],
                            words=[[w[0], w[1], w[2]] for w in mws]))
        else:                                          # model silent this turn → placeholder, keeps the turn-by-turn cadence
            out.append(dict(spk="m", t0=ut[-1][2], t1=ut[-1][2], words=[], silent=True))
    return out


def to_mp3(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # idempotent but never stale: skip only when the mp3 already exists and is not older than the source; re-convert as soon as the source changes
    if dst.exists() and dst.stat().st_size > 0 and dst.stat().st_mtime >= Path(src).stat().st_mtime:
        return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", "1", "-b:a", "64k", str(dst)], check=True)


# ---------------------------------------------------------------- aggregate leaderboards
def leaderboard(task):
    """Aggregate per-system metrics over every item. Returns {system: {...metrics}}."""
    root = "test_set" if is_graded(task) else "tts_review"
    agg = {}
    for sysn in SYSTEMS:
        vals = []
        for d in glob.glob(f"{root}/{task}/*/live_*_{sysn}"):
            rf = f"{d}/{result_name(task)}"
            if os.path.exists(rf):
                try:
                    vals.append(json.load(open(rf)))
                except Exception:  # noqa: BLE001
                    pass
        agg[sysn] = summarize(task, vals)
    return agg


def summarize(task, vals):
    n = len(vals)
    if not n:
        return dict(n=0)
    def mean(f):
        xs = [f(v) for v in vals if f(v) is not None]
        return round(sum(xs) / len(xs), 3) if xs else None
    # unified pass_rate: test_set uses events[0].status, behavior uses eval_item.good (the same verdict as the samples) — feeds the leaderboard card ranking
    if task in IQ:
        npass = sum(1 for v in vals if (v.get("events") or [{}])[0].get("status") == "pass")
        pr = round(npass / n, 3)
    else:
        npass = sum(1 for v in vals if eval_item(task, v).get("good"))
        pr = round(npass / n, 3)
    base = dict(n=n, pass_rate=pr, npass=npass)
    if task == "backchannel":
        base.update(avg_bc=mean(lambda v: v.get("n_backchannel", 0)),
                    floor_take=mean(lambda v: 1.0 if v.get("tor") else 0.0))
    elif task == "pause":
        base.update(bargein=mean(lambda v: 1.0 if v.get("jumped_in") else 0.0))
    elif task == "turn_taking":
        def _clean(v):                                   # replied AND started after the user finished (within window)
            l = _num(v.get("latency_ms"))
            return bool(v.get("tor")) and l is not None and -50 <= l <= 2500
        base.update(resp=mean(lambda v: 1.0 if v.get("tor") else 0.0),
                    clean=mean(lambda v: 1.0 if _clean(v) else 0.0),
                    med_lat=median([v.get("latency_ms") for v in vals            # clean-start latency only (drop overlaps)
                                    if _num(v.get("latency_ms")) is not None and v.get("latency_ms") >= -50]))
    elif task == "user_backchannel":
        base.update(derail=mean(lambda v: 1.0 if v.get("derailed") else 0.0))
    elif task == "interruption":
        base.update(yielded=mean(lambda v: 1.0 if v.get("was_talking") else 0.0),
                    addressed=mean(lambda v: 1.0 if (v.get("relevance") or {}).get("addressed") == "yes" else 0.0))
    return base


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return round((xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2), 1)


def rank_systems(task, lb):
    """Rank the 6 systems best→worst by this category's leaderboard metric (winner on top in the sample grid)."""
    def key(s):
        m = lb.get(s, {})
        if task == "backchannel":                       # more backchannels + less floor-grabbing
            return (-(m.get("avg_bc") or 0), m.get("floor_take") if m.get("floor_take") is not None else 1)
        if task == "pause":                             # fewer barge-ins
            return (m.get("bargein") if m.get("bargein") is not None else 1,)
        if task == "turn_taking":                       # more clean turn starts, then lower latency
            return (-(m.get("clean") or 0), m.get("med_lat") if m.get("med_lat") is not None else 9e9)
        if task == "user_backchannel":                  # less derailed
            return (m.get("derail") if m.get("derail") is not None else 1,)
        if task == "interruption":                      # more interrupts caught and addressed
            return (-(m.get("addressed") or 0), -(m.get("yielded") or 0))
        return (-(m.get("npass") or 0),)                # IQ: more correct
    return sorted(SYSTEMS, key=key)


# ---------------------------------------------------------------- main flow
def timing_totals():
    """The total column of the timing table: per system, the count "handled correctly" / total across all items of the 5 timing scenarios (symmetric with reasoning's TOTAL)."""
    r = {s: {"good": 0, "n": 0} for s in SYSTEMS}
    for task in BEHAV:
        for s in SYSTEMS:
            for d in glob.glob(f"tts_review/{task}/*/live_*_{s}"):
                rf = f"{d}/{result_name(task)}"
                if not os.path.exists(rf):
                    continue
                try:
                    e = eval_item(task, json.load(open(rf)))
                except Exception:  # noqa: BLE001
                    continue
                r[s]["n"] += 1
                if e["good"]:
                    r[s]["good"] += 1
    return r


def para_metrics(task):
    """Per-system metrics for a paralinguistic task (aggregated over every item, does not require all 6 systems — if freezeomni is missing one, it still counts its own)."""
    r = {}
    for s in SYSTEMS:
        evs = []
        for d in glob.glob(f"test_set/{task}/*/live_*_{s}"):
            gf = f"{d}/grade.json"
            if os.path.exists(gf):
                try:
                    evs.append((json.load(open(gf)).get("events") or [{}])[0])
                except Exception:  # noqa: BLE001
                    pass
        n = len(evs) or 1
        nr = sum(1 for e in evs if e.get("status") == "no_response")   # no-response count (more is worse)
        if task == "whisper_production":
            npass = sum(1 for e in evs if e.get("status") == "pass")
            r[s] = dict(n=len(evs), whisper=sum(1 for e in evs if e.get("whisper_ok") == "pass"),
                        npass=npass, no_response=nr, pass_rate=round(npass / n, 3))   # pass_rate feeds the Per-task card
        else:  # volume_understanding
            und = sum(1 for e in evs if e.get("understanding") == "pass")
            r[s] = dict(n=len(evs), understanding=und,
                        adapt=sum(1 for e in evs if e.get("adaptation") == "pass"),
                        adapt_n=sum(1 for e in evs if e.get("condition") == "whisper"),
                        no_response=nr, pass_rate=round(und / n, 3))
    return r


def ig_eval(j):
    """Verdict for one interaction_groundedness item: probe pass rate + coherence → good/warn/bad (colors the sample grid)."""
    s = j.get("summary") or {}
    n = s.get("n") or 0
    npass = s.get("npass") or 0
    rate = s.get("rate") if s.get("rate") is not None else (npass / n if n else 0)
    cls = "good" if rate >= 0.6 else ("warn" if rate >= 0.3 else "bad")
    return dict(good=rate >= 0.6, cls=cls, npass=npass, n_probes=n, rate=round(rate, 3),
                coherence=j.get("coherence"), verdict=j.get("verdict", ""), events=j.get("events") or [])


def ig_metrics():
    """Per-system aggregate of interaction_groundedness: probe pass rate + average coherence."""
    r = {}
    for s in SYSTEMS:
        rows = []
        for d in glob.glob(f"test_set/{IG}/*/live_*_{s}"):
            gf = f"{d}/grade.json"
            if os.path.exists(gf):
                try:
                    rows.append(json.load(open(gf)))
                except Exception:  # noqa: BLE001
                    pass
        npass = sum((v.get("summary") or {}).get("npass", 0) for v in rows)
        nprobe = sum((v.get("summary") or {}).get("n", 0) for v in rows)
        cohs = [v.get("coherence") for v in rows if isinstance(v.get("coherence"), (int, float))]
        r[s] = dict(n=len(rows), npass=npass, n_probes=nprobe,
                    pass_rate=round(npass / nprobe, 3) if nprobe else 0,
                    coherence=round(sum(cohs) / len(cohs)) if cohs else None)
    return r


def ig_item_dirs():
    """{item: {system: folder}}; for interaction, only accept ACTUALLY-GRADED items (grade.json's summary.n>0) —
    avoids empty shells (n=0) that just finished running but haven't gone through grade_interaction showing up as 0/0 in the examples."""
    out = {}
    for sysn in SYSTEMS:
        for d in glob.glob(f"test_set/{IG}/*/live_*_{sysn}"):
            item = Path(d).parent.name
            gf = f"{d}/grade.json"
            if not (os.path.exists(f"{d}/combined.wav") and os.path.exists(gf)):
                continue
            try:
                if (json.load(open(gf)).get("summary") or {}).get("n", 0) > 0:
                    out.setdefault(item, {})[sysn] = d
            except Exception:  # noqa: BLE001
                pass
    return out


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {"systems": [{"id": s, "label": SYS_LABEL[s]} for s in SYSTEMS],
                "behav_tasks": BEHAV, "iq_tasks": IQ, "para_tasks": PARA,
                "timing_totals": timing_totals(), "leaderboards": {}, "examples": {}}

    for task in BEHAV + IQ:
        manifest["leaderboards"][task] = leaderboard(task)
        dirs = item_dirs(task)
        # compute each item's good/bad "split"
        scored = []
        for item, m in dirs.items():
            evals = {}
            for s, d in m.items():
                jsn = json.load(open(f"{d}/{result_name(task)}"))
                e = eval_item(task, jsn)
                e["reason"] = judge_reason(task, jsn)
                evals[s] = e
            ngood = sum(1 for e in evals.values() if e["good"])
            spread = min(ngood, len(SYSTEMS) - ngood)          # closer to a 50/50 split is better
            scored.append((spread, ngood, item, m, evals))
        scored.sort(key=lambda x: (-x[0], -abs(3 - x[1])))     # bigger split first
        if task in PINNED:                                     # pin the specified items (in PINNED order), skip auto-selection
            want = PINNED[task]
            picks = sorted((s for s in scored if s[2] in want), key=lambda s: want.index(s[2]))
        else:
            picks = scored[:K_PER_TASK]
        ranked = rank_systems(task, manifest["leaderboards"][task])   # global rank, used only as a tiebreak within the same verdict
        gpos = {s: i for i, s in enumerate(ranked)}
        cls_rank = {"good": 0, "warn": 1, "bad": 2}
        ex_list = []
        for spread, ngood, item, m, evals in picks:
            # order by "this item's performance": good→warn→bad, then by global strength within a tier, so green sits above red and matches the verdict
            order = sorted(SYSTEMS, key=lambda s: (cls_rank.get(evals[s]["cls"], 3), gpos.get(s, 99)))
            cells = []
            for s in order:
                d = m[s]
                mp3 = AUD / task / item / f"{s}.mp3"
                to_mp3(f"{d}/combined.wav", mp3)
                cells.append(dict(system=s, audio=f"audio/{task}/{item}/{s}.mp3",
                                  **evals[s], timeline=timeline(d),
                                  words=track_words(d, "B_model.parakeet.json"),
                                  uwords=track_words(d, "A_user.parakeet.json")))
            ex_list.append(dict(item=item, spread=spread, cells=cells))
            print(f"  {task}/{item}: spread={spread} ngood={ngood}")
        manifest["examples"][task] = ex_list
        print(f"[{task}] picked {len(ex_list)} items, leaderboard {len(manifest['leaderboards'][task])} systems")

    # ---- alternating_count: turn-discipline variant, usually shown under the Logic puzzle group (not on the IQ /50 table) ----
    alt_lb = leaderboard(ALT)
    manifest["leaderboards"][ALT] = alt_lb
    a_gpos = {s: i for i, s in enumerate(rank_systems(ALT, alt_lb))}
    cls_rank = {"good": 0, "warn": 1, "bad": 2}
    for item, m in sorted(item_dirs(ALT).items()):
        evals = {}
        for s, d in m.items():
            jsn = json.load(open(f"{d}/{result_name(ALT)}"))
            e = eval_item(ALT, jsn)
            e["reason"] = judge_reason(ALT, jsn)
            evals[s] = e
        order = sorted(SYSTEMS, key=lambda s: (cls_rank.get(evals[s]["cls"], 3), a_gpos.get(s, 99)))
        cells = []
        for s in order:
            d = m[s]
            mp3 = AUD / ALT / item / f"{s}.mp3"
            to_mp3(f"{d}/combined.wav", mp3)
            cells.append(dict(system=s, audio=f"audio/{ALT}/{item}/{s}.mp3",
                              **evals[s], timeline=timeline(d),
                              words=track_words(d, "B_model.parakeet.json"),
                              uwords=track_words(d, "A_user.parakeet.json")))
        blk = dict(item=item, spread=0, cells=cells,
                   label="Alternating count · turn-taking discipline (always shown)")
        manifest["examples"].setdefault("logic_puzzle", []).append(blk)
        print(f"  [alternating_count] appended to the logic_puzzle group: item {item}")

    # ---- paralinguistic (whisper production / volume understanding): own board + examples ----
    cls_rank = {"good": 0, "warn": 1, "bad": 2}
    for task in PARA:
        manifest["leaderboards"][task] = para_metrics(task)
    # overall strength rank (used for within-tier ordering in the sample grid, so the worst sinks to the bottom): whisper*2 + understanding + adapt − no-response
    _wlb, _vlb = manifest["leaderboards"]["whisper_production"], manifest["leaderboards"]["volume_understanding"]
    def _pscore(s):
        w, v = _wlb.get(s, {}), _vlb.get(s, {})
        return (w.get("whisper", 0) * 2 + v.get("understanding", 0) + v.get("adapt", 0)
                - w.get("no_response", 0) - v.get("no_response", 0))
    p_ranked = sorted(SYSTEMS, key=lambda s: -_pscore(s))
    pgpos = {s: i for i, s in enumerate(p_ranked)}
    for task in PARA:
        dirs = item_dirs(task)                              # only pick items with all 6 systems for the sample grid
        scored = []
        for item, m in dirs.items():
            evals = {}
            for s, d in m.items():
                jsn = json.load(open(f"{d}/{result_name(task)}"))
                e = eval_item(task, jsn)
                e["reason"] = judge_reason(task, jsn)
                evals[s] = e
            ngood = sum(1 for e in evals.values() if e["good"])
            spread = min(ngood, len(SYSTEMS) - ngood)
            scored.append((spread, ngood, item, m, evals))
        scored.sort(key=lambda x: (-x[0], -abs(3 - x[1])))
        if task in PINNED:                                 # pin the specified items (in order)
            want = PINNED[task]
            picks = sorted((s for s in scored if s[2] in want), key=lambda s: want.index(s[2]))
        else:
            picks = scored[:K_PER_TASK]
        ex_list = []
        for spread, ngood, item, m, evals in picks:
            # within a tier (good→bad), order by overall strength: strongest on top, worst (freezeomni) at the bottom
            order = sorted(SYSTEMS, key=lambda s: (cls_rank.get(evals[s]["cls"], 3), pgpos.get(s, 99)))
            cells = []
            for s in order:
                d = m[s]
                mp3 = AUD / task / item / f"{s}.mp3"
                to_mp3(f"{d}/combined.wav", mp3)
                cells.append(dict(system=s, audio=f"audio/{task}/{item}/{s}.mp3",
                                  **evals[s], timeline=timeline(d),
                                  words=track_words(d, "B_model.parakeet.json"),
                                  uwords=track_words(d, "A_user.parakeet.json")))
            ex_list.append(dict(item=item, spread=spread, cells=cells))
        manifest["examples"][task] = ex_list
        print(f"[{task}] picked {len(ex_list)} items, leaderboard {len(manifest['leaderboards'][task])} systems")

    # ---- Interaction Groundedness: long multi-turn logic grounding. Own board + examples (probe verdicts + dual-track audio) ----
    manifest["ig_task"] = IG
    manifest["leaderboards"][IG] = ig_metrics()
    ig_lb = manifest["leaderboards"][IG]
    ig_ranked = sorted(SYSTEMS, key=lambda s: -(ig_lb.get(s, {}).get("pass_rate") or 0))
    ig_gpos = {s: i for i, s in enumerate(ig_ranked)}
    cls_rank = {"good": 0, "warn": 1, "bad": 2}
    dirs = ig_item_dirs()
    scored = []                                            # pick the scenarios with the biggest across-system split in probe pass rate as examples
    for item, m in dirs.items():
        evals = {s: ig_eval(json.load(open(f"{d}/grade.json"))) for s, d in m.items()}
        ngood = sum(1 for e in evals.values() if e["good"])
        spread = min(ngood, len(evals) - ngood)
        scored.append((spread, ngood, item, m, evals))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    ig_examples = []
    for spread, ngood, item, m, evals in scored[:2]:
        sc_meta = json.load(open(f"{ROOT}/test_set/{IG}/{item}/benchmark.json"))
        u_texts = [t["text"] for t in sc_meta.get("turns", []) if t.get("speaker") == "user"]
        order = sorted(m.keys(), key=lambda s: (cls_rank.get(evals[s]["cls"], 3), ig_gpos.get(s, 99)))
        cells = []
        for s in order:
            d = m[s]
            e = evals[s]
            mp3 = AUD / IG / item / f"{s}.mp3"
            to_mp3(f"{d}/combined.wav", mp3)
            coh = e.get("coherence")
            badge = f'{e["npass"]}/{e["n_probes"]} grounded' + (f' · coherence {coh}' if coh is not None else "")
            cells.append(dict(system=s, audio=f"audio/{IG}/{item}/{s}.mp3",
                              badge=badge, reason=e.get("verdict", ""), **e, timeline=timeline(d),
                              dialogue=ig_dialogue(d, u_texts),
                              words=track_words(d, "B_model.parakeet.json", cap=400),
                              uwords=track_words(d, "A_user.parakeet.json", cap=400)))
        ig_examples.append(dict(item=item, spread=spread, title=sc_meta.get("title"), cells=cells))
        print(f"  [{IG}]/{item}: spread={spread} ngood={ngood} ({sc_meta.get('title')})")
    manifest["examples"][IG] = ig_examples
    print(f"[{IG}] leaderboard {sum(1 for s in ig_lb if ig_lb[s]['n'])} systems, {len(ig_examples)} examples")

    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {DATA/'manifest.json'}; audio in {AUD}")


if __name__ == "__main__":
    main()
