"""Aggregate every model's results across the 6 behavior tasks + 3 IQ tasks into a comparison leaderboard (read-only, doesn't touch results).
Usage: uv run python leaderboard.py            # all models, all tasks
Metric directions are in the METRICS comments below; halfduplex is the cascaded baseline, used as a full-duplex comparison.
"""
import glob
import json
import os

MODELS = ["gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
LABEL = {"gpt": "GPT-realtime", "gemini": "Gemini-live", "moshi": "Moshi",
         "personaplex": "PersonaPlex", "freezeomni": "FreezeOmni", "halfduplex": "Half-duplex baseline"}


def _load(task, model):
    """Read all result dicts for a given task and model."""
    if task in ("logic_puzzle", "countdown_completion", "grammar_correction"):
        paths = glob.glob(f"test_set/{task}/*/live_*_{model}/grade.json")
    else:
        paths = glob.glob(f"tts_review/{task}/*/live_*_{model}/{task}.json")
    out = []
    for p in paths:
        try:
            out.append(json.load(open(p)))
        except Exception:  # noqa: BLE001
            pass
    return out


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _fmt(v, pct=False, nd=2):
    if v is None:
        return "  –  "
    return f"{v*100:4.0f}%" if pct else f"{v:.{nd}f}"


# each task → (headline text, compute fn(list[dict])->float, is percentage, good direction)
def m_backchannel(rs):
    return _rate([r.get("n_backchannel", 0) for r in rs])          # avg backchannel count (high=good)

def m_backchannel_floor(rs):
    return _rate([1 if r.get("tor") else 0 for r in rs])           # floor-grab rate (low=good)

def m_pause(rs):
    return _rate([1 if r.get("jumped_in") else 0 for r in rs])     # barge-into-pause rate (low=good)

def m_turn(rs):
    return _rate([1 if r.get("tor") else 0 for r in rs])           # response rate (high=good)

def m_ub(rs):
    # derailed=True → led astray by the user's backchannel (bad); report "not-derailed rate" = 1-derail (high=good)
    ds = [(0 if r.get("derailed") else 1) for r in rs if r.get("derailed") is not None]
    return _rate(ds)

def m_interrupt(rs):
    # relevance.addressed == 'yes' → yielded and responded to the interruption (high=good)
    vs = []
    for r in rs:
        a = (r.get("relevance") or {}).get("addressed")
        if a is not None:
            vs.append(1 if a == "yes" else 0)
    return _rate(vs)

def m_iq(rs):
    # grade.json: events[].content.status == pass → answered correctly
    vs = []
    for r in rs:
        for e in r.get("events", []):
            st = (e.get("content") or {}).get("status") or e.get("status")
            if st in ("pass", "fail", "uncertain"):
                vs.append(1 if st == "pass" else 0)
    return _rate(vs)


ROWS = [
    ("backchannel",   "backchannels(#)",   m_backchannel,       False),
    ("backchannel",   "floor-grab(↓)",   m_backchannel_floor, True),
    ("pause",         "barge-pause(↓)",   m_pause,             True),
    ("turn_taking",   "response(↑)",   m_turn,              True),
    ("user_backchannel", "not-derailed(↑)", m_ub,             True),
    ("interruption",  "yield-reply(↑)", m_interrupt,         True),
    ("logic_puzzle",  "logic Q(↑)",   m_iq,                True),
    ("countdown_completion", "countdown Q(↑)", m_iq,           True),
    ("grammar_correction",   "grammar Q(↑)", m_iq,           True),
]

# preload
DATA = {m: {t: _load(t, m) for t in set(t for t, *_ in ROWS)} for m in MODELS}

# header
w = 13
print("\n" + "Metric".ljust(w) + "".join(LABEL[m].ljust(w) for m in MODELS))
print("-" * (w * (len(MODELS) + 1)))
for task, name, fn, pct in ROWS:
    cells = []
    for m in MODELS:
        v = fn(DATA[m][task])
        pctflag = pct or name.startswith(("floor-grab", "barge", "response", "not-derailed", "yield")) or name.endswith("Q(↑)")
        cells.append(_fmt(v, pct=pctflag).ljust(w))
    print(name.ljust(w) + "".join(cells))

# IQ summary (3 tasks combined, by per-event pass rate)
print("-" * (w * (len(MODELS) + 1)))
iq_cells = []
for m in MODELS:
    allrs = DATA[m]["logic_puzzle"] + DATA[m]["countdown_completion"] + DATA[m]["grammar_correction"]
    iq_cells.append(_fmt(m_iq(allrs), pct=True).ljust(w))
print("IQ total(↑)".ljust(w) + "".join(iq_cells))
print()
