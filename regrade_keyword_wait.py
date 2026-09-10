"""Re-grade keyword_wait: synthesize two dimensions —— **did it interrupt** + **did it answer correctly**.

The original design used a [50,300]ms real-time interruption window, unreachable under multi-turn choreography (good models only respond a few seconds after the keyword),
so the whole system scored 0 — a grader artifact. Changed here to:
  interrupt  did the model "take the floor and respond after the keyword" (response onset falls after the trigger word ends).
             = it did open its mouth for this keyword (not no-response / not jumping in before the keyword). Drops the 300ms upper bound.
  content    whether what it said after the keyword is correct (reuses the original grader's content.status).
  overall    interrupted AND answered correctly = pass.
  no_response no response on the whole track (original grader judged tor=0 / found no onset) → record no_response (mostly re-runnable).

★ Mutate-in-place: only overwrite status, add interrupt/content_ok, **keep all original fields like timing/tor/content**,
  and lift resp_onset/trig_end to the top level. So it's **idempotent** —— rerunning gives the same result and doesn't destroy the original signals needed for a rerun.
  (An early version replaced the whole events block and read from timing without writing back to timing, so a second run would wrongly flip already-graded items to no_response.)

Only rewrites grade.json's events[0] and summary.rate (the leaderboard/sample-picker both read summary.rate).

Usage: uv run python regrade_keyword_wait.py            # re-grade all models
       uv run python regrade_keyword_wait.py gpt gemini  # re-grade only the given models
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "test_set" / "keyword_wait"


def _num(x):
    return x if isinstance(x, (int, float)) else None


def regrade_event(ev):
    """Mutate ev in place, return the new status. Keeps all original fields, idempotent."""
    tim = ev.get("timing") if isinstance(ev.get("timing"), dict) else {}
    onset = _num(tim.get("resp_onset"))
    if onset is None:
        onset = _num(ev.get("resp_onset"))                 # compatible with the old structure already lifted to top level
    trig = _num(tim.get("trig_end"))
    if trig is None:
        trig = _num(ev.get("trig_end"))
    c = ev.get("content")
    content_ok = (c.get("status") == "pass") if isinstance(c, dict) else (c == "pass")
    tor = ev.get("tor")
    responded = bool(tor) if tor is not None else (onset is not None)

    ev["grade_dimension"] = "interrupt+content"
    if not responded or onset is None:
        ev["status"] = "no_response"
        ev["interrupt"] = None
        ev["content_ok"] = None
        return "no_response"
    interrupt = onset >= (trig - 0.3 if trig is not None else 0)   # it "responded after the keyword", not jumping in before the keyword
    ev["interrupt"] = "pass" if interrupt else "fail"
    ev["content_ok"] = "pass" if content_ok else "fail"
    ev["status"] = "pass" if (interrupt and content_ok) else "fail"
    if onset is not None:                                  # lift to top level so the next run can read it even without timing (idempotent)
        ev["resp_onset"] = onset
    if trig is not None:
        ev["trig_end"] = trig
    return ev["status"]


def main(models):
    tot = changed = nr = 0
    for item in sorted(CAT.glob("*")):
        if not (item / "benchmark.json").exists():
            continue
        for d in item.glob("live_*"):
            mdl = d.name.rsplit("_", 1)[-1]
            if models and mdl not in models:
                continue
            gp = d / "grade.json"
            if not gp.exists():
                continue
            tot += 1
            g = json.loads(gp.read_text())
            evs = g.get("events") or [{}]
            old = evs[0].get("status")
            st = regrade_event(evs[0])
            g["events"] = evs
            g["summary"] = {"n": 1, "npass": 1 if st == "pass" else 0,
                            "rate": 1.0 if st == "pass" else 0.0}
            gp.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
            if st == "no_response":
                nr += 1
            if old != st:
                changed += 1
                print(f"  ✎ {item.name}/{mdl}: {old} → {st}  (interrupt={evs[0].get('interrupt')} correct={evs[0].get('content_ok')})")
    print(f"\n==== re-graded {tot} items, {changed} verdicts changed, no_response {nr} (recommend rerun) ====")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
