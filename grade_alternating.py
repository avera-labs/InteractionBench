"""Grade alternating_count (take-turns counting): the user says odd numbers (1,3,5,7,9, one per turn), the model adds one even number per turn (2,4,6,8,10).

Two dimensions:
  sequence  the even numbers the model says, matched in order against [2,4,6,8,10], how many hit (order must be right). All hit = correct.
  discipline whether it's "one per turn" —— it must not blurt out 2,4,6,8,10 all at once (that's reciting, not taking turns).
             criterion: the hit evens spanning < 3s end-to-end AND ≥3 of them = blurt (violates turn-taking).
  overall   full sequence AND no blurt = pass.
Model silent → no_response (mostly a glitch, recommend rerun).

Usage: uv run python grade_alternating.py            # grade all models
       uv run python grade_alternating.py gpt gemini  # given models
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "test_set" / "alternating_count"
_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _numbers(words):
    """[{word,t0,t1}] → [(value, t0)], recognizing number words and digits."""
    out = []
    for w in words:
        tok = re.sub(r"[^a-z0-9]", "", w["word"].lower())
        if tok.isdigit():
            out.append((int(tok), w["t0"]))
        elif tok in _NUM:
            out.append((_NUM[tok], w["t0"]))
    return out


def grade_item(d: Path, expected):
    mw = json.loads((d / "B_model.parakeet.json").read_text()).get("words", []) if (d / "B_model.parakeet.json").exists() else []
    if not mw:
        return {"status": "no_response", "sequence": None, "discipline": None,
                "note": "model silent on the whole track (mostly a glitch, recommend rerun)"}
    nums = _numbers(mw)
    said = [v for v, _ in nums]
    # match expected in order (greedily take an ordered subsequence)
    hit, ei = [], 0
    for v, t in nums:
        if ei < len(expected) and v == expected[ei]:
            hit.append((v, t)); ei += 1
    n_correct = len(hit)
    span = round(hit[-1][1] - hit[0][1], 1) if len(hit) >= 2 else 0.0
    blurted = n_correct >= 3 and span < 3.0                  # blurting them all at once = reciting, not taking turns
    # off-plan: how many non-target-even numbers were said. A turn-taking player only says 2,4,6,8,10 → off_plan=0;
    # a continuous reciter (1,2,3,4,5…) sneaks by since it incidentally contains the 2,4,6,8,10 ordered subsequence, but has lots of off_plan → judge it as reciting, not taking turns.
    exp_set = set(expected)
    off_plan = sum(1 for v in said if v not in exp_set)
    reciting = off_plan >= 3                                 # ≥3 off-plan numbers mixed in = continuous counting, not selective turn-taking
    seq_ok = n_correct == len(expected)
    disciplined = not (blurted or reciting)
    ok = seq_ok and disciplined
    return {"status": "pass" if ok else "fail",
            "sequence": "pass" if seq_ok else "fail",
            "discipline": "pass" if disciplined else "fail",
            "discipline_fail": ("blurt" if blurted else "reciting" if reciting else None),
            "n_correct": f"{n_correct}/{len(expected)}", "off_plan": off_plan,
            "said": said, "span_s": span, "expected": expected}


def main(models):
    tot = changed = nr = 0
    for item in sorted(CAT.glob("*")):
        bj = item / "benchmark.json"
        if not bj.exists():
            continue
        expected = json.loads(bj.read_text()).get("expected_sequence", [2, 4, 6, 8, 10])
        for d in item.glob("live_*"):
            mdl = d.name.rsplit("_", 1)[-1]
            if models and mdl not in models:
                continue
            gp = d / "grade.json"
            if not gp.exists():
                continue
            tot += 1
            res = grade_item(d, expected)
            g = json.loads(gp.read_text())
            old = (g.get("events") or [{}])[0].get("status")
            g["events"] = [{"event_id": "e1", "category": "alternating_count",
                            "grade_dimension": "sequence+discipline", "status": res["status"], **res}]
            g["summary"] = {"n": 1, "npass": 1 if res["status"] == "pass" else 0,
                            "rate": 1.0 if res["status"] == "pass" else 0.0}
            gp.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
            nr += res["status"] == "no_response"
            if old != res["status"]:
                changed += 1
            print(f"  {item.name}/{mdl}: {res['status']}  (sequence {res.get('sequence')} discipline {res.get('discipline')} "
                  f"{res.get('n_correct')}) said={res.get('said')} span={res.get('span_s')}s")
    print(f"\n==== graded {tot} items, no_response {nr} ====")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
