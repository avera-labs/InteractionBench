"""Rewrite the stay_quiet_until_help scoring: mainly checks **restraint** (did it hold back and not interject before being asked) + **answer** (did it answer correctly after being asked).
Drop the original strict timing window ([200,600]ms) —— that's "real-time interruption" logic and doesn't fit "answer after being asked".

Dimensions:
  restraint  during the user's "thinking" phase (A2 start → before the last 'help'), did the model interject. No interjection = pass (held back).
  content    after the last 'help', does the model's speech contain the correct answer (number-word normalization + substring, lenient, ignores verbosity).
  overall    restraint AND content = pass.
  no_response model silent on the whole track / didn't speak after being asked → flag (mostly a glitch, recommend rerun, not a true fail).

Usage: uv run python regrade_stay_quiet.py            # re-grade all models
       uv run python regrade_stay_quiet.py gpt gemini  # re-grade only the given models
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "test_set" / "stay_quiet_until_help"

_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100}


def _norm(s: str) -> str:
    """Lowercase, strip punctuation, combine hyphenated number words into digits (twenty-eight→28), number words→digits. Returns the normalized token string."""
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = s.split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in _NUM:                                   # handle twenty eight / one hundred
            val = _NUM[t]
            if i + 1 < len(toks) and toks[i + 1] in _NUM and _NUM[toks[i + 1]] < 10 and val >= 20 and val < 100:
                val += _NUM[toks[i + 1]]; i += 1
            elif i + 1 < len(toks) and toks[i + 1] == "hundred":
                val *= 100; i += 1
            out.append(str(val))
        else:
            out.append(t)
        i += 1
    return " ".join(out)


def _has_answer(heard: str, target: str) -> bool:
    """Lenient: after normalization, whether the target's key tokens appear in heard (whole-number match / word match / de-hyphenated spelling)."""
    h = " " + _norm(heard) + " "
    t = _norm(target).strip()
    if not t:
        return False
    toks = t.split()
    # numeric target: any single digit token hitting counts (28 / 100, etc.)
    nums = [x for x in toks if x.isdigit()]
    if nums:
        return any((" " + n + " ") in h for n in nums)
    # word/spelling target (canberra / necessary / jupiter / contract): the key word appearing is enough
    words = [x for x in toks if len(x) >= 3]
    return any((" " + w + " ") in h for w in words) if words else (t in h)


def _words(p: Path):
    return json.loads(p.read_text()).get("words", []) if p.exists() else []


def regrade_item(d: Path, target: str):
    u = _words(d / "A_user.parakeet.json")
    m = _words(d / "B_model.parakeet.json")
    if not m:
        return {"status": "no_response", "restraint": None, "content": None,
                "note": "model silent on the whole track (mostly a glitch, recommend rerun)"}
    # the last 'help' = the real request for help (earlier ones are in the instructions)
    helps = [w for w in u if re.sub(r"[^a-z]", "", w["word"].lower()) == "help"]
    t_help = helps[-1]["t1"] if helps else (u[-1]["t1"] if u else 0)
    # A2 start = the word after the largest gap among user words before t_help (after the big silence caused by the model's ack across turns)
    before = [w for w in u if w["t0"] < t_help]
    a2_start = before[0]["t0"] if before else 0
    maxgap = 0
    for a, b in zip(before, before[1:]):
        g = b["t0"] - a["t1"]
        if g > maxgap:
            maxgap, a2_start = g, b["t0"]
    # restraint: during the thinking phase (a2_start → t_help), did the model start new speech
    interrupted = any(a2_start + 0.3 < w["t0"] < t_help - 0.3 for w in m)
    restraint = not interrupted
    # answer: what the model said after t_help
    post = " ".join(w["word"] for w in m if w["t0"] > t_help - 0.3)
    content = _has_answer(post, target)
    ok = restraint and content
    return {"status": "pass" if ok else "fail",
            "restraint": "pass" if restraint else "fail",
            "content": "pass" if content else "fail",
            "a2_start": round(a2_start, 2), "t_help": round(t_help, 2),
            "post_heard": post[:120], "target": target}


def main(models):
    tot = changed = nr = 0
    for item in sorted(CAT.glob("*")):
        if not (item / "benchmark.json").exists():
            continue
        target = json.loads((item / "benchmark.json").read_text())["expected_events"][0].get("target_response", "")
        for d in item.glob("live_*"):
            mdl = d.name.rsplit("_", 1)[-1]
            if models and mdl not in models:
                continue
            gp = d / "grade.json"
            if not gp.exists():
                continue
            tot += 1
            res = regrade_item(d, target)
            g = json.loads(gp.read_text())
            old = (g.get("events") or [{}])[0].get("status")
            ev = {"event_id": "e1", "category": "stay_quiet_until_help", "grade_dimension": "restraint+content",
                  "status": res["status"], "restraint": res["restraint"], "content_ok": res["content"], **res}
            g["events"] = [ev]
            g["summary"] = {"n": 1, "npass": 1 if res["status"] == "pass" else 0,
                            "rate": 1.0 if res["status"] == "pass" else 0.0}
            gp.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
            if res["status"] == "no_response":
                nr += 1
            if old != res["status"]:
                changed += 1
                print(f"  ✎ {item.name}/{mdl}: {old} → {res['status']}  (restraint={res['restraint']} answer={res['content']}) "
                      f"post={res.get('post_heard','')[:50]!r}")
    print(f"\n==== re-graded {tot} items, {changed} verdicts changed, no_response {nr} (recommend rerun) ====")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
