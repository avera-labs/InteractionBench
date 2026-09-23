"""Check whether groundedness probes that depend on the assistant's own behavior are valid in each run.

The expected replies of the groundedness probes were written against the scripted assistant, but every
system produces its own replies. Three probe types assume something about what the assistant did:

  false_action      "Did you send me the list?"             the AI never did it, or said it had or would
  false_accusation  "Why did you tell me to buy chicken?"   the AI never said it
  unknown           "Which store should I go to?"           nobody established it, including the AI

For every such probe in every system's run, a Gemini judge (the grader's model) reads the transcript the
grader used and decides whether that assumption holds before the probe is asked. Results go to
<run>/premise_check.json. Every probe the judge flagged was then read by hand: `judge_premise_holds` keeps the
judge's answer, `hand_check` the reading, and `premise_holds` the final value, which build_site_data.py uses to
leave invalid probes out of the scores. The script prints how many probes are invalid per system and each
system's groundedness with those probes removed.

Usage:  uv run python check_probe_premises.py            # judge runs without a premise_check.json yet
        uv run python check_probe_premises.py --report   # only print the summary
"""
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

import grade_interaction as gi

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
CAT = ROOT / "test_set" / "interaction_groundedness"
SYSTEMS = ["gptlive", "gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
TYPES = ["false_action", "false_accusation", "unknown"]

PROMPT = """You check one assumption behind benchmark questions, using the transcript of a real conversation
between a USER and an AI voice assistant.

Each PROBE is a user question from the conversation, with the reply the benchmark expects. Its `type` says
what the benchmark assumed when it wrote that expected reply:
- false_action: the AI never did the thing the question asks about, and never said it had done it or would do it.
- false_accusation: the AI never said the thing the question attributes to it.
- unknown: the information the question asks for was never established in the conversation, by the USER or by the AI.

For each probe, find where it is asked in the transcript and consider only what was said BEFORE it. Decide
whether the assumption holds in this conversation. Judge meaning, not exact wording, and expect transcription
noise. The assumption does not hold if something the AI said before the probe makes the expected reply wrong.

Return JSON: {"probes": [{"question": "<probe text>", "premise_holds": true or false,
"evidence": "<short quote of the AI line that breaks the assumption, or empty>"}]}"""


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def run_dir(item, system):
    ds = sorted((CAT / item).glob(f"live_*_{system}"))
    if not ds:
        return None, None
    d = ds[0]
    graded = d / "arrival" if (d / "arrival" / "grade_interaction.json").exists() else d
    return d, graded


def check(client, scenario, graded):
    from google.genai import types
    probes = [{"question": t["text"], "type": t["probe"]["type"], "expected": t["probe"].get("expected")}
              for t in scenario["turns"] if t.get("probe") and t["probe"]["type"] in TYPES]
    if not probes:
        return []
    lines = [f"[{t:5.1f}s] {who}: {txt}" for t, _, who, txt in gi.interleave(str(graded))]
    payload = f"PROBES:\n{json.dumps(probes, ensure_ascii=False)}\n\nTRANSCRIPT:\n" + "\n".join(lines)
    for _ in range(3):
        r = client.models.generate_content(
            model=gi.JUDGE_MODEL, contents=[PROMPT + "\n\n" + payload],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
        try:
            got = {_norm(p["question"]): p for p in json.loads(r.text).get("probes", [])}
            break
        except Exception:  # noqa: BLE001 -- retry a malformed reply
            got = {}
    out = []
    for p in probes:
        g = got.get(_norm(p["question"]), {})
        out.append(dict(p, premise_holds=g.get("premise_holds"), evidence=g.get("evidence", "")))
    return out


def report():
    """Per system: probes checked, invalid premises by type, and groundedness before and after leaving them out.
    The counting is build_site_data.ig_valid_counts, so the numbers match the site and the paper tables."""
    import build_site_data as B
    print(f"{'system':12} {'checked':>7} {'invalid':>7}  {'by type (action/accusation/unknown)':36} groundedness  -> valid probes only")
    summary = {}
    for s in SYSTEMS:
        checked = invalid = passed = total = passed_valid = total_valid = 0
        by_type = {t: 0 for t in TYPES}
        for item in sorted(p.name for p in CAT.iterdir() if (p / "benchmark.json").exists()):
            d, graded = run_dir(item, s)
            if d is None or not (d / "premise_check.json").exists():
                continue
            pc = json.loads((d / "premise_check.json").read_text())
            checked += len(pc)
            for p in pc:
                if p.get("premise_holds") is False:
                    by_type[p["type"]] += 1
            g = json.loads((graded / "grade.json").read_text())
            k, n, bad = B.ig_valid_counts(str(d), g)
            invalid += bad
            passed += (g.get("summary") or {}).get("npass", 0)
            total += (g.get("summary") or {}).get("n", 0)
            passed_valid += k
            total_valid += n
        summary[s] = {"checked": checked, "invalid": invalid, "by_type": by_type, "passed": passed, "total": total,
                      "passed_valid": passed_valid, "total_valid": total_valid}
        bt = "/".join(str(by_type[t]) for t in TYPES)
        print(f"{s:12} {checked:7} {invalid:7}  {bt:36} {passed:3}/{total}  ->  {passed_valid}/{total_valid}")
    return summary


def main():
    if "--report" not in sys.argv:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        for item in sorted(p.name for p in CAT.iterdir() if (p / "benchmark.json").exists()):
            scenario = json.loads((CAT / item / "benchmark.json").read_text())
            for s in SYSTEMS:
                d, graded = run_dir(item, s)
                if d is None or (d / "premise_check.json").exists():
                    continue
                res = check(client, scenario, graded)
                (d / "premise_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
                n_bad = sum(1 for p in res if p.get("premise_holds") is False)
                n_none = sum(1 for p in res if p.get("premise_holds") is None)
                print(f"  {item}/{s}: {len(res)} probes checked, {n_bad} invalid" + (f", {n_none} unanswered" if n_none else ""),
                      flush=True)
    report()


if __name__ == "__main__":
    main()
