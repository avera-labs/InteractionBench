"""Judge every groundedness conversation three times and keep the majority verdict per probe.

Re-judging the same transcript moved GPT-Live's total by five probes of 52 (39 and 44), so one judgment per
conversation is too noisy to rank close systems. For every system and conversation this runs the judge of
grade_interaction.py three times on the transcript the grader uses, stores the three judgments in
<run>/grade_interaction_runs.json, and writes grade_interaction.json and grade.json from the majority:
a probe passes when at least two of the three judgments pass it (a judgment that returns no verdict for a
probe counts as a fail), its label and reason come from a judgment that agrees with the majority, and the
coherence score is the mean of the three. It also prints how often the three judgments agree per probe.

Usage:  uv run python rejudge_ig.py            # judge conversations without grade_interaction_runs.json
        uv run python rejudge_ig.py --report   # only print the agreement and the scores

A judgment that returns no probe verdicts (an unparsable reply) is asked again, up to five times, and
judgments already stored without verdicts are re-asked in a repair pass before the report.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

import grade_interaction as gi

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
CAT = ROOT / "test_set" / "interaction_groundedness"
SYSTEMS = ["gptlive", "gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
K = 3
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def runs():
    for item in sorted(p.name for p in CAT.iterdir() if (p / "benchmark.json").exists()):
        for s in SYSTEMS:
            ds = sorted((CAT / item).glob(f"live_*_{s}"))
            if ds:
                yield item, s, ds[0]


def match(question, judged):
    best = max(judged, key=lambda p: SequenceMatcher(None, norm(question), norm(p.get("question"))).ratio(), default=None)
    if best and SequenceMatcher(None, norm(question), norm(best.get("question"))).ratio() >= 0.75:
        return best
    return None


def majority(scenario, judgments):
    out = []
    for t in scenario["turns"]:
        if not t.get("probe"):
            continue
        got = [match(t["text"], j.get("probes", [])) for j in judgments]
        votes = [bool(g and g.get("pass")) for g in got]
        verdict = sum(votes) * 2 > len(votes)
        pick = next((g for g, v in zip(got, votes) if g and v == verdict), None) or {}
        out.append({"question": t["text"], "type": t["probe"]["type"], "pass": verdict, "votes": sum(votes),
                    "label": pick.get("label") if not verdict else None, "heard": pick.get("heard"),
                    "reason": pick.get("reason")})
    return out


def write(item, s, d, scenario, judgments):
    probes = majority(scenario, judgments)
    cohs = [j.get("coherence") for j in judgments if isinstance(j.get("coherence"), (int, float))]
    out = {"model": s, "item": item, "title": scenario.get("title"), "n_probes": len(probes), "graded": len(probes),
           "passed": sum(p["pass"] for p in probes), "coherence": round(sum(cohs) / len(cohs)) if cohs else None,
           "summary": judgments[0].get("summary"), "probes": probes,
           "judgments": f"majority of {len(judgments)} (grade_interaction_runs.json)"}
    (d / "grade_interaction.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    gi._write_grade_json(str(d), out)


def judge_once(client, scenario, flow, tries=5):
    """One judgment; a reply that cannot be parsed (no probes returned) is asked again rather than counted."""
    for _ in range(tries):
        j = gi.judge(client, scenario, flow)
        if j.get("probes"):
            return j
    return j


def repair(client):
    """Re-ask the judgments that came back without any probe verdicts, then rewrite the majority."""
    for item, s, d in runs():
        p = d / "grade_interaction_runs.json"
        if not p.exists():
            continue
        judgments = json.loads(p.read_text())
        bad = [i for i, j in enumerate(judgments) if not j.get("probes")]
        if not bad:
            continue
        scenario = json.loads((CAT / item / "benchmark.json").read_text())
        flow = gi.interleave(str(d))
        for i in bad:
            judgments[i] = judge_once(client, scenario, flow)
        p.write_text(json.dumps(judgments, ensure_ascii=False, indent=1), encoding="utf-8")
        write(item, s, d, scenario, judgments)
        print(f"  repaired {item}/{s}: judgments {bad} re-asked", flush=True)


def judge_run(args):
    item, s, d, client = args
    scenario = json.loads((CAT / item / "benchmark.json").read_text())
    flow = gi.interleave(str(d))
    judgments = [judge_once(client, scenario, flow) for _ in range(K)]
    (d / "grade_interaction_runs.json").write_text(json.dumps(judgments, ensure_ascii=False, indent=1), encoding="utf-8")
    write(item, s, d, scenario, judgments)
    return f"  {item}/{s}: " + " ".join(str(sum(1 for p in j.get("probes", []) if p.get("pass"))) for j in judgments)


def report():
    print(f"{'system':12} {'score (majority)':>16}   single judgments   probes judged the same all {K} times")
    for s in SYSTEMS:
        k = n = same = total = 0
        singles = [0] * K
        for item, sys_, d in runs():
            if sys_ != s or not (d / "grade_interaction_runs.json").exists():
                continue
            scenario = json.loads((CAT / item / "benchmark.json").read_text())
            judgments = json.loads((d / "grade_interaction_runs.json").read_text())
            g = json.loads((d / "grade_interaction.json").read_text())
            k, n = k + g["passed"], n + g["n_probes"]
            for i, j in enumerate(judgments):
                singles[i] += sum(1 for t in scenario["turns"] if t.get("probe")
                                  and (m := match(t["text"], j.get("probes", []))) and m.get("pass"))
            for p in g["probes"]:
                total += 1
                same += p["votes"] in (0, K)
        print(f"{s:12} {k:10}/{n:<5}   {' '.join(f'{x:2}' for x in singles):>17}   {same}/{total} ({same / max(total, 1):.0%})")


def main():
    if "--report" not in sys.argv:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        todo = [(item, s, d, client) for item, s, d in runs() if not (d / "grade_interaction_runs.json").exists()]
        with ThreadPoolExecutor(max_workers=6) as ex:
            for line in ex.map(judge_run, todo):
                print(line, flush=True)
        repair(client)
    report()


if __name__ == "__main__":
    main()
