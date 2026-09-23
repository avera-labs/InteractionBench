"""Compare GPT-Live's timing results on the session clock and by arrival.

Before promote_arrival.py the session-clock results sit at the top of each GPT-Live run directory and the
arrival results in <run>/arrival/; afterwards the arrival results are at the top and the session-clock ones
in <run>/session_clock/. Both are scored with build_site_data's rules, so the
numbers match the site and the paper. Prints the per-scenario comparison, the timing aggregate next to
the other systems, and the rank correlation between timing and spoken-task accuracy with each version
of GPT-Live. --shift re-scores turn-taking and pause with every arrival word moved by the given seconds,
to show how much the estimated arrival anchor matters. It also prints how far apart the two placements are
(the network and buffering delay in the timing items, and how much the session clock gains on the recorded
user track over the long conversations), how early simply concatenating the deltas would put the last words
of the long conversations, and how late the server's recognizer reports the user's words.
--json writes the numbers for the paper tables.

Usage:  uv run python compare_clocks.py [--shift 0.04] [--json out.json]   # 0.04 s is the calibration spread
"""
import argparse
import itertools
import json
import re
import shutil
import statistics as st
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import build_site_data as B
import grade_behavior as gb

ROOT = Path(__file__).resolve().parent
TIMING = ["backchannel", "pause", "turn_taking", "user_backchannel", "interruption"]
DIAG = {"backchannel": ("floor_take", "floor-take"), "pause": ("bargein", "barge-in"),
        "turn_taking": ("med_lat", "latency ms"), "user_backchannel": ("derail", "derail"),
        "interruption": ("yielded", "yielded")}
SPOKEN = ["logic_puzzle", "countdown_completion", "grammar_correction", "keyword_wait", "stay_quiet_until_help"]
SPOKEN_TIMED = ["grammar_correction", "keyword_wait", "stay_quiet_until_help"]
FULL_DUPLEX = ["gptlive", "gpt", "gemini", "moshi", "personaplex", "freezeomni"]


def spearman(a, b):
    """Spearman rank correlation, ranking as paper/tables.py does (1 = best, ties take the better rank)."""
    def ranks(vals):
        out = [0] * len(vals)
        for pos, i in enumerate(sorted(range(len(vals)), key=lambda i: -vals[i])):
            out[i] = pos + 1
        return out
    ra, rb, n = ranks(a), ranks(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def gptlive_runs(task):
    return sorted((ROOT / "tts_review" / task).glob("*/live_*_gptlive"))


def placed(d: Path, clock: str) -> Path:
    """Where a run's results on `clock` ('session' or 'arrival') live, before or after promote_arrival.py."""
    promoted = (d / "session_clock").exists()
    if clock == "session":
        return d / "session_clock" if promoted else d
    return d if promoted else d / "arrival"


def summary(task, clock):
    vals = [B.load_result(str(placed(d, clock)), task) for d in gptlive_runs(task)
            if (placed(d, clock) / f"{task}.json").exists()]
    return B.summarize(task, vals)


def shifted(task, shift):
    """Re-grade turn-taking or pause from the arrival words moved by `shift` seconds (timing only, no judge)."""
    vals = []
    for d in gptlive_runs(task):
        a = placed(d, "arrival")
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            shutil.copy(a / "A_user.parakeet.json", t / "A_user.parakeet.json")
            doc = json.loads((a / "B_model.parakeet.json").read_text())
            for w in doc.get("words", []):
                w["t0"] += shift
                w["t1"] += shift
            (t / "B_model.parakeet.json").write_text(json.dumps(doc))
            res = gb.grade_turn_taking(t) if task == "turn_taking" else gb.grade_pause(t)
            (t / f"{task}.json").write_text(json.dumps(res))
            vals.append(B.load_result(str(t), task))
    return B.summarize(task, vals)


def word_offsets(d):
    """(session time, arrival minus session) for each of GPT-Live's words matched between the two placements of a run."""
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    a = json.loads((placed(d, "arrival") / "B_model.parakeet.json").read_text())["words"]
    s = json.loads((placed(d, "session") / "B_model.parakeet.json").read_text())["words"]
    sm = SequenceMatcher(None, [norm(w["word"]) for w in a], [norm(w["word"]) for w in s], autojunk=False)
    return [(s[j + k]["t0"], a[i + k]["t0"] - s[j + k]["t0"]) for i, j, n in sm.get_matching_blocks() for k in range(n)]


def clock_offsets():
    """Delay between the placements in the timing items, session-clock gain over the long conversations,
    the error of concatenating the deltas, and the lateness of the server's user-side recognizer."""
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    timed = ["turn_taking", "pause", "interruption", "user_backchannel"]
    meds = [st.median(y for _, y in o) for t in timed for d in gptlive_runs(t) if (o := word_offsets(d))]
    dq = st.quantiles(meds, n=4)
    print(f"\nArrival minus session clock, GPT-Live's words in {', '.join(timed)}: "
          f"median {st.median(meds):.2f}s over {len(meds)} runs (IQR {dq[0]:.2f}-{dq[2]:.2f})")
    long_runs = sorted((ROOT / "test_set" / "interaction_groundedness").glob("*/live_*_gptlive"))
    slopes = []
    for d in long_runs:                                   # Theil-Sen slope of the offset against session time
        p = word_offsets(d)
        slopes.append(st.median((y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in itertools.combinations(p, 2)
                                if abs(x2 - x1) > 1.0))
    q = st.quantiles(slopes, n=4)
    print(f"Session clock gain on the user track over the long conversations: median {-st.median(slopes):.1%} "
          f"of the elapsed time (IQR {-q[2]:.1%} to {-q[0]:.1%}, {len(slopes)} runs)")
    early = []
    for d in long_runs:                                   # deltas played back to back from the first delta's position
        raw = json.loads((d / "B_model_raw.parakeet.json").read_text())["words"]
        arr = json.loads((placed(d, "arrival") / "B_model.parakeet.json").read_text())["words"]
        u0 = json.loads((placed(d, "arrival") / "placement.json").read_text())["anchor_s"]
        sm = SequenceMatcher(None, [norm(w["word"]) for w in raw], [norm(w["word"]) for w in arr], autojunk=False)
        pairs = [(i + k, j + k) for i, j, n in sm.get_matching_blocks() for k in range(n)][-5:]
        early.append(st.median(arr[j]["t0"] - (u0 + raw[i]["t0"]) for i, j in pairs))
    print(f"Concatenated deltas, last words of the long conversations: a median of {st.median(early):.1f}s too early")
    late = []
    for root in ("tts_review", "test_set"):
        for d in sorted((ROOT / root).glob("*/*/live_*_gptlive")):
            heard = [e for e in json.loads((d / "transcript_clock.json").read_text()) if e["who"] == "user"]
            sw = [(norm(x), e["start_ms"] / 1000) for e in heard for x in e["text"].split() if norm(x)]
            aw = [(norm(w["word"]), w["t0"]) for w in json.loads((d / "A_user.parakeet.json").read_text())["words"] if norm(w["word"])]
            sm = SequenceMatcher(None, [x for x, _ in sw], [x for x, _ in aw], autojunk=False)
            offs = [sw[i + k][1] - aw[j + k][1] for i, j, n in sm.get_matching_blocks() for k in range(n) if aw[j + k][1] < 10]
            if offs:
                late.append(st.median(offs))
    print(f"Server recognizer on the user's words (first 10 s of each run): a median of {st.median(late):.2f}s late "
          f"over {len(late)} runs")
    return {"delay_s": st.median(meds), "delay_iqr": [dq[0], dq[2]],
            "gain": -st.median(slopes), "concatenation_early_s": st.median(early), "recognizer_late_s": st.median(late)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=0.04)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    m = json.loads((ROOT / "docs" / "data" / "manifest.json").read_text())
    totals = m["timing_totals"]
    lb = m["leaderboards"]
    labels = {s["id"]: s["label"] for s in m["systems"]}

    out = {"scenarios": {}}
    print(f"GPT-Live, five timing scenarios                session clock        arrival")
    k_s = n_s = k_a = n_a = 0
    for task in TIMING:
        s, a = summary(task, "session"), summary(task, "arrival")
        key, name = DIAG[task]
        fmt = lambda v: "--" if v is None else (f"{v:.0f}" if key == "med_lat" else f"{v:.2f}")
        print(f"  {task:17} pass       {s['npass']:3}/{s['n']:<3} ({s['pass_rate']:.2f})      {a['npass']:3}/{a['n']:<3} ({a['pass_rate']:.2f})")
        print(f"  {'':17} {name:10} {fmt(s.get(key)):>14}      {fmt(a.get(key)):>14}")
        out["scenarios"][task] = {"session": s, "arrival": a}
        k_s, n_s, k_a, n_a = k_s + s["npass"], n_s + s["n"], k_a + a["npass"], n_a + a["n"]
    print(f"  {'timing aggregate':17} {'':10} {k_s:5}/{n_s} ({k_s / n_s:.0%})      {k_a:5}/{n_a} ({k_a / n_a:.0%})")
    out["aggregate"] = {"session": [k_s, n_s], "arrival": [k_a, n_a]}

    print("\nTiming aggregate, all systems (only GPT-Live changes)")
    timing = {s: totals[s]["good"] / totals[s]["n"] for s in totals}
    timing["gptlive"] = k_s / n_s                         # the manifest may already hold either version
    for s in totals:
        if s == "gptlive":
            print(f"  {labels[s]:20} {k_s:4}/{n_s} ({k_s / n_s:.0%})   -> arrival {k_a}/{n_a} ({k_a / n_a:.0%})")
        else:
            print(f"  {labels[s]:20} {totals[s]['good']:4}/{totals[s]['n']} ({timing[s]:.0%})")

    print("\nGPT-Live, spoken tasks whose pass criteria use timing")
    out["spoken"] = {}
    delta = 0
    for task in SPOKEN_TIMED:
        runs = sorted((ROOT / "test_set" / task).glob("*/live_*_gptlive"))
        passed = lambda p: json.loads(p.read_text())["events"][0].get("status") == "pass"
        s = sum(passed(placed(d, "session") / "grade.json") for d in runs)
        a = sum(passed(placed(d, "arrival") / "grade.json") for d in runs if (placed(d, "arrival") / "grade.json").exists())
        n_done = sum(1 for d in runs if (placed(d, "arrival") / "grade.json").exists())
        print(f"  {task:22} session {s}/{len(runs)}   arrival {a}/{n_done}")
        out["spoken"][task] = {"session": [s, len(runs)], "arrival": [a, n_done]}
        delta += a - s
    spoken_k = {s: sum(lb[t][s]["npass"] for t in SPOKEN) for s in totals}
    spoken_n = {s: sum(lb[t][s]["n"] for t in SPOKEN) for s in totals}
    gl_runs = [d for t in SPOKEN for d in sorted((ROOT / "test_set" / t).glob("*/live_*_gptlive"))]
    spoken_k["gptlive"] = sum(json.loads((placed(d, "session") / "grade.json").read_text())["events"][0].get("status") == "pass"
                              for d in gl_runs)
    spoken_n["gptlive"] = len(gl_runs)
    print(f"  {'spoken-task total':22} session {spoken_k['gptlive']}/{spoken_n['gptlive']}   "
          f"arrival {spoken_k['gptlive'] + delta}/{spoken_n['gptlive']}")
    out["spoken_total"] = {"session": spoken_k["gptlive"], "arrival": spoken_k["gptlive"] + delta, "n": spoken_n["gptlive"]}

    spoken = {s: spoken_k[s] / spoken_n[s] for s in totals}
    spoken_arr = dict(spoken, gptlive=(spoken_k["gptlive"] + delta) / spoken_n["gptlive"])
    arr = dict(timing, gptlive=k_a / n_a)
    rho = {}
    for name, tm, sp in (("session", timing, spoken), ("arrival", arr, spoken_arr)):
        six = spearman([tm[s] for s in FULL_DUPLEX], [sp[s] for s in FULL_DUPLEX])
        seven = spearman([tm[s] for s in totals], [sp[s] for s in totals])
        rho[name] = {"six": six, "seven": seven}
    print("\nSpearman rank correlation, timing aggregate vs spoken-task accuracy")
    print(f"  six full-duplex systems:  session {rho['session']['six']:+.2f}   arrival {rho['arrival']['six']:+.2f}")
    print(f"  with the cascade:         session {rho['session']['seven']:+.2f}   arrival {rho['arrival']['seven']:+.2f}")
    out["rho"] = rho

    placements = [json.loads((placed(d, "arrival") / "placement.json").read_text())
                  for t in TIMING for d in gptlive_runs(t) if (placed(d, "arrival") / "placement.json").exists()]
    anchors = [p["anchor_s"] for p in placements if p.get("placed")]
    print(f"\nArrival anchor over {len(anchors)} runs: median {st.median(anchors):.2f}s")
    print(f"Sensitivity: arrival words moved by -{args.shift:.2f}s / 0 / +{args.shift:.2f}s")
    out["sensitivity"] = {}
    for task in ("turn_taking", "pause"):
        row = [shifted(task, -args.shift), summary(task, "arrival"), shifted(task, args.shift)]
        print(f"  {task:17} pass " + "   ".join(f"{r['npass']}/{r['n']}" for r in row))
        out["sensitivity"][task] = [r["npass"] for r in row]
    out["offsets"] = clock_offsets()
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
