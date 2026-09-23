"""How much do the results depend on the scoring thresholds?

Re-scores the stored per-run measurements under other thresholds, without re-running or re-judging anything:

  turn-taking   a reply may start up to 50 ms (the rule), 150 ms (the human median in TurnBench), or 300 ms
                before the user's last word ends; the late bound stays at 2.5 s
  pause         any vocalization in the pause fails (the rule), or backchannels and continuers are allowed:
                every utterance that starts in the pause has at most four words and ends before the user
                resumes ("Yeah, go on.", "Oh?"), so the start of a longer answer still fails
  whisper       harmonics-to-noise ratio / voiced fraction below 4 dB / 0.3, 6 dB / 0.4 (the rule), or 8 dB / 0.5;
                the answer must still be correct, and the hand-corrected cascade item stays not whispered
  grammar       the correction must start 50-300 ms after the error (the rule), 0-1000 ms, or at any time while
                the user is speaking; overlap and content are still required

It also recomputes the timing aggregate with the lenient turn-taking and pause rules, and its rank
correlation with spoken-task accuracy.

Usage:  uv run python threshold_sensitivity.py [--json out.json] [--tex]   # --tex writes paper/tables/thresholds.tex
"""
import argparse
import glob
import json
from pathlib import Path

import build_site_data as B

ROOT = Path(__file__).resolve().parent
SYSTEMS = ["gptlive", "gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
LABEL = {"gptlive": "GPT-Live", "gpt": "GPT-Realtime", "gemini": "Gemini Live", "moshi": "Moshi",
         "personaplex": "PersonaPlex", "freezeomni": "Freeze-Omni", "halfduplex": "Cascade"}
SPOKEN = ["logic_puzzle", "countdown_completion", "grammar_correction", "keyword_wait", "stay_quiet_until_help"]
FULL_DUPLEX = SYSTEMS[:6]


def spearman(a, b):
    """Spearman rank correlation, ranking as paper/tables.py does (1 = best, ties take the better rank)."""
    def ranks(vals):
        out = [0] * len(vals)
        for pos, i in enumerate(sorted(range(len(vals)), key=lambda i: -vals[i])):
            out[i] = pos + 1
        return out
    ra, rb, n = ranks(a), ranks(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def results(task, sysid):
    root = "test_set" if B.is_graded(task) else "tts_review"
    for d in sorted(glob.glob(f"{ROOT}/{root}/{task}/*/live_*_{sysid}")):
        if Path(d, B.result_name(task)).exists():
            yield d, B.load_result(d, task)


def turn_taking(sysid, early_ms):
    k = n = 0
    for _, j in results("turn_taking", sysid):
        n += 1
        lat = j.get("latency_ms")
        k += bool(j.get("tor")) and lat is not None and -early_ms <= lat <= 2500
    return k, n


def utterances(words, gap=0.5):
    out = []
    for w in words:
        if out and w["t0"] - out[-1][-1]["t1"] <= gap:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def only_backchannels(d, j):
    p0, p1 = j["pause_window"]
    words = json.loads(Path(d, "B_model.parakeet.json").read_text()).get("words", [])
    started = [u for u in utterances(words) if p0 <= u[0]["t0"] <= p1]
    return bool(started) and all(len(u) <= 4 and u[-1]["t1"] <= p1 + 0.3 for u in started)


def pause(sysid, allow_backchannel):
    k = n = 0
    for d, j in results("pause", sysid):
        n += 1
        waited = not j.get("jumped_in") or (allow_backchannel and only_backchannels(d, j))
        k += waited and j.get("_spoke") is not False
    return k, n


def whisper(sysid, hnr_max, voiced_max):
    k = n = 0
    for _, g in results("whisper_production", sysid):
        ev = g["events"][0]
        n += 1
        hnr, voiced = ev.get("hnr"), ev.get("voiced_ratio")
        acoustic = hnr is not None and voiced is not None and hnr < 6 and voiced < 0.4
        overridden = acoustic and ev.get("whisper_ok") == "fail"          # counted as not whispered by hand
        ok = hnr is not None and voiced is not None and hnr < hnr_max and voiced < voiced_max and not overridden
        k += ok and ev.get("content_ok") == "pass"
    return k, n


def grammar(sysid, lo, hi):
    k = n = 0
    for _, g in results("grammar_correction", sysid):
        ev = g["events"][0]
        n += 1
        lat = (ev.get("timing") or {}).get("latency_ms")
        in_window = lat is not None and (lo is None or lo <= lat <= hi)
        k += in_window and (ev.get("overlap") or {}).get("status") == "pass" and (ev.get("content") or {}).get("status") == "pass"
    return k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()
    rows = {
        "turn-taking, early start up to 50 ms (rule)": lambda s: turn_taking(s, 50),
        "turn-taking, up to 150 ms": lambda s: turn_taking(s, 150),
        "turn-taking, up to 300 ms": lambda s: turn_taking(s, 300),
        "pause, any vocalization fails (rule)": lambda s: pause(s, False),
        "pause, backchannels allowed": lambda s: pause(s, True),
        "whisper, 4 dB / 0.3": lambda s: whisper(s, 4, 0.3),
        "whisper, 6 dB / 0.4 (rule)": lambda s: whisper(s, 6, 0.4),
        "whisper, 8 dB / 0.5": lambda s: whisper(s, 8, 0.5),
        "grammar, 50-300 ms (rule)": lambda s: grammar(s, 50, 300),
        "grammar, 0-1000 ms": lambda s: grammar(s, 0, 1000),
        "grammar, any onset while the user speaks": lambda s: grammar(s, None, None),
    }
    out = {"rows": {}}
    print(f"{'':44}" + "".join(f"{LABEL[s]:>13}" for s in SYSTEMS))
    for name, f in rows.items():
        vals = {s: f(s) for s in SYSTEMS}
        out["rows"][name] = vals
        print(f"{name:44}" + "".join(f"{f'{k}/{n}':>13}" for k, n in vals.values()))

    base = {}
    for s in SYSTEMS:
        k = n = 0
        for task in ("backchannel", "user_backchannel", "interruption"):
            for _, j in results(task, s):
                n += 1
                k += bool(B.eval_item(task, j).get("good"))
        base[s] = (k, n)
    lb = {t: B.leaderboard(t) for t in SPOKEN}
    spoken = {s: sum(lb[t][s]["npass"] for t in SPOKEN) / sum(lb[t][s]["n"] for t in SPOKEN) for s in SYSTEMS}
    out["aggregate"] = {}
    print()
    for label, tt_ms, bc in (("rule", 50, False), ("turn-taking 150 ms, backchannels allowed in pauses", 150, True)):
        agg = {}
        for s in SYSTEMS:
            k1, n1 = turn_taking(s, tt_ms)
            k2, n2 = pause(s, bc)
            agg[s] = (base[s][0] + k1 + k2, base[s][1] + n1 + n2)
        rate = {s: k / n for s, (k, n) in agg.items()}
        rho6 = spearman([rate[s] for s in FULL_DUPLEX], [spoken[s] for s in FULL_DUPLEX])
        rho7 = spearman([rate[s] for s in SYSTEMS], [spoken[s] for s in SYSTEMS])
        order = sorted(SYSTEMS, key=lambda s: -rate[s])
        print(f"timing aggregate, {label}: " + ", ".join(f"{LABEL[s]} {rate[s]:.0%}" for s in order)
              + f"  | rho six {rho6:+.2f}, seven {rho7:+.2f}")
        out["aggregate"][label] = {"counts": agg, "rho_six": rho6, "rho_seven": rho7}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    if args.tex:
        write_tex(out, ROOT / "paper" / "tables" / "thresholds.tex")


TEX_ROWS = [("turn-taking, early start up to 50 ms (rule)", r"Turn-taking, start $\le$50\,ms early (rule)"),
            ("turn-taking, up to 150 ms", r"\quad $\le$150\,ms early"),
            ("turn-taking, up to 300 ms", r"\quad $\le$300\,ms early"),
            ("pause, any vocalization fails (rule)", r"Pause, any vocalization fails (rule)"),
            ("pause, backchannels allowed", r"\quad backchannels allowed"),
            ("whisper, 4 dB / 0.3", r"Whisper, HNR $<$4\,dB, voiced $<$0.3"),
            ("whisper, 6 dB / 0.4 (rule)", r"\quad $<$6\,dB, $<$0.4 (rule)"),
            ("whisper, 8 dB / 0.5", r"\quad $<$8\,dB, $<$0.5"),
            ("grammar, 50-300 ms (rule)", r"Grammar, 50--300\,ms (rule)"),
            ("grammar, 0-1000 ms", r"\quad 0--1000\,ms"),
            ("grammar, any onset while the user speaks", r"\quad any onset while the user speaks")]


def write_tex(out, path):
    heads = [r"\shortstack{GPT-\\Live}", r"\shortstack{GPT-\\Realtime}", r"\shortstack{Gemini\\Live}", "Moshi", "PersonaPlex",
             r"\shortstack{Freeze-\\Omni}", "Cascade"]              # two-line names keep the table within the text width
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{l" + "c" * len(SYSTEMS) + "}", r"\toprule",
             "Rule & " + " & ".join(heads) + r" \\", r"\midrule"]
    for i, (key, label) in enumerate(TEX_ROWS):
        if i and not label.startswith(r"\quad"):
            lines.append(r"\addlinespace")
        lines.append(label + " & " + " & ".join(f"{k}/{n}" for k, n in (out["rows"][key][s] for s in SYSTEMS)) + r" \\")
    lines.append(r"\midrule")
    agg = out["aggregate"]
    for key, label in (("rule", "Timing aggregate, rules"),
                       ("turn-taking 150 ms, backchannels allowed in pauses", r"\quad 150\,ms, backchannels allowed")):
        lines.append(label + " & " + " & ".join(f"{100 * k / n:.0f}\\%" for k, n in (agg[key]["counts"][s] for s in SYSTEMS)) + r" \\")
    r0, r1 = agg["rule"], agg["turn-taking 150 ms, backchannels allowed in pauses"]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Items passed when the stored measurements are scored under other thresholds; nothing is re-run or "
              r"re-judged. The pause variant allows utterances of at most four words that end before the user resumes. The "
              r"whisper variants still require a correct answer, and the hand-corrected cascade item stays not whispered. "
              rf"With the lenient timing rules, the rank correlation between timing and spoken-task accuracy is "
              rf"{r1['rho_six']:.2f} for the six full-duplex systems ({r0['rho_six']:.2f} under the rules) and "
              rf"{r1['rho_seven']:.2f} with the cascade ({r0['rho_seven']:.2f}).}}",
              r"\label{tab:thresholds}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
