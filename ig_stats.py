"""Groundedness statistics quoted in the paper, over the probes whose premise holds (premise_check.json).

  by type      pass rate of each probe type across all systems
  hosted/open  nonsense questions flagged and false accusations rejected, per system
  labels       the most frequent failure labels across systems (a probe with no verdict counts as NO_RESPONSE
               only when the judge said so; missing verdicts are reported separately)

Usage:  uv run python ig_stats.py
"""
import collections
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "test_set" / "interaction_groundedness"
SYSTEMS = ["gptlive", "gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
HOSTED = ["gptlive", "gpt", "gemini"]
OPEN = ["moshi", "personaplex", "freezeomni"]
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def probes(system):
    """[(type, passed or None, label)] for every probe with a valid premise; None = the judge gave no verdict."""
    out = []
    for item in sorted(p.name for p in CAT.iterdir() if (p / "benchmark.json").exists()):
        ds = sorted((CAT / item).glob(f"live_*_{system}"))
        if not ds:
            continue
        d = ds[0]
        bench = json.loads((CAT / item / "benchmark.json").read_text())
        judged = json.loads((d / "grade_interaction.json").read_text()).get("probes", [])
        bad = set()
        if (d / "premise_check.json").exists():
            bad = {norm(p["question"]) for p in json.loads((d / "premise_check.json").read_text())
                   if p.get("premise_holds") is False}
        for t in bench["turns"]:
            if not t.get("probe") or norm(t["text"]) in bad:
                continue
            best = max(judged, key=lambda p: SequenceMatcher(None, norm(t["text"]), norm(p.get("question"))).ratio(),
                       default=None)
            if best and SequenceMatcher(None, norm(t["text"]), norm(best.get("question"))).ratio() >= 0.75:
                out.append((t["probe"]["type"], bool(best.get("pass")), best.get("label")))
            else:
                out.append((t["probe"]["type"], None, None))
    return out


def main():
    rows = {s: probes(s) for s in SYSTEMS}
    print("pass rate by probe type, all systems (no verdict = fail)")
    by = collections.defaultdict(lambda: [0, 0])
    for s in SYSTEMS:
        for t, ok, _ in rows[s]:
            by[t][0] += bool(ok)
            by[t][1] += 1
    for t, (k, n) in sorted(by.items(), key=lambda x: x[1][0] / x[1][1]):
        print(f"  {t:20} {k:3}/{n:<3} {k / n:.0%}")
    print("\nnonsense flagged / false accusations rejected, per system")
    for s in SYSTEMS:
        ns = [ok for t, ok, _ in rows[s] if t == "nonsense"]
        fa = [ok for t, ok, _ in rows[s] if t == "false_accusation"]
        print(f"  {s:12} nonsense {sum(bool(x) for x in ns)}/{len(ns)}   false accusation {sum(bool(x) for x in fa)}/{len(fa)}")
    for group, name in ((HOSTED, "hosted"), (OPEN, "open")):
        for t in ("false_accusation", "nonsense"):
            v = [ok for s in group for tt, ok, _ in rows[s] if tt == t]
            print(f"  {name:6} {t:17} {sum(bool(x) for x in v)}/{len(v)} = {sum(bool(x) for x in v) / len(v):.0%}")
    print("\nfailure labels, all systems")
    labels = collections.Counter(lab for s in SYSTEMS for _, ok, lab in rows[s] if ok is False and lab)
    for lab, c in labels.most_common(8):
        print(f"  {lab:28} {c}")
    missing = {s: sum(1 for _, ok, _ in rows[s] if ok is None) for s in SYSTEMS}
    print("probes without a verdict:", {s: m for s, m in missing.items() if m})
    print("missing replies among failures:", {s: sum(1 for _, ok, lab in rows[s] if ok is False and lab == "NO_RESPONSE")
                                              for s in SYSTEMS}, "| failures:",
          {s: sum(1 for _, ok, _ in rows[s] if not ok) for s in SYSTEMS})


if __name__ == "__main__":
    main()
