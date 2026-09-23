"""Protocol measurements quoted in the paper, computed from the recordings of every system.

  injection    median times of the injected user speech (Appendix table): the interruption measured from the
               reply onset and from the end of the question, and the two user backchannels measured from the
               reply onset. Reply onset = the first system word that starts no earlier than 0.2 s before the
               question ends.
  silence      in the logic, countdown, and grammar items, the median silence between the end of the system's
               audio and the next user turn
  overlap      two-turn items (the five spoken tasks) in which the next user turn started while the system was
               speaking, by the same audio criterion as `landed`, and how often those items passed
  landed       groundedness probes that started while the system was speaking (at least 60% of the preceding
               0.5 s voiced and the system still sounding just after), and how often they failed; probes whose
               premise does not hold in the run (check_probe_premises.py) are left out, as in the scores

Usage:  uv run python protocol_stats.py
"""
import glob
import json
import re
import statistics as st
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SYSTEMS = ["gptlive", "gpt", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]
SPOKEN = ["logic_puzzle", "countdown_completion", "grammar_correction", "keyword_wait", "stay_quiet_until_help"]


def words(p):
    try:
        return json.loads(Path(p).read_text()).get("words") or []
    except Exception:  # noqa: BLE001
        return []


def utts(ws, gap):
    out = []
    for w in ws:
        if out and w["t0"] - out[-1][-1]["t1"] <= gap:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def injection(s):
    first, second, int_onset, int_qend = [], [], [], []
    for d in glob.glob(f"{ROOT}/tts_review/user_backchannel/*/live_*_{s}"):
        u, m = utts(words(f"{d}/A_user.parakeet.json"), 0.8), words(f"{d}/B_model.parakeet.json")
        if len(u) < 3:
            continue
        qend = u[0][-1]["t1"]
        reply = [w for w in m if w["t0"] >= qend - 0.2]
        if reply:
            first.append(u[1][0]["t0"] - reply[0]["t0"])
            second.append(u[2][0]["t0"] - reply[0]["t0"])
    for d in glob.glob(f"{ROOT}/tts_review/interruption/*/live_*_{s}"):
        j = json.loads(Path(d, "interruption.json").read_text())
        at = j.get("interrupt_at")
        u, m = words(f"{d}/A_user.parakeet.json"), words(f"{d}/B_model.parakeet.json")
        if not at or not u:
            continue
        before = [w for w in u if w["t1"] <= at[0] - 0.3]
        if not before:
            continue
        qend = before[-1]["t1"]
        reply = [w for w in m if w["t0"] >= qend - 0.2]
        int_qend.append(at[0] - qend)
        if reply and reply[0]["t0"] < at[0]:
            int_onset.append(at[0] - reply[0]["t0"])
    med = lambda v: round(st.median(v), 2) if v else None
    return med(int_onset), med(int_qend), med(first), med(second)


def voiced_mask(wav, hop=0.02, thr=0.01):
    x, sr = sf.read(wav)
    x = x if x.ndim == 1 else x[:, 0]
    h = int(hop * sr)
    n = len(x) // h
    return np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1)) >= thr


def during_speech(mask, t, hop=0.02):
    """The system is speaking at time t: >= 60% of the preceding 0.5 s voiced and still sounding just after."""
    i = int(t / hop)
    return 0 < i < len(mask) and mask[max(0, i - int(0.5 / hop)):i].mean() > 0.6 and bool(mask[i:i + int(0.2 / hop)].any())


def next_turns(s, cats, hop=0.02):
    """[(item dir, silence, during)]: silence = next user turn start - end of the system's last voiced frame before it."""
    out = []
    for cat in cats:
        for d in glob.glob(f"{ROOT}/test_set/{cat}/*/live_*_{s}"):
            u = utts(words(f"{d}/A_user.parakeet.json"), 1.0)
            if len(u) < 2:
                out.append((d, None, False))
                continue
            a2 = u[-1][0]["t0"]
            mask = voiced_mask(f"{d}/B_model.wav", hop)
            upto = mask[:int(a2 / hop)]
            if not upto.any():                   # the system said nothing before the next turn
                out.append((d, None, False))
                continue
            last = (len(upto) - 1 - int(np.argmax(upto[::-1]))) * hop + hop
            out.append((d, a2 - last, during_speech(mask, a2, hop)))
    return out


def passed(d):
    g = json.loads(Path(d, "grade.json").read_text())
    return (g.get("events") or [{}])[0].get("status") == "pass"


def landed(s, hop=0.02, thr=0.01):
    toks = lambda t: re.sub(r"[^a-z0-9' ]", " ", (t or "").lower()).split()
    n_l = f_l = n_o = f_o = unmatched = 0
    for item in [f"{i:02d}" for i in range(1, 11)]:
        ds = glob.glob(f"{ROOT}/test_set/interaction_groundedness/{item}/live_*_{s}")
        if not ds:
            continue
        d = ds[0]
        bench = json.loads(Path(ROOT, "test_set/interaction_groundedness", item, "benchmark.json").read_text())
        probes = [t["text"] for t in bench["turns"] if t.get("probe")]
        pc = Path(d, "premise_check.json")
        invalid = {p["question"] for p in json.loads(pc.read_text()) if p.get("premise_holds") is False} if pc.exists() else set()
        gp = json.loads(Path(d, "grade_interaction.json").read_text()).get("probes", [])
        flat = [(tk, w["t0"]) for w in words(f"{d}/A_user.parakeet.json") for tk in toks(w["word"])]
        ftok = [t for t, _ in flat]
        x, sr = sf.read(f"{d}/B_model.wav")
        x = x if x.ndim == 1 else x[:, 0]
        h = int(hop * sr)
        n = len(x) // h
        voiced = np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1)) >= thr
        pos = 0
        for q in probes:
            qt = toks(q)
            L = min(len(qt), 8)
            best = (0.0, None)
            for k in range(pos, len(ftok)):
                r = SequenceMatcher(None, qt[:L], ftok[k:k + L]).ratio()
                if r > best[0]:
                    best = (r, k)
            if best[1] is None or best[0] < 0.6:
                unmatched += 1
                continue
            k = best[1]
            pos = k + 1
            if any(SequenceMatcher(None, qt, toks(b)).ratio() >= 0.75 for b in invalid):
                continue                          # premise does not hold in this run; left out of the score
            i = int(flat[k][1] / hop)
            during = 0 < i < n and voiced[max(0, i - int(0.5 / hop)):i].mean() > 0.6 and voiced[i:i + int(0.2 / hop)].any()
            v, vr = None, 0.0
            for p in gp:
                r = SequenceMatcher(None, qt, toks(p.get("question"))).ratio()
                if r > vr:
                    v, vr = p, r
            ok = bool(v and vr >= 0.7 and v.get("pass"))
            if during:
                n_l, f_l = n_l + 1, f_l + (not ok)
            else:
                n_o, f_o = n_o + 1, f_o + (not ok)
    return n_l, f_l, n_o, f_o, unmatched


def main():
    print("injection medians (s): interruption after reply onset / after question end; backchannels first / second")
    for s in SYSTEMS:
        print(f"  {s:12}", injection(s))
    print("\nsilence before the next user turn in logic, countdown, grammar (median s, from the audio)")
    for s in SYSTEMS:
        g = [x for _, x, _ in next_turns(s, SPOKEN[:3]) if x is not None]
        print(f"  {s:12} {st.median(g):5.2f}  (n={len(g)})" if g else f"  {s:12} --")
    print("\ntwo-turn items where the next user turn started during system speech: count; pass rate overlapped vs not")
    for s in SYSTEMS:
        turns = next_turns(s, SPOKEN)
        ov = [d for d, _, dur in turns if dur]
        rest = [d for d, _, dur in turns if not dur]
        pr = lambda ds: f"{sum(passed(d) for d in ds)}/{len(ds)}" if ds else "-"
        print(f"  {s:12} {len(ov):2} of {len(turns)}   passed {pr(ov)} vs {pr(rest)}")
    print("\ngroundedness probes started during system speech: landed (failed) / not landed (failed) / unmatched")
    for s in SYSTEMS:
        print(f"  {s:12}", landed(s))


if __name__ == "__main__":
    main()
