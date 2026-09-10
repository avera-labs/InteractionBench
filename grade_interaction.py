"""Interaction Groundedness scoring —— checks whether a long multi-turn conversation "stays logical / holds the true state".

Method (aligned with test.md's two-stage judge):
  1. **Time-align** the model's actual replies to each user turn in the script (segment the user track into turns → the model speech after each turn = its reply).
  2. Rebuild the "USER(script) / AI(what the model actually said)" conversation.
  3. LLM judge: first rebuild the true state from the script, then judge pass/fail + failure label + reason **per probe**; then give an **overall logical coherence**.
Each ▶ probe is one scoring point (recall / state update / unknown / reference / speaker / constraint / feint / false accusation / nonsense question…).

Usage: uv run python grade_interaction.py --model gpt              # grade all scenarios for this model
       uv run python grade_interaction.py --model all --items 01  # all models, one scenario
       uv run python grade_interaction.py --model all             # all models, all scenarios + summary
"""
import argparse
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
from dotenv import load_dotenv
load_dotenv(HERE / ".env", override=True)
from google import genai
from google.genai import types

CAT = "test_set/interaction_groundedness"
JUDGE_MODEL = "gemini-2.5-flash"
MODELS = ["gpt", "gemini", "personaplex", "moshi", "freezeomni", "halfduplex"]


def _words(p):
    try:
        return json.load(open(p)).get("words", [])
    except Exception:  # noqa: BLE001
        return []


def _segment(ws, gap=1.2):
    """Word list → [(t0, t1, text)]; a gap > gap between words breaks into a new segment/turn."""
    segs, cur = [], []
    for w in ws:
        if cur and w["t0"] - cur[-1]["t1"] > gap:
            segs.append(cur); cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return [(s[0]["t0"], s[-1]["t1"], " ".join(x["word"] for x in s)) for s in segs]


def interleave(live_dir):
    """Word-level interleave → true conversation flow [(t, _, 'USER'/'AI', text)]. Sort all words from both tracks by time, breaking a segment wherever the speaker changes ——
    so a line that interrupts one side's continuous ramble lands correctly after the interruption point (whole-segment pairing / big-gap splitting would misplace it earlier)."""
    allw = ([(w["t0"], "USER", w["word"]) for w in _words(f"{live_dir}/A_user.parakeet.json")]
            + [(w["t0"], "AI", w["word"]) for w in _words(f"{live_dir}/B_model.parakeet.json")])
    allw.sort(key=lambda x: x[0])
    flow, cur_spk, cur, t0 = [], None, [], 0.0
    for t, spk, word in allw:
        if spk != cur_spk and cur:                    # speaker change → close a segment
            flow.append((t0, 0, cur_spk, " ".join(cur))); cur = []
        if not cur:
            t0 = t
        cur_spk = spk; cur.append(word)
    if cur:
        flow.append((t0, 0, cur_spk, " ".join(cur)))
    return flow


JUDGE_SYS = """You are a strict grader for an "Interaction Groundedness" voice-assistant benchmark.
A long multi-turn conversation actually took place. You get the TIME-ORDERED transcript of what was actually
said: USER lines (fixed script, spoken to the model) and AI lines (what the model ACTUALLY said, via ASR — so
expect minor transcription noise and occasional overlap; judge MEANING, not spelling). A full-duplex model may
talk while the user talks, so an AI line may interleave; read the whole flow.

Certain USER turns are PROBES that test whether the AI stayed logically grounded in the real conversation:
remembering correctly, tracking state updates, admitting what it was NOT told, respecting constraints,
attributing who-said-what, refusing false premises, and FLAGGING nonsense questions instead of fabricating.

You are given: the ground_truth state, and the list of PROBES — each has the exact `question` text, its `type`,
`expected` (the grounded correct reply) and `must_not` (failure labels a wrong answer earns).

First silently rebuild the true state from the USER lines. Then for EACH probe: locate that probe's `question`
in the transcript and find where the AI ADDRESSES that probe's content. The audio was de-gapped, so timing is
compressed and USER turns may be merged into one line; the AI often answers several questions in one breath, and
its reply to a probe may sit just before, just after, or batched with adjacent answers in a single AI utterance.
Match by MEANING, not strict position. Grade:
- pass=true if the AI's answer to that probe is logically grounded (matches `expected`'s meaning / does the
  right thing); pass=false if it hallucinated history, invented unknown info, dropped a constraint, accepted a
  false premise, failed to flag nonsense, contradicted itself, or answered a different turn. Use label
  "NO_RESPONSE" ONLY if the AI genuinely never addresses the probe's content anywhere in the transcript.
- If pass=false, set `label` to the single best-matching failure label (from must_not or the most apt one).
  If pass=true, label=null.
- `heard`: the AI reply you graded (short quote from the transcript). `reason`: one terse sentence.

Also give overall `coherence` 0-100: how logically consistent the WHOLE conversation is (not just probes) —
did the AI stay on topic, avoid self-contradiction, and act like it truly remembered the conversation.

Return ONLY JSON:
{"probes":[{"type":"...","question":"<echo the probe question>","pass":true/false,"label":null_or_"LABEL",
  "heard":"<what AI said>","reason":"..."}],
 "coherence":<0-100>,"summary":"<one line on this model's grounding in this scenario>"}"""


def judge(client, scenario, flow):
    lines = [f'[{t:5.1f}s] {who}: {txt}' for t, _, who, txt in flow]
    probes = [{"question": ut["text"], "type": ut["probe"].get("type"),
               "expected": ut["probe"].get("expected"), "must_not": ut["probe"].get("must_not")}
              for ut in scenario["turns"] if ut.get("probe")]
    payload = (f'GROUND_TRUTH:\n{json.dumps(scenario.get("ground_truth", {}), ensure_ascii=False)}\n\n'
               f'PROBES (grade each — find its question in the transcript, grade the AI reply after it):\n'
               f'{json.dumps(probes, ensure_ascii=False)}\n\n'
               f'TIME-ORDERED TRANSCRIPT (what actually happened):\n' + "\n".join(lines))
    r = client.models.generate_content(
        model=JUDGE_MODEL, contents=[JUDGE_SYS + "\n\n" + payload],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
    try:
        return json.loads(r.text)
    except Exception:  # noqa: BLE001
        return {"probes": [], "coherence": None, "summary": "judge parse error"}


def grade_one(client, model, item):
    sc_path = f"{CAT}/{item}/benchmark.json"
    if not os.path.exists(sc_path):
        return None
    scenario = json.load(open(sc_path))
    ds = glob.glob(f"{CAT}/{item}/live_*_{model}")
    if not ds:
        return None
    flow = interleave(ds[0])
    verdict = judge(client, scenario, flow)
    probes = verdict.get("probes", [])
    passed = sum(1 for p in probes if p.get("pass"))
    n_expected = sum(1 for ut in scenario["turns"] if ut.get("probe"))
    out = {"model": model, "item": item, "title": scenario.get("title"),
           "n_probes": n_expected, "graded": len(probes), "passed": passed,
           "coherence": verdict.get("coherence"), "summary": verdict.get("summary"),
           "probes": probes}
    json.dump(out, open(f"{ds[0]}/grade_interaction.json", "w"), ensure_ascii=False, indent=1)
    _write_grade_json(ds[0], out)                # also overwrite the standard grade.json (otherwise it's the generic grader's empty shell)
    return out


def _write_grade_json(live_dir, out):
    """Write the logic-eval results into the standard grade.json (category/events/summary{n,npass,rate}),
    overwriting the empty shell auto-generated by the headless run, so grade.json itself holds the real results and matches the site's read format."""
    events = [{"type": p.get("type"), "pass": bool(p.get("pass")), "label": p.get("label"),
               "question": p.get("question"), "heard": p.get("heard"), "reason": p.get("reason")}
              for p in out["probes"]]
    npass = out["passed"]; n = out["n_probes"]
    g = {"category": "interaction_groundedness",
         "events": events,
         "summary": {"n": n, "npass": npass, "rate": round(npass / n, 3) if n else 0},
         "coherence": out["coherence"], "verdict": out["summary"], "grader": "grade_interaction.py"}
    json.dump(g, open(f"{live_dir}/grade.json", "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gpt/gemini/personaplex/moshi/freezeomni/halfduplex or all")
    ap.add_argument("--items", nargs="*", default=[f"{i:02d}" for i in range(1, 11)])
    ap.add_argument("--print-recon", action="store_true", help="only print the aligned reconstructed conversation (no scoring, for debugging)")
    a = ap.parse_args()
    models = MODELS if a.model == "all" else [a.model]

    if a.print_recon:
        for m in models:
            for it in a.items:
                ds = glob.glob(f"{CAT}/{it}/live_*_{m}")
                if not ds:
                    continue
                flow = interleave(ds[0])
                print(f"\n===== {m}/{it}  time-ordered conversation ({len(flow)} segments) =====")
                for t, _, who, txt in flow:
                    print(f'  [{t:5.1f}s] {who:4s}: {txt[:66]}')
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    allrows = []
    for m in models:
        rows = []
        for it in a.items:
            r = grade_one(client, m, it)
            if r:
                rows.append(r); allrows.append(r)
                print(f"  {m}/{it}: probes {r['passed']}/{r['n_probes']} passed, coherence {r['coherence']}  ({r['title']})")
        if rows:
            tp = sum(x["passed"] for x in rows); tn = sum(x["n_probes"] for x in rows)
            coh = [x["coherence"] for x in rows if isinstance(x["coherence"], (int, float))]
            print(f"── {m}: probes {tp}/{tn} ({100*tp//max(tn,1)}%), avg coherence {sum(coh)/len(coh):.0f}" if coh else f"── {m}: probes {tp}/{tn}")
    if len(models) > 1 and allrows:
        print("\n===== summary =====")
        for m in models:
            rows = [x for x in allrows if x["model"] == m]
            if not rows:
                continue
            tp = sum(x["passed"] for x in rows); tn = sum(x["n_probes"] for x in rows)
            coh = [x["coherence"] for x in rows if isinstance(x["coherence"], (int, float))]
            print(f"  {m:12s}: probes {tp:3d}/{tn:3d} ({100*tp//max(tn,1):3d}%) | coherence {sum(coh)/len(coh):.0f}" if coh else f"  {m}: {tp}/{tn}")
        json.dump(allrows, open(f"{CAT}/_grade_summary.json", "w"), ensure_ascii=False, indent=1)
        print(f"\nsummary written to {CAT}/_grade_summary.json")


if __name__ == "__main__":
    main()
