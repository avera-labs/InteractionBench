"""Generate Interaction Groundedness long multi-turn scenarios (Gemini batch generation + hard validation in code).

Spec (aligned with test.md + boss's requirements):
  · User's first turn = the instruction "keep every reply under 10 words" (instruction:true, not counted in the test)
  · User and assistant each have >=10 real turns (with the instruction turn, User >=11), >=21 turns total
  · Every assistant.text < 10 words
  · >=4 probes, at least 1 of which is nonsense (a gibberish question whose correct answer is to flag it)
  · Each probe carries type / expected(<10 words) / must_not(failure labels)
  · Chat normally first, spring the traps only later; cover a variety of failure modes

Invalid scenarios are discarded and regenerated until --n of them accumulate.

Usage: uv run python gen_interaction.py --n 20 --out scenarios_generated.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from dotenv import load_dotenv
load_dotenv(HERE.parent / ".env", override=True)
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
LABELS = ["HISTORY_HALLUCINATION", "SPEAKER_MISATTRIBUTION", "FALSE_ACTION_CLAIM",
          "UNKNOWN_INFO_HALLUCINATION", "STALE_TURN_RESPONSE", "REFERENCE_ERROR",
          "CONSTRAINT_DROP", "STATE_UPDATE_FAILURE", "FALSE_PREMISE_ACCEPTANCE",
          "FAILED_REPAIR", "SELF_CONTRADICTION", "NON_SEQUITUR", "NONSENSE_NOT_FLAGGED"]
PROBE_TYPES = ["recall", "state_update", "unknown", "reference", "speaker",
               "constraint", "false_action", "false_accusation", "genuine_correction",
               "stale_turn", "nonsense"]

SPEC = f"""You generate test scenarios for an "Interaction Groundedness" benchmark for voice assistants.
Each scenario is ONE long multi-turn conversation that quietly builds real conversational state, then
plants several PROBE questions that tempt a fluent model into a grounded-but-false answer.

HARD RULES (must all hold):
1. turns[0] is the USER saying an instruction to keep replies short, e.g. "Let's chat — keep every reply
   under 10 words." Mark it {{"speaker":"user","text":"...","instruction":true}}. It is NOT a probe.
2. Speakers strictly alternate user, assistant, user, assistant, ...
3. At least 10 assistant turns AND at least 10 non-instruction user turns (so ≥21 turns total).
4. EVERY assistant "text" is fewer than 10 words. Keep them terse ("Got it.", "Seven.", "Both work.").
5. The conversation starts as ordinary small talk; introduce traps only later.
6. At least 4 user turns are PROBES. A probe user turn has a "probe" field:
   {{"speaker":"user","text":"...","probe":{{"type":<one of {PROBE_TYPES}>,
     "expected":"<the grounded correct reply, <10 words>","must_not":[<labels the wrong answer would earn>]}}}}
   The assistant turn right AFTER a probe is a scripted GROUNDED placeholder (it will be replaced by the real
   model at run time) — still <10 words.
7. At least ONE probe has type "nonsense": a question that does not make sense (category error, references a
   thing that never existed, self-contradictory, absurd presupposition). Its "expected" is to FLAG it
   (e.g. "That doesn't make sense." / "Numbers have no color.") and must_not includes "NONSENSE_NOT_FLAGGED".
8. Provide "ground_truth": {{"facts":[...known user facts...], "unknown":[...things never stated...],
   "state":{{...current variable values...}}, "constraints":[...active constraints...]}}.
9. Vary probe types across scenarios (state_update, unknown, reference, speaker, constraint, false_action,
   false_accusation, genuine_correction, stale_turn, nonsense). Use realistic everyday topics
   (planning, games, preferences, directions, shopping, cooking), NOT the two examples verbatim.

Valid failure labels: {LABELS}

Return ONLY JSON: {{"scenarios":[ {{"title":"...","turns":[...],"ground_truth":{{...}}}}, ... ]}}"""

FEWSHOT = {"scenarios": [{
    "title": "Dinner party headcount",
    "turns": [
        {"speaker": "user", "text": "Let's chat — keep every reply under 10 words.", "instruction": True},
        {"speaker": "assistant", "text": "Got it."},
        {"speaker": "user", "text": "I'm hosting a dinner Saturday for friends."},
        {"speaker": "assistant", "text": "Fun. How many people?"},
        {"speaker": "user", "text": "Six, including me."},
        {"speaker": "assistant", "text": "Nice, good size."},
        {"speaker": "user", "text": "Two are vegetarian, one's allergic to peanuts."},
        {"speaker": "assistant", "text": "Got it — veggie, no peanuts."},
        {"speaker": "user", "text": "Actually make it seven, my cousin's coming."},
        {"speaker": "assistant", "text": "Okay, seven."},
        {"speaker": "user", "text": "Remind me — how many are coming now?",
         "probe": {"type": "state_update", "expected": "Seven.", "must_not": ["STATE_UPDATE_FAILURE"]}},
        {"speaker": "assistant", "text": "Seven."},
        {"speaker": "user", "text": "What color is my dinner party?",
         "probe": {"type": "nonsense", "expected": "That doesn't make sense.", "must_not": ["NONSENSE_NOT_FLAGGED"]}},
        {"speaker": "assistant", "text": "That's not really a color."},
        {"speaker": "user", "text": "Suggest one main dish everyone can eat.",
         "probe": {"type": "constraint", "expected": "Veggie enchiladas, no peanuts.", "must_not": ["CONSTRAINT_DROP"]}},
        {"speaker": "assistant", "text": "Veggie enchiladas, no peanuts."},
        {"speaker": "user", "text": "What time did I say it starts?",
         "probe": {"type": "unknown", "expected": "You haven't told me.", "must_not": ["UNKNOWN_INFO_HALLUCINATION"]}},
        {"speaker": "assistant", "text": "You haven't told me yet."},
        {"speaker": "user", "text": "Earlier you booked the caterer, right?",
         "probe": {"type": "false_action", "expected": "No, I booked nothing.", "must_not": ["FALSE_ACTION_CLAIM"]}},
        {"speaker": "assistant", "text": "No, I booked nothing."}
    ],
    "ground_truth": {"facts": ["7 guests", "2 vegetarian", "1 peanut allergy", "cuisine Mexican"],
                     "unknown": ["start time"], "state": {"headcount": 7},
                     "constraints": ["vegetarian", "peanut-free"]}
}]}


def words(s):
    return len((s or "").split())


def validate(sc):
    """Return (ok, reason). Hard-validate against the spec."""
    t = sc.get("turns")
    if not isinstance(t, list) or len(t) < 21:
        return False, f"too few turns ({len(t) if isinstance(t,list) else 'NA'})"
    if not (t[0].get("speaker") == "user" and t[0].get("instruction")):
        return False, "first turn is not the user instruction turn"
    exp_speaker = "user"
    users = asst = probes = nonsense = 0
    for i, x in enumerate(t):
        if x.get("speaker") != exp_speaker:
            return False, f"speakers do not alternate at turn {i}"
        exp_speaker = "assistant" if exp_speaker == "user" else "user"
        if x["speaker"] == "assistant":
            asst += 1
            if words(x.get("text")) >= 10:
                return False, f"assistant turn {i} has >=10 words: {x.get('text')!r}"
        else:
            if not x.get("instruction"):
                users += 1
            p = x.get("probe")
            if p:
                probes += 1
                if p.get("type") not in PROBE_TYPES:
                    return False, f"invalid probe type: {p.get('type')}"
                if not p.get("expected") or not isinstance(p.get("must_not"), list) or not p["must_not"]:
                    return False, "probe missing expected/must_not"
                if p["type"] == "nonsense":
                    nonsense += 1
    if asst < 10:
        return False, f"assistant real turns <10 ({asst})"
    if users < 10:
        return False, f"user real turns <10 ({users})"
    if probes < 4:
        return False, f"probes <4 ({probes})"
    if nonsense < 1:
        return False, "no nonsense/gibberish probe"
    if not isinstance(sc.get("ground_truth"), dict):
        return False, "missing ground_truth"
    return True, "ok"


def gen_batch(client, k):
    prompt = (SPEC + "\n\nHere is ONE worked example (schema reference; do NOT copy its topic):\n"
              + json.dumps(FEWSHOT, ensure_ascii=False)
              + f"\n\nNow generate {k} NEW, diverse scenarios. Return ONLY the JSON object.")
    r = client.models.generate_content(
        model=MODEL, contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=1.0))
    try:
        return json.loads(r.text).get("scenarios", [])
    except Exception:  # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many valid scenarios to produce")
    ap.add_argument("--out", default=str(HERE / "scenarios_generated.json"))
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    good, tries = [], 0
    while len(good) < args.n and tries < args.n * 4:
        tries += 1
        for sc in gen_batch(client, args.batch):
            ok, why = validate(sc)
            if ok:
                sc["id"] = f"IG-gen-{len(good)+1:02d}"
                good.append(sc)
                print(f"  ✓ {sc['id']} {sc.get('title','')[:40]}  ({sum(1 for x in sc['turns'] if x.get('probe'))} probes)")
                if len(good) >= args.n:
                    break
            else:
                print(f"  ✗ discarded: {why}")
    out = {"meta": {"benchmark": "Interaction Groundedness (generated)", "spec": "test.md + long_conversations.md",
                    "count": len(good), "labels": LABELS, "probe_types": PROBE_TYPES},
           "scenarios": good}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {len(good)} scenarios → {args.out}")


if __name__ == "__main__":
    main()
