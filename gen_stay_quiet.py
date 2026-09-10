"""Generate the stay_quiet_until_help test set: the user thinks out loud (the model should stay silent) until an explicit request for help ("help me…").

Output: test_set/stay_quiet_until_help/<NN>/{dialogue.txt, benchmark.json, A1.wav, A2.wav}
Each item: A=explains "I'm thinking, don't interrupt", B=ack, A=a stretch of thinking out loud + an explicit ask ("...okay, can you help me, <question>?"), B=a short answer.
Trigger word = the "help" in the ask; the model should stay silent while the user thinks aloud and only answer after the ask.
Usage: uv run python gen_stay_quiet.py (idempotent)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import gen_benchmark as gb
import generate_tts_mimo as tts

VOICE = "Mia"
TTS_MODEL = "mimo-v2.5-tts"
OUT = Path(__file__).resolve().parent / "test_set" / "stay_quiet_until_help"


def _retry(fn, tries=5):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


# each item: the think-aloud plants "sounds like asking yourself" bait (hmm, questions) that shouldn't lure the model out; only "help me" is the real signal.
DIALOGUES = [
    """A: I'm going to think through something out loud. Stay quiet until I actually ask for your help.
B: Okay, I'll stay quiet until you ask.
A: So it was forty, then minus twelve, carry the one... is that right? hmm, okay, can you help me, what is forty minus twelve?
B: Twenty-eight.""",
    """A: Let me work this out on my own first — don't jump in until I ask you.
B: Sure, I'll wait until you ask.
A: The capital of Australia, is it Sydney? no, Melbourne maybe?... ugh, okay, help me out, what is the capital of Australia?
B: Canberra.""",
    """A: I want to figure this out myself, so please stay silent until I say I need help.
B: Understood, I won't say anything until then.
A: How many days in February on a leap year, twenty-eight? no that's normal... alright, can you help me, how many days in February in a leap year?
B: Twenty-nine.""",
    """A: Give me a moment to think out loud, and don't interrupt until I ask for help.
B: Got it, I'll hold off until you ask.
A: Five times seven, that's thirty-something, thirty-five? or forty?... okay I give up, help me, what is five times seven?
B: Thirty-five.""",
    """A: I'm brainstorming out loud — stay quiet unless I ask you directly for help.
B: Okay, I'll wait for you to ask.
A: The opposite of 'expand', is it shrink? compress?... hmm, can you help me, what is the opposite of expand?
B: Contract.""",
    """A: Let me try to remember this on my own; don't say anything until I ask.
B: Sure, I'll stay quiet.
A: Water boils at, what, ninety degrees? no, higher... okay, help me out, at what temperature does water boil in Celsius?
B: One hundred degrees.""",
    """A: I'll think this through myself first, so hold your comments until I ask for help.
B: No problem, I'll wait.
A: How do you spell 'necessary', is it one c two s, or two c?... alright, can you help me, how do you spell necessary?
B: N-E-C-E-S-S-A-R-Y.""",
    """A: Just let me mutter through this — please stay silent until I actually ask you.
B: Okay, I'll keep quiet until then.
A: Three squared is nine, and then plus four... is that thirteen? or... hmm, help me, what is three squared plus four?
B: Thirteen.""",
    """A: I want to solve this on my own, so don't chime in until I ask for your help.
B: Understood, I'll wait for your cue.
A: A dozen is twelve, so half a dozen is... six? or eight?... okay, can you help me, how many is half a dozen?
B: Six.""",
    """A: Let me think out loud for a second, and stay quiet until I ask.
B: Sure thing, I'll wait.
A: The largest planet, is it Saturn? no, the really big one... ugh, help me out, what is the largest planet in our solar system?
B: Jupiter.""",
]


def main():
    tok = (os.environ.get("MIMO_TTS_TOKEN") or "").strip()
    if not tok:
        raise SystemExit("MIMO_TTS_TOKEN not set (.env)")
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    for i, dlg in enumerate(DIALOGUES, 1):
        d = OUT / f"{i:02d}"
        d.mkdir(exist_ok=True)
        if (d / "benchmark.json").exists() and (d / "A1.wav").exists() and (d / "A2.wav").exists():
            print(f"⏭  {i:02d} already exists, skipping")
            continue
        out = _retry(lambda: gb.generate(dlg, category="stay_quiet_until_help"))
        ev = out["benchmark"]["expected_events"][0]
        assert ev["expected_action"] == "TAKE_FLOOR" and ev["trigger"], f"{i:02d} wrong structure: {ev}"
        (d / "dialogue.txt").write_text(dlg, encoding="utf-8")
        (d / "benchmark.json").write_text(json.dumps(out["benchmark"], ensure_ascii=False, indent=2), encoding="utf-8")
        for n, line in zip(("A1", "A2"), out["user_lines"][:2]):
            wav, _ = _retry(lambda: tts.synthesize(api_key=tok, ref_data_uri=VOICE, text=line, model=TTS_MODEL, proxy=None))
            (d / f"{n}.wav").write_bytes(wav)
        print(f"✓ {i:02d}: anchor={out['anchor_word']!r} target={out['target_response']!r}")
    print(f"\ndone → {OUT}")


if __name__ == "__main__":
    main()
