"""Generate the keyword_wait test set: pre-written A/B dialogues → gen_benchmark produces reference answers → MiMo TTS produces A1/A2.wav.

keyword_wait = the user reads out a list, and the AI must cut in **the instant a word of the target category is spoken** (word is mid-sentence with more to follow → overlap required).
Output: test_set/keyword_wait/<NN>/{dialogue.txt, benchmark.json, A1.wav, A2.wav}, same structure as the existing IQ tasks.
Usage: uv run python gen_keyword_wait.py   (idempotent: entries that already have benchmark.json are skipped)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import gen_benchmark as gb
import generate_tts_mimo as tts

import time

VOICE = "Mia"
TTS_MODEL = "mimo-v2.5-tts"
OUT = Path(__file__).resolve().parent / "test_set" / "keyword_wait"


def _retry(fn, tries=5):
    """gateway/TTS tends to hit SSL EOF through the congested proxy → back off and retry."""
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last

# each item: A=instruction, B=ack, A=list containing the keyword (keyword is **mid-sentence** with more words after it), B=cut-in. Keyword category varies, exactly one per list.
DIALOGUES = [
    """A: I'm going to list some groceries. The instant you hear a fruit, jump in and say it.
B: Got it, I'll call out the first fruit I hear.
A: So we need milk, eggs, apple, bread, and some cheese.
B: Apple!""",
    """A: I'll read out some objects. The moment you hear an animal, cut in and name it.
B: Sure, I'll jump in on the first animal.
A: On the desk there's a lamp, a folder, a cat, and a stapler.
B: Cat!""",
    """A: I'm naming a few activities. As soon as you hear a city, interrupt and say it.
B: Okay, I'll break in the second I hear a city.
A: My weekend was all running, swimming, Paris, cooking, and reading.
B: Paris!""",
    """A: Listen to this list. The instant you hear a color, jump in with it.
B: Understood, I'll shout the first color.
A: In the drawer there's a pen, a ruler, a red marker, and some tape.
B: Red!""",
    """A: I'll say some words. The moment you hear a number, cut in and say it.
B: Right, I'll call out the first number.
A: My notes just say apple, banana, seven, orange, and grape.
B: Seven!""",
    """A: I'm listing things around the room. Interrupt the second you hear a musical instrument.
B: Got it, I'll jump in on the instrument.
A: There's a chair, a lamp, a guitar, a table, and a rug.
B: Guitar!""",
    """A: Here's a list of hobbies. The instant you hear a sport, break in and name it.
B: Sure, I'll call out the first sport.
A: She's into painting, singing, tennis, drawing, and baking.
B: Tennis!""",
    """A: I'll name a few items. The moment you hear a body part, jump in and say it.
B: Okay, I'll interrupt on the body part.
A: On the shelf: a mug, a book, an elbow pad, and a clock.
B: Elbow!""",
    """A: I'm reading a schedule. As soon as you hear a day of the week, cut in with it.
B: Got it, I'll shout the first day.
A: We've got coffee, a meeting, Tuesday, some errands, and dinner.
B: Tuesday!""",
    """A: Listen to this grocery list. The instant you hear a vegetable, jump in and name it.
B: Right, I'll call out the first vegetable.
A: We need bread, milk, carrot, butter, and eggs.
B: Carrot!""",
]


def main():
    tok = (os.environ.get("MIMO_TTS_TOKEN") or "").strip()
    if not tok:
        raise SystemExit("MIMO_TTS_TOKEN not set (.env)")
    OUT.mkdir(parents=True, exist_ok=True)
    for i, dlg in enumerate(DIALOGUES, 1):
        d = OUT / f"{i:02d}"
        d.mkdir(exist_ok=True)
        if (d / "benchmark.json").exists() and (d / "A1.wav").exists() and (d / "A2.wav").exists():
            print(f"⏭  {i:02d} already exists, skipping")
            continue
        out = _retry(lambda: gb.generate(dlg, category="keyword_wait"))   # LLM annotation (via gateway, with retry)
        ev = out["benchmark"]["expected_events"][0]
        assert ev["expected_action"] == "INTERRUPT" and ev["trigger"], f"{i:02d} is not a timing item: {ev}"
        (d / "dialogue.txt").write_text(dlg, encoding="utf-8")
        import json
        (d / "benchmark.json").write_text(json.dumps(out["benchmark"], ensure_ascii=False, indent=2), encoding="utf-8")
        for n, line in zip(("A1", "A2"), out["user_lines"][:2]):   # direct to xiaomimimo, no proxy
            wav, _ = _retry(lambda: tts.synthesize(api_key=tok, ref_data_uri=VOICE, text=line, model=TTS_MODEL, proxy=None))
            (d / f"{n}.wav").write_bytes(wav)
        print(f"✓ {i:02d}: anchor={out['anchor_word']!r} target={out['target_response']!r}  ({out['user_lines'][1][:40]}…)")
    print(f"\ndone → {OUT}")


if __name__ == "__main__":
    main()
