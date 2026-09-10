"""Turn interaction_groundedness generated scenarios → TTS producing A1/A2/… (each scenario's User turns), placed into test_set.

Each scenario's USER turns (including the first "<10 words" instruction) are TTS'd in order into A1.wav, A2.wav, … (same multi-turn structure as keyword_wait).
The script's AI turns are not TTS'd — at test time A1..An are fed as multiple turns and the model generates its own responses.

Usage: uv run python gen_interaction_tts.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)
import generate_tts_mimo as tts

VOICE = "Mia"                 # user's voice (single speaker, consistent with the other tasks)
TTS_MODEL = "mimo-v2.5-tts"
SRC = HERE / "interaction_groundedness" / "scenarios_generated.json"
OUT = HERE / "test_set" / "interaction_groundedness"


def _retry(fn, n=5):
    import time
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == n - 1:
                raise
            time.sleep(2)


def main():
    tok = (os.environ.get("MIMO_TTS_TOKEN") or "").strip()
    if not tok:
        raise SystemExit("MIMO_TTS_TOKEN not set (.env)")
    scenarios = json.loads(SRC.read_text())["scenarios"]
    OUT.mkdir(parents=True, exist_ok=True)
    for i, sc in enumerate(scenarios, 1):
        nn = f"{i:02d}"
        d = OUT / nn
        d.mkdir(parents=True, exist_ok=True)
        user_turns = [t["text"] for t in sc["turns"] if t["speaker"] == "user"]
        # write benchmark.json (includes probes/ground_truth for later grading)
        (d / "benchmark.json").write_text(json.dumps({
            "category": "interaction_groundedness", "scenario_id": sc.get("id", nn),
            "title": sc.get("title", ""), "n_user_turns": len(user_turns),
            "turns": sc["turns"], "ground_truth": sc.get("ground_truth", {})}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        for k, line in enumerate(user_turns, 1):
            f = d / f"A{k}.wav"
            if f.exists():
                continue
            wav, _ = _retry(lambda: tts.synthesize(api_key=tok, ref_data_uri=VOICE, text=line,
                                                    model=TTS_MODEL, proxy=None))
            f.write_bytes(wav)
        print(f"  {nn} {sc.get('title','')[:34]:34}  {len(user_turns)} User turns → A1..A{len(user_turns)}")
    print(f"\nwrote {len(scenarios)} items → {OUT}")


if __name__ == "__main__":
    main()
