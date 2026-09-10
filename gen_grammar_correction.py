"""Regenerate grammar inputs with various national accents (non-native learners) → test_set/grammar_correction/
(source = the original native-English version test_set/grammar_correction_old/; text and reference answers copied verbatim, only swapping the speaker's accented timbre).

Text (dialogue/benchmark) is copied as-is; only A1 (instruction) + A2 (the sentence with grammar errors) are re-voiced via MiMo TTS's
**director structure** (Role: a person of some country learning English with an accent / Scene / Guidance). No re-testing.
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
import generate_tts_mimo as tts

SRC = Path(__file__).resolve().parent / "test_set" / "grammar_correction_old"
OUT = Path(__file__).resolve().parent / "test_set" / "grammar_correction"

# per item: (country adjective, voice preset) — the user's list of countries + a few more, voices mixed by gender for variety
ITEMS = {
    "01": ("French", "Mia"),
    "02": ("German", "Dean"),
    "03": ("Chinese", "Chloe"),
    "04": ("Japanese", "Milo"),
    "05": ("Korean", "Mia"),
    "06": ("Indian", "Dean"),
    "07": ("Spanish", "Chloe"),
    "08": ("Italian", "Milo"),
    "09": ("Russian", "Mia"),
    "10": ("Brazilian", "Dean"),
}


def director(country: str) -> str:
    return (f"Role: A {country} person in their late twenties who is still learning English and is not "
            f"fluent. They speak English with a strong, authentic {country} accent and often make grammar "
            f"mistakes.\n"
            f"Scene: Practicing spoken English out loud during a one-on-one language lesson.\n"
            f"Guidance: Speak with a genuine, natural {country} accent; clearly non-native and a little "
            f"halting; calm tone; medium intensity.")


def a_lines(dialogue_path: Path):
    """The two 'A:' lines in dialogue.txt = A1 (instruction) / A2 (the sentence with grammar errors)."""
    out = []
    for line in dialogue_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.upper().startswith("A:"):
            out.append(s.split(":", 1)[1].strip())
    return out


def _retry(fn, tries=5):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def main():
    tok = (os.environ.get("MIMO_TTS_TOKEN") or "").strip()
    if not tok:
        raise SystemExit("MIMO_TTS_TOKEN not set")
    OUT.mkdir(parents=True, exist_ok=True)
    for it, (country, voice) in ITEMS.items():
        src = SRC / it
        dst = OUT / it
        dst.mkdir(exist_ok=True)
        if not (src / "dialogue.txt").exists():
            print(f"✗ {it}: source missing dialogue.txt, skipping"); continue
        shutil.copy(src / "dialogue.txt", dst / "dialogue.txt")     # text/reference answers as-is
        shutil.copy(src / "benchmark.json", dst / "benchmark.json")
        lines = a_lines(src / "dialogue.txt")
        if len(lines) < 2:
            print(f"✗ {it}: fewer than 2 A lines"); continue
        style = director(country)
        for name, text in (("A1", lines[0]), ("A2", lines[1])):
            wav, _ = _retry(lambda: tts.synthesize(api_key=tok, ref_data_uri=voice, text=text,
                                                    style=style, model="mimo-v2.5-tts", proxy=None))
            (dst / f"{name}.wav").write_bytes(wav)
        print(f"✓ {it}: {country} accent ({voice})  A2={lines[1][:45]}")
    print(f"\ndone → {OUT}")


if __name__ == "__main__":
    main()
