"""Make the arrival placement GPT-Live's primary recording, and keep the session-clock placement beside it.

After regrade_arrival.py has written <run>/arrival/ for every GPT-Live run, this moves the session-clock
tracks and results at the top of each run directory into <run>/session_clock/ and moves the arrival tracks
and results up in their place. Every other system is placed by arrival already, so afterwards the top-level
files of all seven systems use the same timing reference. The raw stream, the clock log, the server
transcript, the scorecard, and premise_check.json stay at the top. Running it again changes nothing.

Usage:  uv run python promote_arrival.py [--dry-run]
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLACED = ["A_user.wav", "B_model.wav", "combined.wav", "A_user.parakeet.json", "B_model.parakeet.json",
          "B_model.parakeet.json.orig", "grade.json", "grade_interaction.json", "backchannel.json", "pause.json",
          "turn_taking.json", "user_backchannel.json", "interruption.json"]


def main():
    dry = "--dry-run" in sys.argv
    runs = sorted(ROOT.glob("tts_review/*/*/live_*_gptlive")) + sorted(ROOT.glob("test_set/*/*/live_*_gptlive"))
    moved = skipped = missing = 0
    for d in runs:
        a, s = d / "arrival", d / "session_clock"
        if s.exists():
            skipped += 1
            continue
        if not (a / "placement.json").exists():
            missing += 1
            print(f"  ! no arrival placement: {d.relative_to(ROOT)}")
            continue
        if dry:
            moved += 1
            continue
        s.mkdir()
        for name in PLACED:
            if (d / name).exists():
                shutil.move(str(d / name), str(s / name))
        for name in PLACED + ["placement.json"]:
            if (a / name).exists():
                shutil.move(str(a / name), str(d / name))
        shutil.rmtree(a)
        moved += 1
    print(f"{'would promote' if dry else 'promoted'} {moved} runs, {skipped} already done, {missing} without an arrival placement")


if __name__ == "__main__":
    main()
