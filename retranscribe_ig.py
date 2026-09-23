"""Re-transcribe the groundedness recordings that compact_gaps.py transcribed as whole tracks.

compact_gaps.py shortened the long silences in the groundedness recordings of Gemini Live, Moshi, PersonaPlex,
Freeze-Omni, and the cascade, and then ran Parakeet on each whole track. On a whole track Parakeet sometimes drops
an utterance: seven probe questions were missing from Freeze-Omni's user transcripts, so the judge saw no question
and graded a missing reply. This transcribes both tracks segment by segment, as every other recording is
transcribed, and keeps the previous transcript as <track>.parakeet.whole.json. The judgments and premise checks
made from the old transcript are moved aside (grade_interaction_runs.whole.json, premise_check.whole.json), so
rejudge_ig.py and check_probe_premises.py redo them. Runs already done are skipped.

Usage:  uv run python retranscribe_ig.py
"""
import json
from pathlib import Path

import parakeet_local

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "test_set" / "interaction_groundedness"
COMPACTED = ["gemini", "moshi", "personaplex", "freezeomni", "halfduplex"]


def retranscribe(d):
    for name in ("grade_interaction_runs", "premise_check"):
        if (d / f"{name}.json").exists() and not (d / f"{name}.whole.json").exists():
            (d / f"{name}.json").rename(d / f"{name}.whole.json")
    counts = []
    for trk in ("A_user", "B_model"):
        cur, old = d / f"{trk}.parakeet.json", d / f"{trk}.parakeet.whole.json"
        words, _ = parakeet_local.transcribe_wav_segmented(str(d / f"{trk}.wav"))
        if not old.exists():                    # an interrupted earlier pass may have moved it already
            cur.rename(old)
        before = len(json.loads(old.read_text()).get("words") or [])
        cur.write_text(json.dumps({"words": words}))
        counts.append(f"{trk} {before}->{len(words)} words")
    return ", ".join(counts)


def main():
    for item in sorted(p.name for p in CAT.iterdir() if (p / "benchmark.json").exists()):
        for s in COMPACTED:
            for d in sorted((CAT / item).glob(f"live_*_{s}")):
                if (d / "B_model.parakeet.whole.json").exists():
                    continue
                print(f"  {item}/{s}: {retranscribe(d)}", flush=True)


if __name__ == "__main__":
    main()
