"""Check that a gpt-live run's model track really landed where the server said it did.

live_align.py places each speech burst so its words sit at the `start_ms` the server reported. This
re-transcribes the PLACED track and measures whether they actually do -- if the burst detection, the
word matching or the rendering were off, the residual shows up here.

It also prints the user side for context, but does NOT treat it as a reference: input transcripts come
from a streaming recogniser and run a few hundred ms late, which is exactly the trap that put one
item's answer ahead of its question. Only `residual` is a verdict.

Usage: check_live_align.py <run_dir> [run_dir ...]
"""
import json
import sys
from pathlib import Path
from statistics import median

from live_align import _asr_words, _pairs, _server_words


def check(run: Path):
    tr_p = run / "transcript_clock.json"
    if not tr_p.exists():
        print(f"── {run}\n   no transcript_clock.json (run predates session-clock placement)")
        return
    tr = json.loads(tr_p.read_text())
    print(f"── {run}")

    mp = run / "B_model.parakeet.json"
    if mp.exists():
        pairs = _pairs(_server_words(tr, "model"), _asr_words(mp))
        if pairs:
            resid = [st - at for at, st in pairs]
            m = median(resid)
            ok = "aligned" if abs(m) < 0.15 else "systematic offset remains"
            print(f"   model track  residual median {m:+.2f}s  "
                  f"(range {min(resid):+.2f} ~ {max(resid):+.2f}, n={len(resid)})  [{ok}]")
        else:
            print("   model track: no word could be matched to the server transcript")

    up = run / "A_user.parakeet.json"
    if up.exists():
        upairs = _pairs(_server_words(tr, "user"), _asr_words(up))
        if upairs:
            print(f"   user track   input-ASR lag {median(st - at for at, st in upairs):+.2f}s"
                  f"  (n={len(upairs)}, for reference only, not a baseline)")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        check(Path(a))
