"""Re-grade the three IQ categories (logic_puzzle / countdown_completion / grammar_correction).

Rerun each item's grade.json with the improved content judge (grade_live._gemini_local now also feeds the full parakeet whole-track transcript to Gemini,
so the judge doesn't miss the ending cue/completion word in a verbose/repeated-counting reply).
Only overwrite successfully produced results; print the items whose verdict changed.
"""
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=True)
import grade_behavior as gb

KEY = os.environ.get("GEMINI_API_KEY")
CATS = ["logic_puzzle", "countdown_completion", "grammar_correction"]
only = sys.argv[1:] or CATS

changed, errs, n = [], 0, 0
for cat in only:
    for d in sorted(glob.glob(f"test_set/{cat}/*/live_*")):
        gj = Path(d) / "grade.json"
        if not gj.exists():
            continue
        n += 1
        old = json.loads(gj.read_text())
        old_rate = (old.get("summary") or {}).get("rate")
        res = gb.grade_benchmark(d, use_gemini=True, api_key=KEY)
        if "error" in res:
            errs += 1
            print(f"  ERR {d}: {res['error']}", flush=True)
            continue
        new_rate = res["summary"]["rate"]
        gj.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        if old_rate != new_rate:
            heard = ((res["events"][0].get("content") or {}).get("heard") or "")[:60]
            changed.append((d, old_rate, new_rate))
            print(f"  ✎ {d}: rate {old_rate} → {new_rate}  | {heard}", flush=True)

print(f"\n==== re-graded {n} items, {len(changed)} verdicts changed, {errs} errors ====")
for d, o, nw in changed:
    print(f"  {o} → {nw}  {d}")
