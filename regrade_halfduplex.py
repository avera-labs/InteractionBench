"""Re-grade only (no re-simulation): scan all halfduplex results and re-score items containing judge_err using the existing audio.
The judge now retries; running single-process with no proxy contention, it usually passes in one go. Usage:
  uv run python regrade_halfduplex.py            # scan everything
  uv run python regrade_halfduplex.py tts_review/backchannel/5   # a specific directory
"""
import json
import os
import sys

import grade_behavior as gb

API = os.environ.get("GEMINI_API_KEY")
TASK_OF = {"backchannel": "backchannel", "pause": "pause", "user_backchannel": "user_backchannel",
           "interruption": "interruption"}


def _interrupt_text(item_dir):
    """interruption needs the interruption text, recovered from the parent index.txt (aligned with headless_run)."""
    src = os.path.dirname(item_dir)                    # tts_review/interruption
    idx = os.path.join(src, "index.txt")
    iid = os.path.basename(item_dir)                   # NN
    if not os.path.exists(idx):
        return ""
    for line in open(idx, encoding="utf-8"):
        if "\t" in line:
            fn, txt = line.split("\t", 1)
            if fn.endswith("_interrupt.wav") and fn.split("_")[0] == iid:
                return txt.strip()
    return ""


def _has_judge_err(obj):
    return "judge_err" in json.dumps(obj, ensure_ascii=False)


def regrade(folder):
    """folder = a live_*_halfduplex directory. Returns (task, changed?)."""
    parts = folder.replace("\\", "/").split("/")
    # tts_review/<task>/<N>/live_..  |  test_set/<cat>/<N>/live_..
    root = parts[0]
    if root == "test_set":
        rf = os.path.join(folder, "grade.json")
        if not os.path.exists(rf) or not _has_judge_err(json.load(open(rf))):
            return None, False
        item_dir = os.path.dirname(folder)
        if not os.path.exists(os.path.join(folder, "benchmark.json")):
            import shutil
            shutil.copy(os.path.join(item_dir, "benchmark.json"), os.path.join(folder, "benchmark.json"))
        res = gb.grade_benchmark(folder, use_gemini=True, api_key=API)
        json.dump(res, open(rf, "w"), ensure_ascii=False, indent=2)
        return "benchmark", True
    task = parts[1]
    rf = os.path.join(folder, f"{task}.json")
    if not os.path.exists(rf) or not _has_judge_err(json.load(open(rf))):
        return None, False
    if task == "interruption":
        res = gb.grade_interruption(folder, use_gemini=True, api_key=API,
                                    interrupt_text=_interrupt_text(os.path.dirname(folder)))
    elif task == "turn_taking":
        res = gb.grade_turn_taking(folder)
    else:
        res = gb.TASKS[task](folder, use_gemini=True, api_key=API)
    json.dump(res, open(rf, "w"), ensure_ascii=False, indent=2)
    return task, True


def main():
    import glob
    roots = sys.argv[1:] or ["tts_review", "test_set"]
    folders = []
    for r in roots:
        if r.endswith("_halfduplex"):
            folders.append(r)
        else:
            folders += glob.glob(f"{r}/**/live_*_halfduplex", recursive=True)
    fixed = 0
    for f in sorted(folders):
        task, changed = regrade(f)
        if changed:
            fixed += 1
            still = _has_judge_err(json.load(open(os.path.join(
                f, "grade.json" if task == "benchmark" else f"{task}.json"))))
            print(f"  {'✓' if not still else '✗still failing'} {f}  ({task})")
    print(f"==== re-graded {fixed} items containing judge_err ====")


if __name__ == "__main__":
    main()
