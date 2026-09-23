"""Re-grade GPT-Live's timing runs with its audio placed by arrival (arrival_align.py).

For every GPT-Live run of the five timing scenarios, write <run>/arrival/ with the arrival-placed tracks,
their ASR, and the grader's result, calling the graders exactly as headless_run.py does. With --spoken,
do the same for the three spoken tasks whose pass criteria use timing (grammar, keyword-wait,
stay-quiet), applying regrade_keyword_wait.py and regrade_stay_quiet.py as for every other system. The
session-clock files at the top of each run directory are not touched.

Hand corrections: five GPT-Live interruption verdicts were corrected by hand (`manual_correction` in the
session-clock result) because the grader never sent a short correct answer to the judge. If the same
answer is again left unjudged on the arrival timeline, the correction is carried over and marked.

Usage:  uv run python regrade_arrival.py                     # all 140 timing runs
        uv run python regrade_arrival.py turn_taking/18      # one scenario/item
        uv run python regrade_arrival.py --spoken            # the 30 timed spoken-task runs
        uv run python regrade_arrival.py --ig                # the 10 groundedness conversations
        uv run python regrade_arrival.py --content           # place the 41 content-only runs; grades kept
"""
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

import arrival_align
import grade_behavior as gb
import parakeet_local
import regrade_keyword_wait
import regrade_stay_quiet

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
TIMING = ["backchannel", "pause", "turn_taking", "user_backchannel", "interruption"]
SPOKEN = ["grammar_correction", "keyword_wait", "stay_quiet_until_help"]
CONTENT = ["logic_puzzle", "countdown_completion", "whisper_production", "volume_understanding", "alternating_count"]
SESSION_FILES = ["A_user.wav", "B_model.wav", "combined.wav", "A_user.parakeet.json", "B_model.parakeet.json"]


def copy_silent_run(d: Path, dest: Path, result_names):
    """A run in which the model never spoke has nothing to place: its result is the same on either clock."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in SESSION_FILES + result_names:
        if (d / name).exists():
            shutil.copy(d / name, dest / name)
    (dest / "placement.json").write_text(json.dumps(
        {"placed": False, "why": "no model speech in this run, so the result is the same on either clock; "
                                 "files copied from the session-clock placement"}, indent=1))


def silent(d: Path):
    return not json.loads((d / "B_model_raw.parakeet.json").read_text()).get("words")


def carry_transcript_fix(d: Path, dest: Path):
    """Re-apply a hand correction of the session-clock transcript (kept as B_model.parakeet.json.orig)."""
    orig_p = d / "B_model.parakeet.json.orig"
    if not orig_p.exists():
        return None
    before = json.loads(orig_p.read_text())["words"]
    after = json.loads((d / "B_model.parakeet.json").read_text())["words"]
    fixes = [(x["word"], y["word"]) for x, y in zip(before, after) if x["word"] != y["word"]]
    p = dest / "B_model.parakeet.json"
    shutil.copy(p, dest / "B_model.parakeet.json.orig")
    doc = json.loads(p.read_text())
    for wrong, right in fixes:
        for w in doc["words"]:
            if w["word"] == wrong:
                w["word"] = right
                break
        doc["text"] = doc.get("text", "").replace(wrong, right, 1)
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return fixes


def wall_clock_start():
    """Median of (wall time of the first delta) - (its placed position) over the runs placed from their words."""
    from statistics import median
    v = []
    for p in list(ROOT.glob("tts_review/*/*/live_*_gptlive/arrival/placement.json")) + \
             list(ROOT.glob("test_set/*/*/live_*_gptlive/arrival/placement.json")):
        info = json.loads(p.read_text())
        if info.get("placed") and info.get("anchor_from") in ("transcript", None):
            first = json.loads((p.parent.parent / "clock.jsonl").read_text().splitlines()[0])
            v.append(first["t"] - info["anchor_s"])
    return median(v)


def regrade_content():
    """Placement-independent scenarios: write the arrival tracks and their ASR, and keep the existing grade."""
    start = wall_clock_start()
    for task in CONTENT:
        for d in sorted((ROOT / "test_set" / task).glob("*/live_*_gptlive")):
            dest = d / "arrival"
            if (dest / "placement.json").exists():
                continue
            if silent(d):
                copy_silent_run(d, dest, ["grade.json"])
                continue
            info = arrival_align.place(d, dest, fallback_start=start)
            shutil.copy(d / "A_user.parakeet.json", dest / "A_user.parakeet.json")
            transcribe(dest)
            fixes = carry_transcript_fix(d, dest)
            if fixes:
                info["transcript_fix"] = f"hand correction carried over from the session-clock transcript: {fixes}"
            shutil.copy(d / "grade.json", dest / "grade.json")
            info["grade"] = "copied: this scenario's grade does not depend on where the audio is placed"
            (dest / "placement.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            print(f"  {task}/{d.parent.name}: placed, grade kept", flush=True)


def runs(selection=None, spoken=False):
    for task in (SPOKEN if spoken else TIMING):
        base = ROOT / ("test_set" if spoken else "tts_review") / task
        for d in sorted(base.glob("*/live_*_gptlive")):
            if selection and f"{task}/{d.parent.name}" not in selection:
                continue
            yield task, d


def grade_spoken(task, d: Path, dest: Path, key):
    shutil.copy(d / "benchmark.json", dest / "benchmark.json")
    if task == "stay_quiet_until_help":
        target = json.loads((d / "benchmark.json").read_text())["expected_events"][0].get("target_response", "")
        res = regrade_stay_quiet.regrade_item(dest, target)
        g = {"category": task, "events": [{"event_id": "e1", "category": task, "grade_dimension": "restraint+content",
                                           "status": res["status"], "restraint": res["restraint"],
                                           "content_ok": res["content"], **res}]}
        status = res["status"]
    else:
        g = gb.grade_benchmark(dest, use_gemini=True, api_key=key)
        status = g["events"][0].get("status")
        if task == "keyword_wait":
            words = json.loads((dest / "B_model.parakeet.json").read_text()).get("words", [])
            status = regrade_keyword_wait.regrade_event(g["events"][0], words)
    g["summary"] = {"n": 1, "npass": 1 if status == "pass" else 0, "rate": 1.0 if status == "pass" else 0.0}
    (dest / "grade.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    return g


def transcribe(dest: Path):
    wav = dest / "B_model.wav"
    words, text = parakeet_local.transcribe_wav_segmented(str(wav))
    (dest / "B_model.parakeet.json").write_text(
        json.dumps(parakeet_local._doc_for(wav, words, text), ensure_ascii=False), encoding="utf-8")


def grade(task, d: Path, dest: Path, key):
    orig = json.loads((d / f"{task}.json").read_text())
    gemini = bool((orig.get("gemini") or {}).get("asked"))
    k = key if gemini else None
    if task == "interruption":
        res = gb.grade_interruption(dest, use_gemini=gemini, api_key=k, interrupt_text=orig.get("interrupt_text", ""))
        mc = orig.get("manual_correction")
        rel = (res.get("relevance") or {}).get("addressed")
        if mc and rel != "yes" and res.get("was_talking"):
            answer = (orig.get("relevance") or {}).get("heard", "")
            res["relevance"] = dict(orig["relevance"])
            res["manual_correction"] = dict(mc, carried_over="the arrival re-grade again left this answer unjudged: "
                                            f"{answer!r} (judge said {rel!r})")
    elif task == "turn_taking":
        res = gb.grade_turn_taking(dest)
    else:
        res = gb.TASKS[task](dest, use_gemini=gemini, api_key=k)
    (dest / f"{task}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def regrade_ig(key):
    """Groundedness: re-judge each conversation from the arrival-placed tracks, merged by time as before."""
    import grade_interaction as gi
    from google import genai
    client = genai.Client(api_key=key)
    for d in sorted((ROOT / "test_set" / "interaction_groundedness").glob("*/live_*_gptlive")):
        dest = d / "arrival"
        if (dest / "placement.json").exists() and (dest / "grade_interaction.json").exists():
            continue
        info = arrival_align.place(d, dest)
        if not info.get("placed"):
            print(f"  ! ig/{d.parent.name}: {info.get('why')}", flush=True)
            continue
        shutil.copy(d / "A_user.parakeet.json", dest / "A_user.parakeet.json")
        transcribe(dest)
        scenario = json.loads((d.parent / "benchmark.json").read_text())
        verdict = gi.judge(client, scenario, gi.interleave(str(dest)))
        probes = verdict.get("probes", [])
        out = {"model": "gptlive", "item": d.parent.name, "title": scenario.get("title"),
               "n_probes": sum(1 for ut in scenario["turns"] if ut.get("probe")), "graded": len(probes),
               "passed": sum(1 for p in probes if p.get("pass")), "coherence": verdict.get("coherence"),
               "summary": verdict.get("summary"), "probes": probes}
        (dest / "grade_interaction.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        gi._write_grade_json(str(dest), out)
        (dest / "placement.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"  ig/{d.parent.name}: {out['passed']}/{out['n_probes']} probes, coherence {out['coherence']}", flush=True)


def main():
    args = sys.argv[1:]
    if "--ig" in args:
        regrade_ig(os.environ.get("GEMINI_API_KEY"))
        return
    if "--content" in args:
        regrade_content()
        return
    spoken = "--spoken" in args
    selection = set(a for a in args if a != "--spoken") or None
    key = os.environ.get("GEMINI_API_KEY")
    done = 0
    for task, d in runs(selection, spoken):
        dest = d / "arrival"
        result = dest / ("grade.json" if spoken else f"{task}.json")
        if not selection and (dest / "placement.json").exists() and result.exists():
            continue                                   # finished in an earlier pass; placement.json is written last
        if silent(d):
            copy_silent_run(d, dest, ["grade.json" if spoken else f"{task}.json"])
            print(f"  {task}/{d.parent.name}: no model speech, result copied", flush=True)
            continue
        info = arrival_align.place(d, dest)
        if not info.get("placed"):
            print(f"  ! {task}/{d.parent.name}: {info.get('why')}", flush=True)
            continue
        shutil.copy(d / "A_user.parakeet.json", dest / "A_user.parakeet.json")
        transcribe(dest)
        if spoken:
            grade_spoken(task, d, dest, key)
        else:
            grade(task, d, dest, key)
        (dest / "placement.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        done += 1
        print(f"  {task}/{d.parent.name}: anchor {info['anchor_s']:.2f}s from {info['anchor_words']} words "
              f"(sd {info['anchor_sd_s']:.2f})", flush=True)
    print(f"re-graded {done} runs")


if __name__ == "__main__":
    main()
