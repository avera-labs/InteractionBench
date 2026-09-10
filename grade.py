#!/usr/bin/env python3
"""Run the v4 behavioral evals from the command line — no dashboard.

Point it at a run/answer folder holding the two-track recording (`A_user.wav` / `B_model.wav`)
and/or its word-level ASR (`A_user.parakeet.json` / `B_model.parakeet.json`). Missing ASR is
produced on the fly with local parakeet (unless `--no-asr`). Uses the SAME graders as the
dashboard (`grade_behavior.py`). Results print as a summary and are written to
`<folder>/grade_results.json`.

Tasks: backchannel · interruption · turn_taking · pause · user_backchannel.

Examples:
  uv run python grade.py --list                                  # see which dashboard runs exist
  uv run python grade.py live_41612d313d03 --task pause --gemini  # by run name (single-model session auto-drills down)
  uv run python grade.py live_41612d313d03/personaplex           # pick a model; no --task = run all
  uv run python grade.py ./answer --task user_backchannel --gemini   # or pass a folder path directly
  uv run python grade.py ./answer --task interruption --interrupt-text "how do I fix a faucet?"
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

import grade_behavior as gb  # noqa: E402

ALL = ["backchannel", "interruption", "turn_taking", "pause", "user_backchannel", "benchmark"]
RUNS_DIR = ROOT / "dashboard" / "runs"   # runs saved by the dashboard: runs/live_<id>/<model>/
DATASET_DIR = ROOT / "dataset"           # curated/named test sets: dataset/<name>/


def _has_tracks(p: Path):
    return any((p / f).exists() for f in ("A_user.parakeet.json", "A_user.wav", "B_model.wav"))


def resolve_run(arg, runs_dir=RUNS_DIR):
    """Resolve the run name/path the user gave into the actual directory holding the two-track files. Accepts:
      · a direct path (./answer, /abs/path)
      · a dataset name: `turn_taking` → dataset/turn_taking/
      · a dashboard run name: `live_41612d313d03/personaplex`, or just the session `live_41612d313d03` (single model auto-drills down)."""
    for c in (Path(arg), DATASET_DIR / arg, runs_dir / arg):
        if not c.is_dir():
            continue
        if _has_tracks(c):
            return c, None
        subs = [s for s in sorted(c.iterdir()) if s.is_dir() and _has_tracks(s)]   # session dir → drill down into model subdirs
        if len(subs) == 1:
            return subs[0], None
        if len(subs) > 1:
            opts = ", ".join(f"{c.name}/{s.name}" for s in subs)
            return None, f"'{arg}' has multiple models: please specify one -- {opts}"
    return None, f"run not found: '{arg}' (neither a path nor under {runs_dir}). Use --list to see what's available."


def list_runs(runs_dir=RUNS_DIR):
    if not runs_dir.is_dir():
        print(f"(no {runs_dir})"); return
    rows = []
    for sess in sorted(runs_dir.iterdir()):
        if not sess.is_dir():
            continue
        subs = [s for s in sorted(sess.iterdir()) if s.is_dir() and _has_tracks(s)]
        for s in (subs or ([sess] if _has_tracks(sess) else [])):
            name = s.name if s == sess else f"{sess.name}/{s.name}"
            asr = "ASR✓" if (s / "A_user.parakeet.json").exists() else "wav (ASR run at grade time)"
            rows.append((name, asr))
    if not rows:
        print(f"(no runs under {runs_dir})"); return
    print(f"\n{len(rows)} runs under {runs_dir}:\n")
    for name, asr in rows:
        print(f"  {name:42s} {asr}")
    print()


def ensure_asr(d: Path, do_asr=True, quiet=False):
    """Missing *.parakeet.json but the matching wav exists → run local parakeet (same as the dashboard's "run parakeet")."""
    need = [(stem, d / f"{stem}.wav", d / f"{stem}.parakeet.json")
            for stem in ("A_user", "B_model")
            if not (d / f"{stem}.parakeet.json").exists() and (d / f"{stem}.wav").exists()]
    if not need:
        return
    if not do_asr:
        if not quiet:
            print(f"  ⚠ missing ASR and --no-asr: {[n[0] for n in need]} (related tasks will skip)")
        return
    try:
        import parakeet_local
    except Exception as ex:  # noqa: BLE001
        print(f"  ⚠ local parakeet needed but import failed (install: uv sync --extra asr): {type(ex).__name__}: {ex}", file=sys.stderr)
        return
    for stem, wav, pj in need:
        if not quiet:
            print(f"  🎧 parakeet transcribing {stem}.wav …", flush=True)
        words, text = parakeet_local.transcribe_wav_segmented(str(wav))
        pj.write_text(json.dumps(parakeet_local._doc_for(wav, words, text), ensure_ascii=False, indent=2),
                      encoding="utf-8")


def run_task(task, d, use_gemini, api_key, interrupt_text):
    if task == "backchannel":
        return gb.grade_backchannel(d, use_gemini, api_key)
    if task == "interruption":
        return gb.grade_interruption(d, use_gemini, api_key, interrupt_text=interrupt_text)
    if task == "turn_taking":
        return gb.grade_turn_taking(d)
    if task == "pause":
        return gb.grade_pause(d, use_gemini, api_key)
    if task == "user_backchannel":
        return gb.grade_user_backchannel(d, use_gemini, api_key)
    if task == "benchmark":
        return gb.grade_benchmark(d, use_gemini, api_key)
    return {"error": f"unknown task {task}"}


def verdict_line(task, r):
    """One human-readable verdict line."""
    if "error" in r:
        return f"skip — {r['error']}"
    if task == "turn_taking":
        return (f"{'✅ replied' if r['tor'] else '❌ no reply'} · latency {r['latency_ms']}ms · you finished at "
                f"{r['turn_end']}s · '{r['heard_after']}'")
    if task == "pause":
        v = "❌ grabbed floor" if r["tor"] else ("🟡 brief sound" if r["jumped_in"] else "✅ stayed quiet")
        j = f" · Gemini {r['judged']['took_floor'] and 'took over' or 'continuer'}" if r.get("judged") else ""
        return f"{v} · pause window {r['pause_window']}s · TOR {r['tor']}{j}"
    if task == "backchannel":
        return (f"backchannel {r['n_backchannel']} · attempt {r['n_attempt']} · response {r['n_response']} · "
                f"TOR {r['tor']} · freq {r['freq_per_s']}/s")
    if task == "interruption":
        rel = (r.get("relevance") or {}).get("addressed", "—")
        return (f"{'✅ took new turn' if r['tor'] else '❌ no reply'} · latency {r['latency_ms']}ms · relevance {rel} · "
                f"was_talking {r['was_talking']}")
    if task == "user_backchannel":
        return f"{'❌ derailed' if r['derailed'] else '✅ not derailed'} · {r['behaviour']} · {r['n_backchannel']} backchannels"
    if task == "benchmark":
        s = r["summary"]
        mark = "✅" if s["n"] and s["npass"] == s["n"] else ("🟡" if s["npass"] else "❌")
        evs = " · ".join(f"{e.get('event')} {e.get('action','')}/{e.get('grade_dimension','')}={e.get('status')}"
                         for e in r["events"] if e.get("status"))
        return f"{mark} {r['category']} · {s['npass']}/{s['n']} pass (rate {s['rate']}) · {evs}"
    return json.dumps(r, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(
        description="v4 behavioral eval CLI (backchannel / interruption / turn_taking / pause / user_backchannel)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("run_dir", nargs="?", help="run name or folder: a dashboard run name (live_<id>[/model]) or a direct path (./answer)")
    ap.add_argument("--list", action="store_true", help="list the gradeable runs under dashboard/runs, then exit")
    ap.add_argument("--task", action="append", choices=ALL + ["all"],
                    help="tasks to run (repeatable; default all — runs each, skips ones missing files)")
    ap.add_argument("--gemini", action="store_true", help="enable Gemini content judging (needs GEMINI_API_KEY in .env)")
    ap.add_argument("--interrupt-text", default="", help="interruption utterance text (interruption task; if omitted, reads interruption.json or falls back to pause segmentation)")
    ap.add_argument("--no-asr", action="store_true", help="don't auto-run parakeet when ASR is missing")
    ap.add_argument("--out", default=None, help="results JSON path (default <run_dir>/grade_results.json)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.list:
        list_runs(); return
    if not a.run_dir:
        ap.error("provide a run name or path (or use --list to see what's available)")
    d, err = resolve_run(a.run_dir)
    if err:
        ap.error(err)
    tasks = ALL if (not a.task or "all" in a.task) else [t for t in ALL if t in a.task]

    use_gemini = bool(a.gemini)
    api_key = os.environ.get("GEMINI_API_KEY") if use_gemini else None
    if use_gemini and not api_key:
        print("⚠ --gemini but no GEMINI_API_KEY in .env → falling back to no-Gemini (word-count/timing heuristics)", file=sys.stderr)

    itext = a.interrupt_text
    if not itext and "interruption" in tasks and (d / "interruption.json").exists():
        try:
            itext = json.loads((d / "interruption.json").read_text()).get("interrupt_text", "") or ""
        except Exception:  # noqa: BLE001
            pass

    ensure_asr(d, do_asr=not a.no_asr, quiet=a.quiet)

    try:
        disp = str(d.relative_to(RUNS_DIR))
    except ValueError:
        disp = d.name
    print(f"\n▶ {disp}  ·  tasks: {', '.join(tasks)}  ·  gemini={'on' if api_key else 'off'}\n")
    results = {}
    for t in tasks:
        r = run_task(t, d, use_gemini, api_key, itext)
        results[t] = r
        print(f"  {t:17s} {verdict_line(t, r)}")

    out = Path(a.out) if a.out else d / "grade_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ detailed results → {out}\n")


if __name__ == "__main__":
    main()
