"""Voice evaluation dashboard -- local dashboard backend (aiohttp).

Step one only covers the **input side**:
  · POST /api/tts    {text, voice}  → MiMo TTS(preset) generate → save inputs/<id>.wav
  · POST /api/record  upload recording blob(webm/ogg) → convert to wav → save inputs/<id>.wav
  · GET  /inputs/<id>.wav            playback
  · GET  /                           frontend page

Later steps (send to model → grade → compare) add more routes on top.

Run:
  cd instruction_tuning
  env PYTHONPATH=. .venv/bin/python dashboard/server.py
  open http://localhost:8770 in a browser
"""
import asyncio
import io
import json
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import quote, urlparse

from aiohttp import web
from dotenv import load_dotenv
from pydub import AudioSegment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # instruction_tuning/
INPUTS = HERE / "inputs"; INPUTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))      # so live_ws in the same dir can be imported
load_dotenv(ROOT / ".env", override=True)   # .env wins, overriding any empty/stale vars left in the shell

from generate_tts_mimo import synthesize                              # noqa: E402
import grade_behavior as gb                                           # noqa: E402  single source of truth for behavior grading (shared by dashboard + CLI)

PRESET_VOICES = ["Mia", "Chloe", "Milo", "Dean"]   # Mia/Chloe female, Milo/Dean male
TTS_MODEL_PRESET = "mimo-v2.5-tts"
TTS_TOKEN_ENV = "MIMO_TTS_TOKEN"     # a MiMo sk-s token (.env), connects directly to api.xiaomimimo.com (no proxy)


def _tts_token():
    """TTS token -- read from MIMO_TTS_TOKEN in .env."""
    tok = (os.environ.get(TTS_TOKEN_ENV) or "").strip()
    if not tok:
        raise RuntimeError(f"{TTS_TOKEN_ENV} not set (.env) -- TTS needs a MiMo sk-s token")
    return tok


async def index(_req):
    return web.FileResponse(HERE / "index.html")


async def api_tts(req):
    try:
        data = await req.json()
        text = (data.get("text") or "").strip()
        voice = data.get("voice") or "Mia"
        if not text:
            return web.json_response({"error": "text is empty"}, status=400)
        if voice not in PRESET_VOICES:
            voice = "Mia"
        # connect directly to api.xiaomimimo.com (no proxy), token read from MIMO_TTS_TOKEN in .env
        wav_bytes, _usage = synthesize(
            api_key=_tts_token(), ref_data_uri=voice, text=text,
            proxy=None, model=TTS_MODEL_PRESET)
        iid = uuid.uuid4().hex[:12]
        (INPUTS / f"{iid}.wav").write_bytes(wav_bytes)
        (INPUTS / f"{iid}.txt").write_text(text, encoding="utf-8")
        secs = round(len(AudioSegment.from_file(io.BytesIO(wav_bytes))) / 1000.0, 1)
        return web.json_response({"id": iid, "url": f"/inputs/{iid}.wav",
                                  "text": text, "voice": voice, "source": "tts", "secs": secs})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


async def api_record(req):
    try:
        reader = await req.multipart()
        field = await reader.next()
        blob = await field.read()
        if not blob:
            return web.json_response({"error": "empty recording"}, status=400)
        seg = AudioSegment.from_file(io.BytesIO(blob))     # ffmpeg auto-detects webm/ogg
        iid = uuid.uuid4().hex[:12]
        seg.export(INPUTS / f"{iid}.wav", format="wav")
        return web.json_response({"id": iid, "url": f"/inputs/{iid}.wav",
                                  "source": "record", "secs": round(len(seg) / 1000.0, 1)})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


RUNS = HERE / "runs"; RUNS.mkdir(exist_ok=True)


def _combine(input_wav: Path, model_wav: Path, out_dir: Path):
    """Produce the "full conversation" + separate tracks (same timeline, both start at 0, equal length). Returns a url triple."""
    A = AudioSegment.from_file(input_wav)            # user track
    B = AudioSegment.from_file(model_wav)            # model track (model responses already placed at their real timestamps)
    n = max(len(A), len(B))
    A = A + AudioSegment.silent(n - len(A)) if len(A) < n else A
    B = B + AudioSegment.silent(n - len(B)) if len(B) < n else B
    out_dir.mkdir(parents=True, exist_ok=True)
    A.export(out_dir / "A_user.wav", format="wav")
    B.export(out_dir / "B_model.wav", format="wav")
    A.overlay(B).export(out_dir / "combined.wav", format="wav")
    rel = out_dir.relative_to(RUNS)
    return {f"{k}_url": f"/runs/{rel}/{k}.wav" for k in ()} | {
        "combined_url": f"/runs/{rel}/combined.wav",
        "A_url": f"/runs/{rel}/A_user.wav",
        "B_url": f"/runs/{rel}/B_model.wav",
    }


async def api_run(req):
    """{input_id, models:[gemini,...]} → run Live inference per model → produce combined + separate tracks."""
    try:
        data = await req.json()
        iid = data.get("input_id")
        models = data.get("models") or ["gemini"]
        try:
            silence_ms = int(data.get("silence_ms") or 700)
        except (TypeError, ValueError):
            silence_ms = 700
        silence_ms = max(100, min(2000, silence_ms))     # clamp to a sane range
        inp = INPUTS / f"{iid}.wav"
        if not inp.exists():
            return web.json_response({"error": "input does not exist"}, status=400)
        import live_drivers
        results = {}
        for m in models:
            out_dir = RUNS / iid / m
            model_wav = out_dir / "model_raw.wav"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                if m == "gemini":
                    info = await live_drivers.run_gemini(inp, model_wav)
                elif m == "gpt":
                    info = await live_drivers.run_gpt(inp, model_wav, silence_ms=silence_ms)
                else:
                    results[m] = {"error": f"unknown model {m}"}; continue
                tracks = _combine(inp, model_wav, out_dir)
                results[m] = {**info, **tracks}
            except Exception as e:  # noqa: BLE001
                results[m] = {"error": f"{type(e).__name__}: {e}"}
        return web.json_response({"input_id": iid, "results": results})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


async def live_page(_req):
    return web.FileResponse(HERE / "live.html")


async def tts_page(_req):
    return web.FileResponse(HERE / "tts.html")


async def ws_live(req):
    """Human ↔ model real-time conversation: browser mic <-> GPT-Realtime, both tracks written to disk."""
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(req)
    try:
        silence_ms = max(100, min(2000, int(req.query.get("silence_ms") or 500)))
    except (TypeError, ValueError):
        silence_ms = 500
    model = req.query.get("model") or "gpt"
    sid = uuid.uuid4().hex[:12]
    out_dir = RUNS / f"live_{sid}" / model
    import live_ws
    try:
        if model == "gpt":
            info = await live_ws.relay_gpt(ws, silence_ms, out_dir)
        elif model == "gemini":
            info = await live_ws.relay_gemini(ws, out_dir, silence_ms=silence_ms)
        elif model in ("moshi", "personaplex"):        # PersonaPlex = moshi architecture, same WS protocol
            moshi_url = req.query.get("moshi_url") or live_ws.MOSHI_DEFAULT_URL
            info = await live_ws.relay_moshi(ws, out_dir, moshi_url=moshi_url)
        elif model == "freezeomni":                    # FreezeOmni = Flask-SocketIO, raw PCM
            fo_url = req.query.get("fo_url") or live_ws.FO_DEFAULT_URL
            info = await live_ws.relay_freezeomni(ws, out_dir, fo_url=fo_url,
                                                  prompt=req.query.get("prompt") or "")
        else:
            await ws.send_str(json.dumps({"type": "err", "text": f"unknown model {model}"}))
            info = None
        if info is not None and not ws.closed:
            rel = out_dir.relative_to(RUNS)
            await ws.send_str(json.dumps({"type": "done", "secs": info["secs"],
                                          "combined_url": f"/runs/{rel}/combined.wav",
                                          "A_url": f"/runs/{rel}/A_user.wav",
                                          "B_url": f"/runs/{rel}/B_model.wav"}))
    except Exception as e:  # noqa: BLE001
        if not ws.closed:
            await ws.send_str(json.dumps({"type": "err", "text": f"{type(e).__name__}: {e}"}))
    finally:
        if not ws.closed:
            await ws.close()
    return ws


async def api_parakeet(req):
    """Local MLX parakeet produces word-level ASR for a run dir's separate tracks (A_user.wav/B_model.wav).
    {dir: "<runid>/<model>"} → {ok, tracks:{user:{text,words}, model:{...}}}."""
    try:
        data = await req.json()
        rel = (data.get("dir") or "").strip("/")
        d = (RUNS / rel).resolve()
        if not str(d).startswith(str(RUNS.resolve()) + "/") or not d.is_dir():
            return web.json_response({"error": "invalid directory"}, status=400)
        import parakeet_local
        loop = asyncio.get_event_loop()
        tracks = {}
        for tk, fname in (("user", "A_user.wav"), ("model", "B_model.wav")):
            wav = d / fname
            if not wav.exists():
                tracks[tk] = {"error": "missing track"}
                continue
            # parakeet is a blocking MLX call (the first time also loads the model from cache) → push to the thread pool, don't block the event loop.
            # Use the segmented version: on a full track with long silence parakeet drops utterances after the silence, so VAD-split into segments, transcribe each, then stitch absolute timestamps.
            words, text = await loop.run_in_executor(None, parakeet_local.transcribe_wav_segmented, wav)
            doc = parakeet_local._doc_for(wav, words, text)
            (d / (wav.stem + ".parakeet.json")).write_text(
                json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            tracks[tk] = doc                          # full doc (includes track/speaker/sr/text/words/note)
        return web.json_response({"ok": True, "tracks": tracks})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


async def api_download_run(req):
    """Zip up a run dir (full conversation + separate tracks + parakeet) for download, for debugging.
    GET  ?dir=<runid>/<model>                         → only zips the files already in the dir.
    POST {dir, benchmark?, dialogue?}                 → also writes the ground truth (benchmark.json) + dialogue source (dialogue.txt)
                                                        into the dir before zipping.
    → <runid>_<model>.zip (contains a folder of the same name). If parakeet results are missing, run them first."""
    import io
    import zipfile
    bench = dialogue = grade = backchannel = interruption = turn_taking = pause = user_backchannel = None
    if req.method == "POST":
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        rel = (body.get("dir") or "").strip("/")
        bench = body.get("benchmark")
        dialogue = body.get("dialogue")
        grade = body.get("grade")
        backchannel = body.get("backchannel")
        interruption = body.get("interruption")
        turn_taking = body.get("turn_taking")
        pause = body.get("pause")
        user_backchannel = body.get("user_backchannel")
    else:
        rel = (req.query.get("dir") or "").strip("/")
    d = (RUNS / rel).resolve()
    if not str(d).startswith(str(RUNS.resolve()) + "/") or not d.is_dir():
        return web.Response(status=400, text="invalid directory")
    # write ground truth + dialogue source + test results into the dir (zipped along with it)
    if bench is not None:
        (d / "benchmark.json").write_text(
            json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
    if dialogue:
        (d / "dialogue.txt").write_text(str(dialogue), encoding="utf-8")
    if grade is not None:
        (d / "grade.json").write_text(
            json.dumps(grade, ensure_ascii=False, indent=2), encoding="utf-8")
    if backchannel is not None:
        (d / "backchannel.json").write_text(
            json.dumps(backchannel, ensure_ascii=False, indent=2), encoding="utf-8")
    if interruption is not None:
        (d / "interruption.json").write_text(
            json.dumps(interruption, ensure_ascii=False, indent=2), encoding="utf-8")
    if turn_taking is not None:
        (d / "turn_taking.json").write_text(
            json.dumps(turn_taking, ensure_ascii=False, indent=2), encoding="utf-8")
    if pause is not None:
        (d / "pause.json").write_text(
            json.dumps(pause, ensure_ascii=False, indent=2), encoding="utf-8")
    if user_backchannel is not None:
        (d / "user_backchannel.json").write_text(
            json.dumps(user_backchannel, ensure_ascii=False, indent=2), encoding="utf-8")
    # ensure parakeet results exist (run the segmented version if not done yet)
    try:
        import parakeet_local
        loop = asyncio.get_event_loop()
        for _tk, fname in (("user", "A_user.wav"), ("model", "B_model.wav")):
            wav = d / fname
            js = d / (wav.stem + ".parakeet.json")
            if wav.exists() and not js.exists():
                words, text = await loop.run_in_executor(None, parakeet_local.transcribe_wav_segmented, wav)
                js.write_text(json.dumps(parakeet_local._doc_for(wav, words, text),
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass                                              # even if parakeet fails, still zip the audio
    folder = f"{d.parent.name}_{d.name}"                  # live_<sid>_<model>
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(d.iterdir()):
            if f.is_file():
                z.write(f, arcname=f"{folder}/{f.name}")
    return web.Response(body=buf.getvalue(), headers={
        "Content-Type": "application/zip",
        "Content-Disposition": f'attachment; filename="{folder}.zip"',
    })


def _gemini_key():
    """GEMINI_API_KEY -- read from the environment (.env)."""
    return os.environ.get("GEMINI_API_KEY")


async def api_grade_live(req):
    """Grade a live recording: score the separate-track recording against an inline benchmark (locally generated / example_data ground truth), timeline-free.
    {run_dir:"<runid>/<model>", benchmark, variant?, gemini?} -> {ok, category, events, summary}."""
    try:
        data = await req.json()
        run_dir = (data.get("run_dir") or "").strip("/")
        variant = (data.get("variant") or "A").strip()
        use_gemini = bool(data.get("gemini"))
        bench = data.get("benchmark")           # inline ground truth (locally generated / example_data)
        if not bench:
            return web.json_response({"error": "generate/load a benchmark (ground truth) first"}, status=400)
        d = (RUNS / run_dir).resolve()
        if not str(d).startswith(str(RUNS.resolve()) + "/") or not d.is_dir():
            return web.json_response({"error": "invalid recording directory"}, status=400)
        user_j, model_j = d / "A_user.parakeet.json", d / "B_model.parakeet.json"
        if not user_j.exists() or not model_j.exists():
            return web.json_response({"error": "click '🎧 run parakeet transcription' first to produce per-track ASR"}, status=400)

        import grade_live as gl
        api_key = _gemini_key() if use_gemini else None
        gemini_ok = bool(api_key) if use_gemini else True

        def _run():
            user_words = json.loads(user_j.read_text())["words"]
            model_words = json.loads(model_j.read_text())["words"]
            mwav, uwav = d / "B_model.wav", d / "A_user.wav"
            recs = gl.grade_live(bench, user_words, model_words,
                                 model_wav_path=str(mwav) if mwav.exists() else None,
                                 user_wav_path=str(uwav) if uwav.exists() else None,
                                 gemini=use_gemini, api_key=api_key)
            return bench.get("category"), recs

        category, recs = await asyncio.get_event_loop().run_in_executor(None, _run)
        graded = [r for r in recs if "content" in r or "timing" in r]
        npass = sum(r.get("status") == "pass" for r in graded)
        summary = {"n": len(graded), "npass": npass,
                   "rate": round(npass / len(graded), 3) if graded else 0}
        return web.json_response({"ok": True, "category": category, "variant": variant,
                                  "events": recs, "summary": summary,
                                  "gemini": {"asked": use_gemini, "key_found": gemini_ok}})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


async def api_gen_benchmark(req):
    """Auto-generate a "ground-truth" benchmark from raw dialogue text (same schema as the grader).
    {dialogue:"A: ...\\nB: ..."} → {ok, benchmark, user_lines, category, anchor_word, target_response, note}."""
    try:
        data = await req.json()
        dialogue = (data.get("dialogue") or "").strip()
        category = (data.get("category") or "").strip() or None      # None/'auto' → let the LLM decide
        if not dialogue:
            return web.json_response({"error": "paste a dialogue first (each line starting with A: / B:)"}, status=400)
        import gen_benchmark as gb
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: gb.generate(dialogue, category=category))
        return web.json_response({"ok": True, **out})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


def _run_dir_or_error(data):
    """Parse + safely validate run_dir → (Path, None) or (None, 400 response)."""
    run_dir = (data.get("run_dir") or "").strip("/")
    d = (RUNS / run_dir).resolve()
    if not str(d).startswith(str(RUNS.resolve()) + "/") or not d.is_dir():
        return None, web.json_response({"error": "invalid recording directory"}, status=400)
    return d, None


async def _grade_via(req, fn, extra=None):
    """Generic: validate run_dir + parse gemini + run grade_behavior's grader in the thread pool (single source of truth)."""
    try:
        data = await req.json()
        d, err = _run_dir_or_error(data)
        if err is not None:
            return err
        use_gemini = bool(data.get("gemini"))
        api_key = _gemini_key() if use_gemini else None
        kw = {"use_gemini": use_gemini, "api_key": api_key}
        if extra:
            kw.update(extra(data))
        out = await asyncio.get_event_loop().run_in_executor(None, lambda: fn(d, **kw))
        if "error" in out:
            return web.json_response({"error": out["error"]}, status=400)
        return web.json_response({"ok": True, **out})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


async def api_grade_backchannel(req):       # test backchannel ability
    return await _grade_via(req, gb.grade_backchannel)


async def api_grade_interruption(req):      # test "user interrupts model"
    return await _grade_via(req, gb.grade_interruption,
                            extra=lambda data: {"interrupt_text": (data.get("interrupt_text") or "").strip()})


async def api_grade_turn_taking(req):       # test "when to take the turn" smooth turn-taking (no gemini)
    try:
        data = await req.json()
        d, err = _run_dir_or_error(data)
        if err is not None:
            return err
        out = await asyncio.get_event_loop().run_in_executor(None, lambda: gb.grade_turn_taking(d))
        if "error" in out:
            return web.json_response({"error": out["error"]}, status=400)
        return web.json_response({"ok": True, **out})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


async def api_grade_pause(req):             # test "don't jump in on a pause" pause handling
    return await _grade_via(req, gb.grade_pause)


async def api_grade_user_backchannel(req):  # test "don't get derailed by the user's backchannel"
    return await _grade_via(req, gb.grade_user_backchannel)


def build_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_get("/live", live_page)
    app.router.add_get("/tts", tts_page)           # human TTS conversation (type → TTS → pause/resume feeding the model)
    app.router.add_get("/ws/live", ws_live)
    app.router.add_post("/api/tts", api_tts)
    app.router.add_post("/api/record", api_record)
    app.router.add_post("/api/run", api_run)
    app.router.add_post("/api/parakeet", api_parakeet)
    app.router.add_get("/api/download_run", api_download_run)   # zip up the run dir (debug)
    app.router.add_post("/api/download_run", api_download_run)  # POST: also carries the benchmark + dialogue source
    app.router.add_post("/api/grade_live", api_grade_live)
    app.router.add_post("/api/gen_benchmark", api_gen_benchmark)   # raw dialogue → ground truth
    app.router.add_post("/api/grade_backchannel", api_grade_backchannel)  # test backchannel ability
    app.router.add_post("/api/grade_interruption", api_grade_interruption)  # test "user interrupts model"
    app.router.add_post("/api/grade_turn_taking", api_grade_turn_taking)    # test "when to take the turn" smooth turn-taking
    app.router.add_post("/api/grade_pause", api_grade_pause)                # test "don't jump in on a pause" pause handling
    app.router.add_post("/api/grade_user_backchannel", api_grade_user_backchannel)  # test "don't get derailed by the user's backchannel"
    app.router.add_static("/inputs/", INPUTS)
    app.router.add_static("/runs/", RUNS)
    app.router.add_static("/static/", HERE / "static")     # opus-recorder (moshi/personaplex frontend encoding)
    return app


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="InteractionBench dashboard")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8770)),
                    help="listen port (default 8770, override with env PORT)")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = ap.parse_args()
    print(f"InteractionBench dashboard → http://{args.host}:{args.port}")
    web.run_app(build_app(), host=args.host, port=args.port)
