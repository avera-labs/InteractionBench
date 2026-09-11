"""Human ↔ model real-time conversation relay (WebSocket relay).

Browser mic PCM16@24k ──▶ this service ──▶ GPT-Realtime (full-duplex, server VAD natural turn-taking)
Model audio PCM16@24k     ◀── this service ◀── GPT-Realtime  →  browser real-time playback

· User barges in (speech_started) → tell the browser to clear the playback queue (barge-in)
· Two-track recording (user track + model track, same t0 timeline) written to disk, replayed when done
key read from instruction_tuning/.env; clears proxy env (same as live_drivers).
"""
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
from aiohttp import WSMsgType

SR = 24000            # browser capture + recording + GPT/Gemini output are all 24k
GEMINI_IN_SR = 16000  # Gemini input needs 16k (the server resamples the browser's 24k down)
DEFAULT_INSTRUCTIONS = (
    "You are a friendly, concise English voice conversation partner. "
    "Keep replies short and natural, one or two sentences. "
    "If the user asks you to correct their grammar, jump in and correct mistakes as you hear them."
)


def _place(chunks, total):
    """chunks=[(offset_sec, pcm16_bytes)...] placed by offset into a track of total samples (one response concatenated contiguously)."""
    out = np.zeros(total, dtype=np.int16)
    for off, b in chunks:
        a = np.frombuffer(b, dtype="<i2")
        pos = int(round(off * SR))
        end = min(pos + len(a), total)
        if end > pos:
            out[pos:end] = a[:end - pos]
    return out


def _write_tracks(user_pcm: bytes, model_blocks, out_dir: Path):
    """user_pcm: contiguous mic PCM16 bytes (concatenated in order = the real user track, not positioned by read time -- otherwise audio
    piled up during connect gets read out all at once, all landing at ≈0s and overwriting each other, so the start is lost). model_blocks=[(user_off_sec, concat_bytes)]
    anchored on the user track's timeline. Produces A_user.wav / B_model.wav / combined.wav (same timeline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    A = np.frombuffer(bytes(user_pcm), dtype="<i2") if user_pcm else np.zeros(0, dtype=np.int16)

    end = len(A) / SR
    for off, b in model_blocks:
        end = max(end, off + (len(b) // 2) / SR)
    total = max(int(np.ceil(max(end, 0.1) * SR)), len(A))

    Ax = np.zeros(total, dtype=np.int16); Ax[:len(A)] = A       # user track: contiguous, starts at 0
    Bx = _place(model_blocks, total)                            # model track: placed by user-track offset
    sf.write(str(out_dir / "A_user.wav"), Ax, SR, subtype="PCM_16")
    sf.write(str(out_dir / "B_model.wav"), Bx, SR, subtype="PCM_16")
    mix = np.clip(Ax.astype(np.int32) + Bx.astype(np.int32), -32768, 32767).astype(np.int16)
    sf.write(str(out_dir / "combined.wav"), mix, SR, subtype="PCM_16")
    return {"secs": round(total / SR, 1)}


async def relay_gpt(ws, silence_ms: int, out_dir: Path, model: str = "gpt-realtime-2.1",
                    instructions: str = DEFAULT_INSTRUCTIONS):
    """Wire the browser WS to GPT-Realtime, run a whole human conversation, and write both tracks to disk. Returns a dict with the write-out info."""
    from openai import AsyncOpenAI

    # Via proxy: under ClashX proxy mode, gpt/gemini must go through HTTPS_PROXY/ALL_PROXY to reach OpenAI/Google
    # (websockets reads the env proxy automatically; socks5:// is backed by python-socks). The localhost moshi/freezeomni
    # each use proxy=None to bypass the proxy, unaffected. The terminal that starts the server must export ClashX's proxy env.

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user = {"pcm": bytearray(), "n": 0}       # contiguous mic PCM + samples received so far (= user track's current position)
    model_blocks = []                # [(user_off_sec, concat_bytes)] one block per response
    cur = {"off": None, "buf": bytearray()}   # audio of the response currently being received
    stop = asyncio.Event()

    def _flush_block():
        if cur["off"] is not None and cur["buf"]:
            model_blocks.append((cur["off"], bytes(cur["buf"])))
        cur["off"], cur["buf"] = None, bytearray()

    async def _say(obj):
        if not ws.closed:
            await ws.send_str(json.dumps(obj))

    async with client.realtime.connect(model=model) as conn:
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": instructions,
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": SR},
                          "turn_detection": {"type": "server_vad", "silence_duration_ms": int(silence_ms)},
                          "transcription": {"model": "whisper-1"}},
                "output": {"format": {"type": "audio/pcm", "rate": SR}},
            },
        })
        t0 = time.time()
        await _say({"type": "ready"})

        async def from_browser():
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    user["pcm"] += msg.data                    # contiguous concat (mic never stops)
                    user["n"] += len(msg.data) // 2            # int16 sample count
                    try:
                        await conn.input_audio_buffer.append(audio=base64.b64encode(msg.data).decode())
                    except Exception:  # noqa: BLE001
                        pass
                elif msg.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(msg.data)
                    except Exception:  # noqa: BLE001
                        d = {}
                    if d.get("type") == "bye":
                        break
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
            stop.set()

        async def from_model():
            try:
                async for e in conn:
                    if stop.is_set():
                        break
                    t = getattr(e, "type", "")
                    if t in ("response.output_audio.delta", "response.audio.delta") and getattr(e, "delta", None):
                        pcm = base64.b64decode(e.delta)
                        if cur["off"] is None:
                            cur["off"] = user["n"] / SR       # anchor to user track's current position (align the two tracks)
                        cur["buf"] += pcm
                        if not ws.closed:
                            await ws.send_bytes(pcm)
                    elif t in ("response.output_audio.done", "response.done"):
                        _flush_block()
                    elif t in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                        await _say({"type": "model_text", "text": getattr(e, "delta", "") or ""})
                    elif t == "input_audio_buffer.speech_started":
                        await _say({"type": "clear"})          # user barges in → browser stops playing the model
                    elif t == "conversation.item.input_audio_transcription.delta":
                        await _say({"type": "user_text", "text": getattr(e, "delta", "") or ""})
                    elif t == "conversation.item.input_audio_transcription.completed":
                        await _say({"type": "user_text_done", "text": getattr(e, "transcript", "") or ""})
                    elif t == "error":
                        await _say({"type": "err", "text": getattr(getattr(e, "error", None), "message", "") or ""})
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        fb = asyncio.create_task(from_browser())
        fm = asyncio.create_task(from_model())
        await fb
        stop.set()
        await asyncio.sleep(0.2)
        fm.cancel()
        _flush_block()

    return _write_tracks(bytes(user["pcm"]), model_blocks, out_dir)


GPT_LIVE_URL = "wss://api.openai.com/v1/live/sessions"
GPT_LIVE_MODEL = "gpt-live-1"
GPT_LIVE_DELEGATE = os.environ.get("GPT_LIVE_DELEGATE", "gpt-5.5")   # Responses backend; "" = none


async def relay_gpt_live(ws, silence_ms: int, out_dir: Path, model: str = GPT_LIVE_MODEL,
                         instructions: str = DEFAULT_INSTRUCTIONS):
    """Browser WS ↔ GPT-Live (the /live API, not /realtime). Same contract as relay_gpt: writes both tracks, returns the write-out info.

    Differences from GPT-Realtime that matter here:
    · the model track is ONE continuous 24k stream from session start (silent when not speaking, like moshi/gemini),
      not per-response segments -- so it anchors once at the user-track position of the first chunk and stays aligned by its own clock;
    · no server VAD and no speech_started -- it's full-duplex, so there is no barge-in "clear" to forward (silence_ms is unused, kept for dispatch);
    · config goes in session.start (model included) and the URL takes no query parameters;
    · deep reasoning is delegated to a Responses backend (GPT_LIVE_DELEGATE, default gpt-5.5); set it to "" to run the Live model alone.
    """
    import websockets

    key = os.environ["OPENAI_API_KEY"]
    user = {"pcm": bytearray(), "n": 0}       # contiguous mic PCM + samples received so far (= user track's current position)
    model_blocks = []                         # one entry: [(user_off_sec, the whole continuous stream)]
    cur = {"off": None, "buf": bytearray()}
    clk = []                                  # (wall_sec, model_samples, user_samples) sampled on every output delta
    total = {"model": 0}                      # model samples received across all blocks
    raw_pcm = bytearray()                     # every output sample, concatenated -- live_align.py places it
    tr = []                                   # server-clock transcript deltas, for the post-run alignment check
    stop = asyncio.Event()

    def _flush_block():
        if cur["off"] is not None and cur["buf"]:
            model_blocks.append((cur["off"], bytes(cur["buf"])))
        cur["off"], cur["buf"] = None, bytearray()

    async def _say(obj):
        if not ws.closed:
            await ws.send_str(json.dumps(obj))

    session = {
        "model": model,
        "audio": {"format": {"type": "audio/pcm", "rate": SR}},
        "instructions": instructions,
    }
    if GPT_LIVE_DELEGATE:
        session["delegation"] = {"type": "responses", "responses": {"model": GPT_LIVE_DELEGATE}}

    # the TLS handshake to api.openai.com through a local proxy resets now and then -- retry rather than losing the item
    mws = None
    for attempt in range(1, 5):
        try:
            mws = await websockets.connect(GPT_LIVE_URL, additional_headers={"Authorization": f"Bearer {key}"},
                                           max_size=None, open_timeout=30)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[gpt-live] connect attempt {attempt}: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(1.0 * attempt)
    if mws is None:
        await _say({"type": "err", "text": f"can't connect to {GPT_LIVE_URL}"})
        return None

    async with mws:
        await mws.send(json.dumps({"type": "session.start", "session": session}))
        t0 = time.time()
        while True:                                    # wait for session.started before letting audio in
            e = json.loads(await asyncio.wait_for(mws.recv(), 30))
            if e.get("type") == "session.started":
                break
            if e.get("type") == "error":
                await _say({"type": "err", "text": json.dumps(e.get("error") or e)})
                return None
        print(f"[gpt-live] session.started in {time.time()-t0:.2f}s (delegate={GPT_LIVE_DELEGATE or 'none'})", flush=True)
        await _say({"type": "ready"})

        async def from_browser():
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    user["pcm"] += msg.data                    # contiguous concat (mic never stops)
                    user["n"] += len(msg.data) // 2            # int16 sample count
                    try:
                        await mws.send(json.dumps({"type": "session.input_audio.append",
                                                   "audio": base64.b64encode(msg.data).decode()}))
                    except Exception:  # noqa: BLE001
                        pass
                elif msg.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(msg.data)
                    except Exception:  # noqa: BLE001
                        d = {}
                    if d.get("type") == "bye":
                        break
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
            stop.set()

        async def from_model():
            try:
                async for raw in mws:
                    if stop.is_set():
                        break
                    e = json.loads(raw)
                    t = e.get("type", "")
                    if t == "session.output_audio.delta":
                        pcm = base64.b64decode(e["delta"])
                        # The delta stream is NOT a timeline: the server drops silent output frames, so
                        # concatenating it pulls everything after a gap earlier (-1.6s by the end of a 43s
                        # item), and anchoring by arrival instead just swaps that for the network latency
                        # (+1.2s through a proxy). Neither is usable for timing. Keep the raw stream here
                        # and let live_align.py place it on the server's own session clock, post-ASR.
                        if cur["off"] is None:
                            cur["off"] = user["n"] / SR
                        cur["buf"] += pcm
                        raw_pcm.extend(pcm)
                        total["model"] += len(pcm) // 2
                        clk.append((round(time.time() - t0, 3), total["model"], user["n"]))
                        if not ws.closed:
                            await ws.send_bytes(pcm)
                    elif t == "session.output_transcript.delta":
                        # start_ms/end_ms are the server's own session clock -- the only authoritative
                        # timeline we get, so keep it to check the recorded tracks against afterwards
                        tr.append({"who": "model", "start_ms": e.get("start_ms"), "end_ms": e.get("end_ms"),
                                   "text": e.get("delta", ""), "user_s": round(user["n"] / SR, 3)})
                        await _say({"type": "model_text", "text": e.get("delta", "")})
                    elif t == "session.input_transcript.delta":
                        tr.append({"who": "user", "start_ms": e.get("start_ms"), "end_ms": e.get("end_ms"),
                                   "text": e.get("delta", ""), "user_s": round(user["n"] / SR, 3)})
                        await _say({"type": "user_text", "text": e.get("delta", "")})
                    elif t == "error":
                        await _say({"type": "err", "text": json.dumps(e.get("error") or e)})
                        print(f"[gpt-live] error {json.dumps(e)[:300]}", flush=True)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"[gpt-live] reader died: {type(exc).__name__}: {exc}", flush=True)

        fb = asyncio.create_task(from_browser())
        fm = asyncio.create_task(from_model())
        await fb
        stop.set()
        await asyncio.sleep(0.2)
        fm.cancel()
        _flush_block()
        try:
            await mws.send(json.dumps({"type": "session.close"}))
        except Exception:  # noqa: BLE001
            pass

    info = _write_tracks(bytes(user["pcm"]), model_blocks, out_dir)
    (out_dir / "transcript_clock.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1))
    sf.write(str(out_dir / "B_model_raw.wav"),
             np.frombuffer(bytes(raw_pcm), dtype="<i2"), SR, subtype="PCM_16")
    info["clock"] = _clock_report(clk, out_dir)
    return info


def _clock_report(clk, out_dir: Path):
    """Did the two tracks stay on one clock?

    relay_gpt_live anchors the model's continuous stream at the user-track position of its first chunk,
    which is only sound while the model stream, the user stream and the wall clock all advance 1:1.
    Each sample is (wall_sec, model_samples, user_samples); drift is how far the two tracks have pulled
    apart since the anchor, in milliseconds -- the sign says which way the model track is displaced.
    """
    if len(clk) < 2:
        return {"n": len(clk)}
    (w0, m0, u0), (w1, m1, u1) = clk[0], clk[-1]
    span = w1 - w0
    model_rate = (m1 - m0) / SR / span if span > 0 else 0      # model seconds produced per wall second
    user_rate = (u1 - u0) / SR / span if span > 0 else 0       # user seconds consumed per wall second
    drift = [round(((m - m0) - (u - u0)) / SR * 1000) for _, m, u in clk]
    (out_dir / "clock.jsonl").write_text(
        "".join(json.dumps({"t": w, "model_s": round((m - m0) / SR, 3),
                            "user_s": round((u - u0) / SR, 3), "drift_ms": d}) + "\n"
                for (w, m, u), d in zip(clk, drift)))
    return {"n": len(clk), "span_s": round(span, 2),
            "model_rate": round(model_rate, 4), "user_rate": round(user_rate, 4),
            "drift_ms_final": drift[-1], "drift_ms_max": max(drift, key=abs)}


MOSHI_SR = 24000
MOSHI_FRAME = 480                 # 20ms@24k -- matches the official web client's opus-recorder encoderFrameSize:20
MOSHI_DEFAULT_URL = "ws://localhost:8998/api/chat"


async def relay_moshi(ws, out_dir: Path, moshi_url: str = MOSHI_DEFAULT_URL):
    """Browser WS ↔ Moshi (Kyutai, local/GPU server). Protocol: binary with a 1-byte type prefix --
    \\x00 handshake / \\x01 audio (Opus@24kHz, sphn codec) / \\x02 text. Moshi is inherently full-duplex and streams continuously,
    so the model track is also continuous (unlike GPT's segmented responses). The browser is already 24k, no resampling needed. Writes both tracks to disk."""
    import sphn
    import websockets

    user = {"pcm": bytearray(), "n": 0}          # contiguous mic (decoded browser opus) + samples received so far (= user track position)
    model_blocks = []                            # [(off_sec, pcm)] each model speech segment anchored at its user-track position (no wall clock)
    cur = {"off": None, "buf": bytearray(), "last": 0.0}
    MOSHI_GAP = 1.0                              # model-audio arrival gap > this → start a new segment (during the TTS conversation freeze there's no output → big gap)
    stop = asyncio.Event()

    def _flush_block():
        if cur["off"] is not None and cur["buf"]:
            model_blocks.append((cur["off"], bytes(cur["buf"])))
        cur["off"], cur["buf"] = None, bytearray()
    # The browser already encoded opus with opus-recorder (same as the official web UI) → forward it straight to the server; no sphn encoding here
    # (the root cause: the server container can't decode opus encoded by sphn on the Mac). sphn only decodes: downlink reader, uplink recording user_reader.
    reader = sphn.OpusStreamReader(MOSHI_SR)
    user_reader = sphn.OpusStreamReader(MOSHI_SR)

    async def _say(obj):
        if not ws.closed:
            await ws.send_str(json.dumps(obj))

    kw = {"max_size": None, "proxy": None}              # localhost SSH tunnel: disable env proxy (websockets by default shoves wss onto all_proxy/socks → ImportError: python-socks)
    if moshi_url.startswith("wss://"):                  # PersonaPlex/moshi use --ssl self-signed cert → turn off verification
        import ssl as _ssl
        ctx = _ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        kw["ssl"] = ctx
    try:
        mws = await websockets.connect(moshi_url, **kw)
    except Exception as e:  # noqa: BLE001
        await _say({"type": "err", "text": f"can't connect to model server {moshi_url}: {type(e).__name__}: {e}"})
        return None

    import sys as _sys
    def _log(*a):
        print("[moshi-relay]", *a, file=_sys.stderr, flush=True)

    n = {"in": 0, "sent": 0, "recv": 0, "text": 0}
    _log(f"connected to {moshi_url}")

    async with mws:
        t0 = time.time()
        await _say({"type": "ready"})

        # ★ Must wait for moshi's \x00 handshake before forwarding uplink (like the official web UI: send not a single byte before the handshake).
        # Otherwise is_alive() during the server's system-prompt loading will ws.receive() and swallow the browser's early-arriving OGG header (OpusHead)
        # → after the handshake recv_loop gets headerless audio pages → the sphn decode thread crashes → append_bytes reports "closed channel".
        handshake = asyncio.Event()
        q: asyncio.Queue = asyncio.Queue()          # queue the browser opus before the handshake (order preserved, OGG header enqueued first)

        async def from_browser():
            reason = "ws-end"
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    op = bytes(msg.data)                       # OGG/Opus already encoded by the browser's opus-recorder
                    n["in"] += 1
                    q.put_nowait(op)                           # enqueue, released by the forwarder after the handshake
                    try:                                       # recording uses a separate reader, decodes continuously (not gated by the handshake)
                        user_reader.append_bytes(op)
                        while True:
                            pcm = user_reader.read_pcm()
                            if pcm is None or pcm.shape[-1] == 0:
                                break
                            user["pcm"] += (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                            user["n"] += pcm.shape[-1]     # user track sample position (used to anchor model segments)
                    except Exception:  # noqa: BLE001
                        pass
                elif msg.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(msg.data)
                    except Exception:  # noqa: BLE001
                        d = {}
                    if d.get("type") == "bye":
                        reason = "bye"; break
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    reason = f"browser-{msg.type.name}"; break
            q.put_nowait(None)                                 # sentinel: tell the forwarder to wrap up
            _log(f"from_browser exited ({reason}); received {n['in']} browser audio blocks")

        async def forwarder():
            try:                                               # first wait for moshi's handshake, then release the queue (OGG header dequeued first → reaches the server first)
                await asyncio.wait_for(handshake.wait(), timeout=30)
            except asyncio.TimeoutError:
                _log("timed out waiting for moshi handshake (30s), giving up"); stop.set(); return
            _log("got moshi handshake, start forwarding uplink")
            while True:
                op = await q.get()
                if op is None:
                    break
                try:
                    await mws.send(b"\x01" + op)
                    n["sent"] += 1
                except Exception as ex:  # noqa: BLE001
                    _log("send→moshi err:", type(ex).__name__, ex); break
            _log(f"forwarder exited; forwarded {n['sent']} packets to moshi")
            stop.set()

        async def from_moshi():
            try:
                async for m in mws:
                    if stop.is_set():
                        break
                    if not isinstance(m, (bytes, bytearray)) or not m:
                        continue
                    if m[0] == 0:                   # ← moshi handshake: release uplink forwarding
                        if not handshake.is_set():
                            _log("← got moshi \\x00 handshake")
                            handshake.set()
                        continue
                    if m[0] not in (1, 2):
                        continue
                    if m[0] == 1:                   # audio
                        try:
                            reader.append_bytes(bytes(m[1:]))   # sphn 0.1.12: after feeding it, loop read_pcm to pull decoded PCM
                        except Exception:  # noqa: BLE001
                            continue
                        while True:
                            pcm = reader.read_pcm()
                            if pcm is None or pcm.shape[-1] == 0:
                                break
                            now = time.time(); pos = user["n"] / MOSHI_SR    # anchor at user-track position (no wall clock → avoids misalignment during the freeze)
                            if cur["off"] is None:
                                cur["off"] = pos
                                _log(f"model starts speaking @ user track {pos:.2f}s")
                            elif now - cur["last"] > MOSHI_GAP:              # gap too large (freeze) → close the segment, anchor the new segment at the current user-track position
                                _flush_block(); cur["off"] = pos
                            n["recv"] += 1
                            b = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                            cur["buf"] += b; cur["last"] = now
                            if not ws.closed:
                                await ws.send_bytes(b)
                    else:                           # text (the model's own inner-monologue, authoritative) → timestamped, for later time-ordered interleaving
                        n["text"] += 1
                        await _say({"type": "model_text", "text": m[1:].decode(errors="ignore"),
                                    "t": round(time.time() - t0, 3)})
            except (asyncio.CancelledError, Exception) as ex:  # noqa: BLE001
                if not isinstance(ex, asyncio.CancelledError):
                    _log("from_moshi err:", type(ex).__name__, ex)
            _log(f"from_moshi exited; received {n['recv']} moshi audio blocks / {n['text']} text messages")
            stop.set()                              # if moshi drops first, also make from_browser exit, don't hang

        # Note: don't run local parakeet on your track during the session -- it adds local load, and concurrent MLX background-thread calls produce <unk> garbage.
        # Your side's transcription is done in one shot via "click parakeet after finishing" (accurate); the live view only shows the model side (server's authoritative text).
        fb = asyncio.create_task(from_browser())
        fw = asyncio.create_task(forwarder())
        fm = asyncio.create_task(from_moshi())
        await stop.wait()                           # wrap up as soon as either side finishes
        await asyncio.sleep(0.2)
        for _t in (fb, fw, fm):
            _t.cancel()
    _flush_block()
    _log(f"done: browser audio {n['in']} / forwarded {n['sent']} / moshi audio {n['recv']} blocks / text {n['text']} / model speech segments {len(model_blocks)}")
    return _write_tracks(bytes(user["pcm"]), model_blocks, out_dir)


def _to_16k(pcm24: bytes) -> bytes:
    """Browser 24k PCM16 → 16k PCM16 (Gemini / FreezeOmni input)."""
    a = np.frombuffer(pcm24, dtype="<i2").astype(np.float32) / 32768.0
    b = ss.resample_poly(a, 2, 3)                          # 24000*2/3 = 16000
    return (np.clip(b, -1, 1) * 32767).astype("<i2").tobytes()


FO_DEFAULT_URL = "https://localhost:8081"


async def relay_freezeomni(ws, out_dir: Path, fo_url: str = FO_DEFAULT_URL, prompt: str = ""):
    """Browser WS ↔ FreezeOmni (Flask-SocketIO, raw PCM). Uplink 16k, downlink 24k int16.
    Browser sends 24k PCM → resample to 16k → sio.emit('audio', JSON{sample_rate,audio:[bytes]}) (like the official demo.html);
    receive sio 'audio' (24k int16 bytes) → forward straight to the browser for playback. stop_tts=interrupt (clear browser playback). Writes both tracks (both 24k)."""
    import socketio
    import sys as _sys

    def _log(*a):
        print("[fo-relay]", *a, file=_sys.stderr, flush=True)

    user = {"pcm": bytearray(), "n": 0}         # contiguous mic 24k (recording) + samples received so far (= user track's current position)
    model_blocks = []                           # [(off_sec, pcm)] each model "speech segment" anchored at its user-track position
    cur = {"off": None, "buf": bytearray(), "last": 0.0}   # the model segment currently being received
    FO_GAP = 1.0                                # gap between model audio blocks > this → previous segment done, start a new one (keep the silence between segments, don't pull later speech forward)
    stop = asyncio.Event()
    t0 = time.time()
    n = {"in": 0, "out": 0}

    def _flush_block():
        if cur["off"] is not None and cur["buf"]:
            model_blocks.append((cur["off"], bytes(cur["buf"])))
        cur["off"], cur["buf"] = None, bytearray()

    async def _say(obj):
        if not ws.closed:
            await ws.send_str(json.dumps(obj))

    sio = socketio.AsyncClient(ssl_verify=False, reconnection=False)

    @sio.on("audio")
    async def on_audio(data):                   # model TTS output: 24k int16 raw bytes (discrete segments, silent between segments)
        b = data if isinstance(data, (bytes, bytearray)) else (
            data.encode("latin1") if isinstance(data, str) else bytes(data))
        if not b:
            return
        now = time.time()
        pos = user["n"] / SR                    # current user-track position → this model-audio segment is anchored here (no wall clock, avoids connection-latency misalignment)
        if cur["off"] is None:
            cur["off"] = pos
            _log(f"model starts speaking @ user track {pos:.2f}s")
        elif now - cur["last"] > FO_GAP:        # gap from the previous block too large → previous segment done, close it + anchor the new segment at the current user-track position
            _flush_block()
            cur["off"] = pos
            _log(f"model starts speaking again @ user track {pos:.2f}s")
        cur["buf"] += bytes(b)
        cur["last"] = now
        n["out"] += 1
        if not ws.closed:
            await ws.send_bytes(bytes(b))

    @sio.on("stop_tts")
    async def on_stop_tts(*a):
        await _say({"type": "clear"})           # model interrupted → browser clears the playback queue

    @sio.on("too_many_users")
    async def on_too_many(*a):
        await _say({"type": "err", "text": "FreezeOmni: connection limit reached (max_users)"}); stop.set()

    @sio.on("out_time")
    async def on_out_time(*a):
        await _say({"type": "err", "text": "FreezeOmni: session timed out and disconnected"}); stop.set()

    prompt_set = asyncio.Event()

    @sio.on("prompt_success")
    async def on_prompt_ok(*a):
        _log("system prompt set successfully"); prompt_set.set()

    @sio.on("disconnect")
    async def on_disc(*a):
        _log("sio disconnected"); stop.set()

    try:
        await sio.connect(fo_url, transports=["websocket"], wait_timeout=15)
    except Exception as e:  # noqa: BLE001
        await _say({"type": "err", "text": f"can't connect to FreezeOmni {fo_url}: {type(e).__name__}: {e}"})
        return None

    _log(f"connected to {fo_url}")
    await _say({"type": "ready"})
    if prompt.strip():
        await sio.emit("prompt_text", prompt.strip())
        try:                                            # wait for the prompt to take effect before recording, don't let the default (Chinese) persona kick in first
            await asyncio.wait_for(prompt_set.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            _log("⚠️ didn't get prompt_success (2s), continuing anyway")
    await sio.emit("recording-started")

    async def from_browser():
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                pcm24 = bytes(msg.data)                     # browser 24k PCM16
                user["pcm"] += pcm24                        # recording (24k)
                user["n"] += len(pcm24) // 2                # user track sample position (used to anchor model segments)
                n["in"] += 1
                try:                                        # resample to 16k → emit like the official demo (JSON string, audio is a byte list)
                    pcm16 = _to_16k(pcm24)
                    await sio.emit("audio", json.dumps({"sample_rate": 16000, "audio": list(pcm16)}))
                except Exception as ex:  # noqa: BLE001
                    _log("emit audio err:", type(ex).__name__, ex)
            elif msg.type == WSMsgType.TEXT:
                try:
                    d = json.loads(msg.data)
                except Exception:  # noqa: BLE001
                    d = {}
                if d.get("type") == "bye":
                    break
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                break
        stop.set()

    fb = asyncio.create_task(from_browser())
    await stop.wait()
    await asyncio.sleep(0.1)
    fb.cancel()
    for ev in ("recording-stopped",):
        try:
            await sio.emit(ev)
        except Exception:  # noqa: BLE001
            pass
    try:
        await sio.disconnect()
    except Exception:  # noqa: BLE001
        pass
    _flush_block()
    _log(f"done: browser audio {n['in']} / FreezeOmni audio blocks {n['out']} / model speech segments {len(model_blocks)}")
    return _write_tracks(bytes(user["pcm"]), model_blocks, out_dir)


async def relay_gemini(ws, out_dir: Path, silence_ms: int | None = None,
                       model: str = "gemini-3.1-flash-live-preview",
                       instructions: str = DEFAULT_INSTRUCTIONS):
    """Browser WS ↔ Gemini 3.1 Live (turn-based). Input 24k→16k resampled; output 24k forwarded straight. Writes both tracks.
    silence_ms if given: tunes automatic_activity_detection -- smaller + high end sensitivity → jumps in on shorter pauses (more eager to interrupt).
    If not given, uses Gemini's default VAD."""
    from google import genai

    # Via proxy: under ClashX proxy mode, gpt/gemini must go through HTTPS_PROXY/ALL_PROXY to reach OpenAI/Google
    # (websockets reads the env proxy automatically; socks5:// is backed by python-socks). The localhost moshi/freezeomni
    # each use proxy=None to bypass the proxy, unaffected. The terminal that starts the server must export ClashX's proxy env.

    client = genai.Client(vertexai=False, api_key=os.environ["GEMINI_API_KEY"])
    cfg = {
        "response_modalities": ["AUDIO"],
        "system_instruction": instructions,
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }
    if silence_ms:
        cfg["realtime_input_config"] = {
            "automatic_activity_detection": {
                "silence_duration_ms": int(silence_ms),         # silence needed to declare turn end (smaller → jumps in earlier)
                "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
                "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
                "prefix_padding_ms": 20,
            }
        }
    user = {"pcm": bytearray(), "n": 0}       # contiguous mic PCM + samples received so far (= user track's current position)
    model_blocks = []
    cur = {"off": None, "buf": bytearray()}
    stop = asyncio.Event()

    def _flush_block():
        if cur["off"] is not None and cur["buf"]:
            model_blocks.append((cur["off"], bytes(cur["buf"])))
        cur["off"], cur["buf"] = None, bytearray()

    async def _say(obj):
        if not ws.closed:
            await ws.send_str(json.dumps(obj))

    async with client.aio.live.connect(model=model, config=cfg) as sess:
        t0 = time.time()
        await _say({"type": "ready"})

        async def from_browser():
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    user["pcm"] += msg.data                                 # record raw 24k, contiguous concat
                    user["n"] += len(msg.data) // 2
                    try:
                        await sess.send_realtime_input(
                            audio={"data": _to_16k(msg.data), "mime_type": "audio/pcm"})
                    except Exception:  # noqa: BLE001
                        pass
                elif msg.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(msg.data)
                    except Exception:  # noqa: BLE001
                        d = {}
                    if d.get("type") == "bye":
                        break
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
            try:
                await sess.send_realtime_input(audio_stream_end=True)
            except Exception:  # noqa: BLE001
                pass
            stop.set()

        async def from_model():
            # ⚠️ The SDK's sess.receive() returns at the end of each turn (breaks at turn_complete in the source).
            # For multi-turn conversation you must wrap it in an outer while and re-subscribe each turn, otherwise it stops responding after the first turn.
            try:
                while not stop.is_set():
                    async for r in sess.receive():
                        if stop.is_set():
                            break
                        sc = r.server_content
                        if not sc:
                            continue
                        if getattr(sc, "input_transcription", None):
                            await _say({"type": "user_text", "text": sc.input_transcription.text or ""})
                        if getattr(sc, "output_transcription", None):
                            await _say({"type": "model_text", "text": sc.output_transcription.text or ""})
                        if getattr(sc, "interrupted", False):
                            await _say({"type": "clear"})                   # user barges in → stop playback
                        if sc.model_turn:
                            for p in sc.model_turn.parts:
                                if p.inline_data and isinstance(p.inline_data.data, bytes):
                                    if cur["off"] is None:
                                        cur["off"] = user["n"] / SR         # anchor to user track's current position
                                    cur["buf"] += p.inline_data.data
                                    if not ws.closed:
                                        await ws.send_bytes(p.inline_data.data)
                        if getattr(sc, "generation_complete", False) or getattr(sc, "turn_complete", False):
                            _flush_block()
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        fb = asyncio.create_task(from_browser())
        fm = asyncio.create_task(from_model())
        await fb
        stop.set()
        await asyncio.sleep(0.2)
        fm.cancel()
        _flush_block()

    return _write_tracks(bytes(user["pcm"]), model_blocks, out_dir)
