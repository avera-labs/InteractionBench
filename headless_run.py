#!/usr/bin/env python3
"""Headless batch runs: feed pre-recorded user audio to a live model (no browser), record both tracks → parakeet → grade.

In the dashboard the browser does only two things: send mic PCM into /ws/live, and play back the model's PCM.
Here a "fake ws" replaces the browser — it feeds the wav into the relay frame by frame in real time (so pauses/backchannels
land at the right moments naturally), without changing a line of relay logic; when done the relay writes
A_user.wav/B_model.wav/combined.wav itself. Then local parakeet + grade_behavior scoring.

Usage:
  uv run python headless_run.py --model gpt --input tts_review/backchannel/01.wav --task backchannel --gemini
  uv run python headless_run.py --model gpt --input-dir tts_review/backchannel --glob '[0-9][0-9].wav' --task backchannel --gemini
"""
import argparse
import asyncio
import json
import re
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))
load_dotenv(ROOT / ".env", override=True)

import os  # noqa: E402
from aiohttp import WSMsgType  # noqa: E402

import live_ws  # noqa: E402
import parakeet_local  # noqa: E402
import grade_behavior as gb  # noqa: E402

SR = live_ws.SR                 # 24000
FRAME = 960                     # 40ms @24k (samples per frame)
BENCH_TURN_GAP = 8.0            # opus multi-turn: fixed seconds left between turns for the model to answer the previous line
RUNS = ROOT / "dashboard" / "runs"
FREEZEOMNI_DEFAULT_PROMPT_SENTINEL = "You are a friendly assistant. Always reply in English."  # default when --text-prompt is not overridden; freezeomni swaps in a stricter English-only version


# ------------------------------- fake ws -------------------------------
class _Msg:
    __slots__ = ("type", "data")

    def __init__(self, t, d):
        self.type, self.data = t, d


class FakeWS:
    """Impersonates aiohttp WebSocketResponse: async iteration = upstream audio frames (real-time cadence) + bye.
    - gpt/gemini: feed raw PCM16 frames (+ trailing silence);
    - moshi/personaplex: feed OGG pages encoded by opus-recorder (real-time cadence by granule, silence already merged into the pcm).
    Audio the relay sends back (send_bytes) is received but unused (the relay writes it to disk itself), send_str is dropped."""

    def __init__(self, pcm: bytes = None, pages=None, prompt_pcm: bytes = None, injections=None,
                 turns=None, opus_turns=None, opus_nturns=None, turn_end_silence=1.5, turn_max_wait=14.0,
                 max_wait=18.0, realtime=True, tail_silence=10.0):
        self.closed = False
        self._pcm = pcm
        self._pages = pages                           # [(page_bytes, granulepos)] —— moshi/personaplex
        self._prompt_pcm = prompt_pcm                 # reactive: the lead-in sent first (interrupt Q1 / user_backchannel question)
        self._injections = injections or []           # [(pcm_bytes, thr)]: inject this clip once the model has spoken thr seconds total (Q2 / ok-right)
        self._turns = turns                           # [pcm_bytes,...]: multi-turn, send the next line only after the model finishes each one (benchmark)
        self._opus_turns = opus_turns                 # (a1_pages, sil_pages, a2_pages, a2_base_gp): opus adaptive multi-turn (old 2-turn, deprecated)
        self._opus_nturns = opus_nturns               # (header pages, [per-turn audio pages [(pg,dgp)]], silence audio pages [(pg,dgp)]): opus N-turn adaptive
        self._turn_end_silence = turn_end_silence     # model silent this long = this turn is done
        self._turn_max_wait = turn_max_wait
        self._max_wait = max_wait
        self._realtime = realtime
        self._tail = tail_silence
        self.sent_audio = bytearray()
        self.model_speech_sec = 0.0                   # accumulated model speech duration received (summed in send_bytes)
        self.last_recv = 0.0                          # wall clock of the last model audio received (for end-of-turn silence)
        self.first_model_t = None                     # wall-clock moment of the model's first audio frame (≈ where the answer starts in the recording; injections are placed relative to this)
        self.inject_log = []                          # per injection: {at: seconds since start, model_said: seconds the model has spoken}
        self.turn_log = []                            # seconds taken for the model to finish each turn

    async def _pace_pcm(self, pcm):                   # yield a PCM segment frame by frame at real-time cadence
        step = FRAME * 2
        t0 = time.time()
        for i in range(0, len(pcm), step):
            yield _Msg(WSMsgType.BINARY, pcm[i:i + step])
            if self._realtime:
                dt = (i // step + 1) * (FRAME / SR) - (time.time() - t0)
                if dt > 0:
                    await asyncio.sleep(dt)

    async def _gen(self):
        sil = b"\x00\x00" * FRAME
        if self._turns is not None:                   # ---- multi-turn: feed a line → wait for the model to finish → feed the next (benchmark) ----
            for clip in self._turns:
                async for m in self._pace_pcm(clip):
                    yield m
                base = self.model_speech_sec
                self.last_recv = time.time()          # start counting silence from this turn
                t_turn = time.time()
                first_audio = None                    # wall clock of the model's first sound this turn (for buffer-drained check)
                spoke = False
                while True:
                    yield _Msg(WSMsgType.BINARY, sil)
                    if self._realtime:
                        await asyncio.sleep(FRAME / SR)
                    if self.model_speech_sec > base + 0.3:
                        spoke = True
                        if first_audio is None:
                            first_audio = time.time()  # wall clock of the model's first sound this turn
                    now = time.time()
                    # is the buffered audio done playing: wall time elapsed ≥ total audio received this turn (audio arrives faster than it plays, so wait for it to drain)
                    drained = first_audio is None or (now - first_audio) >= (self.model_speech_sec - base) + 0.3
                    if spoke and (now - self.last_recv) > self._turn_end_silence and drained:
                        break                         # model done (silent long enough + buffer drained)
                    if now - t_turn > self._turn_max_wait:
                        break                         # fallback (no reply / too long)
                self.turn_log.append(round(time.time() - t_turn, 1))
            for _ in range(int(1.5 * SR / FRAME)):    # trailing silence to catch the tail
                yield _Msg(WSMsgType.BINARY, sil)
                if self._realtime:
                    await asyncio.sleep(FRAME / SR)
            yield _Msg(WSMsgType.TEXT, json.dumps({"type": "bye"}))
            return
        if self._prompt_pcm is not None:              # ---- reactive injection (gpt/gemini/freezeomni raw PCM) ----
            t0 = time.time()
            async for m in self._pace_pcm(self._prompt_pcm):   # send the lead-in (Q1 / question)
                yield m
            for inj_pcm, thr in self._injections:     # one at a time: wait until "thr seconds have played since the model started" then inject
                # trigger by playback position (model-onset wall clock + thr), not by amount of audio received: streaming models like gemini
                # burst-transmit fast, so using model_speech_sec would cram all injections at the start of the answer and too close together;
                # playback position is what corresponds to the real thr-th second inside the answer.
                w0 = time.time()
                while (time.time() - w0) < self._max_wait:
                    if self.first_model_t is not None and (time.time() - self.first_model_t) >= thr:
                        break
                    yield _Msg(WSMsgType.BINARY, sil)
                    if self._realtime:
                        await asyncio.sleep(FRAME / SR)
                self.inject_log.append({"at": round(time.time() - t0, 2),
                                        "since_model": (round(time.time() - self.first_model_t, 2)
                                                        if self.first_model_t else None)})
                async for m in self._pace_pcm(inj_pcm):        # inject Q2 / backchannel
                    yield m
            for _ in range(int(self._tail * SR / FRAME)):      # trailing silence: catch the reply after injection
                yield _Msg(WSMsgType.BINARY, sil)
                if self._realtime:
                    await asyncio.sleep(FRAME / SR)
            yield _Msg(WSMsgType.TEXT, json.dumps({"type": "bye"}))
            return
        if self._opus_nturns is not None:             # ---- opus N-turn adaptive (personaplex/moshi benchmark) ----
            hdr, turn_ap, sil_ap = self._opus_nturns
            t0 = time.time(); g = 0; seq = 0
            for pg in hdr:                            # OpusHead/OpusTags: send as-is (open the stream), keep seq going
                yield _Msg(WSMsgType.BINARY, pg); seq = _page_seqno(pg)
            for ti, ap in enumerate(turn_ap):         # turn by turn: feed this turn's user speech → wait for the model to finish
                for pg, dgp in ap:                    # feed this turn: rewrite granule/seqno to be continuous, yield first then sleep on granule cadence
                    g += dgp; seq += 1
                    yield _Msg(WSMsgType.BINARY, _rewrite_page(pg, g, seq, clear_eos=True))
                    if self._realtime:
                        dt = g / 48000.0 - (time.time() - t0)
                        if dt > 0:
                            await asyncio.sleep(dt)
                base = self.model_speech_sec; self.last_recv = time.time()
                t_turn = time.time(); first_audio = None; spoke = False; si = 0
                while True:                           # feed silence pages in real time while waiting (user track advances, the model block anchors into this silence → no overlap)
                    pg, dgp = sil_ap[si % len(sil_ap)]; si += 1
                    g += dgp; seq += 1
                    yield _Msg(WSMsgType.BINARY, _rewrite_page(pg, g, seq, clear_eos=True))
                    if self._realtime:
                        dt = g / 48000.0 - (time.time() - t0)
                        if dt > 0:
                            await asyncio.sleep(dt)
                    if self.model_speech_sec > base + 0.3:
                        spoke = True
                        if first_audio is None:
                            first_audio = time.time()
                    now = time.time()
                    drained = first_audio is None or (now - first_audio) >= (self.model_speech_sec - base) + 0.3
                    if spoke and (now - self.last_recv) > self._turn_end_silence and drained:
                        break                         # model done (energy silent long enough + buffer drained)
                    if now - t_turn > self._turn_max_wait:
                        break                         # fallback
                self.turn_log.append(round(time.time() - t_turn, 1))
            yield _Msg(WSMsgType.TEXT, json.dumps({"type": "bye"}))
            return
        if self._opus_turns is not None:              # ---- opus adaptive multi-turn (moshi/personaplex benchmark) ----
            a1p, silp, a2p, a2_base = self._opus_turns
            t0 = time.time()

            async def _turn_end(base):                # wait for the model to finish this turn (silent long enough + buffer drained)
                self.last_recv = time.time(); t_turn = time.time(); first_audio = None; spoke = False
                while True:
                    await asyncio.sleep(FRAME / SR)
                    if self.model_speech_sec > base + 0.3:
                        spoke = True
                        if first_audio is None:
                            first_audio = time.time()
                    now = time.time()
                    drained = first_audio is None or (now - first_audio) >= (self.model_speech_sec - base) + 0.3
                    if spoke and (now - self.last_recv) > self._turn_end_silence and drained:
                        return round(time.time() - t_turn, 1)
                    if now - t_turn > self._turn_max_wait:
                        return round(time.time() - t_turn, 1)

            for pg, gp in a1p:                        # A1: feed in real time by granule
                if self._realtime and gp > 0:
                    dt = gp / 48000.0 - (time.time() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                yield _Msg(WSMsgType.BINARY, pg)
            # wait for the model to answer A1; while waiting, FEED SILENCE PAGES IN REAL TIME (continuous granule) so the user track advances normally and both tracks stay aligned
            base = self.model_speech_sec; self.last_recv = time.time(); t_turn = time.time()
            first_audio = None; spoke = False; si = 0
            g_stop = a1p[-1][1] if a1p else 0; seq = _page_seqno(a1p[-1][0]) if a1p else 0
            while True:
                if si < len(silp):
                    pg, gp = silp[si]; si += 1; g_stop = gp; seq = _page_seqno(pg)
                    if self._realtime and gp > 0:
                        dt = gp / 48000.0 - (time.time() - t0)
                        if dt > 0:
                            await asyncio.sleep(dt)
                    yield _Msg(WSMsgType.BINARY, pg)
                else:
                    await asyncio.sleep(FRAME / SR)
                if self.model_speech_sec > base + 0.3:
                    spoke = True
                    if first_audio is None:
                        first_audio = time.time()
                now = time.time()
                drained = first_audio is None or (now - first_audio) >= (self.model_speech_sec - base) + 0.3
                if spoke and (now - self.last_recv) > self._turn_end_silence and drained:
                    break
                if now - t_turn > self._turn_max_wait:
                    break
            self.turn_log.append(round(time.time() - t_turn, 1))
            # A2: follows the silence already fed — rewrite granule/seqno to be continuous, avoiding jumps
            for pg, gp in a2p:
                seq += 1
                new_gp = int(g_stop + (gp - a2_base))
                pg2 = _rewrite_page(pg, new_gp, seq)
                if self._realtime:
                    dt = new_gp / 48000.0 - (time.time() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                yield _Msg(WSMsgType.BINARY, pg2)
            self.turn_log.append(await _turn_end(self.model_speech_sec))   # wait for the model to answer A2
            yield _Msg(WSMsgType.TEXT, json.dumps({"type": "bye"}))
            return
        if self._pages is not None:                   # ---- OGG page path (moshi/personaplex single segment: backchannel/pause/turn_taking/injection) ----
            t0 = time.time()
            for pg, gp in self._pages:
                yield _Msg(WSMsgType.BINARY, pg)
                if self._realtime and gp > 0:         # granule is in 48k units → target time = gp/48000
                    dt = gp / 48000.0 - (time.time() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
            yield _Msg(WSMsgType.TEXT, json.dumps({"type": "bye"}))
            return
        async for m in self._pace_pcm(self._pcm):     # ---- raw PCM path (gpt/gemini) ----
            yield m
        for _ in range(int(self._tail * SR / FRAME)):  # pad trailing silence: let server_vad commit and record the model's reply
            yield _Msg(WSMsgType.BINARY, sil)
            if self._realtime:
                await asyncio.sleep(FRAME / SR)
        yield _Msg(WSMsgType.TEXT, json.dumps({"type": "bye"}))

    def __aiter__(self):
        return self._gen()

    async def send_str(self, s):        # ready/model_text/user_text… the relay sends back: just drop
        pass

    async def send_bytes(self, b):      # model audio: accumulate speech duration + record the last arrival time (reactive/multi-turn rely on it to gauge progress)
        self.sent_audio += b                          # write everything to disk (including zero frames, to keep the timeline aligned)
        # ⚠ energy gating: models like gemini keep sending zero frames during a "pause" (content = exact 0, RMS = 0);
        # recording last_recv only by "did a frame arrive" would never see silence → hit max_wait → hard-feed the next
        # user line while the model is still talking (human over model, a violation). Only "loud frames" advance the
        # speech duration / silence timer, so last_recv naturally goes stale during a pause.
        if len(b) >= 2:
            _a = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
            loud = _a.size > 0 and float(np.sqrt(np.mean(_a ** 2))) > 0.01
        else:
            loud = False
        if loud:
            if self.first_model_t is None:
                self.first_model_t = time.time()      # wall clock of the model "actually speaking": injections are placed at "onset + thr-second playback position"
            self.model_speech_sec += len(b) / 2 / SR
            self.last_recv = time.time()


# ------------------------------- audio -------------------------------
def load_f32(path: Path) -> np.ndarray:
    """Read as mono float32 @24k ([-1,1])."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    if sr != SR:
        import soxr
        data = soxr.resample(data, sr, SR).astype(np.float32)
    return data


def encode_ogg_pages(pcm_f32: np.ndarray):
    """Encode PCM to OGG/Opus with opus-recorder (node, same as the browser), split into pages and parse granulepos.
    → [(page_bytes, granulepos)]. sphn/the server only accepts this format (sphn's own encoder and ffmpeg are both incompatible)."""
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tf:
        pcm_f32.astype("<f4").tofile(tf.name)
        raw_path = tf.name
    ogg_path = raw_path + ".ogg"
    try:
        subprocess.run(["node", str(ROOT / "opus_encode.js"), raw_path, ogg_path, "128"],
                       check=True, capture_output=True)
        data = Path(ogg_path).read_bytes()
    finally:
        for p in (raw_path, ogg_path):
            try:
                Path(p).unlink()
            except OSError:
                pass
    idxs = [m.start() for m in re.finditer(b"OggS", data)]
    pages = []
    for j, s in enumerate(idxs):
        e = idxs[j + 1] if j + 1 < len(idxs) else len(data)
        pg = data[s:e]
        gp = struct.unpack("<q", pg[6:14])[0] if len(pg) >= 14 else 0
        pages.append((pg, gp))
    return pages


def _split_ogg(pages):
    """encode_ogg_pages output → (header pages [OpusHead/OpusTags], [(audio page, granule delta)]).
    In N-turn adaptive: each turn/silence is encoded separately; take the audio pages + per-page granule delta, and rewrite to continuous granule/seqno when splicing the stream."""
    hdr, audio, prev = [], [], 0
    for pg, gp in pages:
        if b"OpusHead" in pg or b"OpusTags" in pg:    # header page (BOS/tags, granule=0)
            hdr.append(pg)
        elif gp < 0 or gp < prev:                      # trailing EOS placeholder page (gp=-1) or non-monotonic → discard (not real audio)
            continue
        else:                                          # audio page: record the granule delta relative to this segment's start
            audio.append((pg, gp - prev)); prev = gp
    return hdr, audio


# --- OGG page rewrite: in opus adaptive multi-turn, A2 follows real-time silence, so granule/seqno must be made continuous and the CRC recomputed ---
_OGG_CRC = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04c11db7) & 0xffffffff if (_r & 0x80000000) else (_r << 1) & 0xffffffff
    _OGG_CRC.append(_r)


def _ogg_crc(data):
    c = 0
    for b in data:
        c = ((c << 8) & 0xffffffff) ^ _OGG_CRC[((c >> 24) ^ b) & 0xff]
    return c


def _page_seqno(pg):
    return struct.unpack("<I", pg[18:22])[0]


def _rewrite_page(pg, granule, seqno, clear_eos=False):
    """Set a single OGG page's granulepos/seqno to the given values and recompute the CRC (serial untouched, still the same logical stream).
    With clear_eos=True, clear the header-type EOS bit (0x04): in N-turn each segment is encoded separately and its last page carries EOS;
    if not cleared when splicing, the first segment's EOS makes the server think the stream ended and stop speaking."""
    b = bytearray(pg)
    if clear_eos and len(b) > 5:
        b[5] &= ~0x04                              # clear EOS (0x02=BOS / 0x04=EOS)
    b[6:14] = struct.pack("<q", int(granule))
    b[18:22] = struct.pack("<I", int(seqno) & 0xffffffff)
    b[22:26] = b"\x00\x00\x00\x00"
    b[22:26] = struct.pack("<I", _ogg_crc(b))
    return bytes(b)


# ------------------------------- one item -------------------------------
def _pcm16(f32: np.ndarray) -> bytes:
    return (np.clip(f32, -1, 1) * 32767).astype("<i2").tobytes()


async def _run_relay(model, ws, out_dir, silence_ms, url, prompt):
    if model == "gpt":
        return await live_ws.relay_gpt(ws, silence_ms, out_dir)
    if model == "gptlive":
        return await live_ws.relay_gpt_live(ws, silence_ms, out_dir)
    if model == "gemini":
        return await live_ws.relay_gemini(ws, out_dir, silence_ms=silence_ms)
    if model == "freezeomni":                                       # pass an English prompt, otherwise it defaults to a Chinese persona
        return await live_ws.relay_freezeomni(ws, out_dir, fo_url=url, prompt=prompt)
    if model in ("moshi", "personaplex"):
        return await live_ws.relay_moshi(ws, out_dir, moshi_url=url)
    raise SystemExit(f"unknown model {model}")


def _speech_onset(wav_path, th=0.01, hop=0.02, min_run=0.15):
    """Start second of the model's first continuous speech in the recording (RMS envelope threshold + min_run seconds continuous); returns None if no speech."""
    try:
        a, sr = sf.read(str(wav_path))
    except Exception:  # noqa: BLE001
        return None
    if getattr(a, "ndim", 1) > 1:
        a = a[:, 0]
    w = max(1, int(hop * sr))
    need = max(1, int(min_run / hop))
    run = 0
    for i in range(0, max(1, len(a) - w), w):
        if float(np.sqrt(np.mean(a[i:i + w] ** 2))) > th:
            run += 1
            if run >= need:
                return max(0.0, (i - (need - 1) * w) / sr)
        else:
            run = 0
    return None


async def _probe_opus_onset(model, f32, injections, out_dir, silence_ms, url, prompt):
    """Probe pass before opus injection: feed "question + silence" (no injection) to measure the model's onset second.
    A pre-encoded opus stream can't react in real time, so we probe the onset first to place backchannels at "onset + thr" instead of a fixed absolute time.
    On failure, fall back to "end of question" (old behavior). Outputs go to out_dir/_probe and are cleaned up right away so they don't pollute the real results."""
    import shutil
    probe_dir = Path(out_dir) / "_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_tail = max((t for _, t in injections), default=0.0) + 6.0
    pw = FakeWS(pages=encode_ogg_pages(
        np.concatenate([f32, np.zeros(int(probe_tail * SR), dtype=np.float32)])), realtime=True)
    m0 = None
    try:
        await _run_relay(model, pw, probe_dir, silence_ms, url, prompt)
        m0 = _speech_onset(probe_dir / "B_model.wav")
    except Exception as e:  # noqa: BLE001
        print(f"    [opus inject] probe failed {type(e).__name__}: {e}, falling back to end of question", flush=True)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    return m0 if m0 is not None else len(f32) / SR


async def drive_model(model: str, f32: np.ndarray, out_dir: Path, silence_ms: int, tail: float,
                      url: str, prompt: str = "", injections=None, turns_f32=None, turn_max_wait=14.0,
                      opus_gap=5.0, opus_adaptive=False):
    """Call the matching relay per server.py's dispatch rules, feeding the fake ws. When the relay finishes it writes A_user/B_model/combined.wav.
    turns_f32=[f32,...] given = multi-turn (benchmark): feed line by line, wait for the model to finish each (PCM models).
    injections=[(inj_f32, thr)] given = reactive injection: PCM models wait until the model has spoken thr seconds then inject; opus uses fixed pre-encoded intervals."""
    injections = injections or []
    if model == "halfduplex":                                      # half-duplex baseline: local VAD+STT+LLM+TTS offline simulation, no relay
        import halfduplex
        if turns_f32 is not None:                                  # benchmark: multi-turn adaptive — A2 placed right after the model finishes A1 (no fixed hard gap)
            return await asyncio.to_thread(halfduplex.simulate_turns, list(turns_f32), out_dir)
        if injections:                                             # interruption/user_backchannel: injection placed at "thr seconds after the model starts"
            # half-duplex has no reactive path, but the model's onset can be computed analytically: question VAD end te + endpoint + latency.
            # Place the backchannel at onset+thr (thr = thr seconds into the model's speech), same as streaming models: mid-speech, not too early, not too crowded.
            hd_turns = halfduplex._turns_from_vad(f32, 0.5)
            te = hd_turns[-1][1] if hd_turns else len(f32) / SR
            m0 = te + 0.5 + 0.7                                    # model onset ≈ question end + endpoint(0.5) + latency(0.7)
            parts, cur = [f32], len(f32) / SR
            for inj, thr in injections:
                at = m0 + thr
                parts += [np.zeros(int(max(0.0, at - cur) * SR), dtype=np.float32), inj]
                cur = at + len(inj) / SR
            aud = np.concatenate(parts)
        else:                                                      # backchannel/pause/turn_taking: feed the whole segment directly
            aud = f32
        return await asyncio.to_thread(halfduplex.simulate, aud, out_dir)
    if turns_f32 is not None:                                      # multi-turn (benchmark): feed line by line, wait for the model to finish each
        if model in ("moshi", "personaplex") and opus_adaptive:    # opus N-turn adaptive: wait for the model to finish each turn before feeding the next (the human won't talk over the model)
            turn_ap = [_split_ogg(encode_ogg_pages(t))[1] for t in turns_f32]
            hdr = _split_ogg(encode_ogg_pages(turns_f32[0]))[0]     # take the header page from the first turn (open the stream)
            _, sil_ap = _split_ogg(encode_ogg_pages(np.zeros(int(1.0 * SR), dtype=np.float32)))  # 1s silence pages, fed in a loop
            ws = FakeWS(opus_nturns=(hdr, turn_ap, sil_ap), realtime=True, turn_max_wait=turn_max_wait)
        elif model in ("moshi", "personaplex"):                    # opus: one clean pre-encoded stream (A1 + fixed gap + A2).
            # fixed gap: a chatty model rambling past the gap will overrun the next user line; use --opus-adaptive for strict no-overlap.
            GAP = opus_gap
            parts = []
            for i, t in enumerate(turns_f32):
                parts.append(t)
                parts.append(np.zeros(int((GAP if i < len(turns_f32) - 1 else tail) * SR), dtype=np.float32))
            ws = FakeWS(pages=encode_ogg_pages(np.concatenate(parts)), realtime=True)
        else:                                                      # PCM: reactive, send the next line only after the model finishes each one
            ws = FakeWS(turns=[_pcm16(x) for x in turns_f32], realtime=True, turn_max_wait=turn_max_wait)
        info = await _run_relay(model, ws, out_dir, silence_ms, url, prompt)
        if getattr(ws, "turn_log", None):
            print(f"    per-turn time {ws.turn_log}s", flush=True)
        return info
    if model in ("moshi", "personaplex"):                          # opus: pre-encode the whole segment into one OGG stream
        if injections:
            # two passes: first probe the model's onset, then place injections at "onset + thr" (thr = thr seconds into the model's speech, same as PCM reactive).
            # otherwise a fixed absolute time would inject before a slow-starting model (personaplex ~3s latency) speaks, landing the backchannel before the model's onset (wrong).
            m0 = await _probe_opus_onset(model, f32, injections, out_dir, silence_ms, url, prompt)
            print(f"    [opus inject] probed model onset @ {m0:.2f}s → placing injections at onset+{[t for _, t in injections]}s", flush=True)
            parts, cur = [f32], len(f32) / SR
            for inj_f32, thr in injections:
                at = m0 + thr                                      # thr seconds into the model's speech
                parts += [np.zeros(int(max(0.0, at - cur) * SR), dtype=np.float32), inj_f32]
                cur = at + len(inj_f32) / SR
            parts.append(np.zeros(int(tail * SR), dtype=np.float32))
        else:                                                      # backchannel/pause/turn_taking: feed the whole segment directly
            parts = [f32, np.zeros(int(tail * SR), dtype=np.float32)]
        pages = encode_ogg_pages(np.concatenate(parts))
        ws = FakeWS(pages=pages, realtime=True)
    elif injections:                                               # gpt/gemini/freezeomni: reactive injection
        ws = FakeWS(prompt_pcm=_pcm16(f32),
                    injections=[(_pcm16(x), thr) for x, thr in injections],
                    tail_silence=tail, realtime=True)
    else:
        ws = FakeWS(pcm=_pcm16(f32), realtime=True, tail_silence=tail)
    info = await _run_relay(model, ws, out_dir, silence_ms, url, prompt)
    if injections and getattr(ws, "inject_log", None):
        print(f"    injections {ws.inject_log}", flush=True)
    return info


def run_parakeet(out_dir: Path, stems=("A_user", "B_model")):
    for stem in stems:
        wav = out_dir / f"{stem}.wav"
        if not wav.exists():
            continue
        words, text = parakeet_local.transcribe_wav_segmented(str(wav))
        (out_dir / f"{stem}.parakeet.json").write_text(
            json.dumps(parakeet_local._doc_for(wav, words, text), ensure_ascii=False), encoding="utf-8")


def _folder_name(stem: str) -> str:
    """01.wav → '1' (matches the manual 1/ 2/ naming scheme); non-numeric names as-is."""
    return str(int(stem)) if stem.isdigit() else stem


def _valid_result(jf: Path) -> bool:
    """A valid result requires {task}.json to exist and not be an error (a grading error doesn't count as done, needs a re-run)."""
    if not jf.exists():
        return False
    try:
        return "error" not in json.loads(jf.read_text())
    except Exception:  # noqa: BLE001 —— treat bad json as invalid too
        return False


def _result_name(task: str) -> str:
    """Grading result filename. benchmark saves grade.json (to avoid colliding with the ground-truth benchmark.json)."""
    return "grade.json" if task == "benchmark" else f"{task}.json"


def already_done(folder: Path, model: str, task: str) -> bool:
    """This folder already has a VALID result for this model → skip (idempotent). A zip (manual run) counts as complete;
    a plain folder must contain a NON-error result json; wav-only / error-json don't count, so it re-runs."""
    if not folder.is_dir():
        return False
    for p in folder.iterdir():
        if p.name.endswith(f"_{model}.zip"):
            return True
        if p.is_dir() and p.name.endswith(f"_{model}") and _valid_result(p / _result_name(task)):
            return True
    return False


def clean_partial(folder: Path, model: str, task: str):
    """Delete invalid leftovers: plain folders for this model whose result json is missing or an error get removed entirely and re-recorded."""
    import shutil
    if not folder.is_dir():
        return
    for p in folder.iterdir():
        if p.is_dir() and p.name.endswith(f"_{model}") and not _valid_result(p / _result_name(task)):
            shutil.rmtree(p)
            print(f"🧹 cleaned leftover/error {p}", flush=True)


def run_one(model: str, input_wav: Path, task: str, gemini: bool, silence_ms: int, tail: float,
            out_dir: Path, url: str, prompt: str = "", q2_wav: Path = None,
            interrupt_text: str = "", pause_gap: float = 1.5, inject_specs=None, single_feed: bool = False,
            turn_max_wait: float = 14.0, opus_gap: float = 5.0, opus_adaptive: bool = False):
    f32 = load_f32(input_wav)
    injections = turns_f32 = None
    if task == "benchmark" and not (input_wav.parent / "A2.wav").exists():
        # single-turn stimulus (volume_understanding / whisper_production etc.): feed only A1 as one segment, record one reply
        extra = "(single-turn stimulus)"              # turns_f32/injections both None → drive_model feeds f32 as one segment
    elif task == "benchmark":                         # A1 → wait for model → A2 → … (multi-turn, right/wrong grading)
        a2 = load_f32(input_wav.parent / "A2.wav")
        if single_feed:                               # keyword_wait type: instruction + list spoken back to back, fed as one continuous segment → the model inserts at the trigger word as it listens, avoiding an A2-injection collision
            f32 = np.concatenate([f32, np.zeros(int(pause_gap * SR), dtype=np.float32), a2])
            extra = f"(A1+{pause_gap}s pause+A2 single segment)"
        else:
            turns_f32 = [f32, a2]                      # also load A3,A4,… (multi-turn tasks like alternating_count)
            k = 3
            while (input_wav.parent / f"A{k}.wav").exists():
                turns_f32.append(load_f32(input_wav.parent / f"A{k}.wav"))
                k += 1
            extra = f"A1→A{len(turns_f32)} multi-turn"
    elif task == "pause" and q2_wav:                   # A + mid-sentence pause + B, fed as one continuous user utterance (not reactive)
        b = load_f32(q2_wav)
        f32 = np.concatenate([f32, np.zeros(int(pause_gap * SR), dtype=np.float32), b])
        extra = f"(A+{pause_gap}s pause+B)"
    elif inject_specs:                                 # interruption / user_backchannel: reactive injection
        injections = [(load_f32(p), thr) for p, thr in inject_specs]
        extra = f"→inject {'/'.join(p.stem for p, _ in inject_specs)}"
    else:
        extra = f"+{tail:.0f}s trailing silence"
    print(f"▶ {input_wav.name} → {model}  ({extra}, real-time…)", flush=True)
    info = asyncio.run(drive_model(model, f32, out_dir, silence_ms, tail, url, prompt, injections, turns_f32,
                                   turn_max_wait=turn_max_wait, opus_gap=opus_gap, opus_adaptive=opus_adaptive))
    if info is None:
        print("  ✗ connection/relay failed (see the error above)")
        return None
    print(f"  recorded {info['secs']}s, running parakeet…", flush=True)
    run_parakeet(out_dir)
    if model == "gptlive":
        # The Live API's audio deltas carry no timestamps, so the relay can only record the raw stream;
        # placing it on the user timeline needs the session clock, which needs the user track's ASR first.
        import vad_utils

        import live_align
        placed = live_align.place(out_dir, parakeet_local.transcribe_wav_segmented, vad_utils.vad_segments)
        if placed and placed.get("placed"):
            run_parakeet(out_dir, stems=("B_model",))     # re-transcribe the track we just moved
        elif placed:
            print(f"  ! session-clock placement skipped: {placed.get('why')}", flush=True)
    key = os.environ.get("GEMINI_API_KEY") if gemini else None
    if task == "benchmark":                           # ground-truth right/wrong grading
        import shutil
        shutil.copy(input_wav.parent / "benchmark.json", out_dir / "benchmark.json")
        res = gb.grade_benchmark(out_dir, use_gemini=gemini, api_key=key)
    elif task == "interruption":
        res = gb.grade_interruption(out_dir, use_gemini=gemini, api_key=key, interrupt_text=interrupt_text)
    elif task == "turn_taking":
        res = gb.grade_turn_taking(out_dir)                    # takes only the directory, not the gemini arg
    else:
        grader = gb.TASKS[task]
        res = grader(out_dir, use_gemini=gemini, api_key=key)
    (out_dir / _result_name(task)).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        rd = str(out_dir.resolve().relative_to(ROOT))
    except ValueError:
        rd = str(out_dir)
    return {"input": input_wav.name, "run_dir": rd, "result": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt",
                    choices=["gpt", "gptlive", "gemini", "moshi", "personaplex", "freezeomni", "halfduplex"])
    ap.add_argument("--input", type=Path, help="a single user audio wav")
    ap.add_argument("--input-dir", type=Path, help="batch: all wavs in the directory matching --glob")
    ap.add_argument("--glob", default="[0-9][0-9].wav")
    ap.add_argument("--collect-to", type=Path, help="save results by monologue number into <dir>/<N>/live_<id>_<model>/ (no zip); skip if a result for this model already exists")
    ap.add_argument("--task", default="backchannel", choices=list(gb.TASKS))
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--silence-ms", type=int, default=500)
    ap.add_argument("--tail", type=float, default=10.0, help="how many seconds of silence to pad after the user audio (to record the model's reply)")
    ap.add_argument("--turn-max-wait", dest="turn_max_wait", type=float, default=14.0,
                    help="multi-turn PCM: max seconds to wait for the model's answer in a turn before feeding the next line (raise it, e.g. 45, for gemini's long/trailing speech to avoid overlap)")
    ap.add_argument("--opus-gap", dest="opus_gap", type=float, default=5.0,
                    help="multi-turn opus (moshi/personaplex): fixed gap seconds between turns (raise it, e.g. 12, for long dialogues to reduce overlap, then compact the dead air)")
    ap.add_argument("--opus-adaptive", dest="opus_adaptive", action="store_true",
                    help="multi-turn opus: feed silence pages each turn and wait for the model to finish before feeding the next line (strictly no talking over the model; replaces the fixed --opus-gap)")
    ap.add_argument("--url", default="", help="service address for moshi/personaplex/freezeomni (defaults per model if not given)")
    ap.add_argument("--text-prompt", default=FREEZEOMNI_DEFAULT_PROMPT_SENTINEL,
                    help="personaplex system prompt (spliced into the wss URL's text_prompt); for freezeomni, when not overridden, auto-swapped for a stricter English-only pin")
    ap.add_argument("--voice-prompt", default="NATF0.pt",
                    help="personaplex voice .pt file (required by the server, matches the dashboard's voicePrompt dropdown)")
    ap.add_argument("--interrupt-after", type=float, default=3.0,
                    help="interruption task: let the model speak this many seconds (of actual speech) before injecting the interrupt line Q2")
    ap.add_argument("--pause", type=float, default=1.5,
                    help="pause task: seconds of mid-sentence pause between the first half A and the second half B (tests whether the model grabs the floor)")
    ap.add_argument("--single-feed", action="store_true",
                    help="for benchmark, splice A1+pause+A2 into one continuous segment (keyword_wait etc.: the model inserts at the trigger word as it listens, no turn split)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    def model_url():
        if args.url:
            return args.url
        if args.model == "personaplex":                 # wss + system/voice prompt via query (matches the dashboard)
            return (f"wss://localhost:8998/api/chat?text_prompt={quote(args.text_prompt)}"
                    f"&voice_prompt={quote(args.voice_prompt)}")
        if args.model == "moshi":
            return live_ws.MOSHI_DEFAULT_URL             # ws://localhost:8998/api/chat
        if args.model == "freezeomni":
            return live_ws.FO_DEFAULT_URL
        return ""                                        # gpt/gemini use an API, no address
    url = model_url()

    # interruption: <NN>_first.wav (+ _interrupt.wav); pause: <NN>_a.wav (+ _b.wav);
    # user_backchannel: <NN>.wav question + short ok/right backchannels injected from _backchannels/; number = NN
    is_interrupt = args.task == "interruption"
    is_pause = args.task == "pause"
    is_ub = args.task == "user_backchannel"
    is_bench = args.task == "benchmark"              # test_set/<cat>/<NN>/{A1,A2}.wav + benchmark.json, multi-turn right/wrong grading
    src_dir = args.input_dir or (args.input.parent if args.input else None)
    interrupt_texts = {}
    if is_interrupt and src_dir and (src_dir / "index.txt").exists():
        for line in (src_dir / "index.txt").read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                fn, txt = line.split("\t", 1)
                if fn.endswith("_interrupt.wav"):
                    interrupt_texts[fn.split("_")[0]] = txt.strip()
    BC_POOL = ["ok", "right", "mhm", "yeah", "uh-huh", "I-see"]      # short backchannels injected for user_backchannel
    BC_DIR = (src_dir / "_backchannels") if src_dir else None

    glob_pat = ("*_first.wav" if is_interrupt else "*_a.wav" if is_pause
                else "*/A1.wav" if is_bench else args.glob)
    inputs = []
    if args.input:
        inputs = [args.input]
    elif args.input_dir:
        inputs = sorted(args.input_dir.glob(glob_pat))
    else:
        ap.error("give --input or --input-dir")
    if args.limit:
        inputs = inputs[:args.limit]

    collect_to = args.collect_to.resolve() if args.collect_to else None
    out, skipped, failed = [], 0, 0
    for ix, wav in enumerate(inputs):
        sid = uuid.uuid4().hex[:12]
        q2_wav, itext, inject_specs, folder_stem = None, "", None, wav.stem
        if is_interrupt:                                             # NN_first.wav → number NN, Q2, interrupt text
            iid = wav.stem.split("_")[0]
            itext = interrupt_texts.get(iid, "")
            inject_specs = [(wav.with_name(wav.name.replace("_first", "_interrupt")), args.interrupt_after)]
            folder_stem = iid
        elif is_pause:                                               # NN_a.wav → number NN, second half B (concatenated, not injected)
            q2_wav = wav.with_name(wav.name.replace("_a", "_b"))
            folder_stem = wav.stem.split("_")[0]
        elif is_ub:                                                  # question + one backchannel injected at 4s/9s into the model's answer
            b1 = BC_DIR / f"{BC_POOL[ix % len(BC_POOL)]}.wav"
            b2 = BC_DIR / f"{BC_POOL[(ix + 2) % len(BC_POOL)]}.wav"
            inject_specs = [(b1, 4.0), (b2, 9.0)]                     # thr = playback seconds after the model starts (see FakeWS reactive injection)
        if is_bench:
            folder = wav.parent                                          # results saved straight into the test_set item folder
        elif collect_to:
            folder = collect_to / _folder_name(folder_stem)
        else:
            folder = None
        if folder is not None:
            if already_done(folder, args.model, args.task):
                print(f"⏭  {wav.parent.name if is_bench else wav.name} → {args.model} already has a result, skipping", flush=True)
                skipped += 1
                continue
            clean_partial(folder, args.model, args.task)                   # clear last run's crash/error before re-recording
            out_dir = folder / f"live_{sid}_{args.model}"
        else:
            out_dir = RUNS / f"live_{sid}" / args.model
        # freezeomni's Qwen2 base blurts Chinese the moment it hits counting/a Chinese context; give it a stricter English-only pin (unless the user explicitly set --text-prompt)
        run_prompt = args.text_prompt
        if args.model == "freezeomni" and args.text_prompt == FREEZEOMNI_DEFAULT_PROMPT_SENTINEL:
            run_prompt = ("You are an English-only voice assistant. Respond ONLY in English. "
                          "Never speak Chinese or any other language. Say every number as an English "
                          "word (one, two, three, ...), never in Chinese. Keep replies short.")
        try:
            r = run_one(args.model, wav, args.task, args.gemini, args.silence_ms, args.tail, out_dir, url,
                        run_prompt, q2_wav, itext, args.pause, inject_specs, args.single_feed,
                        turn_max_wait=args.turn_max_wait, opus_gap=args.opus_gap,
                        opus_adaptive=args.opus_adaptive)
        except Exception as e:  # noqa: BLE001 —— one item's failure doesn't sink the whole batch
            print(f"  ✗ {wav.name}: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        if r:
            out.append(r)
            res = r["result"]
            brief = {k: res.get(k) for k in ("was_talking", "tor", "latency_ms", "jumped_in", "pause_dur",
                                             "n_backchannel", "behaviour", "derailed", "n_segments") if k in res}
            if isinstance(res.get("relevance"), dict):
                brief["addressed"] = res["relevance"].get("addressed")
            print(f"  ✓ {r['run_dir']}  {brief}", flush=True)
    print(f"\n==== ran {len(out)}/{len(inputs)} items ({args.model} · {args.task}), skipped {skipped}, failed {failed} ====")
    for r in out:
        print(f"  {r['input']:10} {r['run_dir']}")


if __name__ == "__main__":
    main()
