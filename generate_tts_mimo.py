"""
Test Xiaomi MiMo voice-clone TTS.

Endpoint: https://api.xiaomimimo.com/v1/chat/completions
Model:    mimo-v2.5-tts-voiceclone

The reference audio (MP3 or WAV) is base64-encoded and passed as the cloned
voice. The text to synthesize goes in the **assistant** role; the user role is
empty or holds a natural-language style direction. Response audio is base64
16-bit PCM WAV, mono, 24 kHz.

Auth is the `api-key:` header (NOT `Authorization: Bearer`).

Concurrency note: the API allows 1 in-flight request per (api_key, source_IP).
A 2nd simultaneous call from the same combo returns HTTP 429. This script makes
one call at a time, so it stays within that limit.

Usage:
    # API key from .env (MIMO_API_KEY=sk-...) or --api-key
    python generate_tts_mimo.py \
        --reference mixed1.mp3 \
        --text "Hello, this is a test of the MiMo voice clone." \
        --output output_mimo.wav

    # with a natural-language style direction
    python generate_tts_mimo.py \
        --reference mixed1.mp3 \
        --text "Hi [laughs] glad to see you." \
        --style "Speak warmly, slightly tired tone" \
        --output output_mimo.wav
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-tts-voiceclone"

# 10 MB cap on the base64 payload (~7.5 MB raw). Warn past the raw threshold.
MAX_RAW_BYTES = int(7.5 * 1024 * 1024)


def encode_reference(path: Path) -> str:
    """Read a reference audio file and return a base64 data URI."""
    raw = path.read_bytes()
    if len(raw) > MAX_RAW_BYTES:
        print(
            f"WARNING: reference is {len(raw) / 1024 / 1024:.1f} MB raw; "
            f"base64 payload may exceed the 10 MB API cap. "
            f"Use a 15-30 s clip.",
            file=sys.stderr,
        )
    ext = path.suffix.lower()
    if ext == ".wav":
        mime = "audio/wav"
    elif ext in (".mp3", ".mpeg", ".mpg"):
        mime = "audio/mpeg"
    else:
        raise ValueError(f"Reference must be .wav or .mp3, got {ext!r}")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def synthesize(
    api_key: str,
    ref_data_uri: str,
    text: str,
    style: str = "",
    timeout: int = 120,
    proxy: str | None = None,
    model: str = MODEL,
    url: str = API_URL,
) -> tuple[bytes, dict]:
    """Call MiMo TTS and return (wav_bytes, usage_dict).

    Args:
        ref_data_uri: For the voice-clone model, a base64 reference audio data URI.
            For the non-clone model ("mimo-v2.5-tts"), a preset voice id string
            (e.g. "Mia", "Dean"). Either way it goes into the audio.voice field.
        proxy: Optional proxy URL (e.g. "http://user:pass@host:port"). When set,
            the request is routed through it for both http and https. This is how
            "lanes" get distinct outbound IPs to stay under the 1-in-flight-per-
            (api_key, source_IP) cap.
        model: TTS model id. Default mimo-v2.5-tts-voiceclone; pass
            "mimo-v2.5-tts" for preset (non-clone) voices.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": style},  # empty or style/persona direction
            {"role": "assistant", "content": text},  # what gets spoken
        ],
        "audio": {
            "format": "wav",
            "voice": ref_data_uri,
        },
    }
    headers = {
        "api-key": api_key,  # NOT Authorization: Bearer
        "Content-Type": "application/json",
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None

    resp = requests.post(
        url, json=payload, headers=headers, timeout=timeout, proxies=proxies
    )

    if resp.status_code == 429:
        # include MiMo's actual response body (previously dropped), plus our own note, to help pin down which kind of 429 limit this is
        raise RuntimeError(
            f"HTTP 429: {resp.text[:300]} | (local note: usually 1 in-flight per api_key+IP, "
            "wait for the current call to finish or switch IP/lane)"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        audio_b64 = data["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected response shape: {json.dumps(data)[:500]}"
        ) from e

    return base64.b64decode(audio_b64), data.get("usage", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Xiaomi MiMo voice-clone TTS.")
    parser.add_argument(
        "--reference",
        default="mixed1.mp3",
        help="Reference audio file (.wav or .mp3). Default: mixed1.mp3",
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the MiMo voice clone.",
        help="Text to synthesize (goes in the assistant role).",
    )
    parser.add_argument(
        "--style",
        default="",
        help="Optional natural-language style direction (goes in the user role).",
    )
    parser.add_argument(
        "--output",
        default="output_mimo.wav",
        help="Output WAV path. Default: output_mimo.wav",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MIMO_API_KEY"),
        help="MiMo API key. Defaults to MIMO_API_KEY env var.",
    )
    args = parser.parse_args()

    if not args.api_key:
        sys.exit(
            "ERROR: no API key. Set MIMO_API_KEY in .env or pass --api-key sk-..."
        )

    ref_path = Path(args.reference)
    if not ref_path.exists():
        sys.exit(f"ERROR: reference audio not found: {ref_path}")

    print(f"Reference : {ref_path}")
    print(f"Text      : {args.text!r}")
    if args.style:
        print(f"Style     : {args.style!r}")

    ref_data_uri = encode_reference(ref_path)

    t0 = time.time()
    wav_bytes, usage = synthesize(
        api_key=args.api_key,
        ref_data_uri=ref_data_uri,
        text=args.text,
        style=args.style,
    )
    elapsed = time.time() - t0

    out_path = Path(args.output)
    out_path.write_bytes(wav_bytes)

    print(f"\nWrote {out_path} ({len(wav_bytes) / 1024:.1f} KB) in {elapsed:.1f}s")
    print(f"Usage: {usage}")


if __name__ == "__main__":
    main()
