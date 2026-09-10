"""MiMo API transport layer: chat + TTS — direct to Xiaomi's official api.xiaomimimo.com.

  - endpoint: https://api.xiaomimimo.com/v1/chat/completions (official, hardcoded default;
              in the rare case it's needed, override via env var MIMO_API_URL — ordinary users can ignore it)
  - auth:     api-key: <MIMO_API_KEY> (MiMo sk- token, not Bearer)
  - model:    LLM uses mimo-v2.5-pro (gen_benchmark produces the reference answers); TTS uses mimo-v2.5-tts*

Single call, raises RuntimeError on error (message carries `[mimo]: HTTP {status}: {body}`). Retry / concurrency is up to the caller.
"""

import base64
import json
import os

import requests

API_URL = os.getenv("MIMO_API_URL", "https://api.xiaomimimo.com/v1/chat/completions")


def _headers() -> dict:
    key = os.environ.get("MIMO_API_KEY")
    if not key:
        raise RuntimeError("MIMO_API_KEY not set (.env)")
    return {"api-key": key, "Content-Type": "application/json"}  # MiMo sk- token, not Bearer


def gateway_chat(messages: list, model: str, response_format: dict | None = None,
                 timeout: int = 180) -> tuple[str, dict]:
    """Chat/text completion, returns (message.content string, usage dict).

    usage is the API's raw {prompt_tokens, completion_tokens, total_tokens, ...} (may be empty {}).
    """
    payload: dict = {"model": model, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    resp = requests.post(API_URL, json=payload, headers=_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"[mimo]: HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"], data.get("usage") or {}


def gateway_tts(text: str, voice: str, model: str, style: str = "", timeout: int = 120) -> bytes:
    """TTS, returns wav bytes.

    voice: in preset mode it's a voice name (e.g. "Mia"); in voiceclone mode it's a
    `data:audio/...;base64,...` reference-audio data URI.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": style},       # empty or a style instruction
            {"role": "assistant", "content": text},    # the actual content to read aloud
        ],
        "audio": {"format": "wav", "voice": voice},
    }
    resp = requests.post(API_URL, json=payload, headers=_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"[mimo]: HTTP {resp.status_code}: {resp.text[:400]}")
    try:
        audio_b64 = resp.json()["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"[mimo]: no audio in response: {json.dumps(resp.json())[:300]}") from e
    return base64.b64decode(audio_b64)
