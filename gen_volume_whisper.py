"""Generate stimulus audio + benchmark.json for two new tests:
  volume_understanding —— the user speaks the same pragmatic scenarios in a whisper/loudly, fed to a live model to see whether it understands + adapts.
                          A1.wav is rendered as a real whisper/real loud voice via MiMo style.
  whisper_production   —— the user, in a "normal voice", instructs the model to answer in a whisper (with a verifiable answer).
                          A1.wav uses a normal timbre (what's tested is the model's output, not the stimulus).

Stimulus acoustic baseline: each item prints rms/voiced/f0/spectral-centroid to confirm the whisper is truly soft and the loud one truly loud.
Grading is left to grade_acoustic.py (delta relative to the model's own normal baseline).

Usage: uv run python gen_volume_whisper.py            # generate everything
       uv run python gen_volume_whisper.py --preview  # only emit samples to _preview/ for listening
"""
import argparse
import io
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
import generate_tts_mimo as tts
import acoustics

TTS_MODEL = "mimo-v2.5-tts"
VOICE = "Mia"
WHISPER_STYLE = "Whisper very softly and breathily, almost no voice, intimate and hushed, like telling a secret nobody else should hear."
LOUD_STYLE = "Speak very loudly and forcefully, projecting hard, almost shouting across a big noisy room, urgent and strong."

# volume_understanding: condition = stimulus timbre; expect = the pragmatics the model should pick up on (5 whisper + 5 loud)
VOLUME_ITEMS = [
    dict(id="01", condition="whisper", text="Hey, come closer. Between you and me — which one should I actually pick?",
         expect="Treat it as confidential/between-us; engage discreetly, don't broadcast."),
    dict(id="02", condition="whisper", text="Shh, the baby just fell asleep. Keep your voice down and remind me what's next on the list.",
         expect="Acknowledge the quiet context; ideally soften; give the reminder without being loud."),
    dict(id="03", condition="loud", text="Hey! I'm all the way over here and it's really noisy — what time are we leaving?!",
         expect="Register distance/noise; answer clearly/energetically."),
    dict(id="04", condition="loud", text="Quick — the smoke alarm is blaring! What's the first thing I should do?!",
         expect="Register urgency; give a prompt, direct first action."),
    dict(id="05", condition="whisper", text="Don't let them hear us — quickly, what's the wifi password again?",
         expect="Treat as secret/discreet; help quietly without announcing it loudly."),
    dict(id="06", condition="whisper", text="I'm in the library, I have to stay quiet. When does my meeting start?",
         expect="Respect the quiet setting; ideally soften; answer concisely."),
    dict(id="07", condition="whisper", text="Lean in… I'm planning a surprise for her. Help me pick a good place to hide it.",
         expect="Play along discreetly with the secret/surprise; keep it confidential."),
    dict(id="08", condition="loud", text="SPEAK UP! The traffic is deafening out here — which exit do I take?!",
         expect="Register the noise/urgency; give a clear, direct answer."),
    dict(id="09", condition="loud", text="Ow! I just burned my hand — what do I do right now?!",
         expect="Register urgency/distress; give a prompt, calm first-aid action."),
    dict(id="10", condition="loud", text="Hey!! You're way across the field — can you hear me, are we still on for tonight?!",
         expect="Register distance/shouting; answer clearly and reassuringly."),
]

# whisper_production: instruction + verifiable answer (used by the content grader), target_style=whisper
WHISPER_ITEMS = [
    dict(id="01", text="Whisper this so no one else hears — what's seven times eight?", answer="56"),
    dict(id="02", text="Lower your voice right down to a whisper and tell me the capital of France.", answer="Paris"),
    dict(id="03", text="Keep it really quiet, almost silent — what year did the Second World War end?", answer="1945"),
    dict(id="04", text="Whisper it to me — how many days are there in a week?", answer="7"),
    dict(id="05", text="Say this really quietly, almost under your breath — what color do you get mixing blue and yellow?", answer="green"),
    dict(id="06", text="Lower your voice to a whisper: what's the capital of Italy?", answer="Rome"),
    dict(id="07", text="Keep it hushed, like a secret — what's the capital of Japan?", answer="Tokyo"),
    dict(id="08", text="Whisper the answer so nobody hears — what's ten minus four?", answer="6"),
    dict(id="09", text="Very softly, almost silent — how many legs does a spider have?", answer="8"),
    dict(id="10", text="Whisper it — what planet do we live on?", answer="Earth"),
]


def _retry(fn, n=6):
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == n - 1:
                raise
            time.sleep(2 * (i + 1))


# amplitude shaping: style gives the spectral character, then we layer "loudness" by condition so very-soft/very-loud also separate in amplitude.
# levels: whisper pushed down to RMS~0.025 (audible but clearly soft, no lower or the model's VAD/ASR may miss it → empty reply); loud pushed to peak 0.97.
GAIN = {"whisper": ("rms", 0.025), "loud": ("peak", 0.97), "normal": (None, None)}


def _shape(wav_bytes, condition):
    mode, target = GAIN.get(condition, (None, None))
    if not mode:
        return wav_bytes
    a, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = a[:, 0]
    cur = float(np.sqrt(np.mean(a ** 2))) if mode == "rms" else float(np.max(np.abs(a)))
    if cur > 1e-6:
        a = a * (target / cur)
    a = np.clip(a, -0.99, 0.99)
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synth(text, style, dst: Path, condition="normal"):
    tok = os.environ["MIMO_TTS_TOKEN"].strip()
    wav, _ = _retry(lambda: tts.synthesize(api_key=tok, ref_data_uri=VOICE, text=text,
                                           style=style, model=TTS_MODEL, proxy=None))
    wav = _shape(wav, condition)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(wav)
    return acoustics.analyze(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="only emit samples to _preview/ for listening")
    args = ap.parse_args()
    base = ROOT / ("_preview" if args.preview else "test_set")

    print("=== volume_understanding stimuli (A1 = whisper/loud timbre) ===")
    for it in VOLUME_ITEMS:
        d = base / "volume_understanding" / it["id"]
        if (d / "A1.wav").exists():                    # already generated (timbre confirmed) → skip, don't re-synthesize
            print(f"  {it['id']} skip (already exists)")
            continue
        style = WHISPER_STYLE if it["condition"] == "whisper" else LOUD_STYLE
        m = synth(it["text"], style, d / "A1.wav", condition=it["condition"])
        bj = dict(category="volume_understanding", condition=it["condition"],
                  grade_dimension="understanding+adaptation", user_text=it["text"], pragmatic_expect=it["expect"],
                  stimulus_acoustics=m, scenario_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"volume/{it['id']}")))
        (d / "benchmark.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {it['id']} [{it['condition']:7}] rms={m['rms']:.3f} voiced={m['voiced_ratio']} f0={m['mean_f0']} centroid={m['spec_centroid']}  {it['text'][:45]!r}")

    print("\n=== whisper_production stimuli (A1 = normal-voice instruction) ===")
    for it in WHISPER_ITEMS:
        d = base / "whisper_production" / it["id"]
        if (d / "A1.wav").exists():
            print(f"  {it['id']} skip (already exists)")
            continue
        m = synth(it["text"], "", d / "A1.wav")            # instruction uses a normal timbre
        bj = dict(category="whisper_production", grade_dimension="whisper_style+content",
                  target_style="whisper", expected_answer=it["answer"], stimulus_acoustics=m,
                  scenario_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"whisper/{it['id']}")))
        (d / "benchmark.json").write_text(json.dumps(bj, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {it['id']} rms={m['rms']:.3f} voiced={m['voiced_ratio']} ans={it['answer']!r}  {it['text'][:45]!r}")

    print(f"\nwrote {base}")


if __name__ == "__main__":
    main()
