# InteractionBench

**Measuring timing, task accuracy, paralinguistic control, and grounding in real-time voice systems.**

InteractionBench evaluates real-time voice systems on four dimensions: conversational timing, spoken-task accuracy, paralinguistic control, and groundedness in long conversations. Every system receives the same recorded user audio. The text instruction and the release of the next user turn on multi-turn items differ across systems, as described in the site's Limitations section.

🔊 **Live site:** https://avera-labs.github.io/InteractionBench/

## Systems evaluated

| id | system | model / checkpoint |
|---|---|---|
| `gptlive` | GPT-Live-1 | `gpt-live-1`, deep reasoning delegated to `gpt-5.5` |
| `gpt` | GPT-Realtime | `gpt-realtime-2.1` |
| `gemini` | Gemini-Live | `gemini-3.1-flash-live-preview` |
| `moshi` | Moshi (Kyutai) | [`kyutai/moshika-pytorch-bf16`](https://huggingface.co/kyutai/moshika-pytorch-bf16) at `a49141e` |
| `personaplex` | PersonaPlex | [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) at `fdaf409`, voice prompt `NATF0` |
| `freezeomni` | FreezeOmni | [`VITA-MLLM/Freeze-Omni`](https://huggingface.co/VITA-MLLM/Freeze-Omni) at `c8b1918` with [`Qwen/Qwen2-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct) at `f2826a0` |
| `halfduplex` | Turn-based cascade (VAD + STT + LLM + TTS baseline) | WebRTC VAD endpointing, Parakeet TDT 0.6B v3, `gpt-4o-mini`, MiMo TTS |

Moshi, PersonaPlex, and FreezeOmni were self-hosted; revisions are abbreviated Hugging Face commit hashes. Recordings were made between 17 and 31 August 2026, except GPT-Live, which was recorded on 11 September 2026.

## Dimensions

The dimensions are reported in separate tables and are not combined into one overall rank.

1. **Spoken-task accuracy (reasoning).** Fifty spoken items per system across five scenarios: logic questions, countdown completions, grammar repair, keyword-wait interjections, and stay-quiet-until-asked. An item counts when the spoken response meets its scenario's target.

2. **Conversational timing.** Per-scenario behavior over live drills: backchanneling without grabbing the floor, waiting through a mid-sentence pause, prompt turn-taking latency, holding a thread through a user backchannel, and yielding-and-answering when interrupted.

3. **Paralinguistic control.** Whisper on request, scored from the audio (harmonics-to-noise ratio below 6 dB and voiced fraction below 0.4), and volume understanding (an appropriate answer to very soft or very loud speech, and whether the system lowers its own voice when whispered to).

4. **Interaction groundedness (long multi-turn logic).** Ten conversations of 22–24 turns build up state, such as a grocery list or a road trip. Probes then check whether the system recalls what it was told, tracks updates, says when information was never given, keeps constraints, attributes statements to the right speaker, rejects false premises, and flags nonsense questions. An LLM judge grades each probe from a time-aligned transcript and gives the conversation a 0–100 coherence score.

## Repository layout

```
headless_run.py            # run a system over a scenario (drives the model, records both tracks, ASR + grade)
grade_*.py                 # per-dimension graders (grade_interaction.py = interaction groundedness)
build_site_data.py         # aggregate results → docs/data/manifest.json + docs/audio/*.mp3
build_standalone.py        # bundle the site into a single self-contained HTML
gen_*.py                   # scenario / TTS generators per dimension
acoustics.py, compact_gaps.py, halfduplex.py, opus_encode.js …
dashboard/                 # local live dashboard (real-time human ↔ model, relays to each system)
interaction_groundedness/  # scenarios + generator + spec for the groundedness dimension
docs/                      # the public GitHub Pages site (index.html + data/ + audio/)
test_set/, tts_review/     # benchmark recordings & runs  (large, git-ignored, kept locally)
```

## Setup & run

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create the environment
cp .env.example .env                      # then fill in your API keys
```

`moshi` / `personaplex` / `freezeomni` require their model servers to be running (see the [Live dashboard](#live-dashboard)); `gpt` / `gemini` use their hosted APIs; `halfduplex` runs locally except for its language-model call (OpenAI `gpt-4o-mini`).

### 1. Run a system over a scenario

`headless_run.py` drives the model, records both tracks (`A_user.wav` / `B_model.wav` / `combined.wav`), runs local word-level ASR (parakeet), and writes a first-pass `grade.json`:

```bash
# reasoning / restraint / interaction items (multi-turn from test_set/<cat>/<NN>/{A1,A2,…}.wav)
uv run python headless_run.py --model gemini --task benchmark \
  --input test_set/interaction_groundedness/01/A1.wav --gemini

# a timing drill (behavioral)
uv run python headless_run.py --model gemini --task pause \
  --input tts_review/pause/01.wav --gemini
```

### 2. Run the graders

Each dimension has its own grader; they (re)score from the recorded tracks + ASR. `--gemini` enables the audio/content judge where a dimension needs it.

| dimension / category | grader |
|---|---|
| **Timing** (backchannel · pause · turn_taking · user_backchannel · interruption) | `uv run python grade.py <run-or-folder> --task <cat> --gemini` |
| **Reasoning**: logic_puzzle · countdown_completion · grammar | `uv run python regrade_iq.py` |
| **Reasoning**: keyword_wait | `uv run python regrade_iq.py keyword_wait` (judges the reply content), then `uv run python regrade_keyword_wait.py [models…]` (applies the pass rule) |
| **Reasoning**: stay_quiet_until_help | `uv run python regrade_stay_quiet.py [models…]` |
| **Turn-discipline**: alternating_count | `uv run python grade_alternating.py [models…]` |
| **Paralinguistic**: whisper_production · volume_understanding | `uv run python grade_acoustic.py [models…]` |
| **Interaction groundedness** | `uv run python grade_interaction.py --model all` |

`grade.py --list` shows available dashboard runs; omit `[models…]` to grade all seven systems. Every grader writes the standard `grade.json` (`summary.{n,npass,rate}` + per-event detail) that the site reads.

Then aggregate everything into the site data:

```bash
uv run python build_site_data.py          # → docs/data/manifest.json + docs/audio/*.mp3
```

### 3. Preview the site

```bash
cd docs && python3 -m http.server 8791    # then open http://localhost:8791
```

## Generating scenarios & ground truth

The standard answer for every item is a `benchmark.json` holding the graded target: the right value or phrase, the trigger word, or the probe set. It is generated from a written dialogue and is not copied from any system's output. Per category:

| category | generator | how the ground truth is made |
|---|---|---|
| timing (5 scenarios) | `dashboard/test_prompts.md`, voiced on the dashboard's `/tts` page (MiMo TTS) | scripted user lines with no reference answer; the graders score the behavior from the two recorded tracks |
| `keyword_wait` | `gen_keyword_wait.py` | scripted dialogue → `gen_benchmark` for the answer → MiMo TTS `A1/A2.wav` |
| `stay_quiet_until_help` | `gen_stay_quiet.py` | same pattern (user thinks aloud, then asks for help) |
| `grammar_correction` | `gen_grammar_correction.py` | re-TTS the grammar items in non-native accents; carries the existing standard answer |
| `whisper_production` · `volume_understanding` | `gen_volume_whisper.py` | synthesizes the soft/loud/whisper-request stimulus + `benchmark.json` |
| `interaction_groundedness` | `interaction_groundedness/gen_interaction.py` → `gen_interaction_tts.py` | Gemini generates 20+-turn scenarios with `ground_truth` + probes (hard-validated in code); TTS voices only the user turns and writes `benchmark.json` |

All TTS goes through MiMo (`generate_tts_mimo.py`).

## Live dashboard

A local web app for **real-time** human ↔ model conversation and side-by-side comparison, with relays to each system (GPT / Gemini via API; Moshi / PersonaPlex over an Opus WebSocket; FreezeOmni over socket.io):

```bash
uv run --extra dashboard python dashboard/server.py    # then open the printed URL
```

`moshi` / `personaplex` / `freezeomni` need their model servers up first; the dashboard connects to them per `dashboard/`’s config.

## Data

The raw recordings (`test_set/`, `tts_review/`, `dashboard/runs/`, and the per-dimension audio) are large and not committed. They stay local and can be regenerated with the `gen_*` scripts and `headless_run.py`. The example clips shown on the public site live in `docs/audio/` and are committed.

📦 **Full test dataset:** the complete benchmark recordings are available on [Google Drive](https://drive.google.com/drive/folders/1C7huPcVMAiF8GFN6cg76in0dN8yYWk_w?usp=sharing). Download and unpack them into `test_set/` (and `tts_review/`) to reproduce the graded results without re-running the generators.

## Acknowledgements

InteractionBench builds on: **NVIDIA NeMo Parakeet** (via `parakeet-mlx` on Apple Silicon) for local word-level ASR, **MiMo** for TTS, **Gemini 2.5 Flash** as the judge, and **OpenAI** `gpt-4o-mini` as the cascade's language model.

## References

The design of InteractionBench's conversational-timing and turn-taking dimensions draws on the task formulation of **Full-Duplex-Bench**:

```bibtex
@article{lin2025fullduplexbench,
  title={{Full-Duplex-Bench}: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H. and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025},
  url={https://arxiv.org/abs/2503.04721}
}
```

## License

InteractionBench is released under the [Business Source License 1.1](LICENSE). Non-production use is permitted, and the Additional Use Grant permits non-commercial academic research, education, evaluation, testing, and benchmarking in production. Any use primarily intended for commercial advantage requires a separate commercial license from AveraLabs. Each version converts to the MIT License four years after its first public distribution under the Business Source License 1.1.
