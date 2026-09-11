# InteractionBench

**Measuring reasoning, conversational timing, and paralinguistic control in real-time voice systems.**

InteractionBench evaluates full-duplex spoken-dialogue models on what actually makes a voice assistant feel present in a conversation — not just *what* it answers, but *when* it speaks, *how* it sounds, and whether it stays *grounded* across a long exchange. The same recordings are played to every system, so any difference in the result comes from the system.

🔊 **Live site:** https://avera-labs.github.io/InteractionBench/

## Systems evaluated

| id | system |
|---|---|
| `gptlive` | GPT-Live-1 |
| `gpt` | GPT-Realtime |
| `gemini` | Gemini-Live |
| `moshi` | Moshi (Kyutai) |
| `personaplex` | PersonaPlex |
| `freezeomni` | FreezeOmni |
| `halfduplex` | Turn-based cascade (VAD + STT + LLM + TTS baseline) |

## Dimensions

The dimensions are reported in separate tables and are **not** collapsed into one overall rank — each measures something distinct.

1. **Spoken-task accuracy (reasoning).** Fifty spoken items per system across five scenarios: logic questions, countdown completions, grammar repair, keyword-wait interjections, and stay-quiet-until-asked. An item counts when the spoken response meets its scenario's target.

2. **Conversational timing.** Per-scenario behaviour over live drills: backchannelling without grabbing the floor, waiting through a mid-sentence pause, prompt turn-taking latency, holding a thread through a user backchannel, and yielding-and-answering when interrupted.

3. **Paralinguistic control.** Whisper-on-request (graded acoustically — real breathy whisper, HNR < 6 dB, not the model merely *saying* it will whisper) and volume understanding (answering appropriately to very soft / very loud speech, and adapting its own volume).

4. **Interaction groundedness (long multi-turn logic).** Ten 20+-turn conversations quietly build real state — a list, a plan, a game — then plant **probes** that test whether the system stays grounded: recalling a value it was told, tracking an update, admitting what it was *not* told instead of inventing it, keeping a constraint, attributing who-said-what, refusing a false premise, and **flagging a nonsense question** rather than fabricating an answer. Each probe is graded from the transcript of what the system actually said (an LLM judge over a time-aligned reconstruction), plus a 0–100 coherence score for the whole conversation.

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
test_set/, tts_review/     # benchmark recordings & runs  (large — git-ignored, kept locally)
```

## Setup & run

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create the environment
cp .env.example .env                      # then fill in your API keys
```

`moshi` / `personaplex` / `freezeomni` require their model servers to be running (see the [Live dashboard](#live-dashboard)); `gpt` / `gemini` use their hosted APIs; `halfduplex` runs fully locally.

### 1. Run a system over a scenario

`headless_run.py` drives the model, records both tracks (`A_user.wav` / `B_model.wav` / `combined.wav`), runs local word-level ASR (parakeet), and writes a first-pass `grade.json`:

```bash
# reasoning / restraint / interaction items (multi-turn from test_set/<cat>/<NN>/{A1,A2,…}.wav)
uv run python headless_run.py --model gemini --task benchmark \
  --input test_set/interaction_groundedness/01/A1.wav --gemini

# a timing drill (behavioural)
uv run python headless_run.py --model gemini --task pause \
  --input tts_review/pause/01.wav --gemini
```

### 2. Run the graders

Each dimension has its own grader; they (re)score from the recorded tracks + ASR. `--gemini` enables the audio/content judge where a dimension needs it.

| dimension / category | grader |
|---|---|
| **Timing** (backchannel · pause · turn_taking · user_backchannel · interruption) | `uv run python grade.py <run-or-folder> --task <cat> --gemini` |
| **Reasoning** — logic_puzzle · countdown_completion · grammar | `uv run python regrade_iq.py` |
| **Reasoning** — keyword_wait | `uv run python regrade_keyword_wait.py [models…]` |
| **Reasoning** — stay_quiet_until_help | `uv run python regrade_stay_quiet.py [models…]` |
| **Turn-discipline** — alternating_count | `uv run python grade_alternating.py [models…]` |
| **Paralinguistic** — whisper_production · volume_understanding | `uv run python grade_acoustic.py [models…]` |
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

The **standard answer** for every item is a `benchmark.json` (the graded target — the right value/phrase, the trigger word, or the probe set), always produced by an LLM from a written dialogue and **never** hand-copied from a model's output. Per category:

| category | generator | how the ground truth is made |
|---|---|---|
| timing (5, behavioural) | `dashboard/gen_benchmark.py` | LLM turns a bare A/B dialogue into `benchmark.json` — one graded event anchored on a trigger word |
| `keyword_wait` | `gen_keyword_wait.py` | scripted dialogue → `gen_benchmark` for the answer → MiMo TTS `A1/A2.wav` |
| `stay_quiet_until_help` | `gen_stay_quiet.py` | same pattern (user thinks aloud, then asks for help) |
| `grammar_correction` | `gen_grammar_correction.py` | re-TTS the grammar items in non-native accents; carries the existing standard answer |
| `whisper_production` · `volume_understanding` | `gen_volume_whisper.py` | synthesises the soft/loud/whisper-request stimulus + `benchmark.json` |
| `interaction_groundedness` | `interaction_groundedness/gen_interaction.py` → `gen_interaction_tts.py` | Gemini generates 20+-turn scenarios with `ground_truth` + probes (hard-validated in code); TTS voices only the user turns and writes `benchmark.json` |

All TTS goes through MiMo (`generate_tts_mimo.py`).

## Live dashboard

A local web app for **real-time** human ↔ model conversation and side-by-side comparison, with relays to each system (GPT / Gemini via API; Moshi / PersonaPlex over an Opus WebSocket; FreezeOmni over socket.io):

```bash
uv run --extra dashboard python dashboard/server.py    # then open the printed URL
```

`moshi` / `personaplex` / `freezeomni` need their model servers up first; the dashboard connects to them per `dashboard/`’s config.

## Data

The raw recordings (`test_set/`, `tts_review/`, `dashboard/runs/`, and the per-dimension audio) are large and **not** committed — they stay local and are regenerated by the `gen_*` scripts and `headless_run.py`. The curated example clips shown on the public site live in `docs/audio/` and *are* committed.

📦 **Full test dataset:** the complete benchmark recordings are available on [Google Drive](https://drive.google.com/drive/folders/1C7huPcVMAiF8GFN6cg76in0dN8yYWk_w?usp=sharing). Download and unpack them into `test_set/` (and `tts_review/`) to reproduce the graded results without re-running the generators.

## Acknowledgements

InteractionBench builds on: **NVIDIA NeMo Parakeet** (via `parakeet-mlx` on Apple Silicon) for local word-level ASR, **MiMo** for TTS, and **Gemini** / **OpenAI** as content judges.

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
