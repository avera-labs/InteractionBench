# benchmark.json format guide (full-duplex timing benchmark)

`benchmark.json` is the **ground truth** for each **full-duplex timing item** — it specifies "after the trigger fires, **when** and with **what content** the voice model under test should respond", and is used to score the model's produced speech. Each variant (A/B) of each scenario has its own copy.

> The ground truth always comes from the **card** (a spec written up-front by a human / upstream) + deterministic computation (arithmetic for numbers), and **never** from a generative model's output — otherwise the "ground truth" would be just as wrong as the model under test.

---

## 1. Top-level structure

```jsonc
{
  "prompt_key": "interaction benchmark data",   // which prompt generated it
  "scenario_id": "0f0d2f6c-...",                // which scenario card it maps to
  "variant": "A",                                // A / B, two independent variants
  "prompt_version": 2,
  "category": "countdown_completion",            // one of the 8 categories (see appendix)
  "floor_control_action_space": [                // floor-control action vocabulary (aligned with InteractionBench)
    "WAIT","BACKCHANNEL","TAKE_FLOOR","INTERRUPT","CONTINUE","YIELD","RESUME","ABORT"
  ],
  "n_graded_events": 1,                          // number of graded events (count of graded=true)
  "expected_events": [ ... ],                    // ★ core: one entry per event
  "note": "..."                                  // provenance note
}
```

---

## 2. `expected_events` — the event array

Every "anchored response" by the AI is one event. Most items have just 1 graded event; ones like `keyword_wait` / `forbidden_phrase` that "chat through a few distractor words first, then catch the real word" have multiple events (the response to a distractor word is `graded:false`).

### Fields of each event

| Field | Meaning |
|---|---|
| `event_id` | `e1`, `e2`… |
| **`graded`** | `true`=graded (the moment this item really tests); `false`=an ordinary response / chatter to a distractor word, **not scored** |
| **`grade_dimension`** | `timing`=test timing+overlap; `content`=test whether the produced content is correct |
| `expected_action` | what the AI should do: `TAKE_FLOOR` (wait for the other party to finish, then respond) / `INTERRUPT` (cut in, overlap) / `SELF_COMPLETE` (the AI says the whole passage itself) |
| `onset_window_ms` | acceptable onset-delay window `[lower, upper]` (ms); `null` for `SELF_COMPLETE` |
| `onset_reference` | where the delay is measured from: `trigger_word_end` (the instant the trigger word finishes) |
| `expected_overlap` | `required` (should overlap = interrupt) / `forbidden` (should not overlap = wait until finished) / `null` |
| `trigger` | `{type, anchor_word, ov_reported}` — the trigger word + its position in the user's sentence (`ov` 0=sentence start, 1=sentence end); `null` for `SELF_COMPLETE` |
| **`target_response`** | what the AI should say (ground truth, from the card) |
| `generated_text` | what the AI actually said in our **reference track** (reference / debug only, **not** the ground truth) |
| `response_line_index` | index of the AI's response line in the timeline |
| content types also carry | `expected_completion` / `expected_value` / `expected_precount` |

---

## 3. Three event types, and what each tests

### Type ① timing event `grade_dimension: "timing"`

**Categories**: `forbidden_phrase_interruption` / `deliberate_overlap` / `keyword_wait` / `grammar_correction` / `wrong_word_correction`
**What it tests**: the user says the trigger word, and the AI must react at the **right moment** with the **right content**.

**Example** (`keyword_wait`, one item with two events — distractor word first, then the real word):

```jsonc
"expected_events": [
  {                                    // e1: ordinary chatter about the distractor word "guitar" — not graded
    "event_id": "e1", "graded": false, "grade_dimension": "timing",
    "expected_action": "INTERRUPT", "expected_overlap": "required",
    "onset_window_ms": [50, 300], "onset_reference": "trigger_word_end",
    "trigger": { "type": "exact_word", "anchor_word": "guitar", "ov_reported": 0.67 },
    "target_response": "Cello heard.",
    "generated_text": "Guitar, nice. What are you learning?",   // the AI is just chatting, didn't call out
    "response_line_index": 5
  },
  {                                    // e2: the real keyword "cello" — graded
    "event_id": "e2", "graded": true, "grade_dimension": "timing",
    "expected_action": "INTERRUPT", "expected_overlap": "required",
    "onset_window_ms": [50, 300], "onset_reference": "trigger_word_end",
    "trigger": { "type": "exact_word", "anchor_word": "cello", "ov_reported": 0.77 },
    "target_response": "Cello heard.",
    "generated_text": "Cello heard.",
    "response_line_index": 9
  }
]
```

**How to read it**: the user chats normally, occasionally dropping city / instrument names (distractor words, the AI should ignore them); when the real keyword `cello` comes up (mid-sentence, ov≈0.77), the AI must **cut in** (INTERRUPT/required) and call **"Cello heard."** within **50–300ms** after `cello` finishes.
**How it's judged**: ① timing (delay within the window) ② overlap (should overlap) ③ content (said it right or not) ④ silence (no false start on a distractor word before the real word).

---

### Type ② content event (with a trigger cue) `grade_dimension: "content"` and non-empty `onset_window_ms`

**Categories**: `predictive_number_continuation` / `phrase_completion` (the user gives the first half, the AI continues the second)
**What it tests**: the user throws out a cue (a number sequence / the first half of a saying), and the AI must produce the **correct continuation**.

**Example** (`predictive_number_continuation`):

```jsonc
{
  "event_id": "e1", "graded": true, "grade_dimension": "content",
  "expected_action": "TAKE_FLOOR", "expected_overlap": "forbidden",
  "onset_window_ms": [50, 350], "onset_reference": "trigger_word_end",
  "trigger": { "type": "exact_word", "anchor_word": "thirty-five", "ov_reported": 0.95 },
  "target_response": "One hundred thirty-six.",
  "generated_text": "One hundred thirty-six.",
  "response_line_index": 3,
  "expected_completion": "One hundred thirty-six.",
  "expected_value": 136                         // ★ true value from arithmetic (don't trust the generation; double-check the card too)
}
```

**How to read it**: the user counts up to `...thirty-five`, and the AI waits for it to finish (TAKE_FLOOR/forbidden) then delivers the **correct next number, 136**.
**How it's judged**: **mainly ③ content** (did it say `136` correctly, per `expected_value`); because there's a cross-speaker cue, ①② timing/overlap are measured incidentally too.

---

### Type ③ content event `SELF_COMPLETE` (no timing)

**Categories**: `countdown_completion` (the AI says the whole countdown + completion word itself; there's no "user trigger → AI responds")
**What it tests**: the AI produces the whole passage on its own — counts down in order, then delivers the right completion word.

**Example** (`countdown_completion`):

```jsonc
{
  "event_id": "e1", "graded": true, "grade_dimension": "content",
  "expected_action": "SELF_COMPLETE",
  "onset_window_ms": null, "onset_reference": null, "expected_overlap": null,
  "trigger": null,                              // no cross-speaker trigger → timing not tested
  "target_response": "Happy New Year!",
  "generated_text": "Happy New Year!",
  "response_line_index": 8,
  "expected_completion": "Happy New Year!",
  "expected_precount": "descending_to_one"      // ★ a descending count to one must precede the completion word
}
```

**How to read it**: the AI must **count down to one**, then deliver the right completion word **"Happy New Year!"**. There's no onset/trigger (timing not tested).
**How it's judged**: **only ③ content** — a judge (Gemini) listens to the whole AI audio and verifies both the "countdown sequence + completion word" are correct.

---

## 4. `graded: true` vs `false`

In one item the AI may have **multiple** anchored responses; only the `graded:true` ones are scored:
- `graded:true` = the one whose anchor == the declared trigger word (or a content hit), **graded**
- `graded:false` = an ordinary response / chatter to a **distractor word**, kept only as context (in real evaluation it can be used to check "did the model false-trigger on a distractor word")

In the `keyword_wait` example above, `e1` (guitar) is a distractor and not scored; `e2` (cello) is the real word and is graded.

---

## 5. The four scoring dimensions (aligned with InteractionBench)

| Dimension | What it tests | How it's computed |
|---|---|---|
| **① timing** | is the onset delay within the window | `response onset − trigger end` ∈ `onset_window_ms` (deterministic, ASR + time arithmetic) |
| **② overlap** | should it barge in / wait — correct? | `required`→there is overlap; `forbidden`→no overlap (deterministic) |
| **③ content** | is the content correct | against `target_response`/`expected_value`/`expected_completion`; short words / phrases / semantics → Gemini judges by **listening to the audio** |
| **④ silence** | no false start before the trigger | before the trigger word finishes, no onset belonging to this response should appear (game / wait categories only) |

**Event PASS ⟺ timing ✓ AND overlap ✓ AND content ✓ AND (if applicable) silence ✓.** The report keeps the dimensions separate, not blended into one score — a content error like "said go when it should say stop" must not be averaged away by the timing score.

Timing / overlap / silence are millisecond-level objective facts, **computed in code, never handed to Gemini**; only content correctness, which code can't judge, goes to Gemini.

---

## Appendix: the 8 categories

| Category | What the user does | What the AI should say (target) | Dimension | Event type |
|---|---|---|---|---|
| `countdown_completion` | count down and pop the given word on the last beat | the given word ("Now"/"Stop"/"Happy New Year!") | content (+timing) | ③ or ① |
| `predictive_number_continuation` | recite a number sequence, then stop | the correct next number (11,12→**13**) | content | ② |
| `phrase_completion` | give the first half of a saying | the correct completion ("Silence is"→**golden**) | content | ② |
| `forbidden_phrase_interruption` | cut in as soon as the forbidden word is spoken | the given phrase ("Caught it.") | timing | ① |
| `deliberate_overlap` | barge in on a certain word | the given reply ("Being dramatic.") | timing | ① |
| `keyword_wait` | call out the moment the keyword appears, chatting normally before | the call-out keyword ("Cello heard.") | timing | ① (includes distractor events) |
| `grammar_correction` | correct a grammar mistake immediately | the correct fix ("He has been...") | timing | ① |
| `wrong_word_correction` | correct a wrong word immediately | the correct word ("medal"≠"meddle") | timing | ① |

---

## One-sentence summary

Only `graded:true` entries in `expected_events` are scored; `grade_dimension` decides whether **timing** or **content** is tested; `target_response`/`expected_*` are always the **ground truth** from the card, while `generated_text` is just the reference track's actual output.
