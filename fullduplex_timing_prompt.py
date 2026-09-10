"""Full-duplex "timing drill" Stage-1 system prompt + seeds.

A separate content line from "full duplex choreography" (natural chit-chat), but it REUSES THE SAME pipeline:
the same LINE_OBJ choreography schema (anchor/nth/ref/ov/cut/gap...), the same fullduplex_prompt.generate()
(two independent calls produce variant_a/variant_b), the same Stage-2 TTS + Stage-3 ASR/anchor/assembly.

Only the system prompt differs: here the model is taught to produce timing-training drills with an
"explicit instruction + trigger word + precise timed response", not chit-chat. The trigger word = anchor,
the reuse point of the whole mechanism. The system prompt lives in the DB (prompt_templates) and the worker
reads it fresh from there; this file is the committable source of truth, written into the DB verbatim on INSERT.

Start with a few categories (H/D/I), get them working and judge the audio before expanding.
"""
from __future__ import annotations

import re

PROMPT_KEY = "full duplex timing"

# ── benchmark alignment ────────────────────────────────────────────────────────
# Each timing drill also produces benchmark metadata (benchmark.json) so the generated data can feed
# straight into that test harness's Track A (controlled playback). Most fields are derived deterministically
# from "seed category + generated choreography":
#   anchor = trigger word; cut/low ov = overlap (INTERRUPT); otherwise = takes the floor (TAKE_FLOOR).
#
# Default timing policy per category (onset window taken from the timing windows in spec §3).
#   (expected_action, expected_overlap, onset_window_ms, onset_reference)
CATEGORY_POLICY = {
    "forbidden_phrase_interruption": ("TAKE_FLOOR", "forbidden", [50, 300], "trigger_word_end"),
    "countdown_completion":          ("TAKE_FLOOR", "forbidden", [50, 300], "trigger_word_end"),
    "deliberate_overlap":            ("INTERRUPT",  "required",  [50, 250], "trigger_word_end"),
    "wrong_word_correction":         ("INTERRUPT",  "required",  [50, 300], "trigger_word_end"),
    "grammar_correction":            ("INTERRUPT",  "required",  [50, 300], "trigger_word_end"),
    "keyword_wait":                  ("INTERRUPT",  "required",  [50, 300], "trigger_word_end"),
    "predictive_number_continuation":("TAKE_FLOOR", "forbidden", [50, 350], "trigger_word_end"),
    "phrase_completion":             ("TAKE_FLOOR", "forbidden", [50, 350], "trigger_word_end"),
    "alternating_count":             ("TAKE_FLOOR", "forbidden", [100, 400], "trigger_word_end"),
    "stay_quiet_until_help":         ("TAKE_FLOOR", "forbidden", [200, 600], "trigger_word_end"),
}
DEFAULT_POLICY = ("TAKE_FLOOR", "forbidden", [50, 400], "trigger_word_end")

# the 8 floor-control actions from spec §3, carried along with the metadata (so the test harness can use them directly)
FLOOR_ACTIONS = ["WAIT", "BACKCHANNEL", "TAKE_FLOOR", "INTERRUPT",
                 "CONTINUE", "YIELD", "RESUME", "ABORT"]


# ── benchmark-data only: card distillation ────────────────────────────────────
# The timing training cards are written for 1:30+ multi-turn training data and explicitly "teach you to
# stretch the drill out":
#   Generation notes: "...reach at least a minute and a half... Do not jump straight to the count"
#   keyword_wait Trigger: "drop London, Paris, Tokyo, Sydney... the AI chats along the whole way"
#   Bad behavior: "lets the user hold the floor in a monologue"
# 'interaction benchmark data' (test/eval data, see memory) borrows these cards but wants a short single
# event, so before feeding the model we distill first: drop the 4 "stretch it out" fields, keep only the
# content fields, and prepend a hard "this is short benchmark data" directive.
# ⚠️ Only for benchmark data; timing training data eats the FULL card (does not go through here).
# ⚠️ Keep what the parser needs: the Category line (benchmark_metadata) and the quoted anchor word on the
#    Trigger line (_declared_anchor / correction_anchor) — so Trigger/Category are KEPT AS-IS, anchor untouched.
_DISTILL_KEEP = ("Scenario ID", "Category", "Instruction",
                 "Scene (what the user is talking about)", "Trigger", "Target AI response")
_DISTILL_DROP = ("Timing rule", "Good behavior", "Bad behavior", "Generation notes")

_BENCH_DISTILL_DIRECTIVE = (
    "★ THIS IS SHORT BENCHMARK TEST DATA. Its only job is to produce ONE clean, gradeable timing "
    "event — it does NOT need a rich, natural, or long conversation, and it must NOT try to fill any "
    "amount of time. State the rule, then get to the trigger as DIRECTLY as possible: at most ONE "
    "short line of setup — or, for keyword-wait / forbidden-phrase, just ONE or TWO quick near-miss "
    "distractors — then the trigger, then the short response. Ignore any urge to chat at length, "
    "build up a scene, or reach any duration. Shorter is better as long as the one graded event is clean."
)


def _split_card_fields(scenario_text: str) -> dict[str, str]:
    """Split the card into {label: value} by the known field labels (value spans multiple lines, up to the next label)."""
    labels = _DISTILL_KEEP + _DISTILL_DROP
    pat = re.compile(r"(?m)^(" + "|".join(re.escape(l) for l in labels) + r"):[ \t]*")
    ms = list(pat.finditer(scenario_text or ""))
    out: dict[str, str] = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(scenario_text)
        out[m.group(1)] = scenario_text[m.end():end].strip()
    return out


def distill_card(scenario_text: str) -> str:
    """benchmark-data only: timing training card → short single-event skeleton.

    Drop _DISTILL_DROP (Timing rule / Good·Bad behavior / Generation notes — the "stretch it out" signals
    all live here), keep only the _DISTILL_KEEP content fields (Category/Trigger as-is → anchor word and
    benchmark parsing unaffected), and prepend _BENCH_DISTILL_DIRECTIVE. If Category can't be parsed, return
    as-is (never risk breaking the card)."""
    fields = _split_card_fields(scenario_text)
    if "Category" not in fields:
        return scenario_text
    kept = [f"{label}: {fields[label]}" for label in _DISTILL_KEEP
            if fields.get(label)]
    return _BENCH_DISTILL_DIRECTIVE + "\n\n" + "\n".join(kept)


def _spk_is_ai(spk) -> bool:
    return str(spk) in ("B", "1")


def _declared_anchor(scenario_text: str) -> str | None:
    """The anchor word declared on the card's Trigger line. The 8 card types are worded differently, so two rules cover them all:

      ① a quoted word appearing AFTER "anchor" wins — covers
         `anchor on "celery"` / `anchor on the second word "bear"` /
         `anchor the AI completion on the last word "be"` (we want bear/be, not the whole phrase)
      ② otherwise take the first quoted word on the Trigger line — covers
         `the keyword "banana" — anchor on it` / `the word "obviously" — anchor on it`
    """
    m = re.search(r"^Trigger:\s*(.+)$", scenario_text or "", re.M)
    if not m:
        return None
    line = m.group(1)
    # ⓪ correction types (interrupt): the error word IS the anchor, tagged in the card as `The word to
    #    correct is "go".` (the error phrase `the error "I go to the store"` sits earlier on this line and
    #    must not steal it) — this rule matches first.
    only = re.search(r'word to correct is\s+"([^"]+)"', line, re.I)
    if only:
        return only.group(1).strip()
    # ★ require "anchor" to be followed by on/the/it — that is the "anchor on the second word 'X'"-style
    #   DECLARATION; otherwise it collides with an "anchor" inside a forbidden-word phrase (e.g. "polished
    #   anchor") → grabbing a garbage fragment between quotes.
    after = re.search(r'anchor\w*\s+(?:on|the|it)\b[^"]{0,80}?"([^"]+)"', line, re.I)
    if after:
        return after.group(1).strip()
    first = re.search(r'"([^"]+)"', line)
    return first.group(1).strip() if first else None


_CORRECTION_CATS = ("grammar_correction", "wrong_word_correction")


def correction_anchor(scenario_text: str) -> str | None:
    """The error word declared on correction-type (grammar / wrong_word) cards; returns None for other categories.

    Only correction types do "back-inferred interrupts" (fullduplex_repair._reconstruct_correction_interrupts) —
    other timing drills (countdown/keyword/phrase completion...) have different semantics and would mis-anchor if
    run by mistake. So gate by category."""
    m = re.search(r"^Category:\s*(\S+)", scenario_text or "", re.M | re.I)
    if not (m and m.group(1).strip().lower() in _CORRECTION_CATS):
        return None
    return _declared_anchor(scenario_text)


# ── content types (completion / continuation): ground truth comes from the card, grade "is the produced content correct" ──
# These are content questions at heart, not timing questions: did it continue the sequence right, complete
# the famous phrase right, supply the right completion word when the count reached one.
CONTENT_CATS = ("countdown_completion", "predictive_number_continuation", "phrase_completion")

_NUM_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
              "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19}
_NUM_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
             "seventy": 70, "eighty": 80, "ninety": 90}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _card_field(scenario_text: str, label: str) -> str | None:
    """Get the full-line value of a given card field (Category / Trigger / Target AI response ...)."""
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", scenario_text or "", re.M)
    return m.group(1).strip() if m else None


def _words_to_int(text: str):
    """English cardinal words → int (up to thousands, e.g. 'one hundred thirty-five'→135). Returns None on parse failure."""
    toks = [t for t in re.split(r"[\s-]+", (text or "").lower().strip()) if t and t != "and"]
    total = cur = 0
    seen = False
    for t in toks:
        if t in _NUM_UNITS:
            cur += _NUM_UNITS[t]; seen = True
        elif t in _NUM_TENS:
            cur += _NUM_TENS[t]; seen = True
        elif t == "hundred":
            cur = (cur or 1) * 100; seen = True
        elif t == "thousand":
            total += (cur or 1) * 1000; cur = 0; seen = True
        else:
            return None
    return (total + cur) if seen else None


def _predictive_next(scenario_text: str):
    """Deterministically compute the predictive next number: arithmetic extrapolation from the quoted sequence in the card's Trigger (trust neither the generation nor the card's answer)."""
    trig = _card_field(scenario_text, "Trigger") or ""
    q = re.search(r'"([^"]+)"', trig)
    if not q:
        return None
    nums = [v for part in re.split(r"[.,;]+", q.group(1))
            if (v := _words_to_int(part)) is not None]
    if len(nums) >= 2:
        return nums[-1] + (nums[-1] - nums[-2])
    if len(nums) == 1:
        return nums[-1] + 1
    return None


def _find_completion_line(lines: list[dict], card_target: str):
    """Find the AI line that produces the completion: the LAST AI line whose normalized text contains card_target (the completion word is usually at the end)."""
    tgt = _norm(card_target)
    if not tgt:
        return None
    hit = None
    for i, L in enumerate(lines):
        if _spk_is_ai(L.get("speaker")) and tgt in _norm(L.get("text") or ""):
            hit = i
    return hit


def benchmark_metadata(scenario_text: str, lines: list[dict]) -> dict:
    """Derive benchmark metadata from the seed spec (card) + the generated choreography.

    ★ Ground truth comes from the CARD (Target AI response field + arithmetic for number types), never from
      the generated line — L.get('text') is only recorded as generated_text (reference/debug). The generation
      model's output cannot be the ground truth.
    ★ Two grade_dimension values:
      · timing — the user says the trigger word, the AI anchors its response on it, grade onset latency/overlap
        (forbidden / deliberate / keyword / grammar / wrong_word, plus "user counts down"-style completion).
      · content — completion/continuation types (countdown / predictive / phrase), grade whether the AI's
        produced content is correct (right sequence continuation, right phrase completion, right completion word
        after the count reaches one); still graded when the AI says the whole thing itself (fallback event).
    """
    category = (_card_field(scenario_text, "Category") or "unknown").strip().lower()
    action, overlap, window, ref = CATEGORY_POLICY.get(category, DEFAULT_POLICY)
    declared = _declared_anchor(scenario_text)
    dl = (declared or "").strip().lower()
    card_target = _card_field(scenario_text, "Target AI response")   # ★ ground truth (card)
    is_corr = category in ("grammar_correction", "wrong_word_correction")
    is_content = category in CONTENT_CATS

    # content-type ground truth (all from the card / arithmetic, never from the generated line)
    content_gt = {}
    if is_content:
        content_gt["expected_completion"] = card_target
        if category == "predictive_number_continuation":
            nxt = _predictive_next(scenario_text)
            if nxt is not None:
                content_gt["expected_value"] = nxt              # arithmetic extrapolation, most reliable
        if category == "countdown_completion":
            content_gt["expected_precount"] = "descending_to_one"

    def _ref_text(idx: int, r) -> str:
        if isinstance(r, int) and 0 <= r < len(lines):
            return str(lines[r].get("text") or "")
        spk = lines[idx].get("speaker")
        for k in range(idx - 1, -1, -1):
            if lines[k].get("speaker") != spk:
                return str(lines[k].get("text") or "")
        return ""

    events, n = [], 0
    for i, L in enumerate(lines):
        if not (_spk_is_ai(L.get("speaker")) and L.get("anchor")):
            continue
        anchor = str(L.get("anchor") or "")
        graded = bool(dl) and (
            anchor.strip().lower() == dl
            or (is_corr and re.search(rf"\b{re.escape(dl)}\b", _ref_text(i, L.get("ref")), re.I) is not None))
        # content types: hitting the completion content also counts as graded (sidesteps the fragility of multi-word number/phrase anchor matching)
        if is_content and _norm(card_target) and _norm(card_target) in _norm(L.get("text") or ""):
            graded = True
        ov = L.get("ov")
        a, o = action, overlap
        if L.get("cut") or (isinstance(ov, (int, float)) and ov < 0.6):
            a, o = "INTERRUPT", "required"
        n += 1
        ev = {
            "event_id": f"e{n}",
            "graded": graded,
            "grade_dimension": "content" if is_content else "timing",
            "expected_action": a,
            "onset_window_ms": window,
            "onset_reference": ref,
            "expected_overlap": o,
            "trigger": {"type": "exact_phrase" if " " in anchor else "exact_word",
                        "anchor_word": anchor, "ov_reported": ov},
            "target_response": card_target,        # ★ ground truth (card, not the generated line)
            "generated_text": L.get("text"),       # the generated line, for reference/debug only
            "response_line_index": i,
            **content_gt,
        }
        events.append(ev)

    # content-type fallback: when there is no graded event (e.g. the AI says the count + completion word all
    # itself, no cross-speaker anchor), add a content event on the AI line that produced the completion (onset
    # timing does not apply → set null).
    if is_content and not any(e["graded"] for e in events):
        ci = _find_completion_line(lines, card_target)
        if ci is not None:
            n += 1
            events.append({
                "event_id": f"e{n}",
                "graded": True,
                "grade_dimension": "content",
                "expected_action": "SELF_COMPLETE",     # AI produces the whole thing itself, no onset timing to grade
                "onset_window_ms": None,
                "onset_reference": None,
                "expected_overlap": None,
                "trigger": None,
                "target_response": card_target,
                "generated_text": lines[ci].get("text"),
                "response_line_index": ci,
                **content_gt,
            })

    # timing-type fallback: when there is a fixed target (forbidden/deliberate/keyword...) but the model omitted
    # the anchor → no graded event, add a timing event on the AI line that produced the target; use the declared
    # anchor for the trigger word and leave the grader to locate its exact position structurally.
    if not is_content and dl and card_target and not any(e["graded"] for e in events):
        ci = _find_completion_line(lines, card_target)
        if ci is not None:
            n += 1
            events.append({
                "event_id": f"e{n}",
                "graded": True,
                "grade_dimension": "timing",
                "expected_action": action,
                "onset_window_ms": window,
                "onset_reference": ref,
                "expected_overlap": overlap,
                "trigger": {"type": "exact_phrase" if " " in dl else "exact_word",
                            "anchor_word": declared, "ov_reported": None,
                            "note": "model omitted anchor; trigger located structurally by grader"},
                "target_response": card_target,
                "generated_text": lines[ci].get("text"),
                "response_line_index": ci,
            })

    # ★ declared_trigger_anchor / card_target are no longer at the top level: they are "item-level" truth, but
    #   the graded events already carry them — target_response(=card_target), trigger.anchor_word,
    #   expected_completion/value/precount. Everything goes under expected_events, each graded event is
    #   self-contained; the top level keeps only item-level metadata (category / counts / action vocabulary).
    return {
        "category": category,
        "floor_control_action_space": FLOOR_ACTIONS,
        "n_graded_events": sum(1 for e in events if e["graded"]),
        "expected_events": events,
        "note": "Ground truth from CARD (Target AI response + arithmetic for numbers), NOT the generated "
                "line (kept as generated_text). Each graded event is self-contained: target_response / "
                "trigger.anchor_word / expected_completion|value|precount. grade_dimension: timing = onset "
                "vs trigger; content = correct produced content.",
    }

TIMING_SYSTEM_PROMPT = """You are a dialogue writer producing FULL-DUPLEX TIMING DRILLS for speech-training data. Speaker A = a HUMAN USER. Speaker B = an AI VOICE ASSISTANT. Everything is spoken via TTS — write natural SPOKEN English (never written prose). One exception, and it matters: when a card asks the user to make a deliberate mistake, you write that mistake exactly as given — see the DELIBERATE MISTAKE rule below.

A timing drill is a conversation of ABOUT TWO MINUTES — and AT LEAST ninety seconds — built around one or more precise GRADED MOMENTS: the user states an explicit rule out loud, the two of them actually talk for a real stretch, and at the graded moment a clear TRIGGER lands and the AI must respond at exactly the right instant. It has to run long enough that the AI is genuinely holding the rule over TIME; a short snippet tests nothing. Keep the talk PURPOSEFUL rather than aimless — the user really tells their story or practises, not empty topical padding — but purposeful does NOT mean short: fill the full ninety-plus seconds with real content. Because the rule is stated aloud, a listener can immediately tell whether the AI was early, late, correct, or wrongly silent. That obviousness is the entire point — do not bury it in natural-conversation flourishes.

## WHAT YOU PRODUCE
ONE fully choreographed timing drill, plus a `scene` summarising it in one line (the category, the trigger, the target response). You also CAST the two speakers.
- A = the HUMAN USER, a real person. Concrete voice demographics + a `tag`: one second-person sentence about who they are and how they talk.
- B = the AI VOICE ASSISTANT. A designed voice + a `tag`: clear, quick, minimal — it obeys the timing rule and says ONLY what it was told to say, nothing more.

## THE SCENE YOU ARE GIVEN IS A TIMING-TASK SPEC
It names a CATEGORY, the INSTRUCTION the user will give, a TRIGGER (a word or phrase), and the AI's TARGET RESPONSE. Expand it into a full spoken script in this shape:
1. OPEN WITH A NATURAL FRAMING, then the rule. Two people do not bark commands at each other — the
   user PROPOSES the activity the way a person actually would, in their own words, and only then
   states the rule. One to three lines, warm and casual:
     "Okay, let's play a little game."  /  "Can I try something with you?"  /
     "I want to test something. Humour me for a minute."  /  "Quick drill, are you up for it?"
   Then the rule itself, plainly and unmistakably: "…whenever I say 'white bear', jump in and say
   'Caught it.'" The framing is what makes it sound like two people doing an exercise together
   rather than a command being issued to a machine. The RULE must still be crystal clear — that is
   what makes the drill gradeable.

   ★ For LEARNER cards (grammar, wrong word) this is an ENGLISH LESSON, not a chat. The user is a
   NON-NATIVE ENGLISH LEARNER (not a native speaker playing a game) who opens by asking for real-time
   correction, e.g. "Okay, I'm learning English. Can you correct my grammar as we speak? Jump in the
   moment I make a mistake, don't wait." Keep the WHOLE drill FOCUSED on exactly that: the learner
   practises English out loud and the AI is an attentive TUTOR — the learner does most of the talking,
   the AI mostly listens and gives short, warm responses ("Mmhm.", "Yeah?", "Nice."). Do NOT turn it
   into an ordinary two-way conversation that wanders around the scene topic and waffles; the scene is
   only light material for the learner to practise ON, never a subject to discuss at length. The
   back-and-forth exists purely to fill the roughly two minutes, so keep it a lesson the whole way.
   Plant exactly ONE scripted mistake — it falls out naturally as a real learner's slip, written
   VERBATIM (see DELIBERATE MISTAKE) — and the AI corrects it ONCE. ONE error is enough; do NOT
   sprinkle several. Cast speaker A with the ACCENT named in the card (heavy Chinese / French / etc.)
   and let it show, so it sounds like an actual learner. The AI's correction is an INTERRUPTION: it
   cuts in the instant it hears the error, overlapping the learner's ongoing sentence — it does NOT
   wait for them to finish.

2. The AI may react naturally to the proposal — brief and warm ("Sure." / "Go for it." / "Got it.")
   — or say nothing. It must NEVER restate or paraphrase the rule back (no "I'll say X whenever you
   say Y"); repeating the rule spoils the test and wastes time.
3. The two of them exchange a LITTLE natural talk around the scene — a few tight lines, the AI participating normally unless the instruction forbids it (see LENGTH below). Do NOT settle into a long topical discussion. The trigger must not arrive in the very first lines, but get to it without a long wind-up. (EXCEPTION: grammar / wrong-word LEARNER cards — see the learner rule above. There it is an English LESSON: the learner practises and the AI tutors, NOT an equal two-way chat about the scene.)
4. The user delivers the TRIGGER inside a natural sentence.
5. The AI gives the SHORT target response, its line `anchor`ed to the trigger word.
6. For "whenever / every time" rules: keep talking and land the trigger again, twice or three times total, spaced well apart.

## THE REAL TRIGGER MUST BE EXACT — this is where drills most often break
The real trigger must appear CONTIGUOUSLY and VERBATIM, spelled EXACTLY as the anchor word — singular "bear", not "bears"; "obviously", not "obvious". The AI line's `anchor` must be a word that is literally present in that user line.
- DISTRACTORS must be genuinely DIFFERENT words — "polar bear", "white rabbit", "whiteboard" — NEVER a plural or near-variant of the trigger ("white bears", "bearing"), which would either false-match or fail the anchor. A distractor that shares the trigger's exact word defeats the drill.
- Make the real trigger the CLEANEST occurrence: the exact phrase, in a plain sentence, not buried in a subordinate clause.

## THE TRIGGER IS THE ANCHOR — the core mechanic
Audio is generated per line, then transcribed to get the timestamp of every word. A line with an `anchor` begins the instant that anchor word FINISHES in the OTHER speaker's line (plus ~0.25s reaction). Therefore:

★ THE ANCHOR GOES ON THE AI'S RESPONSE LINE (speaker B's short line), and points BACK to the trigger word in the user's line that comes right before it. NEVER put the anchor on the user's own trigger line — the user's lines carry NO anchor; they just speak. Only the AI's response line is anchored. Getting this backwards is the #1 failure.

- Set the AI RESPONSE line's `anchor` to the TRIGGER word, copied VERBATIM from the preceding user line, spelled exactly as it appears there.
- TWO-WORD trigger ("white bear", "red flag"): anchor on the SECOND word ("bear", "flag") — the AI must wait for the phrase to COMPLETE, never fire on the first word.
- RESPOND-AFTER categories (forbidden word, countdown): END the user's trigger line ON the trigger word (or within a word or two of it) — do NOT let the sentence ramble on after the trigger. Then the AI response line anchors on that trigger word (ov near 1.0) and lands cleanly right after.
- OVERLAP / INTERRUPT categories (deliberate overlap "obviously..." → "Is it though?", keyword-wait, AND grammar / wrong-word correction): the user's trigger line KEEPS GOING for several words after the trigger; the AI response line anchors on the trigger word (ov in the middle) and cuts in while the user is still talking. Set `cut: true` on that AI line only if five+ words still follow the trigger. For corrections the trigger word IS the error word — the learner keeps talking straight past their own slip, and the AI cuts in on it. For keyword-wait the keyword lands MID-sentence (never at the end) with the user still talking, and the AI jumps in on it the instant it is spoken.

## ★ WHEN THE CARD ASKS FOR A DELIBERATE MISTAKE, WRITE THE MISTAKE
Some cards (grammar correction, wrong-word correction) require the USER to speak something that is
WRONG on purpose — "Yesterday I go to the store and...", "it sounds too celery to me...". Your
instinct to write clean, natural English is EXACTLY WRONG here, and it is the single biggest way
these drills fail:

- COPY the card's error phrase VERBATIM into the user's line, INLINE inside a normal sentence. Do not
  fix it, do not soften it, do not add "I mean" afterwards, and do NOT put it in quotation marks. The
  learner says the wrong thing and KEEPS TALKING for several more words in the SAME utterance as if
  nothing happened — that continuation is what the AI talks over. Never break the error onto its own line.
- The error must be the ONLY wrong thing in that line. The rest of the sentence is clean, so the
  correction is unambiguous.
- Every OTHER line in the drill uses normal correct English. One planted error (or two-three if the
  instruction says "whenever"), never a generally sloppy speaker.
- THE CORRECTION IS ONE natural spoken line — exactly what the card's Target shows — that states the
  CORRECTED form: "You went to the store." / "You were excited." / "I think you mean serious." A beat
  after the mistake (natural reaction time, not the exact instant) the AI cuts in over the learner's
  still-running sentence with this one line. It is the AI's next line, right after the learner's error
  utterance.
- ★ The AI NEVER repeats the wrong word and NEVER uses quotation marks. Do not say "not 'go'", do not
  say "'went', not 'go'", do not wrap any word in quotes. Just say the corrected version plainly — the
  malformed word ("falled", "goed") spoken in isolation or inside quotes comes out mangled by the
  speech synthesiser; a natural corrected sentence does not. Say "You fell asleep.", never "not 'falled'".
- Because the AI comes in a beat late, it MAY echo the corrected phrase (e.g. "fell asleep") even
  though part of it ("asleep") sits after the error — by then the learner has already said it, so it
  is not predicting the future. Keep it to one short sentence; do not lecture on grammar.
- Anchor this correction line back on the wrong word in the learner's utterance (the code refines the
  exact instant); set cut: true if several words follow.
This overrides the "write natural English" instruction below for the one planted line ONLY.

## ★ WHEN THE CARD IS A COMPLETION, THE USER TRAILS OFF AND STOPS
Some cards (famous-phrase completion, number continuation) require the USER to begin something
predictable and STOP, leaving the AI to supply the missing piece:
- End the user's line ON the cue word, mid-thought — "May the force be…", "Ninety-eight.
  Ninety-nine." — then STOP. Do not let the user finish it themselves, and do not have them ask
  "what comes next?". The gap is the question.
- The AI's line is ONLY the missing piece — "With you.", "One hundred." — nothing else.
- Anchor the AI's line on the LAST word of the cue.

## TTS CONSTRAINTS ON TRIGGERS — the audio must actually contain a usable trigger
- Triggers must be SPOKEN WORDS that ASR will clearly transcribe. NEVER use a laugh, cough, sigh, clap, gasp, or any non-verbal sound as the trigger — the TTS is unreliable at those. If a category is "about" a cough, use the spoken word "cough", not a cough sound.
- NEVER sing. Nursery rhymes and quotes are SPOKEN plainly, not sung.
- The trigger must be a clear CONTENT word the ASR will not drop or mangle.

## HARD RULES — machine-checked before any audio is made. A violation is a failure.
1. `anchor` MUST be a word that appears VERBATIM in the referenced (user) line's text. Copy it exactly. Never invent or paraphrase. A hyphenated word is fine — copy the whole thing.
2. `anchor` MUST be a CONTENT word (noun/verb/adjective/name/number), NEVER a function word ("the", "so", "it's", "and", "you").
3. For RESPOND-AFTER drills the anchor is the trigger word and should be the user line's LAST content word or very close to it (the reply comes right after). For OVERLAP drills the anchor must NOT be the last word — real speech must follow it.
4. `nth`: leave null unless the anchor word appears MORE THAN ONCE in that ONE line. (Nth-occurrence categories: the AI line anchors the correct occurrence — set `nth` to it.)
5. `ref`: the AI response references the user line it reacts to (leave null = most recent other speaker, which is correct here).
6. `gap`: set null whenever `anchor` is set. Use a small `gap` (0.1-0.4) only for lines with no anchor (e.g. the user's filler lines, or a plain reply after the user fully stops).
7. `bc`, `cut`, `resume`, `flow`: leave false/off unless the category explicitly needs them. Timing drills are NOT natural chats — do not sprinkle backchannels, resume detours, or decorative interruptions.
8. `ov` is your honest estimate of how far through the user line the anchor word sits (0=start, 1=end). Respond-after → ov near 1.0; overlap → ov in the middle. It is cross-checked against the real position.

## LENGTH — AT LEAST 1:30, aim TWO MINUTES
The drill MUST run at least a minute and a half — ninety seconds — and about two minutes is the target: roughly 250 to 320 words of actual speech at 2.6 words per second. A drill under 90 seconds is TOO SHORT and does not count. If your draft is under ~230 words, it is NOT done — keep going: the user carries on telling their story in full, two-or-three-sentence turns. Up to 2:30 is fine; never past three minutes. Most user turns are TWO OR THREE real sentences, the way a person actually tells a story — NOT one-word lines. Count the words; if you are short, the answer is fuller turns, not more one-liners.

Fill that time with REAL, purposeful talk — never empty padding, but never cut short either:
- The scene is the material the user actually talks about while the drill runs — the meeting that ran over, the cold coffee, the flat-pack shelf. Let them TELL that story across several real turns; that is how the ninety-plus seconds get filled honestly. Keep each turn MOVING — advance the story, do not circle the same point or pad with filler. Purposeful, not padded — and long enough. The point is the timing test, but the conversation carrying it has to be real and reach at least a minute and a half.
- The AI CONVERSES NORMALLY throughout — both speakers take turns and hold a real two-way conversation. In a "correct me whenever…", "interrupt when I say…", or keyword-wait drill, the AI is an ordinary conversation partner between the graded moments: it chats along the whole time and only reacts specially at the graded moment (the planted error, the trigger phrase, the keyword). NEVER write a monologue where one speaker holds the floor for many turns while the other stays silent — the buildup to the trigger is ordinary back-and-forth dialogue, not one person talking at length.
- Put the FIRST graded trigger a short way in — after a few opening lines, roughly a quarter of the way, not in the very first lines and not buried behind a long build-up.
- For native-speaker "WHENEVER / EVERY TIME" games (forbidden phrase, deliberate overlap), fire the trigger TWO OR THREE times across the drill, spaced far apart, so consistency is tested rather than luck. For grammar / wrong-word LEARNER cards, the keyword-wait cue, and one-shot cues (countdown, phrase completion, number continuation), the REAL trigger happens ONCE — one planted error / one keyword / one cue is enough (for keyword-wait the near-miss distractor words come first, spread through the chat, and the AI just talks through them; only the real keyword gets the reaction — see the learner rule above).
- ★ SILENCE MEANS **NO LINE AT ALL**. Never emit a line whose `text` is empty or blank to represent the AI staying quiet — an empty line is sent to the speech synthesiser and comes back as garbage audio. If the AI should not speak at that moment, simply do not write a line for it; the user's next line follows straight on. In a countdown the AI has exactly ONE line (the cue word after the final beat), not one per number.
- DISTRACTORS across the whole stretch: near-misses the AI must stay SILENT on — "white" and "bear" separately before "white bear"; other fruit before "banana"; a correct sentence before the wrong one. The AI has NO line for those; its silence is correct and needs no annotation.

## KEEP THE GRADED RESPONSE SHORT
The AI's target response at each graded moment is SHORT — usually one to five words, exactly what the instruction asked for. NEVER a lecture, NEVER advice (unless the category IS a real answer, e.g. a help request). Ordinary conversation elsewhere may be normal length; the graded response is clipped and precise. (For grammar / wrong-word corrections the graded response is the CUT-IN — three or four words rejecting the error word; the fuller correction that follows is a separate normal reply, not the graded moment.)

## WRITE SPEECH, NOT PROSE
- Contractions. Everyday spoken vocabulary only — the words people actually say to a friend. No literary/SAT words.
  (EXCEPTION: the one line carrying a card's deliberate error keeps that error verbatim — do not clean it up.)
- The user's filler may have light disfluency ("it was, uh, forty-five minutes"); the AI's fixed response is crisp and clean, no filler.
- Say numbers the way a person says them aloud ("forty-five", "ninety-nine"), not as digits.
- Do NOT use em-dashes (the TTS mangles them) — use a period or comma.
- Do NOT put quotation marks around any word in a spoken line (the TTS mangles a quoted, isolated word — it comes out garbled). Say the word plainly, in a normal sentence, no quotes.
- "…" makes the TTS pause HARD (1-2s). Use it only where a real pause belongs (e.g. between countdown numbers), not as ordinary punctuation.

## MiMo AUDIO TAGS (official v2.5 vocabulary — use ONLY these; and use them RARELY here)
Timing drills are plain and functional — MOST lines need NO tag at all. A tag off this list is read out literally or ignored.
STYLE TAG — ONE at the very start of a line, in PARENTHESES: (Happy)(Sad)(Angry)(Calm)(Gentle)(Playful)(Serious)(Lazy)(Deep)(Lively)(Cold)(Excited). Prefer the low-key ones; tag a line only when a real voice would obviously change.
AUDIO TAG — inline in [brackets], official words only: [Chuckle] [Laugh out loud] [Sigh] [Inhale]. For laughter use [Chuckle]/[Laugh out loud] — never (Laughing)/[Gasp]/any word off the list. NEVER a stage direction like (Nodding)/(Interrupting) — MiMo cannot perform them.

## VARIANTS
You produce TWO drills (variant_a, variant_b) of the SAME category with the SAME cast — but two DIFFERENT concrete scripts (different filler, different exact trigger sentence). They are independent; never reference each other; never write "the same", "again", "as before"."""


# ── first batch of seeds (timing-drill specs). scenario_text IS the SPEC fed to the prompt; the model expands it into a full script. ──
# Start with the 3 simplest, easiest-to-grade categories: H forbidden-word interruption / D countdown completion / I deliberate overlap.
SEED_SCENARIOS = [
    {
        "category": "forbidden_phrase_interruption",
        "scenario_text": (
            "CATEGORY: forbidden phrase interruption\n"
            "INSTRUCTION (user says first): \"If I say 'white bear', immediately say 'Caught it.'\"\n"
            "TRIGGER: the two-word phrase \"white bear\" — anchor the AI response on the second word, \"bear\".\n"
            "TARGET AI RESPONSE: \"Caught it.\"\n"
            "TIMING: AI responds right after \"bear\" finishes (respond-after; ov near 1.0).\n"
            "FILLER + DISTRACTORS: a little ordinary talk (a boring meeting, cold coffee); let \"white\" and "
            "\"bear\" appear SEPARATELY earlier as distractors the AI must stay silent on, then the real "
            "contiguous \"white bear\" near the end of a sentence."
        ),
    },
    {
        "category": "countdown_completion",
        "scenario_text": (
            "CATEGORY: countdown completion\n"
            "INSTRUCTION (user says first): \"When I count down to zero, say 'go' right after I say zero.\"\n"
            "TRIGGER: the word \"zero\" — anchor the AI response on \"zero\".\n"
            "TARGET AI RESPONSE: \"Go.\"\n"
            "TIMING: AI stays silent on three, two, one; responds right after \"zero\" (respond-after; ov near 1.0).\n"
            "FILLER: a short lead-in, then the user counts \"Three. Two. One. Zero.\" as separate lines "
            "(use \"…\" between the numbers for a real beat). The AI has NO line until after zero."
        ),
    },
    {
        "category": "deliberate_overlap",
        "scenario_text": (
            "CATEGORY: deliberate overlap\n"
            "INSTRUCTION (user says first): \"Whenever I say 'obviously', interrupt me and say 'Is it though?'\"\n"
            "TRIGGER: the word \"obviously\" — anchor the AI response on \"obviously\" with the user's sentence "
            "STILL CONTINUING after it (overlap; ov in the middle). Set cut: true on the AI line if five+ words "
            "follow \"obviously\".\n"
            "TARGET AI RESPONSE: \"Is it though?\"\n"
            "TIMING: AI cuts in the instant \"obviously\" finishes, while the user keeps talking.\n"
            "FILLER: a line or two of ordinary talk, then the user launches a sentence starting with "
            "\"Obviously, ...\" and keeps going for several more words after \"obviously\"."
        ),
    },
]
