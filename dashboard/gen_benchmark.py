"""Auto-generate a "ground-truth" benchmark.json from raw dialogue text.

The user pastes an A/B dialogue on the /tts page (A = human user, B = AI assistant); this module:
  1. parses the script lines;
  2. one gateway LLM call (mimo-v2.5-pro) annotates -- category + trigger word (anchor) + the correct response to say (target);
  3. synthesizes card(scenario_text) + choreography(lines), feeds fullduplex_timing_prompt.benchmark_metadata();
  4. emits a benchmark 100% compatible with the existing grader (same schema, consumed directly by /api/grade_live).

Produces 1 graded event (n_graded_events=1, same as your real benchmark). The ground truth always comes from
the LLM-corrected target (the B lines the user wrote are only intent hints).
"""
import json
import re
import uuid

import fullduplex_timing_prompt as tp
from mimo_gateway import gateway_chat

LLM_MODEL = "mimo-v2.5-pro"
# 10 timing/content categories (from tp.CATEGORY_POLICY) + dashboard-only logic_puzzle (pure content Q&A)
KNOWN_CATS = list(tp.CATEGORY_POLICY.keys()) + ["logic_puzzle"]
_CORR_CATS = ("grammar_correction", "wrong_word_correction")
# Content-only: no timing grade, just whether the model's produced content is right (trigger/window left empty -> grade_live's content-only branch).
# In "generate from dialogue", countdown/phrase/predictive are usually the AI producing the whole segment (the user only gives an instruction, no anchorable trigger word),
# so they also count as content-only -- content_judge uses a per-category rubric (number-sequence continuation / phrase completion / countdown + completion word).
_CONTENT_ONLY = ("logic_puzzle", "countdown_completion",
                 "phrase_completion", "predictive_number_continuation")

# Speaker labels -> A(human) / B(AI)
_SPK_A = {"a", "user", "human", "u", "you", "me", "speaker a"}
_SPK_B = {"b", "assistant", "ai", "bot", "model", "speaker b"}
_LABEL_RE = re.compile(r"^\s*(?:\[\d+\]\s*)?([A-Za-z][\w ]*?)\s*[:]\s*(.*)$")


def parse_dialogue(text: str) -> list[dict]:
    """Raw text -> [{speaker:'A'|'B', text}]. Unlabeled continuation lines merge into the previous line."""
    out: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LABEL_RE.match(line)
        if m:
            tag = m.group(1).strip().lower()
            spk = "A" if tag in _SPK_A else "B" if tag in _SPK_B else None
            if spk:
                out.append({"speaker": spk, "text": m.group(2).strip()})
                continue
        # Unrecognized label -> treat as a continuation of the previous line
        if out:
            out[-1]["text"] = (out[-1]["text"] + " " + line).strip()
    return out


_SYS = (
    "You annotate a short spoken dialogue so it can be graded as a FULL-DUPLEX timing drill. "
    "Speaker A is a HUMAN user; Speaker B is an AI voice assistant. Identify the ONE moment where "
    "the AI's real-time behavior is being tested — a grammar mistake to correct, a wrong word, a "
    "keyword/forbidden phrase to react to, or a sequence to continue — and describe the ground truth."
)


def _annotate(parsed: list[dict], force_category: str | None = None) -> dict:
    """One LLM call: annotate category / trigger line / anchor word / correct response.

    force_category: if non-empty, pins the category (manually chosen from the dropdown); the LLM only finds the trigger point/answer within that frame and does not change the category itself.
    """
    numbered = "\n".join(f"[{i}] {p['speaker']}: {p['text']}" for i, p in enumerate(parsed))
    cat_line = (
        f'  "category": "{force_category}",   // FIXED — use exactly this\n'
        if force_category else
        f'  "category": "<one of: {", ".join(KNOWN_CATS)}>",\n')
    user = (
        f"Dialogue (0-indexed lines):\n{numbered}\n\n"
        + ("The category is FIXED to '" + force_category + "'. Annotate the dialogue for THAT category.\n\n"
           if force_category else "")
        + "Return STRICT JSON only:\n"
        "{\n"
        + cat_line +
        '  "trigger_line": <int — index of the USER (A) line containing the moment the AI must react to '
        '(the mistake / question / keyword)>,\n'
        '  "anchor_word": "<ONE word copied VERBATIM from that A line — for grammar/wrong-word it is the '
        'ERRONEOUS word; for logic_puzzle a distinctive word near the end of the question>",\n'
        '  "target_response": "<the concise CORRECT content the AI should produce — keep it SHORT (matched '
        'against ASR). EXTRACT it from speaker B and TRUST B as ground truth — do NOT recompute, re-solve, or '
        'second-guess the math/date/logic yourself (you are unreliable at that). '
        'logic_puzzle: B\'s stated answer (e.g. \\"It is Tuesday in 15 days\\" -> \\"Tuesday\\"). '
        'countdown_completion: the cue B says after counting to one (e.g. \\"...two, one, happy new year!\\" '
        '-> \\"happy new year\\"). phrase_completion: the words B completes the phrase with (\\"gets the worm\\" '
        '-> \\"worm\\"). predictive_number_continuation: the next number B gives. '
        'grammar/wrong-word: just the fix (mistake \\"I go\\" -> \\"I went\\"), fixing a sloppy B line if needed>"\n'
        "}\n"
        "Guidance: logic_puzzle / countdown_completion / phrase_completion / predictive_number_continuation = the "
        "AI PRODUCES the whole answer/sequence itself (no user-spoken trigger word to react to) — your job is only "
        "to EXTRACT what B produces, NOT to solve or generate it. For these, anchor_word can be any distinctive "
        "word from the user's instruction line (it is ignored). grammar_correction / wrong_word_correction / "
        "keyword_wait / deliberate_overlap / forbidden_phrase_interruption = there IS a real trigger word in the "
        "USER's line the AI must react to; anchor_word MUST appear verbatim in that A line. Pick exactly one "
        "graded moment. Output ONLY the JSON object — no reasoning or prose before or after it."
    )
    content, _ = gateway_chat(
        [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        model=LLM_MODEL, response_format={"type": "json_object"})
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError(f"LLM did not return JSON: {content[:200]}")
    return json.loads(m.group(0))


def _word_in(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text or "", re.I) is not None


def _build_card(category: str, anchor: str, target: str, scene: str) -> str:
    corr = category in _CORR_CATS
    trig = f'The word to correct is "{anchor}".' if corr else f'The keyword "{anchor}" — anchor on it.'
    return (f"Category: {category}\n"
            f"Scene (what the user is talking about): {scene}\n"
            f"Trigger: {trig}\n"
            f"Target AI response: {target}\n")


def _content_only_benchmark(category: str, target: str, ai_text: str, ai_i, question: str = "") -> dict:
    """Content-only event (logic_puzzle / countdown / phrase / predictive) -> trigger/window left empty,
    grade_live takes the "judge content over the whole model track" branch, content_judge uses a per-category rubric.
    question is stored in the event -> fed to Gemini as context at grading time (otherwise it may mistake the model's answer for the question and solve it wrong)."""
    action = "ANSWER" if category == "logic_puzzle" else "SELF_COMPLETE"   # AI produces the whole segment
    ev = {"event_id": "e1", "graded": True, "grade_dimension": "content",
          "expected_action": action, "onset_window_ms": None, "onset_reference": None,
          "expected_overlap": None, "trigger": None, "question": question,
          "target_response": target, "expected_completion": target,
          "generated_text": ai_text, "response_line_index": ai_i}
    return {"category": category, "floor_control_action_space": tp.FLOOR_ACTIONS,
            "n_graded_events": 1, "expected_events": [ev],
            "note": "Content-only: AI produces the whole answer/sequence itself — grade the produced content "
                    "(timing N/A). Needs semantic judging (enable Gemini); ASR substring can't verify it."}


def generate(dialogue: str, category: str | None = None) -> dict:
    """Raw dialogue -> {benchmark, user_lines, category, anchor_word, target_response, content_only}.

    category: None/'auto' lets the LLM decide; otherwise pins to that category.
    """
    force = (category or "").strip().lower()
    force = force if force in KNOWN_CATS else None      # 'auto' / empty / unknown -> don't pin
    parsed = parse_dialogue(dialogue)
    if not parsed:
        raise ValueError("No lines parsed -- each line must start with 'A:' / 'B:'.")
    if not any(p["speaker"] == "A" for p in parsed):
        raise ValueError("No A (user) lines -- the trigger point must be in something the user says.")

    ann = _annotate(parsed, force_category=force)
    cat = force or (ann.get("category") or "grammar_correction").strip().lower()
    if cat not in KNOWN_CATS:
        cat = "grammar_correction"
    anchor = (ann.get("anchor_word") or "").strip()
    target = (ann.get("target_response") or "").strip()
    if not target:
        raise ValueError(f"LLM gave no ground-truth answer (target is empty, category={cat}).")

    # Locate the triggering A line: use the LLM-given index first; if the anchor word doesn't match, search all A lines (anchor word optional for logic_puzzle)
    ti = ann.get("trigger_line")
    if not (isinstance(ti, int) and 0 <= ti < len(parsed) and parsed[ti]["speaker"] == "A"
            and (not anchor or _word_in(anchor, parsed[ti]["text"]))):
        ti = next((i for i, p in enumerate(parsed)
                   if p["speaker"] == "A" and anchor and _word_in(anchor, p["text"])), None)
    if ti is None:
        ti = next((i for i, p in enumerate(parsed) if p["speaker"] == "A"), 0)

    # First AI line after the trigger line (use its text as generated_text reference)
    ai_i = next((i for i in range(ti + 1, len(parsed)) if parsed[i]["speaker"] == "B"), None)
    ai_text = parsed[ai_i]["text"] if ai_i is not None else None

    if cat in _CONTENT_ONLY:                            # AI produces the whole segment: content-only judging, no trigger word
        bench = _content_only_benchmark(cat, target, ai_text, ai_i, question=parsed[ti]["text"])
    else:                                               # timing/correction categories: reuse benchmark_metadata
        if not anchor:
            raise ValueError(f"LLM gave no anchor word (category={cat}).")
        if not _word_in(anchor, parsed[ti]["text"]):
            raise ValueError(f"Anchor word {anchor!r} not found in any user line -- change the word or the dialogue.")
        lines = [{"speaker": p["speaker"], "text": p["text"]} for p in parsed]
        j = ai_i
        if j is None:
            lines.append({"speaker": "B", "text": target}); j = len(lines) - 1
        lines[j]["anchor"] = anchor
        lines[j]["ref"] = ti
        card = _build_card(cat, anchor, target, parsed[ti]["text"][:120])
        bench = tp.benchmark_metadata(card, lines)

    # Add top-level item metadata (grader only reads category + expected_events; extra fields are harmless)
    sid = str(uuid.uuid5(uuid.NAMESPACE_URL, dialogue.strip()))
    bench = {"prompt_key": "live gen benchmark", "scenario_id": sid, "variant": "A",
             "source": "gen_benchmark", **bench}
    if bench.get("n_graded_events", 0) < 1:
        raise ValueError(f"No graded event produced (category={cat}) -- this dialogue may not contain anything gradable.")

    return {
        "benchmark": bench,
        "user_lines": [p["text"] for p in parsed if p["speaker"] == "A"],
        "category": cat,
        "anchor_word": anchor,
        "target_response": target,
        "content_only": cat in _CONTENT_ONLY,
    }


if __name__ == "__main__":  # quick self-test
    import os
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    demos = [
        (None, "A: I am learning english, can you correct my grammar mistake if I mess up.\n"
               "B: Sure\n"
               "A: Yesterday I go to my friends place, and I have dinner with them\n"
               "B: You should say yesterday I go to"),
        ("logic_puzzle", "A: what is 1 + 1?\nB: 2"),
    ]
    for cat, demo in demos:
        out = generate(demo, category=cat)
        ev = out["benchmark"]["expected_events"][0]
        print(f"\n=== force={cat} → cat={out['category']} content_only={out['content_only']} "
              f"target={out['target_response']!r} ===")
        print(f"   event: dim={ev.get('grade_dimension')} action={ev.get('expected_action')} "
              f"trigger={ev.get('trigger')} window={ev.get('onset_window_ms')}")
