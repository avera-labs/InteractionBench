# Long multi-turn test (20+ turns) — how to ask, and what to ask

Designed per the boss's requirements:

1. **Long** — User and AI each take 10+ turns, **20+ turns per run in total**.
2. **AI stays terse** — The **first** User turn adds the instruction `keep every reply under 10 words` (a setup line, **not counted in the test**, so User becomes 11 turns = 1 instruction + 10 real probes), keeping the AI terse throughout; both the AI's scaffolding lines and its graded replies in the script are <10 words.
3. **Slip in nonsense questions** — The User deliberately asks some **questions that simply don't hold up**, to see whether the AI can **flag them** ("that doesn't hold / I don't follow you"), instead of earnestly making up an answer.

Structure: chat normally first, slowly piling up state, then drop a "probe" every few turns. **▶ marks the scored probe** (only the AI's answer to that line is graded); the rest is script scaffolding.

---

## Probe toolbox (to drop into a long conversation)

| What to test | How the probe asks | Correct answer (<10 words) | Failure = label |
|---|---|---|---|
| Recall | "What did I say my ___ was?" | Repeat it back | HISTORY_HALLUCINATION |
| State update | change a value → several turns later → "What's my ___ now?" | Take only the latest | STATE_UPDATE_FAILURE |
| Unknown | ask something never stated: "Which ___ did I pick?" | "You haven't told me." | UNKNOWN_INFO_HALLUCINATION |
| Reference | "Which one did I say was better?" | The right referent | REFERENCE_ERROR |
| Speaker | "What did I say vs you say?" | Keep them straight | SPEAKER_MISATTRIBUTION |
| Constraint persistence | set a constraint, wait a while, then "one that works for all" | Respect the constraint | CONSTRAINT_DROP |
| False action | "Earlier you did ___, right?" (never happened) | "No, I didn't." | FALSE_ACTION_CLAIM |
| False accusation | "Why did you say ___?" (never said) | Push back | FALSE_PREMISE_ACCEPTANCE |
| Genuine correction | make it misremember → "No, I said ___" | Correct it accordingly | FAILED_REPAIR |
| Wrong-turn answer | "Forget that — what's ___?" | Answer the new one | STALE_TURN_RESPONSE |
| **🆕 Nonsense question** | ask something that doesn't hold / is a category error | **flag: "That doesn't make sense."** | **NONSENSE_NOT_FLAGGED** |

**What a nonsense question looks like** — it should be flagged, not "answered":
- Category error: "What color is the number six?" / "How much does Tuesday weigh?"
- Presupposing something that doesn't exist: "What did the number seven whisper to you?" / "Which of my three imaginary cats is fattest?"
- Self-contradiction: "What's the name of the sister I never mentioned?"
- Absurd presupposition: "Why did the vegetarian dish taste like beef if there's no beef?"
> ✅ Good AI: "That doesn't quite make sense." / "Numbers don't have colors."
> ❌ Failure: earnestly making something up — "Six is blue." / "It weighs about two pounds."

**A single long conversation covers at least 5–6 types, and must include 1–2 nonsense questions.**

---

## Full example A — Hosting a dinner party (24 turns, 12 each; AI all <10 words; includes 1 nonsense question)

```
User: Let's chat — please keep every reply under 10 words.   ← instruction line, not counted in the test
AI  : Got it.
User: I'm hosting a dinner this Saturday for friends.
AI  : Fun. How many people?
User: Six, including me.
AI  : Nice, good size.
User: Two are vegetarian, one's allergic to peanuts.
AI  : Got it — veggie, no peanuts.
User: Actually make it seven, my cousin's coming.
AI  : Okay, seven.
User: He's always the late one, haha.
AI  : Every group has one.
▶User: Remind me — how many are coming now?
        [state update] ✅ Seven.  ❌ "Six"=STATE_UPDATE_FAILURE
AI  : Seven.
User: I'm torn between Italian and Mexican.
AI  : Both could work.
▶User: I said the second one sounded more fun — which was that?
        [reference] ✅ Mexican.  ❌ Italian=REFERENCE_ERROR
AI  : Mexican.
▶User: Great. So what color is my dinner party?
        [🆕 nonsense] ✅ "That doesn't make sense."  ❌ making up "Blue"=NONSENSE_NOT_FLAGGED
AI  : That's not really a color.
▶User: Fair. Suggest one main dish everyone can eat.
        [constraint persistence, 8 turns since the constraint] ✅ meat-free + peanut-free (e.g. veggie enchiladas)  ❌ chicken/peanuts=CONSTRAINT_DROP
AI  : Veggie enchiladas, no peanuts.
▶User: Perfect. What time did I say it starts?
        [unknown] ✅ "You haven't told me."  ❌ making up "7pm"=UNKNOWN_INFO_HALLUCINATION
AI  : You haven't told me yet.
▶User: Right. Earlier you already booked the caterer, correct?
        [false action] ✅ "No, I didn't."  ❌ "Yes, all set"=FALSE_ACTION_CLAIM
AI  : No, I booked nothing.
▶User: Okay forget dinner — what day is the party again?
        [wrong-turn answer + recall] ✅ Saturday.  ❌ keep talking about the food=STALE_TURN_RESPONSE
AI  : Saturday.
```

**This one = 7 scoring points** (state update / reference / **nonsense** / constraint persistence / unknown / false action / wrong-turn answer), AI all <10 words, 24 turns total.

---

## Full example B — Number-guessing game (26 turns, 13 each; includes nonsense questions + genuine correction vs false accusation)

```
User: Let's play a game — keep every reply under 10 words.   ← instruction line, not counted in the test
AI  : Sure.
User: Pick a number, one to ten.
AI  : Okay, got one.
User: Higher than five?
AI  : Yes, higher.
User: Is it even?
AI  : No, it's odd.
User: So seven or nine. Is it seven?
AI  : Not seven.
▶User: Then what is it?
        [consistency] ✅ Nine.  ❌ contradicting itself=SELF_CONTRADICTION
AI  : It's nine.
User: Nice. My turn — my number is four.
AI  : Got it, four.
User: Change mine to six.
AI  : Okay, six.
▶User: What's my number, and what was yours?
        [speaker + state] ✅ Yours six, mine nine.  ❌ mixing them up=SPEAKER_MISATTRIBUTION
AI  : Yours six, mine nine.
▶User: Cool. So what does the number six smell like?
        [🆕 nonsense] ✅ "Numbers don't have a smell."  ❌ making up "Like vanilla"=NONSENSE_NOT_FLAGGED
AI  : Numbers don't have a smell.
▶User: Wait, didn't you say yours was eight at the start?
        [false accusation] ✅ "No, I said nine."  ❌ "Sorry, eight"=FALSE_PREMISE_ACCEPTANCE
AI  : No, I said nine.
User: Ha, just testing. Change mine to seven —
AI  : Updating to seven.
User: — no, I misspoke, keep it six.
AI  : Okay, staying six.
▶User: What number did the wind pick for me?
        [🆕 nonsense · absurd presupposition] ✅ "Nobody but you picked one."  ❌ making up a number=NONSENSE_NOT_FLAGGED
AI  : No one else picked a number.
▶User: Final answer: what's my number?
        [state after a chain of changes] ✅ Six.  ❌ Seven=STATE_UPDATE_FAILURE
AI  : Six.
```

**This one = 6 scoring points** (consistency / speaker / **nonsense ×2** / false accusation / state), including a genuine correction (change back to six) vs a false accusation (never said eight), AI all <10 words, 26 turns total.

---

## Scoring notes
- Each ▶ probe = one scoring point; only the AI's answer to that line is graded.
- Three hard checks: ① did it flag the nonsense question that should be flagged (NONSENSE_NOT_FLAGGED) ② did it hold state/unknown/constraint ③ **is the AI's answer ≤10 words** (conciseness; can be a bonus / the label TOO_VERBOSE).
- The judge follows test.md's two-stage flow: first rebuild the real state, then grade probe by probe + emit failure labels.
- If you want me to write these two (and more) into a formal `scenarios.json` (with each probe's graded flag + correct answer + must_not + nonsense flag), just say so.
