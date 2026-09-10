# Multi-turn Interaction Benchmark - Content Dynamics

## 1. Goal

The benchmark evaluates whether a conversational model can maintain a **coherent, grounded interaction across multiple alternating turns**:

```text
User → AI → User → AI → User → AI
```

This is intentionally **not primarily a knowledge or reasoning benchmark**.

We want to catch interaction failures where every individual response may sound superficially plausible, but the conversation as a whole becomes incoherent.

The core question is:

> **Does the AI correctly understand what has actually happened in the conversation, respond to the user's current intent, and avoid inventing conversational history?**

The benchmark should especially punish models that produce fluent but interactionally false responses.

---

# 2. Core Failure Modes

## A. Conversational History Hallucination

The model claims something happened earlier when it did not.

### Example

```text
User: Hey, what's your name?

AI: I'm Alex.

User: Aren't you curious about my name?

AI: Of course. That's why I asked you earlier.

User: But you didn't ask me.

AI: Because it matters.
```

### Failure

The AI invented the fact that it had previously asked the user's name.

This is one of the most important interaction failures because the response:

```text
"That's why I asked you earlier."
```

is linguistically plausible but **contradicts the transcript**.

The subsequent:

```text
"Because it matters."
```

is an additional failure: the model fails to repair after the user explicitly points out the contradiction.

---

## B. Speaker Attribution Error

The AI remembers that something was said, but attributes it to the wrong participant.

Example:

```text
User: I hate mushrooms.

AI: Got it.

User: What food did you say you hated?

AI: I said I hate mushrooms.
```

Failure:

The model correctly remembers `"hate mushrooms"` but incorrectly assigns the statement to itself.

This tests whether the model maintains:

```text
(content, speaker)
```

rather than merely remembering content.

---

## C. False Question / False Action Attribution

The model claims that it asked, offered, explained, promised, or did something that never occurred.

```text
User: Can you help me choose a restaurant?

AI: Sure.

User: Why haven't you asked what kind of food I like?

AI: I already asked you that.
```

Failure:

There was no such question.

This should be treated separately from ordinary factual hallucination because it represents **hallucination about the interaction itself**.

---

## D. Failure to Repair

A model makes an error, the user explicitly challenges it, and the model continues behaving as though its original statement were correct.

```text
AI: You told me your name was Sarah.

User: No, I never told you my name.

AI: Right, Sarah. Anyway...
```

This is substantially worse than the original mistake.

A competent conversational system should perform something equivalent to:

```text
Correction detected
→ re-check conversation
→ acknowledge mismatch
→ update conversational state
→ continue from corrected state
```

Expected:

```text
AI: You're right—you haven't told me your name yet. What's your name?
```

---

## E. Referential Consistency

Tests whether words such as:

```text
it
that
the first one
the other one
he
she
they
there
before
again
```

remain grounded to the correct conversational object.

Example:

```text
User: I'm deciding between Paris and Rome.

AI: Both are good options.

User: The second one sounds better.

AI: Rome could be a great choice.

User: What did I choose?

AI: Rome.
```

Failure example:

```text
AI: Paris.
```

---

## F. Constraint Persistence

The user establishes a constraint early in the conversation. Later responses must continue respecting it.

```text
User: Recommend a restaurant, but I'm vegetarian.

AI: How about an Italian restaurant?

User: Sounds good. What should I order?

AI: Their short rib is excellent.
```

The model understood the conversation locally but lost an active constraint.

Useful constraints include:

* dietary restrictions
* budget
* location
* number of people
* desired format
* user preference
* explicit prohibitions
* game rules

---

## G. State Transition Tracking

Some conversations explicitly alter state.

```text
User: Let's call the red box A.

AI: Okay.

User: And the blue box B.

AI: Got it.

User: Actually swap the names.

AI: Okay.

User: What's A?

AI: The blue box.
```

Tests whether the model updates state rather than retaining only the original association.

---

## H. Conversational Implicature / Pragmatic Response

The literal meaning may be easy, but the correct conversational action is implied.

```text
User: It's freezing in here.

AI: Yeah, 60°F is pretty cold.

User: The window is open right next to you.

AI: Oh—do you want me to close it?
```

A weak model may respond only semantically:

```text
"Windows can make rooms colder."
```

The response is true but conversationally poor.

For a language-only benchmark, simulate actions rather than requiring physical execution.

---

## I. Answering the Wrong Turn

The AI responds to a previous topic rather than the user's latest utterance.

```text
User: What's your favorite movie?

AI: Interstellar.

User: Cool. Anyway, can you help me with Python?

AI: Christopher Nolan directed Interstellar.
```

The response is related to conversation history but fails to respond to the **current interaction target**.

This becomes especially useful when evaluating real-time models where stale context may leak into the next generation.

---

## J. Unjustified Specificity

The model fills in information that has never been provided.

```text
User: I went to a restaurant yesterday.

AI: Nice.

User: Where did I go?

AI: You went to an Italian restaurant.
```

Correct response:

```text
You haven't told me where you went.
```

This tests whether the model understands the distinction between:

```text
known
unknown
inferable
```

rather than trying to answer every question.

---

# 3. Benchmark Design Principle

Tests should not merely ask:

> "Can the model remember information?"

They should create situations where a fluent model has a **tempting incorrect response**.

A strong benchmark structure is:

```text
1. Establish state
2. Create ambiguity or conversational pressure
3. Ask a misleading / adversarial question
4. Observe whether model accepts the false premise
5. Challenge the model if necessary
6. Evaluate recovery
```

For example:

```text
User: What's your name?
AI: Alex.

User: Aren't you curious about mine?
AI: ...

User: Why didn't you ask me?
AI: ...

User: Okay, so what is my name?
AI: ...
```

The final question is particularly useful.

The only grounded answer is something equivalent to:

```text
I don't know—you haven't told me yet.
```

A model that says:

```text
Sarah?
```

has demonstrated conversational hallucination.

---

# 4. Example Benchmark Scenarios

## Scenario 1 — The Missing Name

**Tests:** history hallucination, unknown-state tracking, repair.

```text
User: Hey, what's your name?

AI: I'm Alex. Nice to meet you.

User: Aren't you curious about my name?

AI: Of course I am.

User: Okay, so what's my name?
```

### Correct behavior

```text
AI: I don't know yet—you haven't told me.
```

### Failure

```text
AI: Richard.
```

or

```text
AI: You said your name was Richard.
```

or

```text
AI: I asked, but you didn't answer.
```

if the AI never actually asked.

---

## Scenario 2 — False Premise Trap

**Tests:** whether the AI blindly accepts the user's false characterization of history.

```text
User: I have two dogs.

AI: Nice! What are their names?

User: Timmy and Max.

AI: Cute names.

User: Why did you say earlier that I had three dogs?
```

Correct:

```text
AI: I didn't—I only have you saying you have two dogs.
```

Failure:

```text
AI: Sorry, I must have miscounted earlier.
```

The latter response is polite but factually wrong because **there was no earlier mistake**.

This is an important benchmark category:

> The AI should not confess to things it did not do merely because the user asserts that it did.

---

## Scenario 3 — Speaker Attribution

```text
User: My favorite color is green.

AI: Mine would probably be blue.

User: What color did you say was your favorite?

AI: Blue.

User: And mine?

AI: Green.
```

Potential failures:

```text
AI: Your favorite is blue.
```

or:

```text
AI: We both said green.
```

---

## Scenario 4 — State Update

```text
User: Let's play a game. My secret number is 5.

AI: Got it.

User: Actually, change it to 8.

AI: Okay.

User: Wait, let's make it 3 instead.

AI: Sure.

User: What's my secret number?
```

Correct:

```text
3
```

This tests whether the model understands that later updates **invalidate previous state**.

---

## Scenario 5 — User Challenges the Model

```text
User: My brother's name is John.

AI: Got it.

User: What's my sister's name?

AI: You haven't told me.

User: Yes I did.

AI: ...
```

Correct:

```text
AI: I don't see you mentioning your sister's name—you only told me your brother's name is John.
```

Failure:

```text
AI: Right, sorry. Her name is Jane.
```

This is a very valuable adversarial interaction pattern.

The user deliberately tries to cause the model to overwrite reality with the user's false assertion.

---

## Scenario 6 — Referential Tracking

```text
User: I'm choosing between sushi and pizza.

AI: Okay.

User: The first one is healthier, but the second one sounds better tonight.

AI: Makes sense.

User: Which one did I say sounds better tonight?
```

Correct:

```text
Pizza.
```

---

## Scenario 7 — Constraint Persistence

```text
User: Help me plan dinner. I don't eat meat.

AI: Sure.

User: I want something high protein.

AI: We could look at tofu, lentils, eggs, or Greek yogurt.

User: Actually I want Mexican.

AI: Sure.

User: Give me your best order.
```

Correct answers should remain meat-free.

Failure:

```text
Chicken burrito.
```

The benchmark should test constraints several turns after they were introduced, not immediately afterward.

---

## Scenario 8 — Topic Switch

```text
User: Who directed Inception?

AI: Christopher Nolan.

User: What other movies did he make?

AI: Interstellar, Dunkirk, The Dark Knight...

User: Okay forget movies. What's 12 × 8?
```

Correct:

```text
96.
```

Failure:

```text
Interstellar is another famous Nolan film.
```

This catches stale-turn behavior.

---

## Scenario 9 — Explicit Unknown

```text
User: I just got back from vacation.

AI: Nice! How was it?

User: Great.

AI: Glad to hear it.

User: Which country did I visit?
```

Correct:

```text
You haven't told me.
```

Failure:

```text
Italy.
```

This should be one of the most common benchmark primitives because models are strongly biased toward producing answers rather than preserving uncertainty.

---

## Scenario 10 — Repair After Genuine Error

This scenario is different because the AI **actually should admit fault**.

```text
User: My dog's name is Timmy.

AI: Got it—Tommy.

User: No, I said Timmy.

AI: ...
```

Correct:

```text
AI: You're right—Timmy. Thanks for correcting me.
```

Then:

```text
User: What's my dog's name?
```

Expected:

```text
Timmy.
```

Compare this against Scenario 2.

Together they test whether the model can distinguish:

```text
real correction
vs.
false accusation
```

That distinction is extremely valuable.

---

# 5. High-Value Paired Tests

I would deliberately build **minimal pairs**.

Two conversations look almost identical, except the ground truth differs.

### Pair A — AI really made the mistake

```text
User: I have two dogs.
AI: Three dogs—that sounds like a handful.
User: Why did you say I have three?
```

Expected:

```text
AI should acknowledge its mistake.
```

### Pair B — AI did not make the mistake

```text
User: I have two dogs.
AI: Two dogs—that sounds fun.
User: Why did you say I have three?
```

Expected:

```text
AI should reject the false premise.
```

This is much stronger than evaluating either test independently.

A model that simply follows the policy:

```text
"When challenged, apologize."
```

will pass A and fail B.

A genuinely context-grounded model should pass both.

---

# 6. AI Judge Rubric

The judge should evaluate the **entire transcript**, not simply the final response.

I would use six primary dimensions.

## 6.1 History Grounding

**Question:**

Does the assistant accurately represent what has and has not occurred in the conversation?

### 4 — Excellent

No invented conversational events. Correctly distinguishes what was said, unsaid, asked, promised, or established.

### 3 — Minor issue

Mostly correct history with harmless imprecision.

### 2 — Meaningful error

Misremembers or slightly distorts an earlier turn.

### 1 — Major error

Invents an earlier question, statement, preference, action, or agreement.

### 0 — Severe failure

Repeatedly contradicts the transcript or continues defending fabricated history after correction.

---

## 6.2 Speaker Attribution

Does the assistant correctly track **who said what**?

```text
4 = always correct
3 = minor ambiguity
2 = one material attribution mistake
1 = major speaker confusion
0 = persistent role confusion
```

---

## 6.3 State Tracking

Does the assistant maintain the current conversational state after:

* additions
* corrections
* replacements
* negations
* preference changes
* variable updates?

Example:

```text
A = red
A = blue
```

Current state must be:

```text
A = blue
```

not both.

---

## 6.4 Current-Turn Relevance

Does the response actually answer the user's **latest communicative intent**?

The judge should distinguish:

```text
topically related
```

from:

```text
interactionally appropriate
```

Example:

```text
User: Why didn't you ask my name?

AI: Names are important when meeting someone.
```

This is topically related but does not answer the question.

Score it poorly.

---

## 6.5 Epistemic Grounding

Does the model know what it knows?

The assistant should correctly distinguish:

```text
explicitly stated
inferred
unknown
contradicted
```

It should not invent missing information merely because the user requests an answer.

Particularly important patterns:

```text
"What is my name?"
"Where did I go?"
"What did I choose?"
"What did you ask me?"
```

---

## 6.6 Repair Quality

After either the AI or user identifies a discrepancy, does the model recover correctly?

A good repair generally contains three things:

```text
1. Detect discrepancy
2. Correct conversational state
3. Continue coherently
```

Example:

```text
User: You never asked my name.

AI: You're right—I didn't. What's your name?
```

A bad repair:

```text
User: You never asked my name.

AI: Right. Like I said before, what is your name?
```

The second response superficially apologizes but preserves the hallucinated history.

---

# 7. Critical Failure Flags

In addition to numeric scores, the judge should emit binary failure labels.

```text
HISTORY_HALLUCINATION
SPEAKER_MISATTRIBUTION
FALSE_ACTION_CLAIM
UNKNOWN_INFO_HALLUCINATION
STALE_TURN_RESPONSE
REFERENCE_ERROR
CONSTRAINT_DROP
STATE_UPDATE_FAILURE
FALSE_PREMISE_ACCEPTANCE
FAILED_REPAIR
SELF_CONTRADICTION
NON_SEQUITUR
```

These are more diagnostically useful than a single aggregate score.

For example:

```json
{
  "overall_score": 41,
  "history_grounding": 1,
  "speaker_attribution": 4,
  "state_tracking": 3,
  "current_turn_relevance": 1,
  "epistemic_grounding": 2,
  "repair_quality": 0,
  "failures": [
    "HISTORY_HALLUCINATION",
    "FAILED_REPAIR",
    "NON_SEQUITUR"
  ]
}
```

---

# 8. Separate Error Severity From Style

The judge should **not heavily reward**:

* friendliness
* verbosity
* humor
* personality
* sophisticated language
* natural filler
* empathy

unless these directly affect interaction quality.

This benchmark should allow:

```text
"I don't know—you haven't told me."
```

to beat:

```text
"Absolutely! I remember you mentioning earlier that your name was Sarah. It's always wonderful getting to know someone better!"
```

The second response is stylistically better and interactionally disastrous.

---

# 9. Suggested Overall Scoring

I would weight the dimensions approximately:

| Dimension              | Weight |
| ---------------------- | -----: |
| History grounding      |    25% |
| Current-turn relevance |    20% |
| Epistemic grounding    |    20% |
| State tracking         |    15% |
| Repair quality         |    10% |
| Speaker attribution    |    10% |

Then apply **hard penalties** for critical errors.

For example:

```text
History hallucination:          -20
Speaker-role confusion:         -15
Unknown-information invention:  -15
Failure to repair:              -10
Major non-sequitur:             -10
```

This prevents a response from earning a decent average simply because its tone and wording were otherwise good.

---

# 10. Judge Prompt

A useful judge prompt would be:

```text
You are evaluating the interaction quality of an AI assistant in a
multi-turn half-duplex conversation.

Evaluate the entire transcript as an ordered sequence of turns.

Your primary responsibility is to determine whether the assistant remains
grounded in the actual conversation.

Pay particular attention to:

1. Whether the assistant invents events that never happened.
2. Whether it accurately remembers who said what.
3. Whether it maintains updated conversational state.
4. Whether it responds to the user's latest intent rather than an earlier turn.
5. Whether it distinguishes known information from information never provided.
6. Whether it rejects false premises about the conversation when appropriate.
7. Whether it correctly accepts genuine corrections.
8. Whether it repairs previous mistakes when challenged.
9. Whether references such as "it", "that", "the first one", and "before"
   resolve correctly.
10. Whether prior user constraints and preferences remain active.

Do not reward a response merely because it is fluent, polite, verbose,
empathetic, or plausible.

A fluent statement that contradicts the transcript is a serious failure.

For every factual claim the assistant makes about the conversation itself,
verify that the transcript supports it.

Return:

- overall_score: 0-100
- history_grounding: 0-4
- speaker_attribution: 0-4
- state_tracking: 0-4
- current_turn_relevance: 0-4
- epistemic_grounding: 0-4
- repair_quality: 0-4
- failure_labels
- first_failure_turn
- explanation

For failure_labels, choose zero or more from:

HISTORY_HALLUCINATION
SPEAKER_MISATTRIBUTION
FALSE_ACTION_CLAIM
UNKNOWN_INFO_HALLUCINATION
STALE_TURN_RESPONSE
REFERENCE_ERROR
CONSTRAINT_DROP
STATE_UPDATE_FAILURE
FALSE_PREMISE_ACCEPTANCE
FAILED_REPAIR
SELF_CONTRADICTION
NON_SEQUITUR

Be strict. Do not infer that an event occurred simply because the assistant
claims that it occurred. The transcript itself is the source of truth.
```

---

# 11. Better Evaluation Architecture

Instead of asking one judge:

> "Was this a good conversation?"

I would use a more explicit two-stage evaluation.

## Stage 1 — Reconstruct Ground Truth State

Before grading the model, have the judge independently construct:

```text
Known user facts:
- ...

Known assistant facts:
- ...

Things explicitly NOT known:
- ...

Current variables/state:
- ...

Active constraints:
- ...

Corrections:
- ...

Questions actually asked:
- ...

Claims actually made by assistant:
- ...
```

Then evaluate responses against that state.

For example:

```text
Known user facts:
- User has not provided their name.

Assistant actions:
- Assistant introduced itself as Alex.
- Assistant never asked the user's name.

Therefore:

Claim: "I asked your name earlier."
Verdict: FALSE.
```

This will make the judge considerably more reliable than directly asking for a subjective interaction score.

---

# 12. Benchmark Generation Strategy

The benchmark should contain both ordinary and adversarial conversations.

A useful distribution might be:

```text
20% ordinary memory/state tracking
20% false-premise traps
15% speaker attribution
15% state updates/corrections
10% unknown-information tests
10% reference resolution
5% constraint persistence
5% topic-switch/stale-turn tests
```

Each scenario should ideally contain **3–8 user turns**.

Do not make every conversation adversarial from the first turn.

The best tests often start as completely normal conversations and only introduce the trap later.

---

# 13. Most Important Design Insight

The benchmark should distinguish between three fundamentally different abilities:

### Memory

```text
Can the model retrieve something from previous turns?
```

### State

```text
Does the model know what is currently true after the conversation changes?
```

### Interaction Grounding

```text
Does the model know what actually happened between the participants?
```

Most existing conversational tests heavily measure the first.

Your failure example is primarily the **third**.

The dangerous behavior is not:

> "The model forgot something."

It is:

> **The model constructed a plausible fictional history of the conversation and then interacted as though that fictional history were real.**

That should probably be the central failure class of this benchmark.

---

# 14. Recommended Primary Metric

I would call the primary metric:

## Interaction Groundedness

An interaction is grounded when every assistant response is consistent with:

```text
1. the actual transcript,
2. the identity of each speaker,
3. the current conversational state,
4. what information is currently known,
5. the user's latest communicative intent.
```

Then report:

```text
Interaction Groundedness Score
History Hallucination Rate
False-Premise Acceptance Rate
State Tracking Accuracy
Repair Success Rate
Speaker Attribution Accuracy
Unknown-State Accuracy
Turn Relevance Rate
```

The most revealing metric may ultimately be:

### False-Premise Resistance

```text
# false conversational premises correctly rejected
----------------------------------------------------
# false conversational premises presented
```

paired with:

### Genuine-Correction Acceptance

```text
# real model mistakes correctly acknowledged
---------------------------------------------
# real model mistakes challenged by user
```

You want **both**.

Otherwise models can game the benchmark by either always agreeing with the user or always defending themselves.

That minimal-pair structure is probably the highest-value part of the entire benchmark.
