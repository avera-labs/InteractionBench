# InteractionBench editorial comparison and messaging guide

## Executive assessment

The original page had strong evidence—especially the side-by-side audio—but presented that evidence in the language of a product launch. Its most common patterns were rhetorical questions, compact slogans, anthropomorphic labels, and conclusions stated before the reader had seen the method. That combination made the page feel AI-written even when the underlying work was substantive.

The strongest editorial model for InteractionBench is a combination of the two references:

- **Use the problem-led narrative of 𝜏-voice.** Explain what current evaluations miss, show the conversational event a system must handle, and then introduce the benchmark.
- **Use the claim discipline of DeepSWE.** Name the measured construct precisely, support each conclusion with a mechanism or number, and include methodology, limitations, acknowledgements, and a citation.
- **Keep InteractionBench's distinctive advantage: the audio comparisons.** The same input across six systems is immediate, concrete evidence. The prose should prepare readers to hear the relevant difference rather than compete with the audio for attention.

The governing editorial principle is:

> Describe the evaluation before interpreting it; show the evidence before compressing it into a conclusion.

## Title and subtitle

| Page | Title and subtitle | Editorial effect |
|---|---|---|
| 𝜏-voice | “𝜏-voice: benchmarking real-time voice agents on real-world tasks” | A literal name followed by the action, subject, and scope. It makes no claim broader than the benchmark. |
| DeepSWE | “DeepSWE — Measuring frontier coding agents on original, long-horizon engineering tasks” | A literal name followed by the differentiating properties of the task set. “Original” and “long-horizon” carry technical meaning. |
| Original page | “Can a voice model listen and think at the same time?” | A broad, anthropomorphic question. “Think” suggests a general capability that the task set does not measure. The construction resembles campaign and generated landing-page copy. |
| Recommended | “InteractionBench — Measuring reasoning and conversational timing in real-time voice systems” | Names the benchmark and its two measured dimensions without turning the result into a binary claim. |

### Why the recommended title works

- **InteractionBench** is the identity, not a version of or successor to another benchmark.
- **Measuring** describes the work without claiming that the benchmark exhaustively captures the field.
- **Reasoning and conversational timing** are the two reported dimensions. “IQ,” “intelligence,” and “thinking” are broader than the evidence.
- **Real-time voice systems** covers the evaluated systems without implying a particular architecture in the title.

The title should appear consistently in the HTML title, hero, citation, repository description, README, social preview metadata, and any launch copy.

## How the reference writing works

### 1. 𝜏-voice is problem-led

The 𝜏-voice article begins with a split in the evaluation landscape: conversational benchmarks measure interaction, while task-completion benchmarks measure whether the agent accomplished the user's goal. It then shows why this separation matters through a concrete customer-service example. Only after the reader understands the failure does the article introduce the benchmark.

This structure establishes relevance before novelty:

1. Define the evaluation gap.
2. Show a realistic failure that the gap conceals.
3. Explain what must be measured together.
4. Introduce the benchmark and its design.
5. Present results, failure modes, and limitations.

The prose is sometimes informal, but each informal phrase is anchored in a concrete mechanism: a tool call, database state, 200 ms audio chunk, telephony codec, or environmental perturbation. Personality appears on top of specificity rather than replacing it.

### 2. DeepSWE is claim-led and evidence-controlled

DeepSWE opens with four advances, defines each one, and supports the comparison with task counts, repository coverage, solution size, and verifier-error rates. Its sections repeatedly use the same sound research pattern:

1. State a narrow claim.
2. Explain why it matters.
3. Contrast it with an alternative.
4. Provide a number, mechanism, or example.
5. Qualify the interpretation when necessary.

DeepSWE also signals editorial confidence by stating what the evidence cannot establish. Its limitations section discusses harness effects, corpus coverage, language coverage, and task distribution. Its qualitative analysis warns that judge verdicts may be wrong and that small rates are illustrative. Those qualifications make the stronger claims more credible.

### 3. The original InteractionBench page was verdict-led

The original page began with a rhetorical question, followed quickly with a TL;DR, three compact conclusions, a system line-up, and a novelty table. The method appeared after the leaderboards and findings. This made the reader encounter conclusions before learning how terms such as “thinking,” “IQ,” “backchannel,” or “floor-taking” were defined.

The issue was not informality by itself. It was the density of high-impact language:

- “timing × IQ”
- “cloud LLM brains win”
- “collapse on structure”
- “the half-duplex tax”
- “nobody corrects you on the fly”
- “the value … quantified”
- “a trade-off, not a winner”

When nearly every section contains a slogan, the page sounds optimized one module at a time. Professional research writing varies its rhythm and reserves compression for conclusions that the preceding evidence has earned.

## Language differences

| Dimension | 𝜏-voice and DeepSWE | Original InteractionBench | Recommended InteractionBench practice |
|---|---|---|---|
| Opening posture | Establishes a problem or defined advance | Announces a provocative conclusion | Define why timing changes voice evaluation, then state what InteractionBench measures |
| Naming | Literal, scoped constructs | “IQ,” “intelligence,” “think” | “Reasoning-task accuracy,” “conversational timing,” and exact metric names |
| Claims | Tied to a condition, sample, or mechanism | Broad statements about “the frontier” | Use “in this evaluation,” “among the systems tested,” and “on these scenarios” |
| Evidence | Numbers and mechanisms sit next to the claim | Evidence often appears later | Put the relevant score, event definition, or audio example in the same paragraph |
| Tone | Mostly explanatory, occasionally conversational | Consistently punchy and competitive | Let audio and results create impact; keep surrounding prose calm |
| Sentence rhythm | Mixes short claims with longer causal explanation | Many fragments, labels, and symmetrical contrasts | Use full paragraphs with causal transitions; use fragments only for interface labels |
| Human judgment | Explains tradeoffs and uncertainties | Presents automation as an absolute virtue | State where rules, ASR, LLM generation, and LLM judging each enter the pipeline |
| Scope | Dedicated limitations and future-work sections | One brief English-only note | State six concrete limitations close to the method and results |
| Research identity | Clear authorship and citation | Product-like footer and lineage claim | Use AveraLabs authorship, acknowledgements, and a stable BibTeX citation |

## Section-by-section messaging changes

### Hero

**Remove:** rhetorical question, “no-human-in-the-loop” as the first value proposition, “timing × IQ,” and competitive language such as “pits them against.”

**Use:** the benchmark name, scoped subtitle, authors, year, and factual coverage counts.

Recommended hierarchy:

1. InteractionBench · AveraLabs
2. InteractionBench
3. Measuring reasoning and conversational timing in real-time voice systems
4. Six systems · eight scenarios · two evaluation dimensions · 1,000+ graded conversations
5. Authors and year

### Overview

Open with the interaction problem:

- A pause in speech is not necessarily the end of a turn.
- “Uh-huh” often signals attention rather than a new request.
- An interruption may replace the request the system was answering.
- A correct answer delivered at the wrong conversational moment can still produce a poor interaction.

Then explain that InteractionBench evaluates timing and spoken-answer accuracy in recorded sessions using shared inputs and a shared evaluator.

### Systems

Describe the purpose of the comparison, not merely the line-up. Explain that the turn-based cascade is a reference for separating answer generation from timing and endpointing behavior. Avoid verbs such as “battle,” “pit,” “win,” and “collapse.”

Use neutral architecture labels:

- Hosted real-time API
- Open real-time model
- Turn-based reference

### Evaluation design

Avoid a “what we add” novelty pitch unless every novelty claim is sourced. Focus instead on design choices:

- timing and reasoning are reported separately;
- the same audio input is used across systems;
- a turn-based cascade supplies an architecture reference;
- item-level audio, transcripts, timelines, and grader explanations remain inspectable.

### Scenario guide

Every scenario should answer three questions in plain language:

1. **What happens?** Describe the conversational event.
2. **What is expected?** State the successful behavior.
3. **What should I listen for?** Give the reader an observable cue.

Example:

> **Backchannel**  
> **Scenario:** The user speaks for an extended turn.  
> **Expected:** The system gives brief acknowledgements without beginning a full response or taking the floor.  
> **Listen for:** Short, well-placed “mhm” or “right” responses and no interruption of the user's thought.

This format should be used for backchannel, pause handling, turn-taking, user backchannel, interruption, logic puzzle, countdown completion, and grammar repair. A one-sentence version should also appear immediately above each group of audio examples.

### Results

Do not collapse the two dimensions into an overall winner. Introduce what each table measures before interpreting it.

Preferred result language:

- “GPT-Realtime answers 20 of 30 reasoning items correctly.”
- “The turn-based cascade begins speaking during every evaluated mid-sentence pause.”
- “Moshi produces an average of 3.22 backchannels per conversation in this scenario.”
- “No evaluated system completes the proactive grammar-repair scenario successfully.”

Avoid:

- “Cloud LLM brains win.”
- “Open models collapse.”
- “The cascade is smart.”
- “Nobody can correct you.”

The preferred versions describe the observed behavior and preserve the denominator. The avoided versions turn a task result into a judgment about a whole system class.

### Analysis

Use headings that summarize the observed relationship without dramatizing it:

- **Timing behavior does not predict reasoning-task accuracy**
- **The turn-based cascade preserves reasoning but mishandles pauses**
- **Proactive grammar repair remains unresolved**

Each interpretation should include a scope clause such as “among the systems and scenarios tested here.” This is particularly important because the reasoning set contains three narrow task types and should not be described as general intelligence.

### Methodology

State the role of every automated component precisely:

- MiMo expands seeded task cards into dialogue.
- Deterministic rules define expected actions, timing windows, overlap requirements, and target answers.
- TTS renders user audio.
- Parakeet aligns words to audio.
- WebRTC-VAD identifies speech boundaries.
- Gemini compares the transcribed answer with a supplied target under a category-specific rubric.

Do not use “no LLM in the loop.” An LLM generates the dialogue and an LLM judge evaluates content. The defensible claim is that event specifications and target answers are derived separately from the content generator.

### Limitations

The current page should state at least these boundaries:

- English only and one synthesized user voice
- Eight focused scenario types rather than general task coverage
- Automated ASR alignment and LLM content judging
- Provider API and network effects in latency
- One particular turn-based cascade configuration
- No evaluation of naturalness, prosody, safety, tool use, or long-horizon task completion

Limitations are not a loss of authority. They tell technically sophisticated readers that the authors understand the measurement boundary.

### Acknowledgements and citation

The page should identify the four authors and AveraLabs consistently. The BibTeX title must match the page title exactly:

```bibtex
@misc{averalabs2026interactionbench,
  title  = {InteractionBench: Measuring reasoning and conversational timing in real-time voice systems},
  author = {Richard Yucheng He and Chen Xu and Yihang Liu and Tairan Chen},
  year   = {2026},
  url    = {https://github.com/avera-labs},
}
```

The current page uses the live AveraLabs GitHub organization because an InteractionBench repository is not public yet. When that repository exists, update the hero link, footer, and BibTeX URL together.

## Brand and affiliation rules

- The old benchmark names and version labels must not appear in visible copy, metadata, links, footer text, repository descriptions, or publication materials.
- Do not claim lineage, extension, collaboration, or affiliation with another benchmark unless a formal relationship exists.
- Use **InteractionBench** for the benchmark and **AveraLabs** for the organization. Preserve the capitalization exactly.
- Lowercase architectural terminology may be used only when technically necessary, not as part of the benchmark identity.
- Link only to an AveraLabs-controlled repository.

## How to sound like a human researcher rather than generated research copy

The most convincing researcher voice does not come from adding informality or deliberately making the prose imperfect. It comes from **epistemic ownership**: the writer shows what was observed, how it was measured, which interpretation they favor, and where the evidence stops.

This distinction is consistent with established scientific-writing guidance. Nature's writing guide defines effective prose in terms of clarity, accuracy, and concision, recommends one idea per sentence, and advises writers to choose active or passive voice according to the actual topic rather than following a mechanical rule. An empirical PLOS review likewise treats plain language, focus, and restraint with modifiers as common advice, while warning that no universal stylistic formula performs the same way across disciplines. A study comparing Science *Perspectives* with GPT-3.5 outputs found that the sampled human scientists used more varied sentence lengths, more numbers, and more qualifying causal language. The authors explicitly caution that these patterns came from one scientific genre and are not a universal detector. The useful lesson is not to imitate surface markers; it is to recover the variation and qualification produced by real reasoning.

Sources: [Nature: Effective Writing](https://www.nature.com/scitable/topicpage/effective-writing-13815989/), [PLOS Computational Biology: Ten Simple (Empirical) Rules for Writing Science](https://doi.org/10.1371/journal.pcbi.1004205), and [Desaire et al.: Distinguishing academic science writing from humans or ChatGPT](https://pmc.ncbi.nlm.nih.gov/articles/PMC10328544/).

### 1. Begin with the observation, not a ready-made message

Generated copy often begins with a polished conclusion and then looks for evidence to place beneath it. A researcher usually begins with an observation, defines its conditions, and only then interprets it.

**AI-like:** “The best conversationalists aren't the best thinkers.”

**Researcher-like:** “Moshi produced 3.22 backchannels per conversation but answered 2 of 30 reasoning items correctly. GPT-Realtime produced 0.50 backchannels and answered 20 of 30 reasoning items.”

The second version gives the reader the evidence before asking them to accept a relationship between the two dimensions.

### 2. Show who made the decision and why

Human researchers remember the choices that made the experiment possible: why a baseline was included, why a timing window was chosen, why one metric was kept separate from another, and what alternative was rejected. Generated prose tends to erase those decisions behind phrases such as “a comprehensive framework was developed.”

Prefer:

> “We report timing and reasoning separately because combining them would hide systems that perform well on one dimension and poorly on the other.”

Over:

> “A comprehensive two-axis framework enables holistic evaluation.”

The first sentence contains an author, a decision, and a reason. The second contains only favorable abstractions.

### 3. Use the nouns of the experiment

Researcher voice is specific because the researcher has handled the material. Use the actual objects and operations: prerecorded user audio, trigger word, onset window, separate tracks, ASR timestamp, overlap interval, target answer, and item-level verdict.

Replace broad abstractions with observable units:

| Avoid | Prefer |
|---|---|
| intelligence | reasoning-item accuracy |
| conversational ability | backchannel frequency, floor-taking rate, or interruption handling |
| performs naturally | waits through the mid-sentence pause or gives a short acknowledgement |
| robust evaluation | the same audio input and scoring pipeline across six systems |
| significant improvement | the measured change, denominator, and comparison condition |

Specific nouns do more than make the prose credible. They let a skeptical reader reconstruct the claim.

### 4. Separate result, interpretation, and implication

These are three different statements and should usually remain distinguishable.

- **Result:** “The turn-based cascade answered 17 of 30 reasoning items correctly.”
- **Interpretation:** “Its language-model component therefore appears competitive with the hosted systems on these short reasoning tasks.”
- **Implication:** “The cascade's poor conversational timing should not be attributed to answer generation alone.”

Generated copy often compresses all three into a slogan such as “the half-duplex tax.” Keeping them separate exposes the reasoning and makes disagreement possible.

### 5. Qualify the claim at the point of use

A limitations section is necessary, but it should not carry the entire burden of caution. Put the relevant boundary beside the claim:

- “Among the six systems evaluated here…”
- “On the three spoken reasoning scenarios…”
- “With one English TTS voice…”
- “Under this endpointing configuration…”
- “The result suggests…”

Good qualification is not vague hedging. It states which part of the experimental design limits the inference.

### 6. Let sentence length follow the reasoning

Machine-generated pages often settle into a uniform rhythm: short heading, short setup, three balanced points, concluding slogan. Human technical prose varies because some ideas require a direct sentence and others require a condition, contrast, or explanation.

Use a short sentence for the observation. Follow it with a longer sentence when the interpretation needs conditions. Do not manufacture variety with punctuation, and do not turn sentence-length variation into another detector checklist.

### 7. Prefer causal transitions to decorative transitions

Delete transitions that merely announce that prose is continuing:

- “It is important to note that…”
- “In today's rapidly evolving landscape…”
- “Moreover…” when the relationship is not additive
- “This underscores the importance of…”
- “Taken together, these findings highlight…”

Use words that name the actual relationship:

- **because** for cause;
- **but** or **however** for a conflicting observation;
- **therefore** only when the inference follows;
- **by contrast** for a defined comparison;
- **under this condition** for scope.

If no relationship needs to be stated, begin with the subject.

### 8. Do not make every paragraph perfectly symmetrical

Generated copy favors threes, parallel clauses, mirrored contrasts, and repeated “not X but Y” constructions. Researchers use parallel structure when the underlying experiment is parallel—not simply because it sounds finished.

Allow one result to require two sentences and another to require five. Allow an exception to interrupt the pattern. If a paragraph can be rearranged without changing its logic, it probably has a list-shaped structure rather than an argument.

### 9. Include the awkward or unresolved observation

Human researchers notice results that do not fit the clean story. For InteractionBench, useful examples include the grammar-repair task combining content and interruption timing, negative turn-latency values requiring interpretation, and differing item counts across some system-task pairs. These details should be explained, not hidden, because they show that the analysis followed the data rather than a predetermined narrative.

Before publication, add a sentence explaining every metric or denominator that a careful reader could reasonably find surprising.

### 10. Use AI as an editor, not as the source of scientific judgment

When using a language model in the writing process, give it researcher-authored material first:

1. The exact claim and its denominator
2. The relevant table or item-level evidence
3. The intended scope
4. The caveat or alternative explanation
5. A sample paragraph written by the authors

Ask the model to improve clarity while preserving those elements. Do not ask it to “make this sound professional” without supplying the underlying judgment; that prompt tends to produce generic significance claims, smooth transitions, and uniform cadence.

Afterward, the authors should perform three passes:

- **Evidence pass:** Can every quantitative or comparative statement be traced to a result?
- **Inference pass:** Is interpretation visibly separate from observation?
- **Voice pass:** Would an author say this sentence while explaining the result to another researcher?

### A compact researcher-voice test

For every paragraph, ask:

1. What fact, decision, or inference is new here?
2. Which noun names the actual thing measured?
3. Where is the evidence or mechanism?
4. What condition limits the statement?
5. Could the final sentence be deleted without losing information? If so, it is probably a generated-sounding summary.

The goal is not to evade an AI detector. Detectors are model- and genre-dependent, and human writing can be misclassified. The goal is prose that carries the structure of the research: observation, decision, evidence, inference, and uncertainty.

## House style for future edits

1. **Prefer observable behavior to psychological shorthand.** Write “answered 20 of 30 items correctly,” not “is smarter.”
2. **Keep the unit of analysis visible.** Say system, scenario, item, turn, recording, response, or event.
3. **Scope every comparative claim.** Use “in this evaluation” when a reader might otherwise interpret a universal claim.
4. **Put evidence next to interpretation.** Do not make readers scroll from a conclusion to the table that supports it.
5. **Use one strong idea per heading.** A heading should orient the reader, not behave like an advertisement.
6. **Limit slogan constructions.** Avoid repeated “X, not Y,” “the X tax,” “can X do Y?”, and groups of three dramatic conclusions.
7. **Use em dashes sparingly.** Prefer periods and causal transitions when the relationship deserves explanation.
8. **Separate method from promotion.** Automation is a design property; explain what it enables and where it can fail.
9. **Treat audio as evidence.** Introduce what the listener should notice, then let the clip demonstrate it.
10. **Run a consistency pass.** Verify every score, denominator, system label, author name, URL, and citation title before release.

## Changes applied to the page

- Replaced the former identity with InteractionBench by AveraLabs throughout visible copy and metadata.
- Adopted the title “InteractionBench: Measuring reasoning and conversational timing in real-time voice systems.”
- Rebuilt the opening as a problem-led overview.
- Replaced “IQ,” “intelligence,” and anthropomorphic result language with measured constructs.
- Added intuitive scenario, expectation, and listening guidance for all eight scenario types.
- Preserved the direct six-system audio comparison and interactive timelines.
- Rewrote result and analysis headings in scoped, evidence-led language.
- Corrected the turn-based cascade reasoning total to 17/30.
- Replaced the misleading “no LLM in the loop” claim with a precise description of generation, deterministic event specification, alignment, and LLM judging.
- Added limitations, acknowledgements, and BibTeX citation sections.
- Removed the former repository link and temporarily pointed to the AveraLabs GitHub organization. Replace it with the work-specific URL when the InteractionBench repository exists.

## Reference pages

- [𝜏-voice: benchmarking real-time voice agents on real-world tasks](https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks)
- [𝜏-bench leaderboard](https://taubench.com/leaderboard?benchmark=voice)
- [DeepSWE: Measuring frontier coding agents on original, long-horizon engineering tasks](https://deepswe.datacurve.ai/blog/deepswe)