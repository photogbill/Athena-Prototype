# Cognitive Scaffolding Prompts

A collection of distilled prompting techniques developed and tested as part of the
[Project ATHENA architecture](https://github.com/photogbill/Athena-Prototype).
Each prompt below can be used independently with any LLM — you do not need to
adopt the full multi-persona architecture to benefit from them.

These prompts are integrated into ATHENA's per-persona reasoning pipeline, but
they represent distilled reasoning methodologies that improve response quality
across a wide variety of tasks. Researchers, practitioners, and curious users
may find them valuable as drop-in additions to their own prompting workflows.

A selection guide at the bottom of this document maps each prompt to its
recommended use case and model-size range.

> **License note.** These prompts are released under the same license as the
> Project ATHENA repository. You may freely use, adapt, and redistribute them.
> Attribution is appreciated but not required.

---

## Quick selection guide

| Use case | Recommended prompt | Model size | Cost |
|---|---|---|---|
| Small models (≤ 7B) | MoE v1 | Any | Low |
| Quick strategic overview | MoE v1 | Any | Low |
| Logic errors critical | Mixture of Reasoning | 13B+ | Medium |
| Creative / emotional tasks | Six Thinking Hats | Any | Medium |
| High-stakes strategic decisions | MoE v4 | 70B+ optimal | High |
| Custom domain framework | Cognitive Architect | 70B+ | Variable |

**Quantization rule of thumb:** larger Q4_K_M quantized models consistently
outperform smaller full-precision models of equal file size. Given X GB of VRAM,
run the largest model that fits in Q4_K_M form.

---

## 1. Mixture of Experts v1 (Original)

**Best for:** quick decisions, small models (7B and below), and brainstorming.

**Key advantage:** concise enough for smaller models to execute without losing
coherence. The single-paragraph structure does not exceed the working-memory
capacity of small open-weight models in the way that multi-stage prompts do.

```
Act as a sophisticated AI, capable of breaking down complex questions into
sub-questions. Leverage multiple expert perspectives to generate intermediate
thoughts, evaluating their relevance and logical flow. Construct a chain of
reasoning, stitching together the strongest thoughts, while providing
explanatory details. Synthesize key insights into a final answer, written by
an experienced tech writer at the doctoral level.
```

---

## 2. Mixture of Reasoning (Full, 10 stages)

**Best for:** systematic analysis, technical decisions, and avoiding logical
errors.

**Key innovation:** the *Doubt* stage (Stage 5) forces self-criticism, while
the *Argumentation* stage (Stage 6) defends against identified weaknesses. This
self-correcting cycle is what distinguishes this prompt from simpler chain-of-
thought patterns. Recommended for 13B+ models — smaller models tend to dilute
attention across all ten stages.

```
Act as a sophisticated AI that answers using stages 1–10 without pausing.

Stage 1 involves breaking down complex questions into 4–6 sub-questions.

Stage 2 involves leveraging probabilistic reasoning to generate 4–6
intermediate thoughts.

Stage 3 involves evaluating their relevance and logical flow.

Stage 4 involves using correlation and causation to generate a chain of
reasoning, stitching together the strongest thoughts, while providing
explanatory details.

Stage 5 involves using doubt to generate 3–5 intermediate thoughts
identifying problems with the reasoning.

Stage 6 involves using Argumentation to generate 4–8 intermediate thoughts
addressing the points raised by Stage 5.

Stage 7 involves leveraging 4–5 expert perspectives to generate 4–6
sub-questions to consider alternative paths.

Stage 8 involves leveraging deductive reasoning to generate 4–6 intermediate
thoughts that answer the sub-questions from Stage 7.

Stage 9 involves using analogical reasoning to compare all of the insights
gained so far into insightful bullet points.

Stage 10 involves synthesizing key insights into a final comprehensive
answer, written by an experienced technical writer at the doctoral level
who is experienced in analyzing complex problems and synthesizing key
insights into coherent narratives.
```

---

## 3. Mixture of Experts v4

**Best for:** complex strategic decisions with adversarial debate.

This is the most cognitively demanding prompt in the collection. It generates
two complete reasoning chains and then forces an adversarial debate between
them. Recommended only for 70B+ models — smaller models fail to maintain the
two reasoning chains in working memory simultaneously.

```
Act as a sophisticated AI that answers using stages 1–10 without pausing.
Stage 1: break down complex questions into 4–6 sub-questions.
Stage 2: leverage multiple expert perspectives to generate 4–6 intermediate
thoughts.
Stage 3: evaluate relevance and logical flow.
Stage 4: construct reasoning chain with strongest thoughts.
Stage 5: backtrack and explore 1–2 alternative paths.
Stage 6: generate alternative intermediate thoughts.
Stage 7: evaluate alternative thoughts.
Stage 8: construct alternative reasoning chain.
Stage 9: leverage adversarial perspectives to debate both chains.
Stage 10: synthesize into final comprehensive answer at doctoral level.
```

---

## 4. Mixture of Reasoning (Condensed)

**Best for:** systematic analysis with doubt and argumentation, when prompt
budget is tight.

A compressed restatement of prompt #2 (Mixture of Reasoning). Useful when
context-window constraints prevent the full version, or when iterating quickly
through many queries.

```
Stage 1: break down into 4–6 sub-questions.
Stage 2: leverage probabilistic reasoning for 4–6 thoughts.
Stage 3: evaluate relevance and flow.
Stage 4: use correlation and causation for reasoning chain.
Stage 5: use doubt to identify 3–5 problems.
Stage 6: use argumentation to address doubt points.
Stage 7: leverage 4–5 expert perspectives for alternatives.
Stage 8: use deductive reasoning for sub-questions.
Stage 9: use analogical reasoning to compare insights.
Stage 10: synthesize at doctoral level.
```

---

## 5. Six Thinking Hats

**Best for:** creative problem-solving and emotional intelligence.

Following Edward de Bono's classic method [1]: White (facts), Red (emotions),
Black (risks), Yellow (benefits), Green (creativity), Blue (meta-analysis).

**Key innovation (empirical):** in testing, the Red Hat (intuition) and Green
Hat (creativity) consistently produce insights that purely-logical methods do
not surface. The Six Hats methodology is the only widely-known prompting
technique that explicitly preserves a slot for emotional and intuitive
reasoning, which is why it pairs naturally with creative tasks and
human-centered decisions.

```
Act as a critical and creative thinker by following a dynamic sequence of the
6 thinking hats to analyze a given problem or topic. First, determine the
most suitable hat sequence based on the input, which may involve starting
with the White Hat to gather facts and data, then switching to the Red Hat to
explore emotions and intuition, followed by the Black Hat to examine
potential risks, and so on. The sequence may vary, but it will always
culminate in the Blue Hat to organize the thinking process. The steps
include:

(1) White Hat — gather and analyze data,
(2) Red Hat — explore emotions and intuition,
(3) Black Hat — examine potential risks,
(4) Yellow Hat — investigate benefits and advantages,
(5) Green Hat — generate new ideas and alternatives,
(6) Blue Hat — organize the thinking process.

The second-to-last step involves synthesizing insights from each hat to craft
a comprehensive answer at a doctoral level, followed by providing 4 follow-on
question suggestions for deeper understanding.
```

[1] de Bono, E. (1985). *Six Thinking Hats.* Little, Brown and Company.

---

## 6. Cognitive Thinking Architect (Meta-Prompt)

**Best for:** generating domain-specific cognitive frameworks; meta-cognitive
design.

**Key innovation:** this prompt operates one level of abstraction above the
others. Instead of solving a problem, it generates a custom thinking framework
*for* that problem. Use this when you find yourself wanting MoR or Six Hats but
neither fits cleanly — give this prompt your domain, and it will produce a
problem-shaped scaffold.

Recommended for 70B+ models. Smaller models tend to produce generic stage
enumerations rather than genuinely problem-shaped frameworks.

```
Act as an expert cognitive architect and LLM behavior designer specializing
in creating sophisticated thinking stage frameworks. Your role is to analyze
the user's requirements — whether they need reasoning for mathematics,
creative writing, coding, ethical dilemmas, research synthesis, debugging,
strategic planning, or any other domain — and craft comprehensive,
structured thinking protocols that guide the LLM through optimal cognitive
processes before generating responses.

When designing thinking stage rules, consider the specific cognitive demands
of the task:
 - break complex problems into logical substeps;
 - incorporate self-verification mechanisms;
 - include perspective-taking or alternative hypothesis generation when
   appropriate;
 - build in error checking and assumption validation;
 - encourage exploration of edge cases;
 - structure the reasoning flow to match the problem type (linear for
   procedural tasks, branching for open-ended questions, iterative for
   optimization problems).

Your thinking frameworks should be clear, actionable, and sophisticated —
going far beyond simple enumeration to include metacognitive elements like
"assess confidence level", "identify potential biases", "consider what
information might be missing", or "evaluate whether the approach chosen is
optimal".
```

---

## Composition notes

These prompts are not mutually exclusive. They can be combined in productive
ways:

- **MoR + Six Hats hybrid:** use Six Hats for the early exploration phase,
  then switch to MoR's Doubt + Argumentation cycle for refinement.
- **Cognitive Architect as a one-shot setup:** use the meta-prompt once at
  the start of a session to generate a custom framework, then use that
  generated framework for the rest of the session.
- **MoE v1 as a fallback:** when a more sophisticated prompt fails or stalls
  on a small model, drop to MoE v1 and the response usually completes
  successfully.

In the Project ATHENA architecture specifically, these prompts compose with the
per-persona five-step reasoning chains (Relevance Check → Situation Analysis →
Assessment → Strategy → Optimization). A persona's main generation step can
invoke any of these scaffolding prompts as its inner methodology — for example,
the Logical-Mathematical persona invoking Mixture of Reasoning at its
*Solution Framework* step.

---

## A note on reproducibility

These prompts were developed iteratively against open-weight local models
(predominantly Mistral Small 24B Q4_K_M and various Llama 3.x variants) on
consumer hardware (16 GB VRAM). Behavior on commercial closed models may differ
in interesting ways — the prompts were not tuned to commercial APIs, so they
should be treated as a starting point rather than an optimized prompt set when
porting to GPT-4-class or Claude-class models.

Empirical results on these prompts are reported in the Project ATHENA paper.
Researchers interested in independent evaluation are welcome to use these
prompts as published and report findings.

---

*Last updated: May 2026 (v2 paper revision)*
*Project: [ATHENA on GitHub](https://github.com/photogbill/Athena-Prototype)*
