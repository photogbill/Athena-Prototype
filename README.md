# Project ATHENA

**Architecture for Theoretically Holistic Expert Networked Analysis**

A decentralized AI cognitive architecture grounded in Howard Gardner's theory
of multiple intelligences. Eight specialist persona "intelligences" process
queries in **strict cognitive isolation**; a Composer integrates them into a
unified response; a Constitutional persona reviews every synthesis with
word-boundary risk assessment and persistent statistics.

Implemented as a LoLLMs function-call plugin. Open source, local-first, and
runs on consumer hardware (16 GB VRAM minimum).

📄 [**Read the paper**](athena_arxiv_paper_v3_ack_fixed.pdf) (v2 / May 2026, 51 pp)
🧪 [**Run the regression tests**](test_athena.py) (23 architectural-commitment tests)

---

## What makes ATHENA different

Most "multi-agent" LLM systems share state between agents — different prompts
on top of the same model, with context leaking freely between them. ATHENA
makes structural commitments that other systems claim but rarely enforce:

- **Genuine cognitive isolation.** Each persona has its own SQLite memory
  database, its own RAG retrieval, its own reasoning patterns. No persona
  reads another persona's data at any level during processing. The Composer
  is the *only* component that sees all eight perspectives. Verified by the
  regression test suite.
- **Model-graded judgments, not word counts.** Confidence, uncertainty,
  curiosity, tension, and emotional valence signals are produced by a
  consolidated JSON judgment call per persona response, replacing the lexical
  heuristics that early multi-persona systems rely on. Heuristic methods are
  retained as automatic fallback.
- **Eight memory categories** including a META_INTROSPECTION type that
  prevents self-explanations from polluting the analytical RAG corpus.
- **Per-persona heterogeneous models.** Each `SpecialistPersona` can
  optionally be backed by a different GGUF model (e.g. DeepSeek-Math for
  Logical-Mathematical, Codestral for Bodily-Kinesthetic), with a
  `max_active_models` parameter controlling VRAM concurrency. Backward
  compatible — defaults to uniform shared model.
- **Constitutional layer that learns and persists.** Word-boundary regex
  matching (no more `"harm"` matching `"harmless"`), fail-deny on internal
  exception, principle stats that survive restarts, principle evolution from
  observed patterns.
- **Honest about uncertainty.** When the Composer detects high confusion
  across specialist outputs, it explicitly says so rather than producing a
  smoothed-over false consensus. The acknowledged-uncertainty branch now
  persists to the dream-fragment pipeline for later consolidation.
- **Designed for long-term divergent evolution.** Per-persona dream-fragment
  exports stage data for future per-persona LoRA training. Over months, each
  intelligence develops its own specialization trajectory. The Composer's
  own self-RAG over past syntheses lets it scale to handle the increasing
  divergence between specialists.

---

## Architecture at a glance

![Basic per-query processing flow](figure1_basic_flow.png)

A user query enters through `ProjectATHENA.update_context`. Special-command
parsing routes `feedback:`, sleep cycles, and explainability triggers
separately. The Orchestrator computes a `CognitiveState`, picks 1–3
specialist personas based on cognitive affinity (or a model-routed
selection), and dispatches them in isolation. The Composer integrates with
self-RAG and conversation memory. The Constitutional layer reviews the full
output (no truncation) and fails-deny on internal error. See the paper for
the full architecture diagram and the per-persona evolution loop.

---

## Quick start

### Requirements

- **Python** ≥ 3.10
- **LoLLMs** — install via the
  [LoLLMs WebUI](https://github.com/ParisNeo/lollms-webui)
- **A GGUF model** (recommended baseline: Mistral Small 24B Q4_K_M, ~15.4 GB,
  fits on 16 GB VRAM)
- **lollms_client** — required only if you want per-persona heterogeneous
  models. `pip install lollms_client`

### Drop-in install

This is a LoLLMs function-call extension. Place the contents of this repo
inside your LoLLMs custom function-calls folder:

```
<lollms-data-root>/personal_data/custom_function_calls/Athena/
├── function.py
├── config.yaml
├── test_athena.py
├── probe_lollms_swap.py
├── cognitive_scaffolding_prompts.md
├── README.md
└── athena_arxiv_paper_v3_ack_fixed.pdf
```

In the LoLLMs UI, enable the `athena_engine` function call. The default
configuration runs out of the box with a single shared model.

### First run

With defaults (`enable_per_persona_models=false`), all eight personas share
your default LoLLMs model. Send a query — the Orchestrator routes to 1–3
relevant intelligences, each processes it in isolation, and the Composer
delivers a unified response.

Watch the LoLLMs log output for routing decisions and confusion-state
detection.

### Enabling per-persona heterogeneous models

Once you've validated baseline operation:

1. Drop GGUF files into your `athena_models_path`
   (default `data/models/llama_cpp_models`).
2. Set `enable_per_persona_models: true` in the function-call config.
3. Set specific `{persona}_model_path` entries (empty = use shared model).
4. Tune `max_active_models` to your VRAM budget (1 = sequential swap, safe
   on 16 GB; 2+ requires more VRAM but allows concurrent execution).

Suggested mapping at 16 GB VRAM (Mistral Small 24B as default, specialized
smaller models where they measurably beat general capacity):

| Persona | Model | Q4 size |
|---|---|---|
| Linguistic | Mistral Small 24B (default) | 15.4 GB |
| Logical-Mathematical | Qwen2.5-Math-7B | ~5 GB |
| Spatial | Pixtral 12B or Qwen2-VL 7B | ~5–8 GB |
| Musical | Mistral Small (default) | — |
| Bodily-Kinesthetic | Codestral 22B (robotics) / default (other) | 14 GB / — |
| Interpersonal | Mistral Small (default) | — |
| Intrapersonal | DeepSeek-R1-Distill-Llama-8B | ~5 GB |
| Naturalist | Mistral Small (default) | — |

Run `probe_lollms_swap.py` first to validate that lollms_client behaves as
expected on your hardware before relying on this in production:

```bash
python probe_lollms_swap.py
```

The probe verifies model loading, eviction at `max_active_models=1`, shared
registry visibility across multiple LollmsClient instances, and clean
unload.

---

## Verification: the test harness

`test_athena.py` runs 23 structural tests covering the architecture's
distinctive commitments. It does not require a live LoLLMs install — it
uses minimal shims for the lollms imports so it runs in seconds:

```bash
python test_athena.py
```

The tests verify:

- Module loads cleanly under shim conditions
- All architectural constants exist with sensible ranges
- Dataclass shapes are stable (no silent field loss)
- All eight memory types present including `META_INTROSPECTION`
- Public API surface intact (`update_context`, `process_output`,
  `settings_updated`)
- **Cognitive isolation enforced by default** (standard mode passes
  `shared_context=None`)
- Constitutional risk assessment uses **word-boundary** regex
- Model-graded JSON parser robust to markdown fences, prose preamble,
  malformed input
- Conversation memory extraction handles missing / malformed message shapes
- Routing explanation populated in **all three paths** (manual, LLM,
  fallback)
- `principle_stats` table created in schema for persistent constitutional
  state
- `feedback:` command parser recognized
- Bodily-Kinesthetic gates on domain classification (no unconditional code
  emission)
- Meta-introspection filtered from analytical RAG by default
- Response truncations replaced with passthrough constants
- DB schema round-trips cleanly on a fresh DB
- High-confidence engagements raise curiosity satisfaction
- Low-confidence engagements do NOT raise curiosity satisfaction
- v3 per-persona model constants present
- v3 lollms_client import is optional with graceful fallback
- v3 SpecialistPersona accepts per-persona model config
- v3 `_generate` dispatcher falls back to shared model when per-persona
  path unavailable
- v3 all config flags + 8 per-persona model_paths exposed in
  ConfigTemplate

Each test is the architecture's contract layer: a future change that
violates any of these commitments will be caught.

---

## Documentation

| File | Contents |
|---|---|
| [athena_arxiv_paper_v3_ack_fixed.pdf](athena_arxiv_paper_v3_ack_fixed.pdf) | Full paper draft (v2 / May 2026, 51 pp). Architecture description, validation, deployment recipes, full appendices. |
| [athena_arxiv_paper_v2.md](athena_arxiv_paper_v2.md) | Editable Markdown source of the paper. |
| [cognitive_scaffolding_prompts.md](cognitive_scaffolding_prompts.md) | Standalone prompts (MoE v1, MoR, Six Thinking Hats, etc.) usable independently of the architecture. |
| `figure1_basic_flow.{dot,png,svg,pdf}` | Per-query processing flow |
| `figure2_full_architecture.{dot,png,svg,pdf}` | Full v2 architecture with evolutionary training |
| `figure3_per_persona_stack.{dot,png,svg,pdf}` | Eight independent persona stacks |
| `figure4_evolution_loop.{dot,png,svg,pdf}` | Single-persona evolution recurrence |
| `figure5_temporal_scales.{dot,png,svg,pdf}` | Time-scales of memory + divergence trajectory |

Figures are Graphviz-generated from `.dot` source files. To re-render after
edits:

```bash
dot -Tpng -Gdpi=180 figureN_*.dot -o figureN_*.png
```

---

## Configuration reference

The full configuration surface is documented in Appendix B of the paper, but
key flags:

| Setting | Default | Description |
|---|---|---|
| `operation_mode` | standard | standard / collaborative / adversarial |
| `enable_constitutional_persona` | true | Ethical oversight layer |
| `enable_persona_cross_visibility` | false | When True, legacy chained-context mode; default (False) enforces cognitive isolation |
| `enable_model_graded_judgments` | true | LLM judgment vs. heuristic confidence/uncertainty signals |
| `enable_composer_self_rag` | true | Composer retrieves its own past syntheses |
| `enable_conversation_memory` | true | Composer sees last 3 dialogue turns |
| `enable_routing_transparency` | false | Show routing rationale in response footer |
| `enable_per_persona_models` | false | **MASTER SWITCH** for v3 heterogeneous deployment |
| `max_active_models` | 1 | VRAM concurrency (1 = sequential swap) |
| `confusion_expression_threshold` | 0.4 | Composer's confusion-state entry threshold |
| `final_output_format` | visual_dialogue | One of 10 output formats |

Each of the eight personas has `{persona}_enabled`, `{persona}_weight`, and
`{persona}_model_path` settings.

---

## Special commands

The architecture recognizes several special user-query patterns:

| Pattern | What it does |
|---|---|
| `feedback: positive ...` | Record positive feedback in error_autobiography |
| `feedback: negative ...` | Record negative feedback (severity 0.8) |
| `feedback: ...` | Defaults to neutral |
| `explain yourself`, `walk me through`, `show your work`, etc. | Meta-introspection workflow |
| `trigger sleep cycle` | Run dream consolidation; export per-persona JSON |
| `begin dream consolidation` | Alias for sleep cycle |

The full trigger list is in §6.3 of the paper.

---

## Philosophy

ATHENA is grounded in a **relationship paradigm** of AI development: that
alignment and capability emerge from cognitive diversity and structured
relationship rather than from external constraint and uniformity.

Practically, this means:

- The Constitutional Persona functions as integrated ethical intelligence,
  not external filter.
- The architecture is honest about uncertainty rather than papering it over.
- Per-persona LoRA evolution + heterogeneous starting models is designed to
  produce **genuinely divergent specialist intelligences** over months and
  years — not just stylistic variation.

The paper develops this philosophical framing more fully (§1.1, §3.15, §10).

---

## Citation

```bibtex
@misc{duncan2026athena,
  author = {Duncan, William R.},
  title  = {Project ATHENA: A Multi-Persona Cognitive Architecture for
            Transparent and Explainable Artificial Intelligence Grounded
            in Gardner's Theory of Multiple Intelligences},
  year   = {2026},
  note   = {Version 2 (May 2026)},
  url    = {https://github.com/photogbill/Athena-Prototype}
}
```

---

## Acknowledgments

ATHENA is built on the [LoLLMs framework](https://github.com/ParisNeo/lollms)
by ParisNeo, including the `llama_cpp_server` binding and
`max_active_models` capability he added specifically to support
per-persona heterogeneous deployment. The Unsloth project enables efficient
QLoRA training on consumer hardware. Architecture diagrams are generated
with Graphviz from the DOT source files in this repository.

The theoretical foundation is Howard Gardner's *Frames of Mind* (1983) and
*Multiple Intelligences: New Horizons* (2006).

---

## License

[Specify your license here — MIT / Apache 2.0 / CC-BY are common choices for
research code with a paper.]

---

## Contributing & contact

Issues and discussion welcome on the GitHub repository. For research
collaboration or substantive architectural questions, contact the author
directly at william.r.duncan@hotmail.com.

If you build something on top of ATHENA or use the cognitive scaffolding
prompts in your own work, attribution is appreciated but not required.

---

*Current version: v3 (May 2026) — per-persona heterogeneous-model support,
model-graded judgments, Composer self-RAG, per-conversation memory,
persistent constitutional statistics, regression test harness (23 tests).*
