# Project ATHENA

**Architecture for Theoretically Holistic Expert Networked Analysis**

A decentralized AI cognitive architecture grounded in Howard Gardner's theory
of multiple intelligences. Eight specialist persona "intelligences" process
queries in **strict cognitive isolation**; a Composer integrates them into a
unified response; a Constitutional persona reviews every synthesis with
word-boundary risk assessment and persistent statistics. Each persona can run
its own model and slowly evolve in its own direction through use.

Implemented as a LoLLMs function-call plugin. Open source, local-first, runs
on consumer hardware (16 GB VRAM minimum).

📄 [**Research paper**](athena_arxiv_paper_v4.pdf) — full architectural specification (v4, May 2026, 52 pp)
📘 [**White paper**](athena_white_paper_v3.pdf) — accessible overview for contributors and collaborators (10 pp)
🧪 [**Regression tests**](test_athena.py) — 29 architectural-commitment tests

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
  optionally be backed by a different open-weight model (e.g. DeepSeek-Math
  for Logical-Mathematical, Codestral for Bodily-Kinesthetic), with a
  `max_active_models` parameter controlling GPU-memory concurrency. Backward
  compatible — defaults to a uniform shared model.
- **PEFT sleep-cycle training scaffolding.** A `PEFTTrainer` class with method
  dispatch across five hardware tiers (PiSSA-QLoRA, QDoRA, FSDP-QDoRA, GaLore,
  Q-GaLore). Dataset assembly, anchor-set mixing, validation gating, and
  adapter-accumulation strategy are fully implemented; the training loops
  themselves are clearly-marked stubs awaiting contributor hardware.
- **Constitutional layer that learns and persists.** Word-boundary regex
  matching (no more `"harm"` matching `"harmless"`), fail-deny on internal
  exception, principle stats that survive restarts, principle evolution from
  observed patterns.
- **Honest about uncertainty.** When the Composer detects high confusion
  across specialist outputs, it explicitly says so rather than producing a
  smoothed-over false consensus.
- **Designed for long-term divergent evolution.** Per-persona dream-fragment
  exports stage data for per-persona fine-tuning. Over months, each
  intelligence develops its own specialization trajectory.

---

## Architecture at a glance

![Basic per-query processing flow](figure1_basic_flow.png)

A user query enters through `ProjectATHENA.update_context`. Special-command
parsing routes `feedback:`, sleep cycles, and explainability triggers
separately. The Orchestrator computes a `CognitiveState`, picks 1–3
specialist personas based on cognitive affinity (or a model-routed
selection), and dispatches them in isolation. The Composer integrates with
self-RAG and conversation memory. The Constitutional layer reviews the full
output (no truncation) and fails-deny on internal error.

See the research paper for the full architecture diagram, the per-persona
evolution loop, and the temporal divergence trajectory.

---

## Quick start

### Requirements

- **Python** ≥ 3.10
- **LoLLMs** — install via the
  [LoLLMs WebUI](https://github.com/ParisNeo/lollms-webui)
- **A GGUF model** (recommended baseline: Mistral Small 24B Q4_K_M, ~15.4 GB,
  fits on 16 GB VRAM)
- **lollms_client** — required only for per-persona heterogeneous models.
  `pip install lollms_client`
- **PEFT stack** (`torch`, `peft`, `transformers`, `bitsandbytes`) — required
  only if you enable sleep-cycle LoRA/DoRA training.

### Drop-in install

This is a LoLLMs function-call extension. Place the contents of this repo
inside your LoLLMs custom function-calls folder:

```
<lollms-data-root>/personal_data/custom_function_calls/Athena/
├── function.py                       # the complete architecture (~5,570 lines)
├── config.yaml                       # the lollms-UI configuration surface (74 parameters)
├── test_athena.py                    # regression test harness (29 tests)
├── probe_lollms_swap.py              # per-persona model swap diagnostic
├── cognitive_scaffolding_prompts.md  # standalone reusable prompts
├── README.md
├── athena_arxiv_paper_v4.pdf         # research paper
└── athena_white_paper_v3.pdf         # accessible overview
```

In the LoLLMs UI, enable the `athena_engine` function call. The default
configuration runs out of the box with a single shared model.

### First run

With defaults, all eight personas share your default LoLLMs model. Send a
query — the Orchestrator routes to 1–3 relevant intelligences, each processes
it in isolation, and the Composer delivers a unified response. Watch the
LoLLMs log output for routing decisions and confusion-state detection.

---

## Per-persona heterogeneous models

Once baseline operation is validated:

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

## Sleep-cycle PEFT training (scaffolding)

ATHENA is designed so each persona slowly evolves through use. Every day's
significant interactions, unresolved tensions, and emergent curiosities are
consolidated — during a "sleep cycle" — into a per-persona training dataset.
Periodically, that dataset is used to fine-tune the persona's model.

The `PEFTTrainer` class implements the full machinery *around* training —
dataset assembly from per-persona memory, anchor-set mixing to prevent
catastrophic forgetting, a validation gate, and adapter-accumulation strategy.
The five training loops themselves are **clearly-marked scaffolding stubs**:
each comes with a canonical reference-implementation sketch in its docstring,
ready for a contributor with appropriate hardware to fill in.

| `peft_method` | Hardware tier | Notes |
|---|---|---|
| `pissa_qlora` | ~16 GB VRAM (commodity) | PiSSA-initialized QLoRA — adapter training |
| `qgalore` | ~16 GB VRAM, 7-8B persona models | INT4-quantized GaLore — *full* fine-tuning |
| `qdora` | 24–48 GB VRAM | Quantized DoRA |
| `fsdp_qdora` | Multi-GPU | FSDP-distributed QDoRA |
| `galore` | Single 80+ GB GPU | Full fine-tuning via gradient low-rank projection |

All 28 PEFT parameters — method, rank, learning rate, target modules,
quantization details, optimizer, anchor-set path, validation threshold, and
more — are exposed in the LoLLMs UI with tuned defaults. Contributors
implementing a training stub do not need to edit `function.py` to expose
tunables; values come from config.

Sleep-cycle training is **disabled by default** (`enable_lora_training:
false`). When disabled, the sleep cycle still exports dream-fragment JSON for
external processing.

---

## Verification: the test harness

`test_athena.py` runs 29 structural tests covering the architecture's
distinctive commitments. It does not require a live LoLLMs install — it
uses minimal shims for the lollms imports so it runs in seconds:

```bash
python test_athena.py
```

The tests verify, among other commitments:

- Module loads cleanly under shim conditions
- All architectural constants exist with sensible ranges
- All eight memory types present including `META_INTROSPECTION`
- **Cognitive isolation enforced by default** (standard mode passes
  `shared_context=None`)
- Constitutional risk assessment uses **word-boundary** regex
- Model-graded JSON parser robust to markdown fences, prose preamble,
  malformed input
- Conversation memory extraction handles missing / malformed message shapes
- Routing explanation populated in **all three paths** (manual, LLM,
  fallback)
- `principle_stats` table created for persistent constitutional state
- Bodily-Kinesthetic gates on domain classification (no unconditional code
  emission)
- Meta-introspection filtered from analytical RAG by default
- High-confidence engagements raise curiosity satisfaction; low-confidence
  ones do not
- v3 per-persona model dispatch falls back gracefully when libraries absent
- v3.1 PEFT optional imports, `PEFTTrainer` class surface, and all 27+
  PEFT config flags wired through ConfigTemplate

Each test is the architecture's contract layer: a future change that
violates any of these commitments will be caught.

---

## Documentation

| File | Contents |
|---|---|
| [athena_arxiv_paper_v4.pdf](athena_arxiv_paper_v4.pdf) | Research paper (v4, May 2026, 52 pp). Full architecture description, validation, deployment recipes, complete appendices including verbatim persona system prompts. |
| [athena_white_paper_v3.pdf](athena_white_paper_v3.pdf) | White paper (10 pp). Accessible, vision-forward overview for contributors and collaborators. |
| [athena_arxiv_paper_v4.md](athena_arxiv_paper_v4.md) | Editable Markdown source of the research paper. |
| [athena_white_paper_v3.md](athena_white_paper_v3.md) | Editable Markdown source of the white paper. |
| [cognitive_scaffolding_prompts.md](cognitive_scaffolding_prompts.md) | Standalone prompts (MoE v1, MoR, Six Thinking Hats, etc.) usable independently of the architecture. |
| `figure1_basic_flow.{dot,png,svg,pdf}` | Per-query processing flow |
| `figure2_full_architecture.{dot,png,svg,pdf}` | Full architecture with evolutionary training |
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

The full 74-parameter configuration surface is documented in Appendix B of
the research paper. Key flags:

| Setting | Default | Description |
|---|---|---|
| `operation_mode` | standard | standard / collaborative / adversarial |
| `enable_constitutional_persona` | true | Ethical oversight layer |
| `enable_persona_cross_visibility` | false | When True, legacy chained-context mode; default (False) enforces cognitive isolation |
| `enable_model_graded_judgments` | true | LLM judgment vs. heuristic confidence/uncertainty signals |
| `enable_composer_self_rag` | true | Composer retrieves its own past syntheses |
| `enable_conversation_memory` | true | Composer sees last 3 dialogue turns |
| `enable_routing_transparency` | false | Show routing rationale in response footer |
| `enable_per_persona_models` | false | **MASTER SWITCH** for heterogeneous per-persona model deployment |
| `max_active_models` | 1 | GPU-memory concurrency (1 = sequential swap) |
| `enable_lora_training` | false | **MASTER SWITCH** for sleep-cycle PEFT training |
| `peft_method` | pissa_qlora | One of: disabled / pissa_qlora / qdora / fsdp_qdora / galore / qgalore |
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
| `feedback: negative ...` | Record negative feedback (higher severity) |
| `feedback: ...` | Defaults to neutral |
| `explain yourself`, `walk me through`, `show your work`, etc. | Meta-introspection workflow |
| `trigger sleep cycle` | Run dream consolidation; export per-persona JSON; run PEFT training if enabled |
| `begin dream consolidation` | Alias for sleep cycle |

---

## Philosophy

ATHENA is grounded in a **relationship paradigm** of AI development: that
alignment and capability emerge from cognitive diversity and structured
relationship rather than from external constraint and uniformity.

Practically, this means:

- The Constitutional Persona functions as integrated ethical intelligence,
  not external filter.
- The architecture is honest about uncertainty rather than papering it over.
- Per-persona evolution + heterogeneous starting models is designed to
  produce **genuinely divergent specialist intelligences** over months and
  years — not just stylistic variation.

The white paper develops this framing accessibly; the research paper
develops it rigorously (§1.1, §3.15, §10).

---

## How to contribute

ATHENA is built so contributors can plug into a stable API rather than
needing to understand the whole system.

- **Have training hardware?** The highest-value contribution is implementing
  one of the five PEFT training stubs in `PEFTTrainer`. Each is a single
  method with a canonical reference sketch in its docstring, independent of
  the other four. Filling in even one tier closes the evolution loop
  empirically.
- **Want to stress-test?** Install ATHENA as a LoLLMs extension, run real
  queries, report where the architecture behaves unexpectedly. Run
  `probe_lollms_swap.py` to validate the per-persona machinery on your
  hardware.
- **Researcher?** The architecture makes empirically testable claims about
  cognitive diversity, heterogeneous-model deployment, and per-persona
  evolution. The research paper details them. Independent evaluation is
  welcome.

See the white paper's "How you can get involved" section for the full
contributor on-ramp.

---

## Citation

```bibtex
@misc{duncan2026athena,
  author = {Duncan, William R.},
  title  = {Project ATHENA: A Multi-Persona Cognitive Architecture for
            Transparent and Explainable Artificial Intelligence Grounded
            in Gardner's Theory of Multiple Intelligences},
  year   = {2026},
  note   = {Version 4 (May 2026)},
  url    = {https://github.com/photogbill/Athena-Prototype}
}
```

---

## Acknowledgments

ATHENA is built on the [LoLLMs framework](https://github.com/ParisNeo/lollms)
by ParisNeo, including the `llama_cpp_server` binding and `max_active_models`
capability that support per-persona heterogeneous deployment. The PEFT
training scaffolding draws on QLoRA, DoRA, PiSSA, GaLore, and Q-GaLore (see
the research paper's references). The Unsloth project enables efficient
QLoRA training on consumer hardware. Architecture diagrams are generated with
Graphviz from the DOT source files in this repository.

The theoretical foundation is Howard Gardner's *Frames of Mind* (1983) and
*Multiple Intelligences: New Horizons* (2006).

---

## License

[Specify your license here — MIT / Apache 2.0 / CC-BY are common choices for
research code with a paper.]

---

## Contact

Issues and discussion welcome on the GitHub repository. For research
collaboration or substantive architectural questions, contact the author at
william.r.duncan@hotmail.com.

If you build something on top of ATHENA or use the cognitive scaffolding
prompts in your own work, attribution is appreciated but not required.

---

*Current version: v4 (May 2026) — audited cognitive architecture, enforced
cognitive isolation, model-graded judgments, Composer self-RAG,
per-conversation memory, persistent constitutional statistics, per-persona
heterogeneous model support, five-method PEFT training scaffolding, and a
29-test regression harness.*
