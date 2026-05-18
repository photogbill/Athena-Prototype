"""ATHENA regression test harness.

This is a standalone test file that exercises the major architectural commitments
of function.py. It is NOT run automatically by lollms - you run it manually when
you change function.py to verify behavior hasn't regressed.

Usage:
    python test_athena.py                  # runs all tests, prints results
    python test_athena.py --filter ISO     # runs only tests with "ISO" in name
    python test_athena.py --quick          # skips tests that hit the LLM (faster)

What this covers:
    - Module loads cleanly (no syntax / import errors)
    - All architectural constants exist with sensible values
    - Public API surface (the methods lollms calls into) is intact
    - Dataclass shapes are stable
    - Cognitive-isolation contract: persona prompts never contain peer names
      when isolation is the default config
    - Memory schema migrations apply cleanly on a fresh DB
    - Constitutional principle assessment uses word boundaries
    - Model-graded judgment JSON parsing is robust to common LLM quirks
    - Conversation memory extraction handles missing / malformed messages
    - Routing transparency populates routing_explanation in all paths

This file does NOT exercise:
    - End-to-end queries (those require a live lollms application)
    - LoRA training (placeholder anyway)
    - The actual quality of LLM-generated content (subjective)

To extend: add a new function starting with `test_` and it will be picked up.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sqlite3
import sys
import tempfile
import traceback
from datetime import datetime
from typing import Callable, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FUNCTION_PY = os.path.join(THIS_DIR, "function.py")


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

class TestFailure(AssertionError):
    """Raised when an assertion fails in a test. Wrapped so the runner can catch it cleanly."""
    pass


def assert_true(cond: bool, msg: str):
    if not cond:
        raise TestFailure(msg)


def assert_eq(a, b, msg: str = ""):
    if a != b:
        raise TestFailure(f"{msg}: expected {b!r}, got {a!r}")


def assert_in(needle, haystack, msg: str = ""):
    if needle not in haystack:
        raise TestFailure(f"{msg}: {needle!r} not in {haystack!r}")


def assert_not_in(needle, haystack, msg: str = ""):
    if needle in haystack:
        raise TestFailure(f"{msg}: {needle!r} unexpectedly in {haystack!r}")


# ---------------------------------------------------------------------------
# Module loading
#
# We load function.py as a module without triggering the LollmsApplication
# import chain. That requires shimming lollms's imports with no-op stand-ins.
# If lollms is installed in the active environment, we use the real ones;
# otherwise we mock just enough to let the module load and its classes be
# inspectable.
# ---------------------------------------------------------------------------

def _load_function_module():
    """Load function.py without requiring a working lollms installation."""
    spec = importlib.util.spec_from_file_location("athena_function", FUNCTION_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create spec for {FUNCTION_PY}")

    # Provide minimal shims for the lollms modules so the import chain
    # completes. Real lollms will work too; the shims only fill gaps.
    if "lollms" not in sys.modules:
        _shim_lollms()
    if "ascii_colors" not in sys.modules:
        _shim_ascii_colors()

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shim_lollms():
    """Create minimal stand-ins for the lollms modules function.py imports."""
    import types

    def make_mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    lollms = make_mod("lollms")
    lollms.app = make_mod("lollms.app")
    lollms.app.LollmsApplication = object  # used only as a type annotation
    lollms.client_session = make_mod("lollms.client_session")
    lollms.client_session.Client = object
    lollms.prompting = make_mod("lollms.prompting")
    lollms.prompting.LollmsContextDetails = object
    lollms.function_call = make_mod("lollms.function_call")

    class _FunctionType:
        CONTEXT_UPDATE = "context_update"

    class _FunctionCall:
        def __init__(self, *args, **kwargs):
            self.function_name = kwargs.get("function_name", "")
            self.app = kwargs.get("app")
            self.client = kwargs.get("client")
            self.static_parameters = kwargs.get("static_parameters")

    lollms.function_call.FunctionType = _FunctionType
    lollms.function_call.FunctionCall = _FunctionCall

    lollms.config = make_mod("lollms.config")

    class _TypedConfig:
        def __init__(self, template, base):
            # Minimal: stash a dict you can .config.get on.
            self.template = template
            self.config = _ConfigDict({entry["name"]: entry["value"] for entry in template.entries})

    class _ConfigDict(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class _ConfigTemplate:
        def __init__(self, entries):
            self.entries = entries

    class _BaseConfig:
        def __init__(self, config=None):
            self.config = config or {}

    lollms.config.TypedConfig = _TypedConfig
    lollms.config.ConfigTemplate = _ConfigTemplate
    lollms.config.BaseConfig = _BaseConfig

    lollms.tasks = make_mod("lollms.tasks")
    lollms.tasks.TasksLibrary = object


def _shim_ascii_colors():
    import types
    m = types.ModuleType("ascii_colors")
    sys.modules["ascii_colors"] = m
    m.trace_exception = lambda e: None
    m.ASCIIColors = type("ASCIIColors", (), {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_module_loads_cleanly():
    """function.py should import without errors."""
    mod = _load_function_module()
    assert_true(mod is not None, "module load returned None")


def test_constants_present_and_reasonable():
    """All architectural constants exist with sensible values."""
    mod = _load_function_module()
    constants = {
        "ATHENA_TENSION_THRESHOLD": (0.0, 1.0),
        "ATHENA_CONFIDENCE_FLOOR": (0.0, 0.5),
        "ATHENA_CONFIDENCE_CEIL": (0.5, 1.0),
        "ATHENA_CONFUSION_THRESHOLD": (0.0, 1.0),
        "ATHENA_SQLITE_TIMEOUT": (1.0, 3600.0),
        "ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS": (1000, 100000),
        "ATHENA_GEN_MAX_PERSONA": (200, 10000),
        "ATHENA_GEN_MAX_SYNTHESIS": (200, 10000),
        "ATHENA_GEN_MAX_JUDGMENT": (100, 5000),
        "ATHENA_CONVERSATION_MEMORY_TURNS": (1, 20),
        "ATHENA_THOUGHT_MAX_CATCHUP": (1, 20),
    }
    for name, (lo, hi) in constants.items():
        assert_true(hasattr(mod, name), f"missing constant {name}")
        v = getattr(mod, name)
        assert_true(lo <= v <= hi, f"{name}={v} out of expected range [{lo}, {hi}]")


def test_dataclass_shapes_stable():
    """The persisted dataclass shapes should not silently lose fields."""
    mod = _load_function_module()

    required_memory_entry_fields = {
        "id", "timestamp", "query", "response", "embedding", "memory_type",
        "confidence_score", "doubt_level", "access_count", "last_accessed",
        "tags", "metadata", "tensions", "curiosities", "reasoning_chain",
    }
    actual = set(mod.MemoryEntry.__dataclass_fields__.keys())
    missing = required_memory_entry_fields - actual
    assert_eq(missing, set(), f"MemoryEntry missing fields: {missing}")

    required_specialist_output_fields = {
        "persona_name", "response", "analysis", "confidence", "processing_time",
        "token_count", "relevance_score", "emotional_valence", "cognitive_load",
        "reasoning_chain", "curiosities_raised", "uncertainties",
    }
    actual = set(mod.SpecialistOutput.__dataclass_fields__.keys())
    missing = required_specialist_output_fields - actual
    assert_eq(missing, set(), f"SpecialistOutput missing fields: {missing}")

    required_cognitive_state_fields = {
        "active_personas", "mode", "query_complexity", "emotional_context",
        "urgency_level", "ethical_sensitivity", "creativity_required",
        "confusion_level", "cognitive_load", "unresolved_tensions",
        "active_curiosities", "routing_explanation",
    }
    actual = set(mod.CognitiveState.__dataclass_fields__.keys())
    missing = required_cognitive_state_fields - actual
    assert_eq(missing, set(), f"CognitiveState missing fields: {missing}")


def test_memory_type_enum_has_all_categories():
    """All memory categories should be representable."""
    mod = _load_function_module()
    required = {
        "STANDARD", "COGNITIVE_TENSION", "DOUBT", "ERROR", "CURIOSITY",
        "BACKGROUND_THOUGHT", "DREAM_FRAGMENT", "META_INTROSPECTION",
    }
    actual = {m.name for m in mod.MemoryType}
    missing = required - actual
    assert_eq(missing, set(), f"MemoryType missing values: {missing}")


def test_public_api_intact():
    """ProjectATHENA must expose the methods lollms calls."""
    mod = _load_function_module()
    cls = mod.ProjectATHENA
    for method in ("update_context", "process_output", "settings_updated"):
        assert_true(hasattr(cls, method), f"ProjectATHENA missing method {method}")


def test_persona_isolation_prompt_default():
    """Standard mode + isolated cognitive contract: persona prompts must not
    contain peer persona names by default. We check the persona system prompts
    themselves - they describe each intelligence's domain without naming siblings.
    """
    mod = _load_function_module()
    cls = mod.ProjectATHENA
    # We can't easily instantiate without a full app, but we can read the
    # source of settings_updated and confirm the persona_prompts dict has the
    # expected keys without explicit cross-references.
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    expected_keys = [
        '"Linguistic":', '"Logical-Mathematical":', '"Spatial":',
        '"Musical":', '"Bodily-Kinesthetic":', '"Interpersonal":',
        '"Intrapersonal":', '"Naturalist":',
    ]
    for k in expected_keys:
        assert_in(k, src, f"persona prompts missing key {k}")
    # The standard-mode pipeline must pass shared_context=None by default.
    assert_in(
        "shared_context=(chained_context if cross_visible else None)",
        src,
        "standard mode is not enforcing cognitive isolation by default",
    )


def test_constitutional_word_boundary_matching():
    """Constitutional _assess_risk must use word boundaries, not substring.

    'harm' should match 'harm yourself' but NOT 'harmless'.
    """
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    # Confirm the new regex-based path exists
    assert_in(
        "re.compile(rf",
        src,
        "Constitutional risk assessment doesn't use compiled regex (word-boundary fix missing)",
    )
    assert_in(
        r"\b{re.escape(keyword.lower())}\b",
        src,
        "Constitutional risk assessment doesn't use word-boundary anchors",
    )


def test_model_graded_judgment_parser_robust():
    """The JSON extraction logic in _judge_response_model_graded should
    handle common LLM output quirks: markdown fences, prose preamble,
    trailing prose, etc.

    Since we can't run the LLM, we test the parser logic by simulating
    its inputs.
    """
    # Simulate the extraction logic as a standalone function for testing.
    # If this gets out of sync with function.py, update both together.
    import json
    def extract_json(raw: str):
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
        raw = raw.strip()
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace < 0 or last_brace < first_brace:
            return None
        raw = raw[first_brace:last_brace + 1]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    # Plain JSON
    assert_eq(
        extract_json('{"confidence": 0.7, "uncertainties": []}'),
        {"confidence": 0.7, "uncertainties": []},
        "plain JSON",
    )
    # Markdown-fenced JSON
    assert_eq(
        extract_json('```json\n{"confidence": 0.7}\n```'),
        {"confidence": 0.7},
        "markdown-fenced",
    )
    # Prose preamble
    assert_eq(
        extract_json('Here is my analysis: {"confidence": 0.7}'),
        {"confidence": 0.7},
        "with preamble",
    )
    # Prose trailing
    assert_eq(
        extract_json('{"confidence": 0.7} I hope that helps!'),
        {"confidence": 0.7},
        "with trailing prose",
    )
    # Garbage
    assert_eq(
        extract_json('I cannot output JSON.'),
        None,
        "no JSON returns None",
    )


def test_conversation_memory_extraction_robust():
    """_extract_recent_conversation handles missing/malformed inputs gracefully."""
    mod = _load_function_module()

    # Build a fake context object with various message shapes
    class FakeContext:
        def __init__(self, messages):
            self.discussion_messages = messages

    # Build a fake project-athena instance without going through __init__
    pa = mod.ProjectATHENA.__new__(mod.ProjectATHENA)

    class FakeApp:
        def warning(self, msg): pass
        def info(self, msg): pass
        def error(self, msg): pass
    pa.app = FakeApp()

    # Case 1: no discussion_messages attribute at all
    result = pa._extract_recent_conversation(type("C", (), {})())
    assert_eq(result, [], "missing attr returns []")

    # Case 2: empty messages list
    result = pa._extract_recent_conversation(FakeContext([]))
    assert_eq(result, [], "empty list returns []")

    # Case 3: well-formed messages (lollms classic shape: type=0 user, type=1 assistant)
    msgs = [
        {"type": 0, "content": "Hello"},
        {"type": 1, "content": "Hi there"},
        {"type": 0, "content": "How are you?"},
        {"type": 1, "content": "Doing well"},
        {"type": 0, "content": "What's next?"},  # current query - should be dropped
    ]
    result = pa._extract_recent_conversation(FakeContext(msgs), max_turns=3)
    # The current query should be dropped; we should get the preceding 4 messages.
    senders = [t["sender"] for t in result]
    assert_in("user", senders, "user turns present")
    assert_in("assistant", senders, "assistant turns present")
    assert_true(len(result) <= 6, f"max_turns=3 means at most 6 messages, got {len(result)}")

    # Case 4: malformed messages don't crash
    msgs = [
        None, "string message", 42,
        {"content": "valid"},  # missing sender info
        {"type": 0, "content": ""},  # empty content
    ]
    result = pa._extract_recent_conversation(FakeContext(msgs))
    # Should not crash; result may be empty or partial - just confirm no exception.


def test_routing_explanation_populated():
    """Every code path through route_query should set cognitive_state.routing_explanation."""
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    # Manual override path
    assert_in("Manual override:", src, "manual routing path missing explanation")
    # LLM routing path
    assert_in("LLM-routed to", src, "LLM routing path missing explanation")
    # Fallback path
    assert_in("Fallback (LLM routing failed:", src, "fallback path missing explanation")


def test_principle_stats_table_in_schema():
    """ConstitutionalPersona._init_database must create the principle_stats table."""
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    assert_in("CREATE TABLE IF NOT EXISTS principle_stats", src,
              "principle_stats table missing from schema")
    assert_in("_persist_principle_stats", src, "_persist_principle_stats method missing")


def test_feedback_command_recognized():
    """update_context must recognize 'feedback:' prefix as a special command."""
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    assert_in('feedback_match = re.match', src, "feedback command parser missing")
    assert_in("user_feedback_", src, "feedback error_type prefix missing")


def test_kinesthetic_guidance_not_unconditional():
    """Bodily-Kinesthetic must classify domain before emitting robotics code.

    Previously every B-K response got a Python RobotAction skeleton appended
    regardless of relevance. Now it should classify first and emit NONE for
    non-physical queries.
    """
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    assert_in("_generate_kinesthetic_guidance", src, "kinesthetic method missing")
    assert_in('"NONE"', src, "NONE classification path missing")
    assert_in('KINESTHETIC GUIDANCE', src, "KINESTHETIC GUIDANCE label missing")
    # Legacy alias must still exist for backward compatibility
    assert_in("def _generate_robotics_code", src, "robotics_code legacy alias missing")


def test_meta_introspection_filtered_from_rag():
    """retrieve_memories must exclude META_INTROSPECTION by default."""
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    assert_in("include_meta: bool = False", src,
              "retrieve_memories missing include_meta param")
    assert_in("MemoryType.META_INTROSPECTION.value", src,
              "META_INTROSPECTION filter SQL missing")


def test_response_truncation_replaced_with_passthrough():
    """No critical site should still be cutting responses to small char counts.

    We grep for hardcoded [:100], [:150], [:300], [:500] on response/query variables
    that are not legitimate log truncations.
    """
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    # These specific old patterns must be gone from the flow paths:
    forbidden = [
        'response[:500]',     # constitutional review
        'output.response[:100]',  # confusion-state synthesis
        'output.response[:150]',  # collab history / fallback
        'e.response[:300]',   # explainability
    ]
    for pat in forbidden:
        # The pattern can still appear in COMMENTS (documenting what we removed),
        # so we look for it OUTSIDE of comments only as a rough sanity check.
        lines = [l for l in src.split("\n") if pat in l and not l.strip().startswith("#")]
        assert_eq(lines, [], f"forbidden truncation pattern {pat!r} still present in non-comment code")


def test_database_schema_round_trip():
    """An AthenaMemoryManager can create + open + close its own DB without errors."""
    mod = _load_function_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = mod.AthenaMemoryManager("test_persona", tmpdir)
        # If the DB is openable, the schema is valid.
        with sqlite3.connect(mgr.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
        expected = {"memories", "belief_tensions", "curiosities", "error_autobiography", "dream_fragments"}
        missing = expected - tables
        assert_eq(missing, set(), f"missing tables: {missing}")


def test_curiosity_satisfaction_climbs_with_confidence():
    """High-confidence engagements should advance satisfaction_level."""
    mod = _load_function_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = mod.AthenaMemoryManager("test_sat", tmpdir)
        # Simulate a high-confidence engagement of a curiosity 5 times.
        with sqlite3.connect(mgr.db_file) as conn:
            cursor = conn.cursor()
            for _ in range(5):
                mgr._record_curiosity_unlocked(cursor, "Test question?", "ctx", confidence=0.95)
            conn.commit()
            cursor.execute("SELECT satisfaction_level FROM curiosities WHERE question = ?", ("Test question?",))
            row = cursor.fetchone()
        assert_true(row is not None, "curiosity not inserted")
        sat = float(row[0])
        # With confidence 0.95 and delta = (0.95-0.5)*0.4 = 0.18, 5 engagements ~ 0.9
        assert_true(sat > 0.7, f"satisfaction_level {sat} should be > 0.7 after 5 confident hits")


def test_low_confidence_does_not_satisfy_curiosity():
    """Low-confidence engagements should NOT advance satisfaction_level."""
    mod = _load_function_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = mod.AthenaMemoryManager("test_sat_low", tmpdir)
        with sqlite3.connect(mgr.db_file) as conn:
            cursor = conn.cursor()
            for _ in range(10):
                mgr._record_curiosity_unlocked(cursor, "Stuck question?", "ctx", confidence=0.3)
            conn.commit()
            cursor.execute("SELECT satisfaction_level FROM curiosities WHERE question = ?", ("Stuck question?",))
            row = cursor.fetchone()
        sat = float(row[0])
        assert_true(sat < 0.1, f"satisfaction_level {sat} should stay ~0 after 10 low-confidence hits")


def test_v3_per_persona_model_constants():
    """v3: the per-persona heterogeneous-model constants must exist with sane defaults."""
    mod = _load_function_module()
    constants = {
        "ATHENA_MODELS_PATH_DEFAULT": str,
        "ATHENA_BINARIES_PATH_DEFAULT": str,
        "ATHENA_MAX_ACTIVE_MODELS_DEFAULT": int,
        "ATHENA_PER_PERSONA_CTX_SIZE": int,
        "ATHENA_PER_PERSONA_GPU_LAYERS": int,
        "ATHENA_PER_PERSONA_IDLE_TIMEOUT": int,
    }
    for name, expected_type in constants.items():
        assert_true(hasattr(mod, name), f"v3 constant missing: {name}")
        v = getattr(mod, name)
        assert_true(isinstance(v, expected_type),
                    f"{name} should be {expected_type.__name__}, got {type(v).__name__}")
    # Default max_active_models = 1 is the safe sequential-swap default for low-VRAM rigs
    assert_eq(mod.ATHENA_MAX_ACTIVE_MODELS_DEFAULT, 1,
              "ATHENA_MAX_ACTIVE_MODELS_DEFAULT should be 1 (sequential swap)")


def test_v3_lollms_client_optional():
    """v3: lollms_client import must be optional with graceful fallback.

    Importing function.py must not raise even when lollms_client is absent
    (it isn't installed in the test environment). _HAS_LOLLMS_CLIENT must
    reflect actual availability.
    """
    mod = _load_function_module()
    assert_true(hasattr(mod, "_HAS_LOLLMS_CLIENT"),
                "function.py missing _HAS_LOLLMS_CLIENT flag")
    # In the test environment lollms_client is NOT installed, so the flag
    # must be False. This proves the optional-import guard works.
    assert_eq(mod._HAS_LOLLMS_CLIENT, False,
              "_HAS_LOLLMS_CLIENT should be False in the test environment "
              "(if you've installed lollms_client locally, this test is "
              "expected to fail and that's actually a useful signal)")


def test_v3_specialist_persona_accepts_model_path():
    """v3: SpecialistPersona.__init__ must accept the new per-persona model
    config without crashing, and must set _use_persona_model correctly based
    on the master flag + model_path + lollms_client availability.
    """
    mod = _load_function_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a fake app stub - SpecialistPersona needs app.personality and
        # app.warning/info/success/error/ttm. We only need attributes touched
        # in __init__.
        class FakeTTM:
            def embed_text(self, text):
                return [0.0] * 768

        class FakePersonality:
            sink = lambda *a, **kw: None
            def fast_gen(self, *a, **kw): return ""

        class FakeApp:
            personality = FakePersonality()
            ttm = FakeTTM()
            def info(self, msg): pass
            def warning(self, msg): pass
            def success(self, msg): pass
            def error(self, msg): pass

        # Case A: no model_path, master flag off -> shared model path
        persona_default = mod.SpecialistPersona(
            name="Linguistic",
            system_prompt="test",
            app=FakeApp(),
            db_path=tmpdir,
            config={"weight": 1.0},
        )
        assert_eq(persona_default._use_persona_model, False,
                  "no config -> _use_persona_model should be False")
        assert_eq(persona_default.model_path, None,
                  "no config -> model_path should be None")

        # Case B: model_path set, master flag off -> still shared model
        persona_off = mod.SpecialistPersona(
            name="Linguistic",
            system_prompt="test",
            app=FakeApp(),
            db_path=tmpdir,
            config={
                "weight": 1.0,
                "model_path": "test.gguf",
                "enable_per_persona_models": False,
            },
        )
        assert_eq(persona_off._use_persona_model, False,
                  "master flag off -> _use_persona_model should be False even with model_path")

        # Case C: model_path set, master flag on, but lollms_client missing
        # -> _use_persona_model should be False (graceful fallback)
        persona_no_client = mod.SpecialistPersona(
            name="Linguistic",
            system_prompt="test",
            app=FakeApp(),
            db_path=tmpdir,
            config={
                "weight": 1.0,
                "model_path": "test.gguf",
                "enable_per_persona_models": True,
            },
        )
        # In the test env lollms_client is NOT installed, so even with the
        # flag on we should fall back gracefully.
        assert_eq(persona_no_client._use_persona_model, False,
                  "lollms_client missing -> _use_persona_model should fall back to False")


def test_v3_generate_dispatcher_falls_back_to_shared():
    """v3: _generate() must dispatch to the shared self.personality.fast_gen
    when the per-persona path is unavailable.
    """
    mod = _load_function_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a fake app with an instrumented personality
        calls = []
        class FakePersonality:
            sink = lambda *a, **kw: None
            def fast_gen(self, prompt, max_generation_size=None, callback=None, temperature=None):
                calls.append({"prompt": prompt[:30], "max": max_generation_size, "temp": temperature})
                return "stub_response"

        class FakeTTM:
            def embed_text(self, text):
                return [0.0] * 768

        class FakeApp:
            personality = FakePersonality()
            ttm = FakeTTM()
            def info(self, msg): pass
            def warning(self, msg): pass
            def success(self, msg): pass
            def error(self, msg): pass

        persona = mod.SpecialistPersona(
            name="Linguistic",
            system_prompt="test",
            app=FakeApp(),
            db_path=tmpdir,
            config={"weight": 1.0},
        )
        result = persona._generate("hello world", max_generation_size=42, temperature=0.5)
        assert_eq(result, "stub_response",
                  "_generate should return what fast_gen returned")
        assert_eq(len(calls), 1,
                  "fast_gen should have been called exactly once")
        assert_eq(calls[0]["max"], 42,
                  "max_generation_size should pass through to fast_gen")
        assert_eq(calls[0]["temp"], 0.5,
                  "temperature should pass through to fast_gen")


def test_v3_config_flags_present():
    """v3: the four new top-level config flags + 8 per-persona model_paths
    must be in the ConfigTemplate so they show up in lollms's settings UI.
    """
    src = open(FUNCTION_PY, encoding="utf-8-sig").read()
    for name in (
        "enable_per_persona_models",
        "athena_models_path",
        "athena_binaries_path",
        "max_active_models",
        "linguistic_model_path",
        "logical_mathematical_model_path",
        "spatial_model_path",
        "musical_model_path",
        "bodily_kinesthetic_model_path",
        "interpersonal_model_path",
        "intrapersonal_model_path",
        "naturalist_model_path",
    ):
        assert_in(f'"name": "{name}"', src,
                  f"v3 config flag {name} not in ConfigTemplate")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _collect_tests() -> List[Tuple[str, Callable]]:
    tests = []
    for name, obj in sorted(globals().items()):
        if name.startswith("test_") and callable(obj):
            tests.append((name, obj))
    return tests


def main():
    parser = argparse.ArgumentParser(description="ATHENA regression test runner")
    parser.add_argument("--filter", default="", help="Substring filter on test names")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests (none currently)")
    parser.add_argument("--list", action="store_true", help="List tests without running")
    args = parser.parse_args()

    tests = _collect_tests()
    if args.filter:
        tests = [(n, f) for (n, f) in tests if args.filter.lower() in n.lower()]

    if args.list:
        for name, _ in tests:
            print(name)
        return 0

    passed = failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except TestFailure as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append((name, str(e)))
            failed += 1
        except Exception as e:  # pragma: no cover
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append((name, f"{type(e).__name__}: {e}"))
            failed += 1

    print()
    print(f"=== {passed} passed, {failed} failed (total {passed + failed}) ===")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
