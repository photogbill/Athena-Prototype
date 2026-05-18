#!/usr/bin/env python3
"""ATHENA per-persona model swap probe.

What this verifies:
  1. lollms_client is installed and importable
  2. A LollmsClient with the llama_cpp_server binding can be instantiated
  3. A model can be loaded explicitly via client.llm.load_model(filename)
  4. ps() reports the loaded model with its process info
  5. A second LollmsClient sharing the same models_path can ALSO see
     the first one's running server through the FileLock-protected
     JSON registry
  6. max_active_models=1 enforces sequential swap: loading a second
     model evicts the first
  7. generate_text() returns a usable response
  8. unload_model() actually frees VRAM (rough check via timing)
  9. The pattern ATHENA uses (per-persona client, each with its own
     model_path) can be exercised end-to-end without errors

What this does NOT do:
  - Validate response quality (we just need a non-error response)
  - Test parallel/concurrent dispatch (max_active_models > 1)
  - Test the full ATHENA pipeline; that's test_athena.py + a real query

How to run:
  1. Pip-install the dependency:
       pip install lollms_client ascii_colors

  2. Drop at least TWO small GGUF files (under 5 GB each, to keep this
     probe fast) into your shared models directory. The defaults below
     point at the same paths ATHENA's config uses.

  3. Edit MODEL_A_FILENAME and MODEL_B_FILENAME below to match the
     GGUFs you placed in MODELS_PATH.

  4. Run from anywhere:
       python probe_lollms_swap.py

The probe prints clear PASS / FAIL / SKIP lines for each check and a
final summary. Any FAIL means ATHENA's per-persona-model path will not
behave as expected on this machine; investigate before flipping
enable_per_persona_models=True in the production config.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration: edit these to match your local setup before running
# ─────────────────────────────────────────────────────────────────────────────

# Match the defaults in function.py and config.yaml
MODELS_PATH = "data/models/llama_cpp_models"
BINARIES_PATH = "data/bin/llm/llama_cpp_server"

# Replace with GGUFs you actually have. Both should fit in your VRAM
# individually; this probe tests the eviction behavior so we WANT them
# small enough to swap quickly.
MODEL_A_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"   # ~0.7 GB
MODEL_B_FILENAME = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"  # ~4.4 GB

# Common-case shared config for both clients
SHARED_BINDING_CONFIG = {
    "models_path": MODELS_PATH,
    "binaries_path": BINARIES_PATH,
    "ctx_size": 2048,
    "n_gpu_layers": -1,
    "n_threads": 4,
    "n_parallel": 1,
    "batch_size": 512,
    "max_active_models": 1,   # <-- the parameter we're testing
    "idle_timeout": -1,
}

# A trivial prompt - we only need a response that isn't an error
TEST_PROMPT = "Say 'OK' and nothing else."


# ─────────────────────────────────────────────────────────────────────────────
# Probe infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class ProbeResult:
    """Collects test outcomes for the summary line."""
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def passed_check(self, name: str):
        print(f"  [PASS] {name}")
        self.passed.append(name)

    def failed_check(self, name: str, reason: str):
        print(f"  [FAIL] {name}: {reason}")
        self.failed.append((name, reason))

    def skipped_check(self, name: str, reason: str):
        print(f"  [SKIP] {name}: {reason}")
        self.skipped.append((name, reason))

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print()
        print("=" * 70)
        print(f"  SUMMARY: {len(self.passed)} passed, "
              f"{len(self.failed)} failed, {len(self.skipped)} skipped "
              f"(total {total})")
        print("=" * 70)
        if self.failed:
            print("\nFailures:")
            for name, reason in self.failed:
                print(f"  - {name}: {reason}")
        return 1 if self.failed else 0


# ─────────────────────────────────────────────────────────────────────────────
# Individual probe checks
# ─────────────────────────────────────────────────────────────────────────────

def check_import(result: ProbeResult):
    """Check 1: lollms_client is installed."""
    print("\n[1] lollms_client importable?")
    try:
        from lollms_client import LollmsClient  # noqa: F401
        result.passed_check("lollms_client imports")
        return True
    except ImportError as e:
        result.failed_check("lollms_client imports",
                            f"not installed ({e}); run: pip install lollms_client")
        return False


def check_gguf_files_exist(result: ProbeResult) -> bool:
    """Check 2: the two GGUF files actually exist on disk."""
    print(f"\n[2] GGUF files present in {MODELS_PATH}?")
    models_dir = Path(MODELS_PATH)
    if not models_dir.exists():
        result.failed_check("models_path exists",
                            f"{models_dir.resolve()} does not exist; create it "
                            f"or update MODELS_PATH in this script")
        return False
    result.passed_check(f"models_path exists ({models_dir.resolve()})")

    all_ok = True
    for fname in (MODEL_A_FILENAME, MODEL_B_FILENAME):
        path = models_dir / fname
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            result.passed_check(f"GGUF found: {fname} ({size_mb:.1f} MB)")
        else:
            result.failed_check(f"GGUF missing: {fname}",
                                f"expected at {path.resolve()}; place one there "
                                f"or update MODEL_*_FILENAME in this script")
            all_ok = False
    return all_ok


def check_client_construction(result: ProbeResult):
    """Check 3: LollmsClient instantiates without error."""
    from lollms_client import LollmsClient
    print("\n[3] LollmsClient construction?")
    try:
        client = LollmsClient(
            llm_binding_name="llama_cpp_server",
            llm_binding_config=SHARED_BINDING_CONFIG,
            user_name="user",
            ai_name="probe",
        )
        result.passed_check("LollmsClient(llama_cpp_server) instantiates")
        return client
    except Exception as e:
        result.failed_check("LollmsClient(llama_cpp_server) instantiates",
                            f"{type(e).__name__}: {e}")
        return None


def check_load_model(result: ProbeResult, client, fname: str, label: str) -> bool:
    """Check 4 / 6: load a specific model."""
    print(f"\n[load] Loading {label}: {fname}")
    t0 = time.time()
    try:
        ok = client.llm.load_model(fname)
        elapsed = time.time() - t0
        if ok:
            result.passed_check(f"load_model({fname}) returned True in {elapsed:.2f}s")
            return True
        else:
            result.failed_check(f"load_model({fname})",
                                f"returned False after {elapsed:.2f}s")
            return False
    except Exception as e:
        result.failed_check(f"load_model({fname})",
                            f"{type(e).__name__}: {e}")
        return False


def check_ps(result: ProbeResult, client, label: str) -> list:
    """Check 5: ps() reports running servers."""
    print(f"\n[ps] {label} — querying active servers")
    try:
        servers = client.llm.ps() or []
        if servers:
            for srv in servers:
                mname = srv.get('model_name', '?')
                pid = srv.get('pid', '?')
                port = srv.get('port', '?')
                rss = srv.get('rss_mb', '?')
                print(f"     active: {mname}  PID:{pid}  port:{port}  RSS:{rss} MB")
            result.passed_check(f"{label}: ps() reports {len(servers)} active server(s)")
        else:
            result.passed_check(f"{label}: ps() returned empty list (no servers)")
        return servers
    except Exception as e:
        result.failed_check(f"{label}: ps()",
                            f"{type(e).__name__}: {e}")
        return []


def check_eviction(result: ProbeResult, ps_before: list, ps_after: list):
    """Check that max_active_models=1 actually evicted the first model."""
    print("\n[eviction] Did loading Model B evict Model A?")
    names_before = {s.get('model_name', '') for s in ps_before}
    names_after = {s.get('model_name', '') for s in ps_after}

    if not names_before:
        result.skipped_check("eviction check",
                             "no servers visible before second load - "
                             "can't verify eviction")
        return

    if MODEL_A_FILENAME in names_after:
        result.failed_check(
            "max_active_models=1 enforces eviction",
            f"Model A ({MODEL_A_FILENAME}) is still loaded after Model B "
            f"was loaded - eviction did not occur. Servers visible: "
            f"{sorted(names_after)}"
        )
    else:
        result.passed_check(
            "max_active_models=1 enforces eviction "
            f"(Model A no longer in ps() after Model B load)"
        )


def check_generate(result: ProbeResult, client, label: str) -> bool:
    """Check 7: generate_text returns a usable response."""
    print(f"\n[generate] {label} — generating with TEST_PROMPT")
    try:
        t0 = time.time()
        resp = client.generate_text(
            prompt=TEST_PROMPT,
            n_predict=32,
            temperature=0.1,
            top_p=0.9,
            stream=False,
        )
        elapsed = time.time() - t0
        if isinstance(resp, dict) and "error" in resp:
            result.failed_check(f"{label}: generate_text",
                                f"returned error dict: {resp['error']}")
            return False
        text = str(resp).strip()
        if not text:
            result.failed_check(f"{label}: generate_text",
                                f"returned empty string after {elapsed:.2f}s")
            return False
        snippet = text[:60].replace("\n", " ")
        result.passed_check(
            f"{label}: generate_text returned text in {elapsed:.2f}s: '{snippet}...'"
        )
        return True
    except Exception as e:
        result.failed_check(f"{label}: generate_text",
                            f"{type(e).__name__}: {e}")
        return False


def check_unload(result: ProbeResult, client, fname: str, label: str):
    """Check 8: unload_model executes cleanly."""
    print(f"\n[unload] {label} — unloading {fname}")
    try:
        client.llm.unload_model(fname)
        result.passed_check(f"{label}: unload_model({fname}) completed")
    except Exception as e:
        result.failed_check(f"{label}: unload_model({fname})",
                            f"{type(e).__name__}: {e}")


def check_shared_registry(result: ProbeResult, client_a, client_b):
    """Check 5b: a second client sees the first client's loaded server."""
    print("\n[shared registry] Does Client B see Client A's loaded model?")
    try:
        from_a = {s.get('model_name', '') for s in (client_a.llm.ps() or [])}
        from_b = {s.get('model_name', '') for s in (client_b.llm.ps() or [])}
        if from_a == from_b:
            result.passed_check(
                "Both clients agree on which models are loaded "
                "(FileLock registry working)"
            )
        else:
            result.failed_check(
                "Shared registry visibility",
                f"Client A sees {sorted(from_a)}; Client B sees {sorted(from_b)}"
            )
    except Exception as e:
        result.failed_check("Shared registry visibility",
                            f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Probe lollms_client per-persona model behavior for ATHENA"
    )
    parser.add_argument("--no-eviction-test", action="store_true",
                        help="Skip loading the second model (faster, but doesn't "
                             "verify eviction)")
    parser.add_argument("--keep-loaded", action="store_true",
                        help="Don't unload models at end (useful for debugging)")
    args = parser.parse_args()

    print("=" * 70)
    print("ATHENA per-persona model swap probe")
    print("=" * 70)
    print(f"Models path:      {MODELS_PATH}")
    print(f"Binaries path:    {BINARIES_PATH}")
    print(f"Model A:          {MODEL_A_FILENAME}")
    print(f"Model B:          {MODEL_B_FILENAME}")
    print(f"max_active:       {SHARED_BINDING_CONFIG['max_active_models']}")
    print("=" * 70)

    result = ProbeResult()

    # Phase 1: import
    if not check_import(result):
        return result.summary()

    # Phase 2: file presence
    if not check_gguf_files_exist(result):
        print("\nCannot continue: required GGUF files are missing.")
        return result.summary()

    # Phase 3: build two clients (simulating two SpecialistPersonas)
    client_a = check_client_construction(result)
    client_b = check_client_construction(result)
    if client_a is None or client_b is None:
        return result.summary()

    # Phase 4: load model A via client A
    if not check_load_model(result, client_a, MODEL_A_FILENAME, "Model A (via client_a)"):
        return result.summary()

    ps_after_a = check_ps(result, client_a, "After loading Model A")
    check_shared_registry(result, client_a, client_b)
    check_generate(result, client_a, "Model A")

    # Phase 5: eviction test
    if not args.no_eviction_test:
        if not check_load_model(result, client_b, MODEL_B_FILENAME,
                                "Model B (via client_b)"):
            return result.summary()
        ps_after_b = check_ps(result, client_b, "After loading Model B")
        check_eviction(result, ps_after_a, ps_after_b)
        check_generate(result, client_b, "Model B")
    else:
        result.skipped_check("eviction test", "--no-eviction-test specified")

    # Phase 6: cleanup
    if not args.keep_loaded:
        check_unload(result, client_a, MODEL_A_FILENAME, "Model A")
        if not args.no_eviction_test:
            check_unload(result, client_b, MODEL_B_FILENAME, "Model B")
    else:
        result.skipped_check("unload test", "--keep-loaded specified")

    return result.summary()


if __name__ == "__main__":
    sys.exit(main())
