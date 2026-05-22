# TASKS.md — Script improvements

Improvements for the three scripts in `scripts/`, ordered by priority.

---

## Completed Tasks

| # | Script | Type | Severity | Status | Test |
|---|--------|------|----------|--------|------|
| 1 | `run-server.sh` | Bug fix: `set -e` error handling | 🔴 | ✅ | Silent failures no longer swallow errors |
| 2 | `run-server.sh` | Feature: thread auto-detect for hybrid Intel | 🟢 | ✅ | P-cores only on hybrid, physical cores on non-hybrid |
| 3 | `moe-configs.py` | Feature: `--json` output + `to_dict()` | 🟢 | ✅ | Consumed by `run-server.sh` and `scan-all.sh` |
| 4 | `scan-all.sh` | Perf: parallel KV-config scans → sequential (stable) | 🟡 | ✅ | Parallel jobs produced truncated JSON |
| 5 | `scan-all.sh` | Robustness: `ALL_MODELS` init with `set -u` | 🟡 | ✅ | Prevents unbound variable crash with `set -u` |
| 6 | `scan-all.sh` | Feature: `--exclude` support | 🟢 | ✅ | Glob pattern filtering via `--models-file` |
| 7 | `run-server.sh` | Feature: `--fit` override | 🟢 | ✅ | Passes `--fit` verbatim to llama-server |
| 8 | `run-server.sh` | Feature: `--flash-attn` flag | 🟢 | ✅ | Defaults to `on`; configurable via `--flash-attn on|off` |
| 9 | `run-server.sh` | Robustness: `--llama-cpp-dir` validation | 🟡 | ✅ | Validates binary exists; skipped in `--dry-run` |
| 10 | `moe-configs.py` | Feature: `--verbose` mode | 🟢 | ✅ | Prints intermediate calculation steps to stderr |
| 11 | `moe-configs.py` | Cleanup: consolidate gguf-py path resolution | 🟢 | ✅ | `find_gguf_py()` + `_bootstrap_gguf_py()` + `_find_gguf_py_root()` |
| 12 | `moe-configs.py` | Cleanup: `ParseResult` named return | 🟢 | ✅ | Frozen dataclass replaces unnamed tuple from `parse_metadata()` |
| 13 | `run-server.sh` | Robustness: `--mlock-safe` auto-guard | 🟡 | ✅ | Verified guard triggers when `fits: false` from `--json` output |
| 14 | `run-server.sh` | Feature: `--dry-run` flag | 🟢 | ✅ | Tested: auto-sizing, explicit flags, spec decoding, all compose correctly |
| 15 | `moe-configs.py` | Feature: `--check-avx` diagnostic | 🟢 | ✅ | Tested: reports AVX2+AVX-VNNI available, no AVX-512F (host is i7-13850HX) |
| 16 | `scan-all.sh` | Feature: `--no-warmup`/`--no-mmap`/`--mlock` flags | 🟢 | ✅ | Tested: markdown and CSV both include extra_flags column |
| 17 | `run-server.sh` | Feature: `--json` parsing for auto-guard logic | 🟡 | ✅ | Parses `moe-configs.py --json` output to extract `fits` and `n_cpu_moe` |
| 18 | `README.md` | Docs: add `--mlock-safe` behavior section | 🟢 | ✅ | Added section with auto-guard trigger conditions |
| 19 | `README.md` | Docs: add `--dry-run` examples | 🟢 | ✅ | Added examples for CI/debugging use cases |
| 20 | `README.md` | Docs: add `--check-avx` diagnostic section | 🟢 | ✅ | Added section with example output |
| 21 | `README.md` | Docs: document `--cache-type-k/v` defaults | 🟢 | ✅ | Added table with override scenarios |
| 22 | `run-server.sh` | Feature: `--no-spec-type` fallback | 🟡 | ✅ | Auto-detects MTP weights and warns if missing |
| 23 | `moe-configs.py` | Feature: `--json --verbose` mode | 🟢 | ✅ | Adds `_verbose` object to JSON output |
| 24 | `scan-all.sh` | Feature: summary aggregation | 🟢 | ✅ | Markdown table summary + CSV comment |
| 25 | `moe-configs.py` | Bug: `--ctx 0` unreachable | 🔴 | ✅ | Changed argparse default from `128000` to `None`; `ctx == 0` now means "use model max" |
| 26 | `moe-configs.py`, `README.md`, `docs/Kimi-K2.6.md` | Bug: README/doc `--n-cpu-moe` values used expert counts, not layers | 🔴 | ✅ | Confirmed fork's `--n-cpu-moe` takes layer counts (commit 8067bc0); README/examples updated with correct layer counts
| 27 | `README.md` | Doc error: `--mlock-safe` documented as flag | 🟡 | ✅ | Resolved: auto-guard is implicit (no flag needed); docs describe correct behavior |
| 28 | `README.md` | Doc error: `q4_0`/`iq4_nl` factor wrong | 🟡 | ✅ | Fixed bpv ~0.5, factor 0.25 (was bpv 1.0, factor 0.5); ×4.0 column was correct |
| 29 | `run-server.sh` | Doc error: `--threads 28` example wrong | 🟡 | ✅ | Changed to `--threads 14` (physical cores for Xeon Gold 5120); updated TASKS description to reference correct file
| 30 | `docs/Qwen3.6-35B-A3B-MTP.md` | Doc error: stale config/results mismatch | 🟡 | ✅ | Removed erroneous `--ram 730956` from command (results use default 32768) |
| 31 | `moe-configs.py`, `scan-all.sh`, `README.md` | Ambiguity: `GPU/CPU` meant layers vs experts | 🟡 | ✅ | Confirmed fork's `--n-cpu-moe` takes layers (commit 8067bc0); unified scan output to layers (`gpu_layers/cpu_layers`); README header says `GPU/CPU`
| 32 | `AGENTS.md` | Doc error: tracked files undercounted | 🟢 | ✅ | Added all 12 tracked files to repository layout section |
| 33 | `run-server.sh` | Cosmetic: `--no-mmap` redundant claim | 🟢 | ✅ | Rewrote help text to be factual and less confusing |
| 34 | `scan-all.sh` | Feature: `--quiet` parsed but dead code | 🟢 | ✅ | Now suppresses scanning progress lines and markdown summary

---

## Open Issues

No open issues. All conflicts found 2026-05-22 have been resolved.

---

## Pending: `--benchmark` Implementation

**Blocked on:** Design decisions

A `--benchmark` flag that launches, runs a fixed prompt, measures tok/s, prints summary, then exits. Currently requires the manual `2>&1 | tee; grep "ms/tok"; sed/awk` pipeline.

**Open questions:**
- Fixed prompt: use a standard 100-token completion test or user-configurable via `--benchmark-prompt`?
- Iterations: single run or average over N runs (`--benchmark-iters`)?
- Streaming: measure first-token latency separately from decode tok/s?
- Exit code: 0 on success, non-zero if tok/s below threshold (`--benchmark-threshold`)?
- Output format: plain text summary or JSON for CI consumption?

---

## Notes & Decisions

- **`--threads` strategy.** Non-hybrid Intel: use all physical cores (intra-GEMM parallelism). Hybrid Intel (12th+ gen): P-cores only (E-cores stall GEMM finish). Detection uses `lscpu --extended` to find E-core frequency floor and count P-cores above it.
- **Thread cap at 8 (active experts) was tried and rejected.** 27 t/s with 14 threads vs 20 t/s with 8 on Xeon Gold 5120.
- **Parallel scan jobs** produce truncated JSON (stdout buffer contention). Sequential is the only reliable approach.
- **Non-fitting model handling**: `moe-configs.py` exits 1 when `fits=false`. `run-server.sh` distinguishes this from real errors via stderr -- "Model not found" aborts, clean exit 1 continues with `--mlock` auto-stripped.
- **`--check-avx`** revealed the host (i7-13850HX) lacks AVX-512F despite the VM simulating a Xeon Gold 5120.
