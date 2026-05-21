# TASKS.md — Script improvements

Improvements for the three scripts in `scripts/`, ordered by priority.

---

## Completed Tasks

### Original Tasks (1–12)

| # | Script | Type | Status |
|---|--------|------|--------|
| 1 | `run-server.sh` | Bug fix: `set -e` error handling | ✅ |
| 2 | `run-server.sh` | Feature: thread auto-detect for hybrid Intel | ✅ |
| 3 | `moe-configs.py` | Feature: `--json` output + `to_dict()` | ✅ |
| 4 | `scan-all.sh` | Perf: parallel KV-config scans → sequential (stable) | ✅ |
| 5 | `scan-all.sh` | Robustness: `ALL_MODELS` init with `set -u` | ✅ |
| 6 | `scan-all.sh` | Feature: `--exclude` support | ✅ |
| 7 | `run-server.sh` | Feature: `--fit` override | ✅ |
| 8 | `run-server.sh` | Feature: `--flash-attn` flag | ✅ |
| 9 | `run-server.sh` | Robustness: `--llama-cpp-dir` validation | ✅ |
| 10 | `moe-configs.py` | Feature: `--verbose` mode | ✅ |
| 11 | `moe-configs.py` | Cleanup: consolidate gguf-py path resolution | ✅ |
| 12 | `moe-configs.py` | Cleanup: `ParseResult` named return | ✅ |

### New Tasks (13–16)

| # | Script | Type | Status | Test |
|---|--------|------|--------|------|
| 13 | `run-server.sh` | Robustness: `--mlock-safe` auto-guard | ✅ | Verified guard triggers when `fits: false` from `--json` output |
| 14 | `run-server.sh` | Feature: `--dry-run` flag | ✅ | Tested: auto-sizing, explicit flags, spec decoding, all compose correctly |
| 15 | `moe-configs.py` | Feature: `--check-avx` diagnostic | ✅ | Tested: reports AVX2+AVX-VNNI available, no AVX-512F (host is i7-13850HX) |
| 16 | `scan-all.sh` | Feature: `--no-warmup`/`--no-mmap`/`--mlock` flags | ✅ | Tested: markdown and CSV both include extra_flags column |
| 17 | `run-server.sh` | Feature: `--json` parsing for auto-guard logic | ✅ | Parses `moe-configs.py --json` output to extract `fits` and `n_cpu_moe` |

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

### `--quiet` in `scan-all.sh`

The `--quiet` flag is parsed but never used — the full table is always printed. Could suppress the scanning progress lines and only print fitting models, but hasn't been needed yet.

---

## Documentation Gaps (New Tasks)

| # | File | Type | Status | Notes |
|---|------|------|--------|-------|
| 18 | `README.md` | Docs: add `--mlock-safe` behavior section | ✅ | Added section with auto-guard trigger conditions |
| 19 | `README.md` | Docs: add `--dry-run` examples | ✅ | Added examples for CI/debugging use cases |
| 20 | `README.md` | Docs: add `--check-avx` diagnostic section | ✅ | Added section with example output |
| 21 | `README.md` | Docs: document `--cache-type-k/v` defaults | ✅ | Added table with override scenarios |
| 22 | `run-server.sh` | Feature: `--no-spec-type` fallback | ✅ | Auto-detects MTP weights and warns if missing |
| 23 | `moe-configs.py` | Feature: `--json --verbose` mode | ✅ | Adds `_verbose` object to JSON output |
| 24 | `scan-all.sh` | Feature: summary aggregation | ✅ | Markdown table summary + CSV comment |

---

## Notes & Decisions

- **`--threads` = physical cores (non-hybrid), P-cores only (hybrid).** On non-hybrid Intel (Xeon Gold 5120), `--threads 14` beats 8 because `--threads` controls intra-GEMM parallelism inside each expert's MLP — more threads = faster matrix multiplies. On hybrid Intel (12th+ gen Core), only P-cores are used because E-cores bottleneck the parallel GEMM finish. Detection uses `lscpu --extended` to identify E-core frequency floor (lowest MAXMHZ) and counts all cores with frequency above it — this handles the case where different P-cores report different MAXMHZ values (5100 vs 5300 MHz on i7-13850HX).
- **Thread cap at 8 (active experts) was tried and rejected.** On the Xeon Gold 5120, `--threads 8` gave 20 t/s vs 27 t/s with 14 threads. Each expert's GEMM is itself multi-threaded, so more threads inside each GEMM matters more than avoiding "idle" threads.
- **Parallel scan jobs** were tried but produce truncated JSON in this environment (stdout buffer contention between concurrent `moe-configs.py` + `gguf-py` processes). Sequential is the only reliable approach.
- **`--dry-run`** skips server binary validation since the command is only composed, not launched. Useful for CI, debugging, and pasting commands into different terminals.
- **Non-fitting model handling**: `moe-configs.py` exits with code 1 when `fits=false`. `run-server.sh` now distinguishes this from real errors by capturing stderr — "Model not found" triggers a proper abort, while a clean stderr with exit code 1 produces a warning and continues (with `--mlock` auto-stripped by the guard).
- **`--check-avx`** revealed the host CPU (i7-13850HX) lacks AVX-512F despite the VM simulating a Xeon Gold 5120. The build with `-DGGML_AVX512=ON -DGGML_NATIVE=ON` falls back to AVX2 at runtime.
- **`--mlock-safe` auto-guard** prevents OOM by disabling `--mlock` when the sizing plan shows the model doesn't fit in the RAM budget.
