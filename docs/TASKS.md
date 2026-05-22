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

---

## Open Issues — Docs/Script Conflicts (Found 2026-05-22)

These are conflicts between documents and actual code/behavior. Ordered by severity.

| # | Severity | File(s) | Type | Description |
|---|----------|---------|------|-------------|
| 25 | 🔴 | `README.md`, `moe-configs.py` | Bug | **`--ctx 0` does nothing.** README says `--ctx 0` uses the model's trained max, but `argparse` default is `128000`, so `--ctx 0` sets `ctx=0` (not `None`). Result: 0-token context. The documented feature is unreachable. |
| 26 | 🔴 | `README.md`, `moe-configs.py` | Bug | **`--n-cpu-moe 237` on a 40-layer model.** README canonical command uses `--n-cpu-moe 237` for Qwen3.6-35B-A3B (40 layers). `moe-configs.py`'s `Plan.n_cpu_moe` returns `cpu_layers`, which can be at most 40. Either the README example is wrong, or `--n-cpu-moe` in this fork actually accepts expert counts (contradicting the docstring). |
| 27 | 🟡 | `README.md` | Doc error | **`--mlock-safe` as a flag that doesn't exist.** README documents `--mlock-safe` as a toggleable flag, but `run-server.sh` has no such argument. The auto-guard is implicit and always-on. Either remove the flag from docs or add the CLI option. |
| 28 | 🟡 | `README.md` | Doc error | **`q4_0` / `iq4_nl` factor column is wrong.** README table lists bpv=1.0 and factor=0.5 (implying 2x smaller). Code has bpe=0.5 -> factor=0.25. The "x4.0 smaller" column is correct; factor and bpv are both wrong in the doc. |
| 29 | 🟡 | `README.md` | Doc error | **`--threads 28` example contradicts docs.** README `--threads` example for Kimi-K2.6 uses `--threads 28`, but the documentation says non-hybrid Intel uses physical cores (14 for Xeon Gold 5120). The canonical Kimi-K2.6.md command uses `--threads 14` (correct). |
| 30 | 🟡 | `docs/Qwen3.6-35B-A3B-MTP.md` | Doc error | **Config results don't match command.** Command passes `--ram 730956`, but results block shows budget 32768 (default). Headroom math (26112 + 6656 = 32768) confirms the results are from a different run than the command shown. |
| 31 | 🟡 | `README.md` | Ambiguity | **`GPU/CPU` column ambiguous.** README table header says `GPU/CPU` but two tools use different meanings: `moe-configs.py --scan` prints `gpu_layers/cpu_layers`; `scan-all.sh` prints `gpu_experts/cpu_experts`. Values don't sum to a consistent total (57+71=128 layers; 19+237=256 experts). |
| 32 | 🟢 | `AGENTS.md`, `README.md` | Doc error | **Tracked files undercounted.** AGENTS.md lists 4 tracked files; the repo also tracks `requirements.txt`, `LICENSE`, `scripts/run-server.sh`, `scripts/scan-all.sh`, and 5 files in `docs/`. |
| 33 | 🟢 | `run-server.sh`, `README.md` | Cosmetic | **`--no-mmap` default claim.** README says llama-server defaults to `--no-mmap` and the flag is redundant. But `run-server.sh` doesn't add it by default. If it truly defaults to no-mmap in llama-server, the canonical command should drop it. |

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

### `--quiet` in `scan-all.sh` (Issue 34)

The `--quiet` flag is parsed but never used -- the full table is always printed. Could suppress the scanning progress lines and only print fitting models, but hasn't been needed yet.

---

## Notes & Decisions

- **`--threads` strategy.** Non-hybrid Intel: use all physical cores (intra-GEMM parallelism). Hybrid Intel (12th+ gen): P-cores only (E-cores stall GEMM finish). Detection uses `lscpu --extended` to find E-core frequency floor and count P-cores above it.
- **Thread cap at 8 (active experts) was tried and rejected.** 27 t/s with 14 threads vs 20 t/s with 8 on Xeon Gold 5120.
- **Parallel scan jobs** produce truncated JSON (stdout buffer contention). Sequential is the only reliable approach.
- **Non-fitting model handling**: `moe-configs.py` exits 1 when `fits=false`. `run-server.sh` distinguishes this from real errors via stderr -- "Model not found" aborts, clean exit 1 continues with `--mlock` auto-stripped.
- **`--check-avx`** revealed the host (i7-13850HX) lacks AVX-512F despite the VM simulating a Xeon Gold 5120.
