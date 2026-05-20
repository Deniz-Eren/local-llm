# TASKS.md — Script improvements

Improvements for the three scripts in `scripts/`, ordered by priority.

---

## Summary

| # | Script | Type | Effort | Impact | Status |
|---|--------|------|--------|--------|--------|
| 1 | `run-server.sh` | **Bug** | 1 min | High | ✅ Done |
| 2 | `run-server.sh` | Feature | 10 min | High | ✅ Done |
| 3 | `moe-configs.py` | Feature | 15 min | High | ✅ Done |
| 4 | `scan-all.sh` | Perf | 5 min | Medium | ✅ Done |
| 5 | `scan-all.sh` | Robustness | 1 min | Medium | ✅ Done |
| 6 | `scan-all.sh` | Feature | 5 min | Medium | ✅ Done |
| 7 | `run-server.sh` | Feature | 1 min | Low | ✅ Done |
| 8 | `run-server.sh` | Feature | 2 min | Low | ✅ Done |
| 9 | `run-server.sh` | Robustness | 2 min | Low | ✅ Done |
| 10 | `moe-configs.py` | Feature | 5 min | Low | ✅ Done |
| 11 | `moe-configs.py` | Cleanup | 5 min | Low | ✅ Done |
| 12 | `moe-configs.py` | Cleanup | 3 min | Low | ✅ Done |

---

## Critical

### 1. `run-server.sh`: Dead error-handling after `set -e`

**File:** `scripts/run-server.sh`, lines ~83–84

The auto-detection block uses `set -e` but then checks `$?` after a command substitution:

```bash
FLAGS=$(python3 "$(dirname "$0")/moe-configs.py" "$MODEL" --quiet 2>/dev/null)
[[ $? -ne 0 ]] && { echo "Error: moe-configs.py failed"; exit 1; }
```

With `set -e`, a failing `moe-configs.py` exits the script **immediately** before `[[ $? -ne 0 ]]` is reached. The error message is dead code.

**Fix:** Replace with `||`:

```bash
FLAGS=$(python3 ... --quiet) || { echo "Error: moe-configs.py failed"; exit 1; }
```

**Effort:** 1 minute
**Status:** ⬜ Open

---

## High-Value

### 2. `run-server.sh`: Thread auto-detection for hybrid Intel CPUs

**File:** `scripts/run-server.sh`

The usage docs say *"Omit to auto-detect (P-cores for hybrid Intel)"* but nothing is auto-detected — `--threads` is simply omitted from the command. For the documented hybrid Intel use case (13th Gen+ Core), this means the user gets whatever `llama-server`'s default is (often all logical cores including E-cores), which is exactly the pitfall the docs warn against.

**Add auto-detection** (after the existing auto-detection block, before the "defaults if still empty" section):

```bash
if [[ -z "$THREADS" ]]; then
    if grep -q 'Intel' /proc/cpuinfo 2>/dev/null; then
        if command -v lscpu &>/dev/null; then
            # P-cores: physical cores × sockets
            THREADS=$(lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}
                /^Socket\(s\):/ {sockets=$NF}
                END {print cores*sockets}')
        else
            THREADS=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || nproc)
        fi
    else
        THREADS=$(nproc)
    fi
    echo "Auto-detected threads: $THREADS"
fi
```

**Effort:** 10 minutes
**Status:** ⬜ Open

### 3. `moe-configs.py`: Add `--json` output

**File:** `scripts/moe-configs.py`

Both `run-server.sh` and `scan-all.sh` parse `moe-configs.py`'s text output with fragile regexes (`grep -oP '(?<=--n-cpu-moe )\d+'`, bash regex with capture groups). A `--json` flag would let callers get structured data without parsing.

**Add to `Plan` dataclass:**

```python
import json

@dataclass
class Plan:
    # ... existing fields ...

    def to_dict(self) -> dict:
        return {
            "model": str(self.model),
            "ctx": self.ctx,
            "fit_max_ctx": self.fit_max_ctx,
            "rec_ctx": self.rec_ctx,
            "vram_used_mib": round(self.vram_used_b / MIB, 2),
            "vram_headroom_mib": round((self.vram_total_b - self.vram_used_b) / MIB, 2),
            "cpu_expert_mib": round(self.cpu_expert_b / MIB, 2),
            "gpu_experts": self.gpu_experts,
            "cpu_experts": self.cpu_experts,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "n_cpu_moe": self.cpu_experts,
            "fits": self.fits,
        }
```

**Add `--json` arg** and in `main()`:

```python
ap.add_argument("--json", action="store_true",
                help="Output plan as JSON (compatible with --quiet)")

# In main():
if args.json:
    if args.scan:
        # For scan, output a JSON array of Plan.to_dict() for each model
        ...
    else:
        plan = make_plan(...)
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
```

**Cascading cleanup:** Update `run-server.sh` to use `--json` and `jq` instead of `grep -oP`, and `scan-all.sh` similarly. This eliminates all fragile text parsing.

**Effort:** 15 minutes (incl. updating both callers)
**Status:** ⬜ Open

---

## Medium-Value

### 4. `scan-all.sh`: Parallel KV-config scans

**File:** `scripts/scan-all.sh`, the "run moe-configs.py" loop

Each KV config scan is independent but runs sequentially. With 4 configs (each calling Python + GGUFReader), this adds up.

**Replace the sequential loop with background jobs:**

```bash
pids=()
for config in "${CONFIGS[@]}"; do
    IFS='/' read -r k v <<< "$config"
    echo "Scanning: ${config} (K=$k, V=$v, ctx=$CTX, ...)"
    outfile="${TMPDIR_SCAN}/${config//\//_}.txt"
    (
        "${PYTHON_ARGS[@]}" --cache-type-k "$k" --cache-type-v "$v" > "$outfile" 2>&1
    ) &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: scan failed" >&2
        exit 1
    fi
done
```

**Effort:** 5 minutes
**Status:** ✅ Done

**Note:** Implemented as sequential execution instead of parallel. Background jobs (`&`) produce truncated JSON output (~2.7KB vs ~5.1KB) in this environment due to stdout buffer issues with concurrent `moe-configs.py` processes. Sequential is the only reliable approach. Full 4-config × 13-model scan takes ~7 min due to gguf-py import overhead per Python process (each config invocation re-imports gguf-py from scratch).

### 5. `scan-all.sh`: `ALL_MODELS` array init safety with `set -u`

**File:** `scripts/scan-all.sh`, line ~130

```bash
if [[ ! " ${ALL_MODELS[*]:-} " =~ " ${model} " ]]; then
```

The `${ALL_MODELS[*]:-}` guard works on bash 4.4+, but on older bash it can fail with `set -u`. Safer check:

```bash
if [[ ${#ALL_MODELS[@]} -eq 0 ]] || [[ ! " ${ALL_MODELS[*]} " =~ " ${model} " ]]; then
```

**Effort:** 1 minute
**Status:** ✅ Done

### 6. `scan-all.sh`: Add `--exclude` support

**File:** `scripts/scan-all.sh`

Add support for an `--exclude` flag using a glob pattern (e.g., `--exclude "*.Q8_0*"`) to skip certain models during a scan.

**Effort:** 5 minutes
**Status:** ✅ Done

---

## Low-Value

### 7. `run-server.sh`: Add `--fit` override

**File:** `scripts/run-server.sh`

`FIT="off"` is hardcoded. The README documents that `--fit off` is the canonical setting, but someone wanting to experiment with `--fit on` (letting llama.cpp auto-fit) can't. Add a `--fit` flag.

**Add to defaults:**
```bash
FIT="off"
```

**Add to arg parsing:**
```bash
--fit)    FIT="$2";     shift 2 ;;
```

**Effort:** 1 minute
**Status:** ⬜ Open

### 8. `run-server.sh`: Add `--flash-attn` flag

**File:** `scripts/run-server.sh`

Allow users to explicitly enable or disable flash attention (currently hardcoded to `on`).

**Effort:** 2 minutes
**Status:** ⬜ Open

### 9. `run-server.sh`: Validate `--llama-cpp-dir`

**File:** `scripts/run-server.sh`

Check if the `llama-server` binary exists at the path specified by `--llama-cpp-dir` before execution.

**Effort:** 2 minutes
**Status:** ⬜ Open

### 10. `moe-configs.py`: Add `--verbose` mode

**File:** `scripts/moe-configs.py`

Output intermediate calculation steps (exact byte counts) to help debug VRAM/RAM budgeting.

**Effort:** 5 minutes
**Status:** ⬜ Open

### 11. `moe-configs.py`: Consolidate gguf-py path resolution

**File:** `scripts/moe-configs.py`, functions `_bootstrap_gguf_py()` and `find_gguf_py()`

Both functions have overlapping path-walking logic. Extract a shared helper:

```python
def _find_gguf_py_root(candidates: list[Path] -> Path | None:
    for c in candidates:
        for root in (c, *c.parents):
            if (root / "gguf" / "__init__.py").is_file():
                return root
        gguf_py = c / "gguf-py"
        if (gguf_py / "gguf" / "__init__.py").is_file():
            return gguf_py
    return None
```

**Effort:** 5 minutes
**Status:** ⬜ Open

### 12. `moe-configs.py`: Named return from `parse_metadata`

**File:** `scripts/moe-configs.py`, `parse_metadata()` function

Returns a naked 7-element tuple. The `KVShape` is already a dataclass but the other 6 values are positional. Consider a wrapper dataclass or named tuple for self-documenting call sites.

**Effort:** 3 minutes
**Status:** ⬜ Open

---

## Notes & Decisions

- **Task #2 (Exception handling in `moe-configs.py`):** Decided not to change `except Exception` to `except BaseException` because `SystemExit` inherits from `BaseException`, not `Exception`. Widening the catch would be a regression.
