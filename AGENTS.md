# AGENTS.md — `local-llm`

## Project overview

A personal workbench for running a large **MoE LLM** locally on a single laptop with a tight VRAM budget, using a custom build of llama.cpp.

The hard constraint that governs every decision in this repo:

> The model must run within the configured VRAM and RAM budget at the longest context the host budget allows. VRAM, RAM, and KV cache type are parameters of `scripts/moe-configs.py` (`--vram`, `--ram`, `--cache-type-k`, `--cache-type-v`) so sizing tracks the host machine, not a hardcoded GPU.

If a change risks busting that budget, flag it explicitly. Do not "simplify" configs in ways that violate it.

## Host budget (defaults; configurable)

`scripts/moe-configs.py` defaults reflect the current development host. Override with the flags below when running on different hardware; do not hardcode new values into the script.

- **VRAM:** `--vram` MiB (default `6144` = 6 GiB). The usable VRAM budget is this value; the script reports actual used VRAM so overhead is visible.
- **RAM:** `--ram` MiB (default `32768` = 32 GiB), the RAM budget available to llama.cpp. Subtract your OS / other-process overhead before passing.
- **KV cache:** `--cache-type-k` (default `turbo4`) and `--cache-type-v` (default `turbo3_tcq`). Choices: `turbo2`, `turbo3`, `turbo2_tcq`, `turbo3_tcq`, `turbo4`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`.

VRAM must hold, in order of priority:
1. all **dense** (non-expert) weights,
2. the **TurboQuant / TCQ-compressed KV cache**, and
3. as many **active expert tensors** as still fit.

The remaining experts live in RAM and run on CPU when routed.

## Repository layout

Tracked files only:

- `README.md` — append-only scratchpad of working commands, tested `llama-server` invocations, and measured tokens/sec, RAM, VRAM numbers.
- `scripts/moe-configs.py` — heuristic that picks `--n-cpu-moe` for a given GGUF.
- `AGENTS.md` — this file.
- `.gitignore` — excludes the cloned llama.cpp fork directory (`llama.cpp/`) so the working checkout can live alongside the repo files without being committed.

External to the repo (not tracked, paths supplied by the caller):

- The llama.cpp fork checkout. Conventionally cloned as `./llama.cpp/` so the binary lands at `./llama.cpp/build/bin/llama-server`, but any path works — substitute as needed.
- GGUF model files. Referred to as `<model-path>` in the documentation; there is no required directory.

## Build, run, sizing

`README.md` is the source of truth:

- Toolchain and llama.cpp fork: `README.md` § Build.
- `cmake` build flags: `README.md` § Build.
- Canonical `llama-server` invocation and per-model settings: `README.md` § Run, § Models on disk — recommended settings.
- `--n-cpu-moe` sizing via `scripts/moe-configs.py`: `README.md` § Sizing, § Sizing precedence.
- KV-cache types and the `turbo*` table: `README.md` § KV-cache strategy.

Do not duplicate those commands or numbers here. Update `README.md` and link, don't fork.

## Test / lint / CI

There are **no tests, no linters, and no CI** in this repo. Do not fabricate them. Validation is empirical: run `llama-server` with the canonical command, watch `nvidia-smi`, record tokens/sec in `README.md`.

## Don't

- Don't commit the llama.cpp fork checkout (`llama.cpp/` and anything under it, including its `build/` tree) or any GGUF model files.
- Don't propose configs that violate the configured VRAM/RAM budget — e.g. `--n-gpu-layers 999` with no `--n-cpu-moe`, full-precision KV, or moving experts back to GPU wholesale.
- Don't use `-nkvo` — KV must stay on the GPU in this build.
- Don't use `-ngl <small>` partial dense offload. We only run MoE models; the offload knob is `--n-cpu-moe`, not `-ngl`.
- Don't omit `-c <ctx>`; the default training context busts VRAM.
- Don't bake host-specific VRAM/RAM values into `scripts/moe-configs.py`. Sizing is parameterized via `--vram`, `--ram`; change the defaults only if the development host actually changes, and never by hardcoding values inside `make_plan()`.
- Don't restructure `README.md` into prose docs; keep it as a flat, append-friendly log.
- Don't rewrite recorded numbers (t/s, RAM, VRAM). They are empirical measurements, not estimates.

## When editing

- New helper scripts belong in `scripts/` alongside `moe-configs.py`.
- If you change a flag in the canonical command, update `README.md` and explain why in the same commit.
