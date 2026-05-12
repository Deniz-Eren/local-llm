# local-llm

A workbench for running large **Mixture-of-Experts** LLMs locally on consumer hardware with a tight VRAM budget. Dense weights and KV cache stay GPU-resident; experts live in RAM and run on CPU when routed (`--n-cpu-moe`). KV size is brought under control with quantized cache types from a llama.cpp fork.

Includes a sizing tool (`scripts/moe-configs.py`) that reads any GGUF, takes your VRAM/RAM as parameters, and prints the `--n-gpu-layers / --n-cpu-moe / -c` flags that fit.

## How it works

A 35B MoE has only ~3B parameters active per token (8 of 256 experts on Qwen3.6-35B-A3B). The other ~31B can sit cold in slow memory at no per-token cost — provided we can route the active 8 into compute quickly.

`--n-cpu-moe N` keeps `N` experts pinned in RAM and **runs their MLP on CPU threads in place**; it does *not* copy expert weights to the GPU. Per token: GPU runs attention + dense layers, the router picks 8 experts, those 8 MLPs run wherever their weights live, and only the small output activations cross PCIe to be summed back into the residual stream. Throughput is gated by CPU MLP compute, not PCIe bandwidth on weight transfers.

The KV cache is the other VRAM consumer. At 262144 tokens it would be ~20 GiB at FP16 — far past a 6 GiB budget. The fork's TurboQuant / TCQ KV types compress this ~5× (`turbo3_tcq` = 3.25 bpv) at ~97% of `q8_0` decode speed and constant cost across context, so KV stays GPU-resident at any context. The lossless `turbo4` (4.25 bpv, ~3.8× compression) is the safe default.

## KV-cache strategy

KV is GPU-resident on this build. The fork ships **Trellis-Coded Quantization (TCQ)** KV types named `turbo4` (4.25 bpv, lossless), `turbo3_tcq` / `turbo3`, and `turbo2_tcq` / `turbo2`. The `turbo4` type is scalar-like in quality (no TCQ trellis) but still compressed; `turbo3_tcq` and `turbo2_tcq` use the trellis codebook for quality that matches or beats FP16.

| K / V pair                        | bpv  | KLD @2K / @7K   | KV size (rel.)         | Use when                                         |
|-----------------------------------|------|-----------------|------------------------|--------------------------------------------------|
| **`turbo4` / `turbo3_tcq`**       | 3.75 | lossless / 0.058 | +15% vs baseline     | **Default.** Lossless keys, tight values.        |
| `turbo4` / `turbo4`               | 4.25 | lossless        | +31% vs baseline       | Maximum quality, VRAM headroom to spare.         |
| `turbo3_tcq` / `turbo3_tcq`       | 3.25 | 0.058 / 0.074   | baseline               | Tighter KV, still beats FP16 at short ctx.       |
| `turbo3_tcq` / `turbo2_tcq`       | 2.75 | 0.078 / 0.101   | −15%                   | Stretch to longer contexts.                      |
| `turbo2_tcq` / `turbo2_tcq`       | 2.25 | 0.101 / 0.136   | −31%                   | Maximum compression, accept some quality loss.   |

Scalar `turbo3` and `turbo2` (no trellis) have the same bpv as their TCQ counterparts but slightly higher KLD; they consume identical VRAM.

The default `turbo4`/`turbo3_tcq` pair gives lossless keys (4.25 bpv) while compressing values at 3.25 bpv — asymmetric, with no quality loss on K while keeping V ~5× compressed. No KLD numbers exist for this exact asymmetric pair; the values KLD matches symmetric `turbo3_tcq` (0.058/@7K=0.074). `q8_0` (~1.06 bytes/elem) works as a plain CUDA fallback if TCQ ever misbehaves on a new model.

# Build

llama.cpp fork that adds the `turbo*` KV types with CUDA kernels (KV stays on the GPU, no `-nkvo` required):

- repo: `git@github.com:spiritbuun/buun-llama-cpp.git`, branch: `master`

This fork is a **temporary** dependency: once upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) lands TurboQuant / TCQ support, this repo will switch back to upstream.

```bash
git clone -b master git@github.com:spiritbuun/buun-llama-cpp.git llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-O3" \
  -DCMAKE_CXX_FLAGS="-O3"
cmake --build build -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES=86` targets the RTX A1000's sm_86 — adjust for your GPU. `GGML_CUDA_FA_ALL_QUANTS=ON` is required so flash-attention kernels are compiled for the quantized KV types; `-fa on` with `turbo*` KV silently falls back without it.

### AVX-512 build

On CPUs with AVX-512F support, build with `GGML_AVX512=ON`. The `-march` is handled automatically by `-DGGML_NATIVE=ON`:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_AVX512=ON \
  -DCMAKE_C_FLAGS="-O3" \
  -DCMAKE_CXX_FLAGS="-O3"
```

Two parts are required:
- **`-DGGML_AVX512=ON`** — cmake preprocessor define that enables the AVX-512 code paths in ggml source.
- **`-DGGML_NATIVE=ON`** — auto-detects the CPU and passes `-march=native` to the compiler so it emits the correct instruction set (including AVX-512F on compatible CPUs).

Without the cmake option, the AVX-512 blocks aren't compiled at all, so you get AVX2 only — half the per-cycle GEMM throughput on the CPU-side expert MLPs.

#### Checking which AVX-512 extensions your CPU supports

To check what your CPU supports:

```bash
# Check if the CPU advertises AVX-512 extensions
grep -oP 'avx512(f|bw|vl|vnni|bf16|vbmi)' /proc/cpuinfo
```

Or use `lscpu`:

```bash
lscpu | grep -i avx
```

Then enable the matching cmake options:

| Extension | cmake flag | First microarch | What it speeds up |
|-----------|-----------|-----------------|-------------------|
| AVX-512F (base) | `GGML_AVX512=ON` | Skylake | All 512-bit GEMM paths |
| AVX-512BW | `GGML_AVX512_BW=ON` | Cascade Lake | Byte/word GEMM ops |
| AVX-512VL | *(bundled with F)* | Skylake | Vector length 256/512 |
| AVX-512VNNI | `GGML_AVX512_VNNI=ON` | Cascade Lake | INT8 matmul (MoE experts) |
| AVX-512BF16 | `GGML_AVX512_BF16=ON` | Cooper Lake / Zen 4 | BF16 GEMM |
| AVX-512VBMI | `GGML_AVX512_VBMI=ON` | Cannon Lake | Vector byte/word ops |

# Sizing

## Single model

```bash
python3 scripts/moe-configs.py <model-path>
```

Reports the VRAM/RAM breakdown and prints the `--n-gpu-layers / --n-cpu-moe / -c` flags to use. Host budget is configurable via `--vram` and `--ram` (both in MiB). Default `--vram` is 6144 (6 GiB); default `--ram` is 32768 (32 GiB, i.e. the budget available to llama.cpp after subtracting OS overhead). KV cache types default to `turbo4` (keys) and `turbo3_tcq` (values). Default context is `--ctx 128000`; pass `--ctx 0` to use the model's trained max, or any other value to stretch as far as VRAM allows.

## Directory scan

```bash
python3 scripts/moe-configs.py --scan <models-dir>     # pick the best-fitting GGUF
```

Evaluates every `.gguf` in `<dir>` and prints a markdown table with the best-fit flags.

## Multi-config scan

```bash
./scripts/scan-all.sh <models-dir> --vram 6144 --ram 32768 --ctx 128000
```

Scans all models across multiple KV cache configurations (turbo4/turbo3_tcq, turbo3_tcq/turbo3_tcq, turbo4/turbo4, turbo3_tcq/turbo2_tcq) and outputs a combined markdown or CSV table (`--format csv`). See `./scripts/scan-all.sh --help` for all options.

VRAM is allocated in strict order:

1. **Dense backbone** — always GPU-resident.
2. **KV cache** — capped to whatever fits after dense, and rounded down to a multiple of 256 (llama.cpp pads `n_ctx` up to that multiple).
3. **Experts** — fill whatever VRAM is left. The rest go to RAM.

There is no expert floor: if dense + KV consume the budget, every expert goes to RAM and per-token routing pulls them from CPU. The script's verdict surfaces this when `gpu_experts < active`.

# Run

## Quick start with `run-server.sh`

The repo includes `scripts/run-server.sh` to launch the server with MoE expert routing and TurboQuant KV without manually composing flags. It auto-detects `--n-cpu-moe`, `-c`, and KV types from `moe-configs.py`:

```bash
# Auto-size from GGUF
./scripts/run-server.sh --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf

# Explicit expert count, 128K context, custom port
./scripts/run-server.sh -m ~/models/Kimi-K2.6-UD-Q4_K_XL.gguf \
    --n-cpu-moe 382 --ctx 128000 --port 8081 --alias kimi-k2.6

# Tighter KV, fewer threads for a hybrid CPU
./scripts/run-server.sh -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
    --n-cpu-moe 237 -ctk turbo3_tcq -ctv turbo3_tcq \
    --threads 8 --alias qwen3.6
```

See `./scripts/run-server.sh --help` for all options.

## Empirically working command

On this hardware (RTX A1000 6 GiB, 32 GiB RAM):

```bash
./llama.cpp/build/bin/llama-server \
  -m ~/Downloads/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --alias qwen3.6-35b \
  --n-gpu-layers 999 \
  --n-cpu-moe 237 \
  -ctk turbo4 \
  -ctv turbo3_tcq \
  -c 128000 \
  -fa on \
  --fit off \
  -np 1 \
  --threads 8 \
  --host 0.0.0.0 --port 8080 \
  --no-mmap
```

Key flags:

- `--n-gpu-layers 999` + `--n-cpu-moe N` — every non-expert tensor on GPU, `N` experts pinned in RAM. Re-derive `N` and `-c` with `scripts/moe-configs.py` whenever model or ctx changes. The default context is 128000 tokens.
- `-ctk turbo4 -ctv turbo3_tcq` — default asymmetric KV: lossless keys (4.25 bpv), tight values (3.25 bpv TCQ). GPU-resident. **Do not use `-nkvo`.** With `-c 128000` (default), this costs ~4.6 GiB VRAM for the KV cache.
- `-fa on` — flash attention; required for efficient quantized KV. Needs `GGML_CUDA_FA_ALL_QUANTS=ON` at build time.
- `--fit off` — honor `--n-cpu-moe` verbatim instead of llama.cpp's auto-fit.
- `-np 1` — single slot; multiple slots duplicate KV state.
- `--no-mmap` — load experts into anonymous RAM so they stay process-resident. Add `--mlock` for steady-state benchmarking (see below).

## Pinning expert weights (`--no-mmap --mlock`)

Without locking, the kernel can evict CPU-side experts under memory pressure, causing multi-second stalls when faulted back in.

- `--no-mmap` — load with `read()` into anonymous heap; pages owned by the process. Safe alone, no privilege required.
- `--mlock` — `mlock()` weight pages so the kernel cannot evict them. Requires raised `RLIMIT_MEMLOCK` (the default 64 KiB silently caps multi-GiB models without aborting).

| Situation                              | Flags                                                                        |
|----------------------------------------|------------------------------------------------------------------------------|
| Casual interactive use, FIT=OK         | neither (default mmap)                                                       |
| Stable tok/s, FIT=OK                   | `--no-mmap --mlock`                                                          |
| FIT=OK under memory pressure           | `--no-mmap --mlock`                                                          |
| RAM-over row in the table              | **do not** use `--mlock` — it will OOM the box; rely on mmap-paging instead. |

Verify with `cat /proc/meminfo | grep Mlocked` after start: it should jump by the model's on-disk size. If not, `mlock()` is being silently denied — usually a `ulimit -l` set in a different shell. Raise via `ulimit -l <KiB>` (as root, in the same shell) or permanently via `memlock` in `/etc/security/limits.conf`.

## CPU thread count (`--threads`)

`--threads` sets the number of CPU worker threads used to run the expert MLPs that `--n-cpu-moe` keeps in RAM. On hybrid Intel CPUs (Alder Lake and later, including 12th–14th Gen Core and Core Ultra) **set this to the number of P-cores only**. E-cores have lower per-core throughput and a different cache hierarchy; mixing them into the same parallel MLP gemm causes the P-cores to wait on the slowest E-core finisher every step, dropping decode tok/s. Hyper-threading siblings on the P-cores add contention for the same vector units and also hurt; one thread per P-core is the right setting.

This development host is a **13th Gen Intel Core i7-13850HX**: 8 P-cores + 12 E-cores, 28 logical threads total. Canonical setting: `--threads 8` (one per P-core).

For any other CPU, look up the **P-core count specifically** (not total cores, not total threads) and use that:

| CPU class                                  | `--threads` rule                                         |
|--------------------------------------------|----------------------------------------------------------|
| Intel hybrid (12th Gen+ Core, Core Ultra)  | number of P-cores (e.g. i7-13850HX → 8)                  |
| Intel non-hybrid (11th Gen and earlier Xeon/Core) | number of physical cores (ignore HT siblings)     |
| AMD Ryzen / EPYC (Zen 2+)                  | number of physical cores (ignore SMT siblings)           |
| Apple Silicon                              | number of P-cores                                        |

Pinning helps too: `taskset -c 0-7 ./llama-server ...` (or the P-core CPU-list from `lscpu --extended`) keeps the scheduler from migrating workers onto E-cores or HT siblings under load.

## K/V cache types (`--cache-type-k / --cache-type-v`)

The script's default KV pair is **`turbo4` for keys** and **`turbo3_tcq` for values** (`--cache-type-k turbo4 --cache-type-v turbo3_tcq`). Keys are lossless (4.25 bpv) while values use 3.25 bpv TCQ — this asymmetric pairing gives lossless KV for the attention numerator while keeping values ~5× compressed.

All supported types with their relative costs:

| Type           | bpv  | Factor vs fp16 | KV size (rel.) |
|----------------|------|----------------|----------------|
| `turbo4`       | 4.25 | 0.266          | ×3.8 smaller |
| `turbo3_tcq`   | 3.25 | 0.203          | ×4.9 smaller |
| `turbo3`       | 3.25 | 0.203          | ×4.9 smaller |
| `turbo2_tcq`   | 2.25 | 0.141          | ×7.1 smaller |
| `turbo2`       | 2.25 | 0.141          | ×7.1 smaller |
| `f16` / `bf16` | 2.0  | 1.0            | baseline |
| `q8_0`         | ~1.0 | ~0.515         | ×1.9 smaller |
| `q5_1`         | ~1.3 | ~0.33          | ×3.0 smaller |
| `q5_0`         | ~1.25| ~0.312         | ×3.2 smaller |
| `q4_1`         | ~1.1 | ~0.275         | ×3.6 smaller |
| `q4_0`         | 1.0  | 0.5            | ×4.0 smaller |
| `iq4_nl`       | 1.0  | 0.5            | ×4.0 smaller |
| `f32`          | 4.0  | 2.0            | ×0.5 (2× larger) |

The `--scan` table below uses the default asymmetric pair (`turbo4`/`turbo3_tcq`). To try tighter compression, pass `--cache-type-k turbo3_tcq --cache-type-v turbo3_tcq` (symmetric turbo3, ~14% smaller KV, slightly more KLD at long context) or `--cache-type-k turbo3_tcq --cache-type-v turbo2_tcq` (aggressive, ~30% smaller than default).

## Prefill vs. decode speed

During a file read (the prompt), token throughput hits ~200 tok/s. During reasoning and output, it drops to ~20 tok/s. This is the prefill/decode gap inherent to autoregressive transformers, amplified by MoE routing on CPU.

**Prefill** is batched: the entire prompt is tokenized and every token is processed in parallel via large GPU matrix multiplies. The GPU is fully utilized.

**Decode** is sequential: each new token requires a full forward pass, and the output becomes the next input. This is memory-bandwidth bound — the model is read but only one token is produced. On this hardware with 128K context, only ~19 of the 256 experts fit on GPU — the router picks 8 active experts per token, and several of those live in RAM. After the GPU runs attention + dense layers, the router's selected experts are fetched over PCIe and their MLP gemms run on a single CPU thread (per step). The GPU then waits for that CPU execution to finish before summing back into the residual stream. Decode latency is gated by single-threaded CPU MLP compute, not GPU throughput.

# Models on disk — recommended settings

Generated with `python3 scripts/moe-configs.py --scan <models-dir> --cache-type-k turbo4 --cache-type-v turbo3_tcq --ctx 128000` (default asymmetric KV pair, 128K context). 30B variants run at `context_length = 40960` (model max); 35B variants at `-c 128000` are well below their 262144 trained ctx — the 6 GiB VRAM budget is the binding constraint. FIT respects both the 6 GiB VRAM budget and the ~32 GiB RAM budget.

The default `--ctx 128000` is a practical sweet spot for long-term focus on this hardware. At this context the KV cache (with `turbo4`/`turbo3_tcq`) costs ~4.6 GiB — leaving just enough headroom for a reasonable number of experts on GPU. This lets the model attend to multi-page documents, long codebases, and extended conversations without hitting the VRAM wall.

| Model file                           | Size  | `--n-cpu-moe` | GPU exp     | `-c`     | VRAM used | RAM used   | tokens/sec | FIT      |
|--------------------------------------|------:|--------------:|------------:|---------:|----------:|-----------:|:----------:|----------|
| Qwen3-30B-A3B-Q2_K.gguf              |  11 G |            71 |   57 / 128  |    40960 |  6080 MiB |   5551 MiB |            | **OK**   |
| Qwen3-30B-A3B-Q3_K_S.gguf            |  13 G |            81 |   47 / 128  |    40960 |  6053 MiB |   7518 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-MXFP4_MOE.gguf       |  18 G |           239 |   17 / 239  |   128000 |  6101 MiB |  16932 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-Q8_0.gguf            |  18 G |           247 |    9 / 247  |   128000 |  6033 MiB |  31492 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-UD-IQ3_S.gguf        |  22 G |           215 |   41 / 215  |   128000 |  6105 MiB |   9270 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-UD-Q4_K_S.gguf       |  19 G |           237 |   19 / 237  |   128000 |  6076 MiB |  16181 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf      |  18 G |           239 |   17 / 239  |   128000 |  6142 MiB |  17514 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf      |  18 G |           242 |   14 / 242  |   128000 |  6143 MiB |  21549 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf      |  18 G |           245 |   11 / 245  |   128000 |  6091 MiB |  26609 MiB |            | **OK**   |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf      |  18 G |           247 |    9 / 247  |   128000 |  6120 MiB |  32882 MiB |            | **ram**   |
| gemma-4-26B-A4B-it-MXFP4_MOE.gguf    |  18 G |           116 |   12 / 116  |   128000 |  6096 MiB |  12062 MiB |            | **OK**   |
| gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf   |  19 G |           120 |    8 / 120  |   128000 |  6082 MiB |  18508 MiB |            | **OK**   |
| gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf   |  18 G |           122 |    6 / 122  |   128000 |  6025 MiB |  22705 MiB |            | **OK**   |

All 35B and gemma-4 variants land at `-c 128000` (default context): dense + KV consume ~6 GiB VRAM, leaving 0–40 GPU experts depending on quant quality. UD-Q4_K_S is the recommended default — good quant quality, 19 GPU experts keep the active set mostly on-GPU, and ~16 GiB RAM headroom for safety. Q8_K_XL exceeds the RAM budget; do not `--mlock` it. Q8_0 is the best overall fit if you need maximum quality and can tolerate more CPU expert routing (only 9 GPU experts).

# Profiling

```bash
sudo nsys profile -o llama_profile ./llama.cpp/build/bin/llama-server ...
```

Open `llama_profile.nsys-rep` in NVIDIA Nsight Systems. Useful llama-server logging flags: `--perf --verbosity 4 --log-verbosity 4`.

# Browser

llama-server ships a built-in chat UI. Once the server is running, open `http://localhost:8080` to test the model interactively before wiring it into a coding agent.

# Claude Code redirect

`~/.claude/llamacpp.settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8080",
    "ANTHROPIC_AUTH_TOKEN": "local-dev",
    "ANTHROPIC_MODEL": "qwen3.6-35b"
  }
}
```

Or via environment:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:8080/v1"
export ANTHROPIC_AUTH_TOKEN="local-development"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_MODEL="qwen3.6-35b"

claude --settings ~/.claude/llamacpp.settings.json
```

`--alias qwen3.6-35b` on llama-server makes `ANTHROPIC_MODEL` resolve correctly.

# OpenCode redirect

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (local)",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "qwen3.6-35b": {}
      }
    }
  },
  "model": "llamacpp/qwen3.6-35b"
}
```

The model id under `models` must match `--alias` on llama-server. Pick the active model at runtime with `opencode` → `/models`, or pin it with the top-level `model` key as above.

# Pi redirect

`~/.pi/agent/models.json`:

```json
{
  "providers": {
    "llama.cpp (local)": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "Qwen3.6-35b"
        }
      ]
    }
  }
}
```

Point Pi at the same `llama-server` instance running locally. The provider name (`"llama.cpp (local)"`) is an arbitrary label; the `baseUrl` must match the llama-server address, and the model `id` should correspond to the model loaded.

# Future platform — Kimi K2.6

Kimi K2.6 is a sparse MoE model much larger than the Qwen3.6 family. We are profiling its fit across different host platforms.

See **[docs/kimi-k2.6.md](docs/kimi-k2.6.md)** for full details: experiment host hardware, model metadata, merge instructions, AVX-512 build, sizing scans, and hypothetical RTX 5090 / RTX 4090 platform analysis.

| Host | VRAM | GPU experts @128K | RAM used | VRAM headroom | Recommendation |
|------|------|-------------------|----------|---------------|----------------|
| RTX 5090 | 32 GiB | 12 / 384 | 527310 MiB | ~125 MiB | ✅ Production target |
| RTX 4090 | 24 GiB | 6 / 384 | 535815 MiB | ~438 MiB | ⚠️ Viable, fewer GPU experts |
| T4 | 16 GiB | 0 / 384 | 544320 MiB | ~750 MiB | ⚠️ Viable, fewer GPU experts |

# References

- buun-llama-cpp (fork): https://github.com/spiritbuun/buun-llama-cpp
- TCQ paper / dataset: https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache
- Qwen3.6 35B-A3B GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main
- Gemma 4 26B-A4B-it GGUFs: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main
- Kimi-K2.6 UD-Q4_K_XL GGUFs: https://huggingface.co/unsloth/Kimi-K2.6-GGUF/tree/main/UD-Q4_K_XL (full details in [docs/kimi-k2.6.md](docs/kimi-k2.6.md))

# License

MIT License. See [LICENSE](LICENSE) for the full text.
