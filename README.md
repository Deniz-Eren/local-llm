# local-llm

A workbench for running large **Mixture-of-Experts** LLMs locally on consumer hardware with a tight VRAM budget. Dense weights and KV cache stay GPU-resident; experts live in RAM and run on CPU when routed (`--n-cpu-moe`). KV size is brought under control with quantized cache types from a llama.cpp fork.

Includes a sizing tool (`scripts/moe-configs.py`) that reads any GGUF, takes your VRAM/RAM as parameters, and prints the `--n-gpu-layers / --n-cpu-moe / -c` flags that fit.

## How it works

A 35B MoE has only ~3B parameters active per token (8 of 256 experts on Qwen3.6-35B-A3B). The other ~31B can sit cold in slow memory at no per-token cost — provided we can route the active 8 into compute quickly.

`--n-cpu-moe N` keeps `N` experts pinned in RAM and **runs their MLP on CPU threads in place**; it does *not* copy expert weights to the GPU. Per token: GPU runs attention + dense layers, the router picks 8 experts, those 8 MLPs run wherever their weights live, and only the small output activations cross PCIe to be summed back into the residual stream. Throughput is gated by CPU MLP compute, not PCIe bandwidth on weight transfers.

The KV cache is the other VRAM consumer. At 262144 tokens it would be ~20 GiB at FP16 — far past a 6 GiB budget. The fork's TurboQuant / TCQ KV types compress this ~5× (`turbo3_tcq` = 3.25 bpv) at ~97% of `q8_0` decode speed and constant cost across context, so KV stays GPU-resident at any context.

## KV-cache strategy

KV is GPU-resident on this build. The fork ships **Trellis-Coded Quantization (TCQ)** KV types named `turbo4`, `turbo3` / `turbo3_tcq`, and `turbo2` / `turbo2_tcq`.

| K / V pair                        | bpv  | KLD @2K / @7K   | KV size (rel.)         | Use when                                         |
|-----------------------------------|------|-----------------|------------------------|--------------------------------------------------|
| `turbo4` / `turbo4`               | 4.25 | lossless        | +20%                   | Maximum quality, VRAM headroom to spare.         |
| **`turbo3_tcq` / `turbo3_tcq`**   | 3.25 | 0.058 / 0.074   | baseline (canonical)   | **Default.** Beats FP16 short ctx, within 2% long. |
| `turbo3_tcq` / `turbo2_tcq`       | 2.75 | 0.078 / 0.101   | −15%                   | Stretch to longer contexts.                      |
| `turbo2_tcq` / `turbo2_tcq`       | 2.25 | 0.101 / 0.136   | −31%                   | Maximum compression, accept some quality loss.   |

KLD numbers from the upstream Qwen3.5-27B Q6_K bench. `q8_0` (~1.06 bytes/elem) works as a plain CUDA fallback if TCQ ever misbehaves on a new model.

# Build

llama.cpp fork that adds the `turbo*` KV types with CUDA kernels (KV stays on the GPU, no `-nkvo` required):

- repo: `git@github.com:spiritbuun/buun-llama-cpp.git`, branch: `master`

This fork is a **temporary** dependency: once upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) lands TurboQuant / TCQ support, this repo will switch back to upstream.

```
git clone -b master git@github.com:spiritbuun/buun-llama-cpp.git llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES=86` targets the RTX A1000's sm_86 — adjust for your GPU. `GGML_CUDA_FA_ALL_QUANTS=ON` is required so flash-attention kernels are compiled for the quantized KV types; `-fa on` with `turbo*` KV silently falls back without it.

# Sizing

```
python3 scripts/moe-configs.py <model-path> --ctx 262144
python3 scripts/moe-configs.py --scan <models-dir> --ctx 262144     # pick the best-fitting GGUF
```

Reports the VRAM/RAM breakdown and prints the flags to use. Host budget is configurable via `--vram`, `--ram`, `--tax`, `--os-reserve` (all in MiB). Defaults: 6144 / 32768 / 650 / 5120, giving an effective ~27 GiB RAM budget for llama.cpp. Override on the command line; do not edit defaults inside the script.

VRAM is allocated in strict order:

1. **Dense backbone** — always GPU-resident.
2. **KV cache** — capped to whatever fits after dense, and rounded down to a multiple of 256 (llama.cpp pads `n_ctx` up to that multiple).
3. **Experts** — fill whatever VRAM is left. The rest go to RAM.

There is no expert floor: if dense + KV consume the budget, every expert goes to RAM and per-token routing pulls them from CPU. The script's verdict surfaces this when `gpu_experts < active`.

# Run

Empirically working command on this hardware (RTX A1000 6 GiB, 32 GiB RAM):

```
./llama.cpp/build/bin/llama-server \
  -m ~/Downloads/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
  --alias qwen3.6-35b \
  --n-gpu-layers 999 \
  --n-cpu-moe 256 \
  -ctk turbo3_tcq \
  -ctv turbo3_tcq \
  -c 185344 \
  -fa on \
  --fit off \
  -np 1 \
  --threads 8 \
  --host 0.0.0.0 --port 8080 \
  --no-mmap
```

Key flags:

- `--n-gpu-layers 999` + `--n-cpu-moe N` — every non-expert tensor on GPU, `N` experts pinned in RAM. Re-derive `N` and `-c` with `scripts/moe-configs.py` whenever model or ctx changes.
- `-ctk / -ctv turbo3_tcq` — symmetric TCQ 3-bit KV, GPU-resident. **Do not use `-nkvo`.**
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

## Prefill vs. decode speed

During a file read (the prompt), token throughput hits ~200 tok/s. During reasoning and output, it drops to ~20 tok/s. This is the prefill/decode gap inherent to autoregressive transformers, amplified by MoE routing on CPU.

**Prefill** is batched: the entire prompt is tokenized and every token is processed in parallel via large GPU matrix multiplies. The GPU is fully utilized.

**Decode** is sequential: each new token requires a full forward pass, and the output becomes the next input. This is memory-bandwidth bound — the model is read but only one token is produced. On this hardware the gap is wider because `--n-cpu-moe 256` puts all experts in RAM. After the GPU runs attention + dense layers, the router picks 8 active experts whose weights are pulled from RAM over PCIe and their MLP gemms run on a single CPU thread (per step). The GPU then waits for that CPU execution to finish before summing back into the residual stream. Decode latency is gated by single-threaded CPU MLP compute, not GPU throughput.

# Models on disk — recommended settings

Generated with `python3 scripts/moe-configs.py --scan <models-dir>`. 30B variants run at `context_length = 40960` (model max); 35B variants are capped well below their 262144 trained ctx by the 6 GiB VRAM budget. FIT respects both the 6 GiB VRAM budget and the ~27 GiB RAM budget.

| Model file                           | Size  | `--n-cpu-moe` | GPU exp     | `-c`     | VRAM used | RAM used   | FIT      |
|--------------------------------------|------:|--------------:|------------:|---------:|----------:|-----------:|----------|
| Qwen3-30B-A3B-Q2_K.gguf              |  11 G |            77 |   51 / 128  |    40960 |  6140 MiB |   6020 MiB | **OK**   |
| Qwen3-30B-A3B-Q3_K_S.gguf            |  13 G |            86 |   42 / 128  |    40960 |  6119 MiB |   7982 MiB | **OK**   |
| Qwen3.6-35B-A3B-UD-IQ3_S.gguf        |  13 G |           256 |    0 / 256  |   220416 |  6140 MiB |  11038 MiB | **OK**   |
| Qwen3.6-35B-A3B-UD-Q4_K_S.gguf       |  20 G |           256 |    0 / 256  |   192768 |  6144 MiB |  17478 MiB | **OK**   |
| Qwen3.6-35B-A3B-MXFP4_MOE.gguf       |  21 G |           256 |    0 / 256  |   185344 |  6144 MiB |  18136 MiB | **OK**   |
| Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf      |  21 G |           256 |    0 / 256  |   185344 |  6144 MiB |  18760 MiB | **OK**   |
| Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf      |  25 G |           256 |    0 / 256  |   185344 |  6144 MiB |  22796 MiB | **OK** (best fit) |
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf      |  30 G |           256 |    0 / 256  |   185344 |  6144 MiB |  27804 MiB | borderline (RAM 156 MiB over) |
| Qwen3.6-35B-A3B-Q8_0.gguf            |  35 G |           256 |    0 / 256  |   185856 |  6141 MiB |  32640 MiB | RAM over |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf      |  36 G |           256 |    0 / 256  |   183552 |  6140 MiB |  34080 MiB | RAM over |
| gemma-4-26B-A4B-it-MXFP4_MOE.gguf    |  16 G |           128 |    0 / 128  |   188672 |  5492 MiB |  13310 MiB | **OK**   |
| gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf   |  22 G |           128 |    0 / 128  |   188672 |  5492 MiB |  19742 MiB | **OK**   |
| gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf   |  26 G |           128 |    0 / 128  |   184832 |  5491 MiB |  23822 MiB | **OK**   |

All 35B variants land at `--n-cpu-moe 256` (= 0 GPU experts): dense + KV consume the VRAM budget, leaving none for experts. UD-Q5_K_XL is the best fit — highest-quality quant whose experts still fit in RAM (~4.8 GiB headroom) at `-c 185344`. UD-Q6_K_XL is 156 MiB over and runs with a small `-c` reduction. Q8_0 / UD-Q8_K_XL exceed RAM by 5–6 GiB; do not `--mlock` them.

# Profiling

```
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

```
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

# Future platform — Kimi K2.6

Scoping a heavier host for Kimi K2.6 (sparse MoE, much larger than the Qwen3.6 family). Target hardware under consideration:

- **GPU:** NVIDIA RTX 5090, 32 GiB VRAM
- **RAM:** 1 TiB

Sizing approach: build the full GGUF locally, then run `scripts/moe-configs.py` against it with `--vram 32768 --ram 1048576` to get the recommended `--n-cpu-moe / -c` and the dense + KV + expert breakdown. The verdict tells us whether the platform clears the budget at the desired context, and how much headroom remains for further quants or longer ctx.

We are testing with the **UD-Q4_K_XL** quant — 14 shards totalling ~544 GiB on disk, merging to a single `Kimi-K2.6-UD-Q4_K_XL.gguf` of ~544 GiB (583.7 GB).

## Merging the split GGUF

Unsloth ships Kimi-K2.6 as 14 split shards. Merge them into a single GGUF with `llama-gguf-split` from the same build used to run the server:

```
./llama.cpp/build/bin/llama-gguf-split --merge \
  ~/Downloads/models/Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf \
  ~/Downloads/models/Kimi-K2.6-UD-Q4_K_XL.gguf
```

Pass only the **first** shard (`00001-of-00014`) plus the desired output path; `llama-gguf-split` discovers the remaining shards from the filename pattern and refuses if any of `00002`..`00014` are missing. All 14 files must be present in the same directory before merging.

After the merge succeeds, the shards can be deleted; only the merged file is needed at runtime.

## Sizing run

Run `scripts/moe-configs.py` against the merged GGUF on each candidate host:

```
python3 scripts/moe-configs.py \
  ~/Downloads/models/Kimi-K2.6-UD-Q4_K_XL.gguf \
  --vram <vram-mib> --ram <ram-mib> --ctx 262144
```

Model fixed facts (from the GGUF metadata): 61 layers, 384 experts, 8 active per token, dense backbone 12343 MiB, one expert ~1418 MiB, all experts ~544320 MiB, KV cache ~6736 MiB at `turbo3_tcq` and `-c 262144`.

| GPU       | VRAM   | RAM   | `--n-cpu-moe` | GPU exp     | `-c`     | VRAM used | RAM used     | FIT     |
|-----------|-------:|------:|--------------:|------------:|---------:|----------:|-------------:|---------|
| RTX 5090  | 32 GiB | 1 TiB |           375 |   9 / 384   |   262144 | 31837 MiB |  531562 MiB  | **OK**  |
| RTX 4090  | 24 GiB | 1 TiB |           381 |   3 / 384   |   262144 | 23332 MiB |  540067 MiB  | **OK**  |

The RTX 5090 + 1 TiB RAM row clears the budget at full 262144 ctx with ~280 MiB VRAM headroom and ~500 GiB RAM headroom — i.e. the GPU is the binding constraint, not RAM. The RAM headroom is large enough to absorb a heavier KV pair (`turbo4`/`turbo4`) or a higher quant if desired.

The RTX 4090 + 1 TiB RAM row also fits at full ctx but holds only 3 GPU experts vs the 8 active per token, so on average ~5 of the 8 active expert MLPs per token run on CPU instead of GPU. Throughput is then dominated by CPU MLP compute. The 5090 row clears 8 GPU experts (with one to spare), so the active set runs entirely on the GPU and per-token throughput is GPU-bound.

# References

- buun-llama-cpp (fork): https://github.com/spiritbuun/buun-llama-cpp
- TCQ paper / dataset: https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache
- Qwen3.6 35B-A3B GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main
- Gemma 4 26B-A4B-it GGUFs: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main
- Kimi-K2.6 UD-Q4_K_XL GGUFs: https://huggingface.co/unsloth/Kimi-K2.6-GGUF/tree/main/UD-Q4_K_XL

# License

MIT License. See [LICENSE](LICENSE) for the full text.
