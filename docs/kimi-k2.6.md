# Kimi K2.6 — Profiling Notes

Kimi K2.6 is a sparse MoE model much larger than the Qwen3.6 family. We are profiling its fit across different host platforms.

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 530 GB | DDR4 ECC |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Model

We are testing with the **UD-Q4_K_XL** quant from Unsloth — 14 shards totalling ~544 GiB on disk, merged to a single `Kimi-K2.6-UD-Q4_K_XL.gguf` (~544 GiB).

### Merging the split GGUF

Unsloth ships Kimi-K2.6 as 14 split shards. Merge them into a single GGUF with `llama-gguf-split` from the same build used to run the server:

```bash
./llama.cpp/build/bin/llama-gguf-split --merge \
  ~/Downloads/models/Kimi-K2.6/Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf \
  ~/Downloads/models/Kimi-K2.6/Kimi-K2.6-UD-Q4_K_XL.gguf
```

Pass only the **first** shard (`00001-of-00014`) plus the desired output path; `llama-gguf-split` discovers the remaining shards from the filename pattern and refuses if any of `00002`..`00014` are missing. All 14 files must be present in the same directory before merging.

After the merge succeeds, the shards can be deleted; only the merged file is needed at runtime.

### Model metadata

| Property | Value |
|----------|-------|
| Layers | 61 |
| Experts (total) | 384 |
| Active per token | 8 |
| Dense backbone | 12343 MiB |
| One expert | ~1418 MiB |
| All experts | ~544320 MiB (~531 GiB) |
| Trained context | 262144 |

## Build with AVX-512

The Xeon Gold 5120 (Skylake-SP) supports AVX-512F — build with `GGML_AVX512=ON`. `-DGGML_NATIVE=ON` handles `-march=native` automatically.

```bash
cd llama.cpp
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
cmake --build build -j$(nproc)
```

Key flags:
- **`-DGGML_AVX512=ON`** — enables the AVX-512 code paths in ggml source (Skylake-SP supports AVX-512F, CD, ER, PF, VL only — not VBMI, VNNI, or BF16).
- **`-DGGML_NATIVE=ON`** — auto-detects the CPU and passes `-march=native` so the compiler emits the right instruction set (including AVX-512F on the Xeon Gold 5120).
- **`-DCMAKE_C/CXX_FLAGS="-O3"`** — maximum optimization level for CPU-side compute.
- **`-DCMAKE_CUDA_ARCHITECTURES=75`** — targets the T4's Turing architecture (sm_75).
- **`-DCMAKE_BUILD_TYPE=Release`** — optimize build type.

AVX-512 accelerates the CPU-side expert MLPs that `--n-cpu-moe` keeps in RAM. Without it, the Xeon Gold 5120 falls back to AVX2 (256-bit), halving the per-cycle throughput of the expert GEMM kernels.

## Sizing

The host has 16384 MiB VRAM and 530 GiB RAM. Note: the T4 ships with ECC memory enabled by default, which reserves ~1 GiB for error correction — reducing usable VRAM to 15360 MiB. Disable it before profiling:

```bash
sudo nvidia-smi -e 0
# Disabled ECC support for GPU 00000000:00:05.0.
# All done.
# Reboot required.
```

After disabling ECC, `nvidia-smi` reports the full 16384 MiB:

| GPU | Name | Memory-Usage |
|-----|------|-------------|
| 0 | Tesla T4 | 0 MiB / 16384 MiB |

With `turbo3_tcq` KV at `-c 20000`, the script fits the model:

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/Kimi-K2.6/Kimi-K2.6-UD-Q4_K_XL.gguf \
  --vram 16384 --ram 542720 \
  --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq \
  --ctx 20000
```

Result:

```
Context:          19968  (model max: 262144, VRAM-fit max: 157184)
  KV cache:             513.12 MiB
  Experts on GPU (  2):    2835.00 MiB
  VRAM used:          15691.53 MiB
  RAM used:          541485.00 MiB
  Flags: --n-cpu-moe 382 -c 19968 -ctk turbo3_tcq -ctv turbo3_tcq
```

**Note:** We use `-c 20000` (rounded to 19968 by CTX_PAD alignment) as a practical profiling limit — it keeps the KV cache small (~513 MiB) for faster iteration during experimentation. The hardware can easily support much larger contexts: the VRAM-fit max is 157184 tokens at `turbo3_tcq`, and 128000 is the preferred default for long-term work. Lower `-c` simply keeps VRAM profiling manageable while the model is still being profiled.

### Config strategy comparison

| Configuration | KV cache | GPU experts | VRAM used | RAM used | Notes |
|---------------|----------|-------------|-----------|----------|-------|
| **`@ 20K — profiling`** | | | | | |
| `turbo3_tcq` / `turbo3_tcq` | 513 MiB | 2 / 382 | 15692 MiB | 541485 MiB | Default for profiling |
| `turbo4` / `turbo4` | 671 MiB | 2 / 382 | 15850 MiB | 541485 MiB | Lossless keys |
| `turbo4` / `turbo3_tcq` | 597 MiB | 2 / 382 | 15775 MiB | 541485 MiB | Asymmetric: lossless K, tight V |
| `turbo3_tcq` / `turbo2_tcq` | 439 MiB | 2 / 382 | 15617 MiB | 541485 MiB | Tightest |
| **`@ 128K — long-term`** | | | | | |
| `turbo3_tcq` / `turbo3_tcq` | 3289 MiB | 0 / 384 | 15633 MiB | 544320 MiB | 0 GPU; 8 active experts on CPU |
| `turbo4` / `turbo4` | 4035 MiB | 0 / 384 | 16379 MiB | 544320 MiB | Lossless; 0 GPU; VRAM nearly full |
| `turbo4` / `turbo3_tcq` | 3825 MiB | 0 / 384 | 16169 MiB | 544320 MiB | Asymmetric; 0 GPU |
| `turbo3_tcq` / `turbo2_tcq` | 2813 MiB | 0 / 384 | 15156 MiB | 544320 MiB | Tightest; still 0 GPU at 128K |

At this VRAM budget only 2 of 384 experts fit on GPU (128K context: 0). The remaining experts run on CPU — by design — and the 530 GiB RAM budget accommodates all 382 CPU-side experts (~531 GiB) with only ~1.2 GiB headroom.

## Canonical run command

```bash
./llama.cpp/build/bin/llama-server \
  -m ~/Downloads/models/Kimi-K2.6/Kimi-K2.6-UD-Q4_K_XL.gguf \
  --alias kimi-k2.6 \
  --n-gpu-layers 999 \
  --n-cpu-moe 382 \
  -ctk turbo3_tcq \
  -ctv turbo3_tcq \
  -c 20000 \
  -fa on \
  --fit off \
  --threads 28 \
  --host 0.0.0.0 --port 8080
```

- `--n-cpu-moe 382` — only 2 experts fit on the 16 GiB GPU; the rest run on CPU.
- `-c 20000` — profiling context; increase toward 128000 for long-term use when ready.
- `--threads 28` — set to the full thread count for the Xeon Gold 5120 (14 physical cores × 2 threads each) since it's a non-hybrid processor.
- `--no-mmap` is omitted here — add it if experts need to stay pinned in process RAM (see [main README Run section](../README.md#run) for guidance).

### Hypothetical future platforms

The sizing script can also evaluate what a heavier host would enable. These are hypothetical setups used for capacity planning — real hardware may differ.

#### RTX 5090 (32 GiB VRAM)

| KV config | `--n-cpu-moe` | GPU exp | `-c` | VRAM used | RAM used | FIT |
|-----------|--------------:|--------:|-----:|----------:|---------:|-----|
| `turbo3_tcq` | 372 | 12 / 384 | 128000 | 32643 MiB | 527310 MiB | **OK** |
| `turbo4` / `turbo4` | 11 / 384 | 11 / 384 | 128000 | 32643 MiB | 528727 MiB | **OK** |
| `turbo4` / `turbo3_tcq` | 12 / 384 | 12 / 384 | 128000 | 32643 MiB | 527310 MiB | **OK** |

At 128K context with `turbo3_tcq` the RTX 5090 holds **12 GPU experts**. The remaining 125 MiB VRAM headroom is thin — bumping to `turbo4`/`turbo4` (lossless K+V) pushes KV from 3289 MiB to 4035 MiB, still fitting with ~280 MiB headroom. At full 262144 context the VRAM-fit max is 262144, and RAM headroom is ~500 GiB — the GPU is the binding constraint.

**Verdict:** RTX 5090 is the best fit at 128K context and the clear production target.

#### RTX 4090 (24 GiB VRAM)

| KV config | `--n-cpu-moe` | GPU exp | `-c` | VRAM used | RAM used | FIT |
|-----------|--------------:|--------:|-----:|----------:|---------:|-----|
| `turbo3_tcq` | 378 | 6 / 384 | 128000 | 24138 MiB | 535815 MiB | **OK** |
| `turbo4` / `turbo4` | 6 / 384 | 6 / 384 | 128000 | 24138 MiB | 535815 MiB | **OK** |
| `turbo4` / `turbo3_tcq` | 7 / 384 | 7 / 384 | 128000 | 24138 MiB | 534397 MiB | **OK** |

At 128K context with `turbo3_tcq` the RTX 4090 holds **6 GPU experts** (2 fewer than the 8 active per token, meaning every token routes at least 2 experts from RAM). The 438 MiB VRAM headroom is generous enough that even `turbo4`/`turbo4` still fits.

**Verdict:** The RTX 4090 fits at 128K context with 6 GPU experts. It is viable but offers fewer GPU experts than the 5090 — more GPU experts reduces PCIe expert fetches, which is an opportunistic optimization rather than a requirement.

#### RTX 4090 (24 GiB VRAM) at max context

At `-c 262144` the RTX 4090 drops to **3 GPU experts**, since the larger KV cache leaves less room on the GPU.

| KV config | `--n-cpu-moe` | GPU exp | `-c` | VRAM used | RAM used | FIT |
|-----------|--------------:|--------:|-----:|----------:|---------:|-----|
| `turbo3_tcq` | 381 | 3 / 384 | 262144 | 23332 MiB | 540067 MiB | **OK** |

**Verdict:** At 262144 context the RTX 4090 fits with 3 GPU experts. Fewer GPU experts means more PCIe traffic for expert weight fetches — a valid trade-off if the larger context is needed.

---

### Summary: host comparison for Kimi K2.6 at 128K

| Host | VRAM | GPU experts | RAM used | VRAM headroom | Recommendation |
|------|------|------------|----------|---------------|----------------|
| RTX 5090 | 32 GiB | 12 / 384 | 527310 MiB | ~125 MiB | ✅ Production target |
| RTX 4090 | 24 GiB | 6 / 384 | 535815 MiB | ~438 MiB | ⚠️ Viable, fewer GPU experts |
| T4 | 16 GiB | 0 / 384 | 544320 MiB | ~750 MiB | ⚠️ Viable, fewer GPU experts |

## References

- Kimi-K2.6 UD-Q4_K_XL GGUFs: https://huggingface.co/unsloth/Kimi-K2.6-GGUF/tree/main/UD-Q4_K_XL
- Intel Xeon Gold 5120 spec sheet: https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html
- NVIDIA T4 / TU104 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf

## License

MIT License. See [LICENSE](LICENSE) for the full text.
