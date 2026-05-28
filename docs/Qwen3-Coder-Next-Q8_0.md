# Qwen3-Coder-Next-Q8_0 — Profiling Notes

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Final Run Command Used

Stable at 14 tokens/s, context +224k.
```bash
run-server.sh --model Qwen3-Coder-Next/Q8_0/Qwen3-Coder-Next-Q8_0-00001-of-00003.gguf --n-cpu-moe 48 -c 224512 -ctk q8_0 -ctv q8_0 --alias "Qwen3-Coder-Next-Q8_0" --threads 14 --no-mmap --mlock
```

## Model

Configuration script:
```
./scripts/moe-configs.py Qwen3-Coder-Next/Q8_0/Qwen3-Coder-Next-Q8_0.gguf --ctx 262144 --vram 16384 --ram 740000 --cache-type-k q8_0 --cache-type-v q8_0 --compute-overhead 3000
NOTE: Qwen3-Coder-Next-Q8_0.gguf: requested --ctx 262144 clamped to 224512 (VRAM-fit max=224512).
```

Configuration results:
```
NOTE: Qwen3-Coder-Next-Q8_0.gguf: requested --ctx 262144 clamped to 224512 (VRAM-fit max=224512).

═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3-Coder-Next/Q8_0/Qwen3-Coder-Next-Q8_0.gguf
  Layers:          48
  Experts (total): 512  (active per token: 10)
  Context:         224512  (model max: 262144, VRAM-fit max: 224512)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2541.56 MiB
  All experts:            78336.00 MiB
  One expert:               153.00 MiB
  KV cache (q8_0, eff 0.515x):   10839.72 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2541.56 MiB
  KV cache:               10839.72 MiB
  Compute/MTP buffer:      3000.00 MiB
  Experts on GPU (  0 layers,   0):       0.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16381.28 MiB  ( 16.00 GiB)
  Headroom:                   2.72 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 48 layers, 512):   78336.00 MiB
  Headroom:              661664.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK
  -> Only 0 experts on GPU; per-token routing needs 10 active. On average 10 of the active expert MLPs per token will run on CPU instead of GPU (slower per-token compute). Reduce --n-cpu-moe N to pin fewer layers to CPU, thereby keeping more layers (and their experts) on GPU.

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 48 -c 224512 -ctk q8_0 -ctv q8_0

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  78,336.00 MiB
  Single Expert Weight:   153.00 MiB
  Active Experts/token:   10
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   153.00
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256         40.9
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512         81.8 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        163.6
  PCIe 5.0 x16                    64,000      50,049        327.1

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   1,530.00
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       10.8
  DDR4-2666         42,656      18,306       12.0
  DDR4-3200         51,200      21,973       14.4 ◄ baseline
  DDR4-3600         57,600      24,719       16.2
  DDR5-4800         76,800      32,959       21.5
  DDR5-5600         89,600      38,452       25.1
  DDR5-6000         96,000      41,199       26.9
  DDR5-7200        115,200      49,438       32.3

═══════════════════════════════════════════════════════════════════
```

# `--n-cpu-moe 48` pins all 48 layers to CPU RAM; 0 layers on GPU. For shared-expert Qwen3-Coder-Next (512 experts across all 48 layers), 0 GPU layers ≈ 0 GPU experts, meaning all 10 active experts per token run on CPU.

# Alternative Quantizations

## Qwen3-Coder-Next UD-Q4_K_XL

Lower quantization — smaller weights, more GPU capacity.

```bash
python3 scripts/moe-configs.py \
  ~/models/Qwen3-Coder-Next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144 --compute-overhead 3000
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3-Coder-Next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
  Layers:          48
  Experts (total): 512  (active per token: 10)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2564.62 MiB
  All experts:            44740.00 MiB
  One expert:                87.38 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    5756.93 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2564.62 MiB
  KV cache:                5756.93 MiB
  Compute/MTP buffer:      3000.00 MiB
  Experts on GPU (  5 layers,  53):    4660.42 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   15981.96 MiB  ( 15.61 GiB)
  Headroom:                 402.04 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 43 layers, 459):   40079.58 MiB
  Headroom:              699920.42 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 43 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  44,740.00 MiB
  Single Expert Weight:   87.38 MiB
  Active Experts/token:   10
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   87.38
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256         71.6
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        143.2 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        286.4
  PCIe 5.0 x16                    64,000      50,049        572.8

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   873.83
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       18.9
  DDR4-2666         42,656      18,306       20.9
  DDR4-3200         51,200      21,973       25.1 ◄ baseline
  DDR4-3600         57,600      24,719       28.3
  DDR5-4800         76,800      32,959       37.7
  DDR5-5600         89,600      38,452       44.0
  DDR5-6000         96,000      41,199       47.1
  DDR5-7200        115,200      49,438       56.6

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q4_K_XL puts **5 GPU layers (53 GPU experts)** on the GPU — an improvement over Q8_0 (0 GPU layers). 43 layers on CPU. Predicted 25.1 t/s on DDR4-3200 — nearly double the Q8_0 speed (14.4 t/s). The turboKV cache (turbo4/turbo3_tcq) is smaller than q8_0, freeing VRAM for GPU experts.

## Qwen3-Coder-Next UD-Q8_K_XL

Higher-fidelity quantization — larger weights but still better than Q8_0.

```bash
python3 scripts/moe-configs.py \
  ~/models/Qwen3-Coder-Next/UD-Q8_K_XL/Qwen3-Coder-Next-UD-Q8_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144 --compute-overhead 3000
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3-Coder-Next/UD-Q8_K_XL/Qwen3-Coder-Next-UD-Q8_K_XL.gguf
  Layers:          48
  Experts (total): 512  (active per token: 10)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2566.87 MiB
  All experts:            79776.00 MiB
  One expert:               155.81 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    5756.93 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2566.87 MiB
  KV cache:                5756.93 MiB
  Compute/MTP buffer:      3000.00 MiB
  Experts on GPU (  3 layers,  32):    4986.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16309.80 MiB  ( 15.93 GiB)
  Headroom:                  74.20 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 45 layers, 480):   74790.00 MiB
  Headroom:              665210.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 45 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  79,776.00 MiB
  Single Expert Weight:   155.81 MiB
  Active Experts/token:   10
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   155.81
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256         40.2
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512         80.3 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        160.6
  PCIe 5.0 x16                    64,000      50,049        321.2

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   1,558.12
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       10.6
  DDR4-2666         42,656      18,306       11.7
  DDR4-3200         51,200      21,973       14.1 ◄ baseline
  DDR4-3600         57,600      24,719       15.9
  DDR5-4800         76,800      32,959       21.2
  DDR5-5600         89,600      38,452       24.7
  DDR5-6000         96,000      41,199       26.4
  DDR5-7200        115,200      49,438       31.7

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q8_K_XL puts **3 GPU layers (32 GPU experts)** — better than Q8_0 (0 GPU layers). 45 layers on CPU. Tight VRAM fit with only 74 MiB headroom. Predicted 14.1 t/s — same as Q8_0 but with GPU experts offloading some work. Both use turboKV cache.

## Comparison summary

| Quant | GPU layers | GPU experts | Context | VRAM headroom | Est. t/s (DDR4-3200) |
|-------|-----------:|------------:|--------:|--------------:|---------------------:|
| UD-Q4_K_XL | 5 | 53 | 262K | 402 MiB | 25.1 |
| UD-Q8_K_XL | 3 | 32 | 262K | 74 MiB | 14.1 |
| Q8_0 | 0 | 0 | 224K | 3 MiB | 14.4 |

Note: Q8_0 uses q8_0 q8_0 KV cache (10840 MiB at 224K), while the UD variants use turbo4/turbo3_tcq (5757 MiB at 262K). The smaller turboKV cache is what enables GPU experts for the UD variants.
