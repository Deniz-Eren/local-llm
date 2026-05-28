# Gemma-4-26B-A4B-it — Profiling Notes

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## MXFP4_MOE

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/gemma-4-26B-A4B-it-MXFP4_MOE.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           gemma-4-26B-A4B-it-MXFP4_MOE.gguf
  Layers:          30
  Experts (total): 128  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2459.20 MiB
  All experts:            13310.01 MiB
  One expert:               103.98 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    4844.29 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2459.20 MiB
  KV cache:                4844.29 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 11 layers,  47):    4880.34 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16183.83 MiB  ( 15.80 GiB)
  Headroom:                 200.17 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 19 layers,  81):    8429.68 MiB
  Headroom:              731570.32 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 19 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  13,310.01 MiB
  Single Expert Weight:   103.98 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   26.00
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        240.7
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        481.3 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        962.6
  PCIe 5.0 x16                    64,000      50,049      1,925.2

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   831.88
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       19.8
  DDR4-2666         42,656      18,306       22.0
  DDR4-3200         51,200      21,973       26.4 ◄ baseline
  DDR4-3600         57,600      24,719       29.7
  DDR5-4800         76,800      32,959       39.6
  DDR5-5600         89,600      38,452       46.2
  DDR5-6000         96,000      41,199       49.5
  DDR5-7200        115,200      49,438       59.4

═══════════════════════════════════════════════════════════════════
```

## Alternative Quantizations

### UD-Q6_K_XL

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf
  Layers:          30
  Experts (total): 128  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2459.20 MiB
  All experts:            19741.92 MiB
  One expert:               154.23 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    4844.29 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2459.20 MiB
  KV cache:                4844.29 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU (  7 layers,  30):    4606.45 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   15909.94 MiB  ( 15.54 GiB)
  Headroom:                 474.06 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 23 layers,  98):   15135.47 MiB
  Headroom:              724864.53 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 23 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  19,741.92 MiB
  Single Expert Weight:   154.23 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   38.56
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        162.2
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        324.5 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        649.0
  PCIe 5.0 x16                    64,000      50,049      1,298.0

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   1,233.87
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       13.4
  DDR4-2666         42,656      18,306       14.8
  DDR4-3200         51,200      21,973       17.8 ◄ baseline
  DDR4-3600         57,600      24,719       20.0
  DDR5-4800         76,800      32,959       26.7
  DDR5-5600         89,600      38,452       31.2
  DDR5-6000         96,000      41,199       33.4
  DDR5-7200        115,200      49,438       40.1

═══════════════════════════════════════════════════════════════════
```

### UD-Q8_K_XL

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf
  Layers:          30
  Experts (total): 128  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2518.98 MiB
  All experts:            23821.89 MiB
  One expert:               186.11 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    4844.29 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2518.98 MiB
  KV cache:                4844.29 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU (  6 layers,  26):    4764.38 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16127.65 MiB  ( 15.75 GiB)
  Headroom:                 256.35 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 24 layers, 102):   19057.51 MiB
  Headroom:              720942.49 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 24 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  23,821.89 MiB
  Single Expert Weight:   186.11 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   46.53
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        134.5
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        268.9 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        537.8
  PCIe 5.0 x16                    64,000      50,049      1,075.7

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   1,488.87
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       11.1
  DDR4-2666         42,656      18,306       12.3
  DDR4-3200         51,200      21,973       14.8 ◄ baseline
  DDR4-3600         57,600      24,719       16.6
  DDR5-4800         76,800      32,959       22.1
  DDR5-5600         89,600      38,452       25.8
  DDR5-6000         96,000      41,199       27.7
  DDR5-7200        115,200      49,438       33.2

═══════════════════════════════════════════════════════════════════
```

## Comparison summary

| Quant | GPU layers | GPU experts | Context | VRAM headroom | Est. t/s (DDR4-3200) |
|-------|-----------:|------------:|--------:|--------------:|---------------------:|
| MXFP4_MOE | 11 | 47 | 262K | 200 MiB | 26.4 |
| UD-Q6_K_XL | 7 | 30 | 262K | 474 MiB | 17.8 |
| UD-Q8_K_XL | 6 | 26 | 262K | 256 MiB | 14.8 |

**Verdict:** Gemma-4-26B-A4B-it fits well on 16 GiB VRAM with all quantizations. MXFP4_MOE offers the best balance — 11 GPU layers and 26 t/s. The UD-Q6_K_XL has the most VRAM headroom (474 MiB) but slower generation. All use the smaller turboKV cache (4844 MiB at 262K) thanks to Gemma 4's different attention pattern compared to Qwen3.

## References

- Gemma-4 GGUFs: https://huggingface.co/unsloth
