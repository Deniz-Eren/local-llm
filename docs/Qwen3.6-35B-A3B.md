# Qwen3.6-35B-A3B — Profiling Notes

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Model

Configuration script:
```
python3 scripts/moe-configs.py \
  ~/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Configuration results:
```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2543.10 MiB
  All experts:            32640.00 MiB
  One expert:               127.50 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2543.10 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 10 layers,  64):    8160.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   15902.46 MiB  ( 15.53 GiB)
  Headroom:                 481.54 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 30 layers, 192):   24480.00 MiB
  Headroom:              715520.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 30 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  32,640.00 MiB
  Single Expert Weight:   127.50 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   63.75
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256         98.1
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        196.3 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        392.5
  PCIe 5.0 x16                    64,000      50,049        785.1

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   1,020.00
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       16.2
  DDR4-2666         42,656      18,306       17.9
  DDR4-3200         51,200      21,973       21.5 ◄ baseline
  DDR4-3600         57,600      24,719       24.2
  DDR5-4800         76,800      32,959       32.3
  DDR5-5600         89,600      38,452       37.7
  DDR5-6000         96,000      41,199       40.4
  DDR5-7200        115,200      49,438       48.5

═══════════════════════════════════════════════════════════════════
```

This is the **non-MTP variant** of Qwen3.6-35B-A3B (no draft token prediction). It shares the same architecture (40 layers, 256 experts, 8 active per token) as the MTP variant in [Qwen3.6-35B-A3B-MTP.md](Qwen3.6-35B-A3B-MTP.md).

### Comparison: non-MTP vs MTP (Q8_0)

| Variant | GPU layers | GPU experts | Context | VRAM headroom | Est. t/s (DDR4-3200) |
|---------|-----------:|------------:|--------:|--------------:|---------------------:|
| **non-MTP Q8_0** | 10 | 64 | 262K | 482 MiB | 21.5 |
| **MTP Q8_0** | 8 | 51 | 262K | 676 MiB | 21.5 |

The non-MTP variant actually fits **more GPU layers** (10 vs 8) because it has no MTP draft overhead in the compute buffer, freeing VRAM. The MTP variant uses `--spec-type draft-mtp` which requires the compute buffer to reserve space for draft tokens. Both produce the same generation speed when MTP is not used.

### Run command

```bash
run-server.sh --model Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf \
  --n-cpu-moe 30 -c 262144 -ctk turbo4 -ctv turbo3_tcq \
  --alias "Qwen3.6-35B-A3B" --threads 14 --no-mmap --mlock
```

## References

- Qwen3.6 35B-A3B standard GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF

# Alternative Quantizations

## UD-IQ3_S

Lowest quantization — smallest weights, most GPU layers.

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-UD-IQ3_S.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          1994.66 MiB
  All experts:            11038.00 MiB
  One expert:                43.12 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          1994.66 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 33 layers, 211):    9106.35 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16300.37 MiB  ( 15.92 GiB)
  Headroom:                  83.63 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU (  7 layers,  45):    1931.65 MiB
  Headroom:              738068.35 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 7 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  11,038.00 MiB
  Single Expert Weight:   43.12 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   21.56
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        290.2
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        580.4 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024      1,160.8
  PCIe 5.0 x16                    64,000      50,049      2,321.5

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   344.94
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       47.8
  DDR4-2666         42,656      18,306       53.1
  DDR4-3200         51,200      21,973       63.7 ◄ baseline
  DDR4-3600         57,600      24,719       71.7
  DDR5-4800         76,800      32,959       95.6
  DDR5-5600         89,600      38,452      111.5
  DDR5-6000         96,000      41,199      119.4
  DDR5-7200        115,200      49,438      143.3

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-IQ3_S puts **33 GPU layers (211 GPU experts)** — the most of any non-MTP quantization. Only 7 layers on CPU. Extremely fast: 63.7 t/s on DDR4-3200. Trade-off: lowest fidelity, smallest expert weights (43 MiB).

## UD-Q4_K_S

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2436.65 MiB
  All experts:            17478.00 MiB
  One expert:                68.27 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2436.65 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 20 layers, 128):    8739.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16375.01 MiB  ( 15.99 GiB)
  Headroom:                   8.99 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 20 layers, 128):    8739.00 MiB
  Headroom:              731261.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 20 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  17,478.00 MiB
  Single Expert Weight:   68.27 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   34.14
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        183.3
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        366.5 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        733.1
  PCIe 5.0 x16                    64,000      50,049      1,466.1

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   546.19
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       30.2
  DDR4-2666         42,656      18,306       33.5
  DDR4-3200         51,200      21,973       40.2 ◄ baseline
  DDR4-3600         57,600      24,719       45.3
  DDR5-4800         76,800      32,959       60.3
  DDR5-5600         89,600      38,452       70.4
  DDR5-6000         96,000      41,199       75.4
  DDR5-7200        115,200      49,438       90.5

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q4_K_S puts **20 GPU layers (128 GPU experts)** — half the layers on GPU. 20 layers on CPU. Predicted 40.2 t/s. Tight fit with only 9 MiB headroom.

## UD-Q4_K_XL

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2554.11 MiB
  All experts:            18760.00 MiB
  One expert:                73.28 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2554.11 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 18 layers, 115):    8442.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16195.47 MiB  ( 15.82 GiB)
  Headroom:                 188.53 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 22 layers, 141):   10318.00 MiB
  Headroom:              729682.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 22 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  18,760.00 MiB
  Single Expert Weight:   73.28 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   36.64
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        170.7
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        341.5 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        683.0
  PCIe 5.0 x16                    64,000      50,049      1,365.9

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   586.25
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       28.1
  DDR4-2666         42,656      18,306       31.2
  DDR4-3200         51,200      21,973       37.5 ◄ baseline
  DDR4-3600         57,600      24,719       42.2
  DDR5-4800         76,800      32,959       56.2
  DDR5-5600         89,600      38,452       65.6
  DDR5-6000         96,000      41,199       70.3
  DDR5-7200        115,200      49,438       84.3

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q4_K_XL puts **18 GPU layers (115 GPU experts)**. 22 layers on CPU. Predicted 37.5 t/s. Good balance between fidelity and speed.

## UD-Q5_K_XL

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2554.11 MiB
  All experts:            22796.00 MiB
  One expert:                89.05 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2554.11 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 15 layers,  96):    8548.50 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16301.97 MiB  ( 15.92 GiB)
  Headroom:                  82.03 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 25 layers, 160):   14247.50 MiB
  Headroom:              725752.50 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 25 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  22,796.00 MiB
  Single Expert Weight:   89.05 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   44.52
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        140.5
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        281.0 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        562.1
  PCIe 5.0 x16                    64,000      50,049      1,124.1

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   712.38
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       23.1
  DDR4-2666         42,656      18,306       25.7
  DDR4-3200         51,200      21,973       30.8 ◄ baseline
  DDR4-3600         57,600      24,719       34.7
  DDR5-4800         76,800      32,959       46.3
  DDR5-5600         89,600      38,452       54.0
  DDR5-6000         96,000      41,199       57.8
  DDR5-7200        115,200      49,438       69.4

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q5_K_XL puts **15 GPU layers (96 GPU experts)**. 25 layers on CPU. Predicted 30.8 t/s. Higher fidelity than Q4 but fewer GPU experts.

## UD-Q6_K_XL

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2554.11 MiB
  All experts:            27804.00 MiB
  One expert:               108.61 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2554.11 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 12 layers,  77):    8341.20 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16094.67 MiB  ( 15.72 GiB)
  Headroom:                 289.33 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 28 layers, 179):   19462.80 MiB
  Headroom:              720537.20 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 28 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  27,804.00 MiB
  Single Expert Weight:   108.61 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   54.30
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        115.2
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        230.4 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        460.8
  PCIe 5.0 x16                    64,000      50,049        921.6

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   868.88
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       19.0
  DDR4-2666         42,656      18,306       21.1
  DDR4-3200         51,200      21,973       25.3 ◄ baseline
  DDR4-3600         57,600      24,719       28.4
  DDR5-4800         76,800      32,959       37.9
  DDR5-5600         89,600      38,452       44.3
  DDR5-6000         96,000      41,199       47.4
  DDR5-7200        115,200      49,438       56.9

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q6_K_XL puts **12 GPU layers (77 GPU experts)**. 28 layers on CPU. Predicted 25.3 t/s. Higher fidelity, fewer GPU experts.

## UD-Q8_K_XL

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
  Layers:          40
  Experts (total): 256  (active per token: 8)
  Context:         262144  (model max: 262144, VRAM-fit max: 262144)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2579.42 MiB
  All experts:            34080.00 MiB
  One expert:               133.12 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    1199.36 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:          2579.42 MiB
  KV cache:                1199.36 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 10 layers,  64):    8520.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16298.78 MiB  ( 15.92 GiB)
  Headroom:                  85.22 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU ( 30 layers, 192):   25560.00 MiB
  Headroom:              714440.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 30 -c 262144 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  34,080.00 MiB
  Single Expert Weight:   133.12 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   66.56
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256         94.0
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        188.0 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024        376.0
  PCIe 5.0 x16                    64,000      50,049        751.9

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   1,065.00
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       15.5
  DDR4-2666         42,656      18,306       17.2
  DDR4-3200         51,200      21,973       20.6 ◄ baseline
  DDR4-3600         57,600      24,719       23.2
  DDR5-4800         76,800      32,959       30.9
  DDR5-5600         89,600      38,452       36.1
  DDR5-6000         96,000      41,199       38.7
  DDR5-7200        115,200      49,438       46.4

═══════════════════════════════════════════════════════════════════
```

**Verdict:** UD-Q8_K_XL puts **10 GPU layers (64 GPU experts)** — matches the non-MTP Q8_0 from the main directory. 30 layers on CPU. Predicted 20.6 t/s. Slightly different backbone size due to different quantization method.

## Complete non-MTP comparison

| Quant | GPU layers | GPU experts | Context | VRAM headroom | Est. t/s (DDR4-3200) |
|-------|-----------:|------------:|--------:|--------------:|---------------------:|
| UD-IQ3_S | 33 | 211 | 262K | 84 MiB | 63.7 |
| UD-Q4_K_S | 20 | 128 | 262K | 9 MiB | 40.2 |
| UD-Q4_K_XL | 18 | 115 | 262K | 189 MiB | 37.5 |
| UD-Q5_K_XL | 15 | 96 | 262K | 82 MiB | 30.8 |
| UD-Q6_K_XL | 12 | 77 | 262K | 289 MiB | 25.3 |
| Q8_0 | 10 | 64 | 262K | 482 MiB | 21.5 |
| UD-Q8_K_XL | 10 | 64 | 262K | 85 MiB | 20.6 |

All quantizations fit on 16 GiB VRAM at 262K context. The UD variants use turboKV cache (same as the main Q8_0), while the Q8_0 quantization uses the standard q8_0 KV format. The UD-IQ3_S offers the best speed (63.7 t/s) with the most GPU experts, but lowest fidelity. UD-Q8_K_XL matches Q8_0 in GPU layer count but with slightly lower generation speed due to higher per-expert data volume.
