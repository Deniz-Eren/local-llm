# Qwen3-30B-A3B — Profiling Notes

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Q2_K

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3-30B-A3B-Q2_K.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
NOTE: Qwen3-30B-A3B-Q2_K.gguf: requested --ctx 262144 clamped to 40960 (model_max_ctx=40960).

═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3-30B-A3B-Q2_K.gguf
  Layers:          48
  Experts (total): 128  (active per token: 8)
  Context:         40960  (model max: 40960, VRAM-fit max: 40960)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:           723.35 MiB
  All experts:            10008.00 MiB
  One expert:                78.19 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):     899.52 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:           723.35 MiB
  KV cache:                 899.52 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 48 layers, 128):   10008.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   15630.87 MiB  ( 15.26 GiB)
  Headroom:                 753.13 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU (  0 layers,   0):       0.00 MiB
  Headroom:              740000.00 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 0 -c 40960 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  10,008.00 MiB
  Single Expert Weight:   78.19 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   19.55
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        320.1
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        640.1 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024      1,280.2
  PCIe 5.0 x16                    64,000      50,049      2,560.5

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   625.50
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       26.3
  DDR4-2666         42,656      18,306       29.3
  DDR4-3200         51,200      21,973       35.1 ◄ baseline
  DDR4-3600         57,600      24,719       39.5
  DDR5-4800         76,800      32,959       52.7
  DDR5-5600         89,600      38,452       61.5
  DDR5-6000         96,000      41,199       65.9
  DDR5-7200        115,200      49,438       79.0

═══════════════════════════════════════════════════════════════════
```

## Alternative Quantization: Q3_K_S

```bash
python3 scripts/moe-configs.py \
  ~/Downloads/models/others/Qwen3-30B-A3B-Q3_K_S.gguf \
  --vram 16384 --ram 740000 --ctx 262144
```

Result:

```
NOTE: Qwen3-30B-A3B-Q3_K_S.gguf: requested --ctx 262144 clamped to 40960 (model_max_ctx=40960).

═══════════════════════════════════════════════════════════════════
  MODEL SUMMARY
═══════════════════════════════════════════════════════════════════
  Model:           Qwen3-30B-A3B-Q3_K_S.gguf
  Layers:          48
  Experts (total): 128  (active per token: 8)
  Context:         40960  (model max: 40960, VRAM-fit max: 40960)

═══════════════════════════════════════════════════════════════════
  TENSOR SIZES
═══════════════════════════════════════════════════════════════════
  Dense backbone:           790.99 MiB
  All experts:            11880.00 MiB
  One expert:                92.81 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):     899.52 MiB

═══════════════════════════════════════════════════════════════════
  VRAM PLAN — Budget: 16384 MiB
═══════════════════════════════════════════════════════════════════
  Dense backbone:           790.99 MiB
  KV cache:                 899.52 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU ( 43 layers, 115):   10642.50 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16333.01 MiB  ( 15.95 GiB)
  Headroom:                  50.99 MiB

═══════════════════════════════════════════════════════════════════
  RAM PLAN — Budget: 740000 MiB
═══════════════════════════════════════════════════════════════════
  Experts on CPU (  5 layers,  13):    1237.50 MiB
  Headroom:              738762.50 MiB

═══════════════════════════════════════════════════════════════════
  VERDICT
═══════════════════════════════════════════════════════════════════
  VRAM: OK
  RAM:  OK

═══════════════════════════════════════════════════════════════════
  LLAMA-SERVER FLAG
═══════════════════════════════════════════════════════════════════
  --n-gpu-layers 999 --n-cpu-moe 5 -c 40960 -ctk turbo4 -ctv turbo3_tcq

═══════════════════════════════════════════════════════════════════
  HARDWARE FORECAST — Performance Projections
═══════════════════════════════════════════════════════════════════
  Total MoE Weight Pool:  11,880.00 MiB
  Single Expert Weight:   92.81 MiB
  Active Experts/token:   8
  Micro-Batch Size (ubatch): 512 tokens

  ── Phase 1: Prefill (PCIe DMA Weight Streaming) ─────────────────
  Data per token (MiB):   23.20
  Efficiency modifier:    82%

  PCIe Config                 Raw (MB/s)  Eff (MiB/s)  Prefill (t/s)
  --------------------------  ----------  -----------  ------------
  PCIe 3.0 x8 / 4.0 x4             8,000       6,256        269.6
  PCIe 3.0 x16 / 4.0 x8           16,000      12,512        539.2 ◄ baseline
  PCIe 4.0 x16                    32,000      25,024      1,078.5
  PCIe 5.0 x16                    64,000      50,049      2,157.0

  ── Phase 2: Token Generation (System RAM CPU Compute) ───────────
  Data per token (MiB):   742.50
  Efficiency modifier:    45%

  RAM Config    Raw (MB/s)  Eff (MiB/s)   Gen (t/s)
  ------------  ----------  -----------  ----------
  DDR4-2400         38,400      16,479       22.2
  DDR4-2666         42,656      18,306       24.7
  DDR4-3200         51,200      21,973       29.6 ◄ baseline
  DDR4-3600         57,600      24,719       33.3
  DDR5-4800         76,800      32,959       44.4
  DDR5-5600         89,600      38,452       51.8
  DDR5-6000         96,000      41,199       55.5
  DDR5-7200        115,200      49,438       66.6

═══════════════════════════════════════════════════════════════════
```

## Comparison summary

| Quant | GPU layers | GPU experts | Context | VRAM headroom | Est. t/s (DDR4-3200) |
|-------|-----------:|------------:|--------:|--------------:|---------------------:|
| **Q2_K** | 48 (all) | 128 (all) | 40K | 753 MiB | 35.1 |
| **Q3_K_S** | 43 | 115 | 40K | 51 MiB | 29.6 |

**Verdict:** Qwen3-30B-A3B is a **perfect fit** on 16 GiB VRAM. The Q2_K quantization puts all 48 layers (128 experts) on GPU — zero CPU expert routing. Extremely fast at 35 t/s with 40K context. Q3_K_S is also viable but very tight (51 MiB headroom) with 5 layers on CPU.

Note: Qwen3-30B-A3B has a model max context of only 40960 tokens (unlike Qwen3.6-35B-A3B at 262K). This is the primary trade-off vs the larger context models.

## References

- Qwen3 MoE GGUFs: https://huggingface.co/unsloth/Qwen3-MoE-GGUF
