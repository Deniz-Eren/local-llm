# MiniMax M2.7 — Profiling Notes

MiniMax M2.7 is a sparse MoE model much larger than the Qwen3.6 family. We are profiling its fit across different host platforms.

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Final Run Command Used

Worked but unusable at 3-4 tokens/s, good token context +128k but tool calls regularly failed for unknown reasons:
```bash
run-server.sh --model MiniMax-M2.7-Q8_0.gguf --n-cpu-moe 62 -c 92416 -ctk q8_0 -ctv q4_0 --alias "MiniMax-M2.7-Q8_0" --threads 14 --no-mmap --mlock
```

## Model

Configuration script:
```
./scripts/moe-configs.py ~/models/MiniMax-M2.7-GGUF/Q8_0/MiniMax-M2.7-Q8_0.gguf --ctx 196608 --vram 16384 --ram 740000 --cache-type-k q8_0 --cache-type-v q4_0 --compute-overhead 3600
```

Configuration results:
```
Model:            MiniMax-M2.7-Q8_0.gguf
Layers:           62
Experts (total):  256  (active per token: 8)
Context:          92416  (model max: 196608, VRAM-fit max: 92416)

=== Tensor sizes ===
  Dense backbone:          4201.49 MiB
  All experts:           227664.00 MiB
  One expert:               889.31 MiB
  KV cache (K=q8_0, V=q4_0, eff 0.383x):    8561.11 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          4201.49 MiB
  KV cache:                8561.11 MiB
  Compute/MTP buffer:      3600.00 MiB
  Experts on GPU (  0 layers,   0):       0.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16362.60 MiB  ( 15.98 GiB)
  Headroom:                  21.40 MiB

=== RAM plan (budget 740000 MiB) ===
  Experts on CPU ( 62 layers, 256):  227664.00 MiB
  Headroom:              512336.00 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK
  -> Only 0 layers (0 experts) on GPU; per-token routing needs 8 active. On average 8 of the active expert MLPs per token will run on CPU instead of GPU (slower per-token compute). Offload fewer layers (`--n-cpu-moe N`, smaller `N`) to put more layers — and more experts — on GPU.

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 62 -c 92416 -ctk q8_0 -ctv q4_0
```

# References

- MiniMax-M2.7 229B GGUFs: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF
