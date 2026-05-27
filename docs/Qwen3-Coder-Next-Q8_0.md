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
Model:            Qwen3-Coder-Next/Q8_0/Qwen3-Coder-Next-Q8_0.gguf
Layers:           48
Experts (total):  512  (active per token: 10)
Context:          224512  (model max: 262144, VRAM-fit max: 224512)

=== Tensor sizes ===
  Dense backbone:          2541.56 MiB
  All experts:            78336.00 MiB
  One expert:               153.00 MiB
  KV cache (q8_0, eff 0.515x):   10839.72 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          2541.56 MiB
  KV cache:               10839.72 MiB
  Compute/MTP buffer:      3000.00 MiB
  Experts on GPU (  0 layers,   0):       0.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16381.28 MiB  ( 16.00 GiB)
  Headroom:                   2.72 MiB

=== RAM plan (budget 740000 MiB) ===
  Experts on CPU ( 48 layers, 512):   78336.00 MiB
  Headroom:              661664.00 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK
  -> Only 0 experts on GPU; per-token routing needs 10 active. On average 10 of the active expert MLPs per token will run on CPU instead of GPU (slower per-token compute). Reduce --n-cpu-moe N to pin fewer layers to CPU, thereby keeping more layers (and their experts) on GPU.

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 48 -c 224512 -ctk q8_0 -ctv q8_0

# `--n-cpu-moe 48` pins all 48 layers to CPU RAM; 0 layers on GPU. For shared-expert Qwen3-Coder-Next (512 experts across all 48 layers), 0 GPU layers ≈ 0 GPU experts, meaning all 10 active experts per token run on CPU.
