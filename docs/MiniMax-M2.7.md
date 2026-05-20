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
run-server.sh --model MiniMax-M2.7-Q8_0.gguf --n-cpu-moe 256 -c 149760 -ctk turbo4 -ctv turbo3_tcq --alias "MiniMax-M2.7-Q8_0" --threads 8 --no-mmap --mlock
```

## Model

run-server.sh --n-cpu-moe 256 -c 149760 -ctk turbo4 -ctv turbo3_tcq --alias "MiniMax-M2.7-Q8_0" --threads 8 --no-mmap --mlock

./scripts/moe-configs.py --gguf-py-path ../buun-llama-cpp/  --vram 16384 --ram 730956 ~/models/MiniMax-M2.7-GGUF/Q8_0/MiniMax-M2.7-Q8_0.gguf
Model:            MiniMax-M2.7-Q8_0.gguf
Layers:           62
Experts (total):  256  (active per token: 8)
Context:          128000  (model max: 196608, VRAM-fit max: 196608)

=== Tensor sizes ===
  Dense backbone:          4201.49 MiB
  All experts:           227664.00 MiB
  One expert:               889.31 MiB
  KV cache (K=turbo4, V=turbo3_tcq, eff 0.234x):    7261.75 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          4201.49 MiB
  KV cache:                7261.75 MiB
  Experts on GPU (  5):    4446.56 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts)
  -------------------------------------
  Used:                   15909.80 MiB  ( 15.54 GiB)
  Headroom:                 474.20 MiB

=== RAM plan (budget 730956 MiB) ===
  Experts on CPU (251):  223217.44 MiB
  Headroom:              507738.56 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK
  -> Only 5 experts on GPU; per-token routing needs 8 active. On average 3 of the active expert MLPs per token will run on CPU instead of GPU (slower per-token compute). Reduce --ctx or use a smaller quant if you need more GPU experts.

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 251 -c 128000 -ctk turbo4 -ctv turbo3_tcq

