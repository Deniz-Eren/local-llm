# Qwen3.6-35B-A3B-Q8_0 — Profiling Notes

For this test we use the main llama.cpp repo to utilize MTP and as such drop TurboQuant since that hasn't been merged yet.

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Final Run Command Used

These flags use `--spec-type` and `--spec-draft-n-max` as script options (now configurable in `run-server.sh`).

Stable at 18 tokens/s, good token context +128k.
```bash
run-server.sh --model Qwen3.6-35B-A3B-Q8_0.gguf --n-cpu-moe 256 -c 262144 -ctk q8_0 -ctv q8_0 --alias "Qwen3.6-35B-A3B-Q8_0" --threads 14 --no-mmap --mlock --spec-type draft-mtp --spec-draft-n-max 3
```

## Model

Model:            Qwen3.6-35B-A3B-Q8_0.gguf
Layers:           40
Experts (total):  256  (active per token: 8)
Context:          262144  (model max: 262144, VRAM-fit max: 262144)

=== Tensor sizes ===
  Dense backbone:          2543.10 MiB
  All experts:            32640.00 MiB
  One expert:               127.50 MiB
  KV cache (q8_0, eff 0.515x):   10547.20 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          2543.10 MiB
  KV cache:               10547.20 MiB
  Experts on GPU ( 25):    3187.50 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts)
  -------------------------------------
  Used:                   16277.80 MiB  ( 15.90 GiB)
  Headroom:                 106.20 MiB

=== RAM plan (budget 730956 MiB) ===
  Experts on CPU (231):   29452.50 MiB
  Headroom:              701503.50 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 231 -c 262144 -ctk q8_0 -ctv q8_0
