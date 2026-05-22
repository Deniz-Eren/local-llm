# Qwen3.6-35B-A3B-MTP-Q8_0 — Profiling Notes

For this test we use the main llama.cpp repo to utilize MTP and as such drop TurboQuant since that hasn't been merged yet.

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Final Run Command Used

These flags use `--spec-type` and `--spec-draft-n-max` as script options (now configurable in `run-server.sh`).

Stable at 27 tokens/s, good token context +250k.
```bash
run-server.sh --model Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf --n-cpu-moe 32 -c 262144 -ctk q8_0 -ctv q8_0 --alias "Qwen3.6-35B-A3B-Q8_0" --threads 14 --no-mmap --mlock --spec-type draft-mtp --spec-draft-n-max 3
```

## Model

Configuration script:
```
./scripts/moe-configs.py --gguf-py-path ../llama.cpp/  --vram 16384 --ram 730956 ~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf --cache-type-k q8_0 --cache-type-v q8_0 --ctx 262144
```

Configuration results:
```
Model:            Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf
Layers:           40
Experts (total):  256  (active per token: 8)
Context:          262144  (model max: 262144, VRAM-fit max: 262144)

=== Tensor sizes ===
  Dense backbone:          2543.10 MiB
  All experts:            32640.00 MiB
  One expert:               127.50 MiB
  KV cache (q8_0, eff 0.515x):    2636.80 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          2543.10 MiB
  KV cache:                2636.80 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU (  8 layers,  51):    6528.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   15707.90 MiB  ( 15.34 GiB)
  Headroom:                 676.10 MiB

=== RAM plan (budget 32768 MiB) ===
  Experts on CPU ( 32 layers, 205):   26112.00 MiB
  Headroom:                6656.00 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 32 -c 262144 -ctk q8_0 -ctv q8_0
```

# References

- Qwen3.6 35B-A3B MTP GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
- Qwen3.6 35B-A3B standard GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
