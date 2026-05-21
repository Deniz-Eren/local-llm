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
Layers:           41
Experts (total):  256  (active per token: 8)
Context:          262144  (model max: 262144, VRAM-fit max: 262144)

=== Tensor sizes ===
  Dense backbone:          2583.45 MiB
  All experts:            33456.00 MiB
  One expert:               130.69 MiB
  KV cache (q8_0, eff 0.515x):   10810.88 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          2583.45 MiB
  KV cache:               10810.88 MiB
  Experts on GPU ( 22):    2875.12 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts)
  -------------------------------------
  Used:                   16269.46 MiB  ( 15.89 GiB)
  Headroom:                 114.54 MiB

=== RAM plan (budget 730956 MiB) ===
  Experts on CPU (234):   30580.88 MiB
  Headroom:              700375.12 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 234 -c 262144 -ctk q8_0 -ctv q8_0
```

# References

- Qwen3.6 35B-A3B MTP GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
- Qwen3.6 35B-A3B standard GGUFs: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
