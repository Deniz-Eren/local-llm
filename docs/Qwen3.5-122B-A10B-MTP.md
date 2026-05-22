# Qwen3.5-122B-A10B-Q8_0 — Profiling Notes

For this test we use the main llama.cpp repo to utilize MTP and as such drop TurboQuant since that hasn't been merged yet.

## Experiment host hardware

| Component | Model | Details |
|-----------|-------|--------|
| **CPU** | Intel Xeon Gold 5120 | 14 cores / 28 threads, Skylake-SP, AVX-512, 2.20 GHz ([spec sheet](https://www.intel.com/content/www/us/en/products/sku/120474/intel-xeon-gold-5120-processor-19-25m-cache-2-20-ghz/specifications.html)) |
| **RAM** | 740 GB | DDR4 ECC (test server) |
| **GPU** | NVIDIA TU104-895-A1 (T4) | 16 GB GDDR6 (16384 MiB), 4096 CUDA cores, Tensor Cores ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)) |

## Final Run Command Used

These flags use `--spec-type` and `--spec-draft-n-max` as script options (now configurable in `run-server.sh`).

Stable at 10 tokens/s, good token context +200k.
```bash
run-server.sh --model Qwen3.5-122B-A10B-MTP-GGUF/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf --n-cpu-moe 48 -c 262144 -ctk q8_0 -ctv q8_0 --alias "Qwen3.5-122B-A10B-MTP-Q8_0" --threads 14 --no-mmap --mlock --spec-type draft-mtp --spec-draft-n-max 3
```

## Model

Configuration script:
```
./scripts/moe-configs.py ~/models/Qwen3.5-122B-A10B-Q8_0/Qwen3.5-122B-A10B-Q8_0.gguf --ctx 262144 --vram 16384 --ram 740000 --cache-type-k q8_0 --cache-type-v q8_0
```

Configuration results:
```
Model:            Qwen3.5-122B-A10B-Q8_0/Qwen3.5-122B-A10B-Q8_0.gguf
Layers:           49
Experts (total):  256  (active per token: 8)
Context:          262144  (model max: 262144, VRAM-fit max: 262144)

=== Tensor sizes ===
  Dense backbone:          6450.98 MiB
  All experts:           119952.00 MiB
  One expert:               468.56 MiB
  KV cache (q8_0, eff 0.515x):    3164.16 MiB

=== VRAM plan (budget 16384 MiB) ===
  Dense backbone:          6450.98 MiB
  KV cache:                3164.16 MiB
  Compute/MTP buffer:      4000.00 MiB
  Experts on GPU (  1 layers,   5):    2448.00 MiB
  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)
  -------------------------------------
  Used:                   16063.14 MiB  ( 15.69 GiB)
  Headroom:                 320.86 MiB

=== RAM plan (budget 740000 MiB) ===
  Experts on CPU ( 48 layers, 251):  117504.00 MiB
  Headroom:              622496.00 MiB

=== Verdict ===
  VRAM: OK
  RAM:  OK
  -> Only 5 layers (40 experts) on GPU; per-token routing needs 8 active. On average 3 of the active expert MLPs per token will run on CPU instead of GPU (slower per-token compute). Offload fewer layers (`--n-cpu-moe N`, smaller `N`) to put more layers — and more experts — on GPU.

=== llama-server flag ===
  --n-gpu-layers 999 --n-cpu-moe 48 -c 262144 -ctk q8_0 -ctv q8_0
```

# References

- Qwen3.5 122B-A10B MTP GGUFs: https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF
