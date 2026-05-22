#!/usr/bin/env python3
# Copyright (c) 2026 Deniz Eren
# Licensed under the MIT License. See the LICENSE file at the repository
# root for the full license text.
"""
Size a MoE GGUF for the local-llm split:
  - VRAM holds: dense weights + TurboQuant KV cache + as many active experts as fit
  - RAM  holds: the remaining (CPU-side) experts

Prints a full breakdown and the recommended `--n-cpu-moe` value for llama-server.
With --scan <dir>, evaluates every .gguf in <dir> and picks the best-fitting one.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── gguf-py bootstrap (consolidated) ────────────────────────────────────────
MIB = 1024 ** 2
GIB = 1024 ** 3


def _find_gguf_py_root(candidate: Path) -> Path | None:
    """Walk from *candidate* up the tree looking for gguf/__init__.py.
    Also check the sibling gguf-py directory if candidate looks like a
    llama.cpp checkout root."""
    for root in (candidate, *candidate.parents):
        if (root / "gguf" / "__init__.py").is_file():
            return root
        # Also try if root is the llama.cpp root — gguf-py sits beside
        # src/, models/, etc. as a sibling directory.
        gguf_py = root / "gguf-py"
        if (gguf_py / "gguf" / "__init__.py").is_file():
            return gguf_py
    return None


def _bootstrap_gguf_py() -> None:
    candidates: list[Path] = [
        Path.cwd() / "llama.cpp" / "gguf-py",
        Path.cwd() / "gguf-py",
    ]
    script_dir = Path(__file__).resolve().parent
    for p in (script_dir, *script_dir.parents):
        candidates.extend([
            p / "llama.cpp" / "gguf-py",
            p / "gguf-py",
        ])
    for c in candidates:
        root = _find_gguf_py_root(c)
        if root is not None:
            sys.path.insert(0, str(root))
            return
    raise SystemExit(
        "Could not locate gguf-py under ./llama.cpp/gguf-py or ./gguf-py. "
        "Pass --gguf-py-path /path/to/gguf-py to point at it explicitly."
    )


def find_gguf_py(explicit: str | None) -> None:
    if explicit:
        p = Path(explicit).expanduser()
        root = _find_gguf_py_root(p)
        if root is not None:
            sys.path.insert(0, str(root))
            return
        # Fallback: try the original logic for backwards compatibility
        for cand in (p, *p.parents):
            if (cand / "gguf" / "__init__.py").is_file():
                sys.path.insert(0, str(cand))
                return
            gguf_py = cand / "gguf-py"
            if (gguf_py / "gguf" / "__init__.py").is_file():
                sys.path.insert(0, str(gguf_py))
                return
        raise SystemExit(
            f"--gguf-py-path does not point at a gguf-py package: {p}\n"
            "Pass the gguf-py directory (e.g. .../llama.cpp/gguf-py) or any "
            "path inside it."
        )
    _bootstrap_gguf_py()


# Bytes per element for each supported KV cache type (fp16 = 2.0).
# These are approximate and include quantization overhead (scales, etc.).
# The "factor" relative to fp16 is bpe / 2.0.
CACHE_TYPE_BPE: dict[str, float] = {
    "f32":       4.0,     # 16 bits  → factor 2.0
    "f16":       2.0,     # 16 bits  → factor 1.0
    "bf16":      2.0,     # 16 bits  → factor 1.0
    "q8_0":      1.03,    # 8-bit    → factor ~0.515
    "q5_1":      0.66,    # ~5.3 bit → factor ~0.33
    "q5_0":      0.625,   # 5 bit    → factor ~0.3125
    "q4_1":      0.55,    # ~4.4 bit → factor ~0.275
    "q4_0":      0.5,     # 4 bit    → factor 0.25
    "iq4_nl":    0.5,     # 4 bit NLO → factor 0.25
    "turbo4":    0.531,   # 4.25 bpv, lossless → factor ~0.266
    "turbo3_tcq":0.406,   # 3.25 bpv, TCQ → factor ~0.203
    "turbo3":    0.406,   # 3.25 bpv, scalar → factor ~0.203
    "turbo2_tcq":0.281,   # 2.25 bpv, TCQ → factor ~0.141
    "turbo2":    0.281,   # 2.25 bpv, scalar → factor ~0.141
}
CACHE_TYPE_K_DEFAULT = "turbo4"
CACHE_TYPE_V_DEFAULT = "turbo3_tcq"

# llama.cpp rounds n_ctx UP to a multiple of CTX_PAD when it allocates the KV
# cache (`cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256)` in src/llama-context.cpp).
# We round our suggested -c value DOWN to the same alignment so the value the
# user passes matches the value llama.cpp actually allocates — no surprise
# upward padding past the budgeted VRAM.
CTX_PAD = 256


def align_ctx_down(n: int) -> int:
    """Largest multiple of CTX_PAD that is <= n, with a CTX_PAD floor."""
    return max(CTX_PAD, (n // CTX_PAD) * CTX_PAD)


def _valid_cache_types() -> str:
    return ", ".join(sorted(CACHE_TYPE_BPE))


# ── Plan dataclass (Task 3: to_dict for JSON output) ───────────────────────
@dataclass
class Plan:
    model: Path
    layers: int
    experts: int
    active: int
    dense_b: float
    expert_b: float
    per_expert_b: float
    kv_b: float
    gpu_layers: int
    cpu_layers: int
    gpu_experts: int
    cpu_experts: int
    vram_used_b: float
    vram_total_b: float
    ram_budget_b: float
    cpu_expert_b: float
    gpu_expert_b: float
    compute_overhead_b: float
    ctx: int
    model_max_ctx: int
    fit_max_ctx: int
    rec_ctx: int
    cache_type_k: str
    cache_type_v: str
    bpe_k: float
    bpe_v: float
    experts_per_layer: int

    def to_dict(self, verbose: bool = False) -> dict:
        result = {
            "model": str(self.model),
            "ctx": self.ctx,
            "fit_max_ctx": self.fit_max_ctx,
            "rec_ctx": self.rec_ctx,
            "vram_used_mib": round(self.vram_used_b / MIB, 2),
            "vram_headroom_mib": round((self.vram_total_b - self.vram_used_b) / MIB, 2),
            "compute_overhead_mib": round(self.compute_overhead_b / MIB, 2),
            "cpu_expert_mib": round(self.cpu_expert_b / MIB, 2),
            "gpu_layers": self.gpu_layers,
            "cpu_layers": self.cpu_layers,
            "gpu_experts": self.gpu_experts,
            "cpu_experts": self.cpu_experts,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "n_cpu_moe": self.n_cpu_moe,
            "experts_per_layer": self.experts_per_layer,
            "fits": self.fits,
        }
        # Task 23: add verbose fields when requested
        if verbose:
            result["_verbose"] = {
                "dense_b_mib": round(self.dense_b / MIB, 2),
                "expert_b_mib": round(self.expert_b / MIB, 2),
                "per_expert_b_mib": round(self.per_expert_b / MIB, 2),
                "kv_b_mib": round(self.kv_b / MIB, 2),
                "effective_factor": round(self.effective_factor, 3),
            }
        return result

    @property
    def effective_factor(self) -> float:
        """Effective compression factor relative to fp16 (averaged over K+V)."""
        return (self.bpe_k + self.bpe_v) / 4.0

    @property
    def kv_type_str(self) -> str:
        k, v = self.cache_type_k, self.cache_type_v
        return k if k == v else f"K={k}, V={v}"

    @property
    def vram_ok(self) -> bool:
        return self.vram_used_b <= self.vram_total_b

    @property
    def ram_ok(self) -> bool:
        return self.cpu_expert_b <= self.ram_budget_b

    @property
    def fits(self) -> bool:
        return self.vram_ok and self.ram_ok

    @property
    def n_cpu_moe(self) -> int:
        # --n-cpu-moe always takes a **layer count**: the number of MoE layers
        # whose expert tensors (blk.{i}.ffn_*_exps) are placed on the CPU.
        # This is true for both shared-expert (Qwen3) and per-layer-expert
        # (Mixtral, DeepSeek) architectures.
        return self.cpu_layers


# ── KV geometry dataclass ───────────────────────────────────────────────────
@dataclass
class KVShape:
    """Per-architecture KV-cache geometry. Models with interleaved sliding-
    window attention (Gemma 3/4, Cohere2, etc.) split layers into:

      * **full** layers whose KV grows linearly with `ctx`, and
      * **SWA** layers whose KV is bounded by a fixed window and may use
        different (typically smaller) per-head key/value dims.

    KV bytes at context `ctx` is:

        swa_const_b  +  ctx * full_per_tok_b

    where `swa_const_b` is independent of `ctx`. For models without SWA the
    SWA fields are zero and the formula collapses to the old linear form.

    K and V cache types can differ (e.g. turbo4 for keys, turbo3 for values),
    so bpe_k and bpe_v are tracked separately.
    """
    n_full_layers: int
    n_swa_layers: int
    n_head_kv: int
    key_len: int
    val_len: int
    key_len_swa: int
    val_len_swa: int
    sliding_window: int
    bpe_k: float   # bytes per element for K cache type
    bpe_v: float   # bytes per element for V cache type

    @property
    def full_per_layer_b(self) -> float:
        """Bytes per full layer, per token (for K and V combined)."""
        return self.n_head_kv * (self.key_len * self.bpe_k + self.val_len * self.bpe_v)

    @property
    def swa_per_layer_b(self) -> float:
        """Bytes per SWA layer, per token (for K and V combined)."""
        return self.n_head_kv * (self.key_len_swa * self.bpe_k + self.val_len_swa * self.bpe_v)

    @property
    def swa_const_b(self) -> float:
        """Constant bytes for SWA layers (bounded by sliding window)."""
        return self.n_swa_layers * self.sliding_window * self.swa_per_layer_b

    @property
    def full_per_tok_b(self) -> float:
        """Bytes per token from full-attention layers."""
        return self.n_full_layers * self.full_per_layer_b

    def bytes_at(self, ctx: int) -> float:
        return self.swa_const_b + ctx * self.full_per_tok_b


def _validate_cache_type(name: str) -> str:
    """Validate a cache type name, raising SystemExit if invalid."""
    if name not in CACHE_TYPE_BPE:
        raise SystemExit(
            f"Invalid cache type '{name}'. Valid types: {_valid_cache_types()}"
        )
    return name


# ── Named return for parse_metadata (Task 12: ParseResult) ─────────────────
@dataclass(frozen=True)
class ParseResult:
    """Named return from parse_metadata()."""
    layers: int
    experts: int
    active: int
    dense_b: int
    expert_b: int
    model_max_ctx: int
    kv_shape: "KVShape"
    # Number of experts per MoE layer.  For **shared-expert** Qwen3 MoE models
    # (where all layers share the same pool), this equals the total expert
    # count (e.g. 256).  For **per-layer-expert** models (Mixtral, etc.) it
    # is the per-layer count (e.g. 8).  Used to convert cpu_experts into the
    # --n-cpu-moe flag value.
    experts_per_layer: int


# ── parse_metadata (Task 12: returns ParseResult) ──────────────────────────
def parse_metadata(model: Path, cache_type_k: str, cache_type_v: str) -> ParseResult:
    """Return a ParseResult with named fields for layers, experts, sizes,
    and KVShape.

    Reads per-layer KV geometry straight from the GGUF so models with
    interleaved sliding-window attention (Gemma 3/4 etc.) are not
    over-counted by treating every layer as a full-attention layer."""
    from gguf import GGUFReader  # type: ignore

    r = GGUFReader(str(model))

    def find_field(suffix: str):
        for name, field in r.fields.items():
            if name.endswith(suffix):
                return field
        return None

    def field_uint(suffix: str, default: int | None = None) -> int:
        f = find_field(suffix)
        if f is None:
            if default is not None:
                return default
            raise ValueError(f"Could not find *.{suffix} in {model.name}")
        return int(f.parts[f.data[0]][0])

    def field_uint_array(suffix: str) -> list[int] | None:
        """Return the field as a list of ints if it's an array, else None."""
        f = find_field(suffix)
        if f is None or len(f.data) <= 1:
            return None
        return [int(f.parts[i][0]) for i in f.data]

    layers = field_uint("block_count")
    experts = field_uint("expert_count")
    active = field_uint("expert_used_count")
    model_max_ctx = field_uint("context_length")

    n_head_kv = field_uint("attention.head_count_kv")
    key_len = field_uint("attention.key_length")
    val_len = field_uint("attention.value_length")

    # Full attention interval (some models only maintain KV cache for every Nth layer)
    # Field name varies by model architecture:
    #   - "qwen35moe.full_attention_interval" for Qwen3.5/3.6 MoE
    #   - "qwen3moe.full_attention_interval" for Qwen3 MoE
    #   - "attention.full_attention_interval" for other models
    #   - absent (default to 1 = every layer stores KV).
    full_attn_interval = 1
    for suffix in ("qwen35moe.full_attention_interval",
                   "qwen3moe.full_attention_interval",
                   "attention.full_attention_interval"):
        try:
            full_attn_interval = field_uint(suffix)
            break
        except (ValueError, KeyError):
            pass
    if full_attn_interval is None or full_attn_interval < 1:
        full_attn_interval = 1

    # SWA geometry. Models without SWA have no swa_layers array and no
    # sliding_window key, in which case all layers are treated as full.
    key_len_swa = field_uint("attention.key_length_swa", default=key_len)
    val_len_swa = field_uint("attention.value_length_swa", default=val_len)
    sliding_window = field_uint("attention.sliding_window", default=0)

    # `attention.sliding_window_pattern` is one of:
    #   - a length-n_layer BOOL array of per-layer is_swa flags (Gemma 4),
    #   - a scalar period P meaning layer is SWA when (il + 1) % P != 0
    #     (Gemma 3 convention; period 1 means "no SWA"),
    #   - absent (no SWA).
    swa_arr = field_uint_array("attention.sliding_window_pattern")
    if swa_arr is not None and len(swa_arr) == layers:
        n_swa_layers = sum(1 for v in swa_arr if v)
    else:
        period = field_uint("attention.sliding_window_pattern", default=0)
        if period > 1 and sliding_window > 0:
            n_swa_layers = sum(1 for il in range(layers)
                               if (il + 1) % period != 0)
        else:
            n_swa_layers = 0
    if sliding_window <= 0:
        # No window size declared → can't bound SWA layers; treat as full.
        n_swa_layers = 0
    n_full_layers = layers - n_swa_layers

    # Apply full_attention_interval: only every Nth full layer actually stores KV cache
    # llama.cpp allocates KV for layers where (layer_idx + 1) % interval == 0
    n_kv_layers = n_full_layers // full_attn_interval

    kv_shape = KVShape(
        n_full_layers=n_kv_layers,  # Use KV-storing layers, not total full layers
        n_swa_layers=n_swa_layers,
        n_head_kv=n_head_kv,
        key_len=key_len,
        val_len=val_len,
        key_len_swa=key_len_swa,
        val_len_swa=val_len_swa,
        sliding_window=sliding_window,
        bpe_k=CACHE_TYPE_BPE[cache_type_k],
        bpe_v=CACHE_TYPE_BPE[cache_type_v],
    )

    dense_b = 0
    expert_b = 0
    # Detect experts-per-layer from tensor shapes.  The last dimension of
    # any _exps tensor tells us how many experts each MoE layer has.
    # For **shared-expert** Qwen3 MoE models all layers share the same pool
    # so last_dim == total expert_count.  For **per-layer-expert** models
    # (Mixtral, DeepSeek, …) each layer has its own smaller set (e.g. 8).
    experts_per_layer = experts  # fallback: assume total == per-layer
    for t in r.tensors:
        if "_exps" in t.name and len(t.shape) > 0:
            experts_per_layer = int(t.shape[-1])
            break

    for t in r.tensors:
        if "_exps" in t.name:
            expert_b += int(t.n_bytes)
        else:
            dense_b += int(t.n_bytes)

    return ParseResult(
        layers=layers,
        experts=experts,
        active=active,
        dense_b=dense_b,
        expert_b=expert_b,
        model_max_ctx=model_max_ctx,
        kv_shape=kv_shape,
        experts_per_layer=experts_per_layer,
    )


def kv_bytes(ctx: int, kv_shape: KVShape) -> float:
    return kv_shape.bytes_at(ctx)


def max_fit_ctx(kv_shape: KVShape, dense_b: float,
                vram_mib: int, overhead_mib: float = 4000.0) -> int:
    """Largest context such that dense + KV fits in VRAM. KV is
    `swa_const + ctx * full_per_tok` (linear in ctx), so closed-form.

    *overhead_mib* is reserved for compute buffers (FlashAttention scratch,
    MTP draft weights, context-length spikes, etc.) and is subtracted from
    the budget before evaluating how much space remains for the KV cache."""
    vram_for_kv_b = (vram_mib - overhead_mib) * MIB - dense_b - kv_shape.swa_const_b
    if vram_for_kv_b <= 0:
        return 0
    if kv_shape.full_per_tok_b <= 0:
        # No full-attention layers: KV is bounded by the SWA window for every
        # layer, so any ctx fits as far as KV is concerned. Return a sentinel
        # large enough that `model_max_ctx` and `--ctx` always clamp first.
        return math.inf
    return int(vram_for_kv_b // kv_shape.full_per_tok_b)


# ── make_plan (with --verbose support, Task 10) ────────────────────────────
def make_plan(model: Path, vram_mib: int, ram_mib: int,
              ctx: int | None,
              cache_type_k: str, cache_type_v: str,
              verbose: bool = False,
              compute_overhead_mib: float = 4000.0) -> "Plan":
    """Build a sizing Plan with this VRAM allocation precedence:

      1. **Dense backbone:** all non-expert tensors are placed on the GPU.
      2. **KV cache:** target ctx defaults to 128000 (or the user-supplied
         `--ctx`, or the model's trained `context_length` if `--ctx 0`).
         It is capped to `model_max_ctx` and further to whatever fits in
         VRAM after the dense backbone, then rounded down to a multiple of
         CTX_PAD so the value matches what llama.cpp actually allocates.
      3. **Experts:** any VRAM left over after dense + KV is filled with as
         many experts as fit. The remaining experts live in RAM and run on
         CPU when routed.

    There is no expert floor: if the dense backbone + KV cache leave no
    room, all experts go to RAM and the model pages experts from CPU on
    every routed token. The verdict surfaces this via `gpu_experts == 0`.
    """
    result = parse_metadata(model, cache_type_k, cache_type_v)
    if result.experts <= 0:
        raise ValueError("expert_count is 0 — this script targets MoE models only.")

    kv_shape = result.kv_shape
    per_expert_b = result.expert_b / result.experts
    # Per-layer expert weight: every layer has its own named expert tensors
    # (blk.{i}.ffn_*_exps).  Whether experts are shared across layers (Qwen3)
    # or per-layer (Mixtral), `--n-cpu-moe` always takes a **layer count**.
    per_layer_expert_b = result.expert_b / result.layers

    if verbose:
        print(f"  [verbose] model={model.name}", file=sys.stderr)
        print(f"  [verbose] dense_b={result.dense_b}  expert_b={result.expert_b}", file=sys.stderr)
        print(f"  [verbose] per_expert_b={per_expert_b}  per_layer_expert_b={per_layer_expert_b}  layers={result.layers}", file=sys.stderr)
        print(f"  [verbose] experts_per_layer={result.experts_per_layer}  (total={result.experts})", file=sys.stderr)
        print(f"  [verbose] n_full_layers={kv_shape.n_full_layers}  n_swa_layers={kv_shape.n_swa_layers}", file=sys.stderr)
        print(f"  [verbose] full_per_tok_b={kv_shape.full_per_tok_b}  swa_const_b={kv_shape.swa_const_b}", file=sys.stderr)

    # Step 2a: largest ctx that fits with dense backbone + compute overhead reserved.
    overhead_b = compute_overhead_mib * MIB
    fit_ctx_raw = max_fit_ctx(kv_shape, result.dense_b, vram_mib, compute_overhead_mib)
    fit_max_ctx = align_ctx_down(min(result.model_max_ctx, max(0, fit_ctx_raw)))

    if verbose:
        print(f"  [verbose] fit_ctx_raw={fit_ctx_raw}  fit_max_ctx={fit_max_ctx}", file=sys.stderr)

    # Step 2b: cap the requested (or default) ctx by model_max and VRAM-fit.
    # ctx == 0 or ctx is None → use model's trained max context.
    if ctx is None or ctx == 0:
        requested_ctx = result.model_max_ctx
    else:
        requested_ctx = ctx
    chosen_ctx = align_ctx_down(min(requested_ctx, result.model_max_ctx, fit_ctx_raw))
    if ctx is not None and ctx > 0 and chosen_ctx < align_ctx_down(ctx):
        reasons = []
        if requested_ctx > result.model_max_ctx:
            reasons.append(f"model_max_ctx={result.model_max_ctx}")
        if requested_ctx > fit_ctx_raw:
            reasons.append(f"VRAM-fit max={fit_max_ctx}")
        print(f"NOTE: {model.name}: requested --ctx {ctx} clamped to "
              f"{chosen_ctx} ({', '.join(reasons) or 'CTX_PAD alignment'}).",
              file=sys.stderr)

    rec_ctx = align_ctx_down(min(result.model_max_ctx, max(1, fit_ctx_raw)))

    kv_b = kv_bytes(chosen_ctx, kv_shape)

    vram_total_b = vram_mib * MIB
    ram_budget_b = ram_mib * MIB

    # Step 3: fill remaining VRAM with layers, reserving space for compute
    # buffers (FlashAttention scratch, MTP draft weights, etc.).
    # Every layer has its own expert tensors (blk.{i}.ffn_*_exps), so VRAM
    # per layer is expert_b / num_layers.  --n-cpu-moe takes a layer count.
    vram_after_kv_b = vram_total_b - result.dense_b - kv_b - overhead_b
    if vram_after_kv_b <= 0 or per_layer_expert_b <= 0:
        gpu_layers = 0
    else:
        gpu_layers = max(0, min(result.layers, int(vram_after_kv_b / per_layer_expert_b)))
    cpu_layers = result.layers - gpu_layers

    gpu_expert_b = gpu_layers * per_layer_expert_b
    cpu_expert_b = cpu_layers * per_layer_expert_b

    # Derive per-expert counts for display and RAM budget checks
    gpu_experts = round(gpu_expert_b / per_expert_b) if per_expert_b > 0 else 0
    cpu_experts = result.experts - gpu_experts
    # Report includes the overhead so headroom reflects truly free VRAM.
    vram_used_b = result.dense_b + kv_b + gpu_expert_b + overhead_b

    if verbose:
        print(f"  [verbose] chosen_ctx={chosen_ctx}  kv_b={kv_b}", file=sys.stderr)
        print(f"  [verbose] vram_after_kv={vram_after_kv_b}  gpu_layers={gpu_layers}  cpu_layers={cpu_layers}", file=sys.stderr)
        print(f"  [verbose] vram_used_b={vram_used_b}", file=sys.stderr)

    return Plan(
        model=model, layers=result.layers, experts=result.experts, active=result.active,
        dense_b=result.dense_b, expert_b=result.expert_b, per_expert_b=per_expert_b,
        kv_b=kv_b, gpu_layers=gpu_layers, cpu_layers=cpu_layers,
        gpu_experts=gpu_experts, cpu_experts=cpu_experts,
        vram_used_b=vram_used_b, vram_total_b=vram_total_b,
        ram_budget_b=ram_budget_b, cpu_expert_b=cpu_expert_b,
        gpu_expert_b=gpu_expert_b, compute_overhead_b=overhead_b, ctx=chosen_ctx,
        model_max_ctx=result.model_max_ctx, fit_max_ctx=fit_max_ctx, rec_ctx=rec_ctx,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        bpe_k=CACHE_TYPE_BPE[cache_type_k],
        bpe_v=CACHE_TYPE_BPE[cache_type_v],
        experts_per_layer=result.experts_per_layer,
    )


def fmt_mib(b: float) -> str:
    return f"{b / MIB:>9.2f} MiB"


def fmt_gib(b: float) -> str:
    return f"{b / GIB:>6.2f} GiB"


def flag_line(p: Plan) -> str:
    return (f"--n-gpu-layers 999 --n-cpu-moe {p.n_cpu_moe} -c {p.ctx} "
            f"-ctk {p.cache_type_k} -ctv {p.cache_type_v}")


def print_full_report(p: Plan, vram_mib: int, ram_mib: int) -> None:
    print(f"Model:            {p.model}")
    print(f"Layers:           {p.layers}")
    print(f"Experts (total):  {p.experts}  (active per token: {p.active})")
    print(f"Context:          {p.ctx}  "
          f"(model max: {p.model_max_ctx}, VRAM-fit max: {p.fit_max_ctx})")
    print()
    print("=== Tensor sizes ===")
    print(f"  Dense backbone:        {fmt_mib(p.dense_b)}")
    print(f"  All experts:           {fmt_mib(p.expert_b)}")
    print(f"  One expert:            {fmt_mib(p.per_expert_b)}")
    eff = p.effective_factor
    print(f"  KV cache ({p.kv_type_str}, eff {eff:.3f}x):  {fmt_mib(p.kv_b)}")
    print()
    print(f"=== VRAM plan (budget {vram_mib} MiB) ===")
    print(f"  Dense backbone:        {fmt_mib(p.dense_b)}")
    print(f"  KV cache:              {fmt_mib(p.kv_b)}")
    print(f"  Compute/MTP buffer:    {fmt_mib(p.compute_overhead_b)}")
    print(f"  Experts on GPU ({p.gpu_layers:>3} layers, {p.gpu_experts:>3}):  {fmt_mib(p.gpu_expert_b)}")
    print(f"  (precedence: dense -> KV cache (capped to fit) -> experts layer-by-layer)")
    print(f"  -------------------------------------")
    print(f"  Used:                  {fmt_mib(p.vram_used_b)}  ({fmt_gib(p.vram_used_b)})")
    print(f"  Headroom:              {fmt_mib(p.vram_total_b - p.vram_used_b)}")
    print()
    print(f"=== RAM plan (budget {int(p.ram_budget_b / MIB)} MiB) ===")
    print(f"  Experts on CPU ({p.cpu_layers:>3} layers, {p.cpu_experts:>3}):  {fmt_mib(p.cpu_expert_b)}")
    print(f"  Headroom:              {fmt_mib(p.ram_budget_b - p.cpu_expert_b)}")
    print()
    print("=== Verdict ===")
    print(f"  VRAM: {'OK' if p.vram_ok else 'OVER BUDGET'}")
    print(f"  RAM:  {'OK' if p.ram_ok else 'OVER BUDGET'}")
    if not p.vram_ok:
        print("  -> Dense + KV cache alone exceed the VRAM budget. Reduce --ctx, "
              "use a smaller quant, or accept a smaller --n-gpu-layers split.")
    if not p.ram_ok:
        print("  -> CPU-side experts exceed the RAM budget. Use a smaller quant.")
    if p.gpu_experts < p.active:
        print(f"  -> Only {p.gpu_experts} experts on GPU; per-token routing needs "
              f"{p.active} active. On average {p.active - p.gpu_experts} of the "
              "active expert MLPs per token will run on CPU instead of GPU "
              "(slower per-token compute). Reduce --n-cpu-moe N to pin fewer "
              "layers to CPU, thereby keeping more layers (and their experts) on GPU.")
    print()
    print("=== llama-server flag ===")
    print(f"  {flag_line(p)}")


def scan_dir(scan_path: Path, vram: int, ram: int,
             ctx: int | None, quiet: bool, json_output: bool,
             cache_type_k: str, cache_type_v: str,
             compute_overhead_mib: float,
             models_file: Path | None = None) -> int:
    if models_file is not None:
        # Read explicit model list from file (one path per line)
        ggufs = [Path(p.strip()) for p in models_file.read_text().strip().splitlines()
                 if p.strip() and Path(p.strip()).exists()]
        if not ggufs:
            raise SystemExit(f"No valid models found in {models_file}")
    else:
        ggufs = sorted(scan_path.glob("*.gguf"))
        if not ggufs:
            raise SystemExit(f"No .gguf files found in {scan_path}")

    rows = []
    for g in ggufs:
        try:
            p = make_plan(g, vram, ram, ctx, cache_type_k, cache_type_v,
                          compute_overhead_mib=compute_overhead_mib)
            rows.append((g, p, None))
        except Exception as e:  # noqa: BLE001
            rows.append((g, None, str(e)))

    fitting = [(g, p) for g, p, err in rows if p is not None and p.fits]
    if ctx is not None:
        # Prefer models that can reach the full requested context.
        # Among those, prefer larger models with more GPU layers (the unit
        # of MoE offload).
        fitting.sort(key=lambda gp: (
            0 if gp[1].ctx >= ctx else 1,              # can reach ctx first
            -(gp[1].dense_b + gp[1].expert_b),          # larger models first
            gp[1].cpu_layers,                           # fewer CPU layers
        ))
    else:
        # Auto mode: largest models with most GPU layers.
        fitting.sort(key=lambda gp: (
            -(gp[1].dense_b + gp[1].expert_b),
            gp[1].cpu_layers,
        ))
    best = fitting[0] if fitting else None

    if ctx is not None and best and best[1].ctx < ctx:
        print(f"NOTE: no model fits the VRAM budget at ctx={ctx}; "
              f"best reaches {best[1]}. Reduce --ctx or use a smaller quant.",
              file=sys.stderr)

    # --json output: emit JSON array for callers (scan-all.sh, etc.)
    if json_output:
        plans = []
        for g, p, err in rows:
            if p is not None:
                plans.append(p.to_dict())
            else:
                plans.append({"error": err})
        print(json.dumps(plans, indent=2))
        return 0

    if quiet:
        if not best:
            print("# no model fits the budget", file=sys.stderr)
            return 1
        g, p = best
        print(f"# {g.name}")
        print(f"-m {g} {flag_line(p)}")
        if p.gpu_experts < p.active:
            print(f"# WARNING: {p.gpu_experts} GPU experts < {p.active} active; "
                  f"{p.active - p.gpu_experts} of {p.active} active experts run on CPU per token",
                  file=sys.stderr)
        return 0

    name_w = max(len(g.name) for g, _, _ in rows)
    ctx_label = "auto (per-model)" if ctx is None else str(ctx)
    print(f"Scanning {scan_path}  (vram={vram} MiB, ram={ram} MiB, "
          f"k={cache_type_k}, v={cache_type_v}, ctx={ctx_label})")
    print()
    header = (f"{'MODEL'.ljust(name_w)}  {'CTX':>7}  {'MAX':>7}  "
              f"{'VRAM used':>11}  {'RAM used':>11}  {'GPU/CPU lay':>12}  FIT  FLAG")
    print(header)
    print("-" * len(header))
    for g, p, err in rows:
        if err:
            print(f"{g.name.ljust(name_w)}  {'-':>7}  {'-':>7}  {'-':>11}  "
                  f"{'-':>11}  {'-':>12}  ERR  {err}")
            continue
        fit = "OK " if p.fits else ("vram" if not p.vram_ok else "ram ")
        gpu_cpu = f"{p.gpu_layers}/{p.cpu_layers}"
        print(f"{g.name.ljust(name_w)}  {p.ctx:>7}  {p.model_max_ctx:>7}  "
              f"{p.vram_used_b/MIB:>8.0f} MiB  {p.cpu_expert_b/MIB:>8.0f} MiB  "
              f"{gpu_cpu:>12}  {fit:>3}  {flag_line(p)}")
    print()
    if best:
        g, p = best
        print(f"Best fit: {g.name}")
        print(f"  ./llama.cpp/build/bin/llama-server -m {g} {flag_line(p)} -fa on")
        return 0
    print("No model fits the VRAM+RAM budget at the given context length.", file=sys.stderr)
    return 1


# ── AVX-512 diagnostic ────────────────────────────────────────────────────
def check_avx() -> None:
    """Read /proc/cpuinfo and report which instruction sets the CPU supports.
    Useful for verifying the build matches the runtime CPU — especially
    when building on a host different from the execution host (e.g. CI, cross-compilation, VM)."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text().lower()
    except FileNotFoundError:
        print("⚠ /proc/cpuinfo not found — skipping AVX-512 check (non-Linux?)", file=sys.stderr)
        return

    flags = set(cpuinfo.split())

    avx512_exts = [
        ("avx512f",       "GGML_AVX512=ON"),
        ("avx512bf16",    "GGML_AVX512_BF16=ON"),
        ("avx512vnni",    "GGML_AVX512_VNNI=ON"),
        ("avx512vbmi",    "GGML_AVX512_VBMI=ON"),
        ("avx512bw",      "GGML_AVX512_BW=ON"),
    ]

    # Also report the instruction sets that ARE available
    available_exts = [
        "avx2", "avx_vnni", "avx512f", "avx512bf16", "avx512vnni",
        "avx512vbmi", "avx512bw", "sse4_2", "sse4_1", "ssse3",
    ]
    available = sorted(set(e for e in available_exts if e in flags))

    avx512_supported = [f"{cmake} ({name})" for name, cmake in avx512_exts if name in flags]
    avx512_missing   = [f"{cmake} (CPU lacks {name})" for name, cmake in avx512_exts if name not in flags]

    print(f"Available: {', '.join(available)}")
    if avx512_supported:
        print(f"AVX-512:   {', '.join(avx512_supported)}")
    else:
        print("AVX-512:   none")
    if avx512_missing:
        print(f"⚠ Missing (will fall back to slower paths): {', '.join(avx512_missing)}")
    if not avx512_exts or not any(e in flags for e in ["avx512f", "avx512bf16", "avx512vnni"]):
        if "avx2" in flags:
            print("ℹ No AVX-512 — AVX2 paths will be used.")
        elif "avx_vnni" in flags and "avx2" in flags:
            print("ℹ AVX2 + VNNI available (good for GEMM), but no full AVX-512F.")
        elif "avx" in flags:
            print("ℹ AVX (256-bit) available but no AVX2 or AVX-512.")
    sys.exit(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-avx", action="store_true",
                    help="Check /proc/cpuinfo for AVX-512 extensions and exit.")
    ap.add_argument("gguf", nargs="?", help="Path to a .gguf model file (omit when using --scan)")
    ap.add_argument("--scan", metavar="DIR",
                    help="Scan a directory of .gguf files and recommend the best-fitting model")
    ap.add_argument("--vram", type=int, default=6144,
                    help="Total VRAM in MiB (default: 6144 = 6 GiB)")
    ap.add_argument("--ram",  type=int, default=32768,
                    help="RAM budget available to llama.cpp in MiB (default: 32768 = 32 GiB). "
                         "Subtract your OS / other-process overhead before passing.")
    ap.add_argument("--cache-type-k", default=CACHE_TYPE_K_DEFAULT,
                    help=f"KV cache type for keys (default: {CACHE_TYPE_K_DEFAULT}). "
                         f"Choices: {_valid_cache_types()}")
    ap.add_argument("--cache-type-v", default=CACHE_TYPE_V_DEFAULT,
                    help=f"KV cache type for values (default: {CACHE_TYPE_V_DEFAULT}). "
                         f"Choices: {_valid_cache_types()}")
    ap.add_argument("--ctx",  type=int, default=None,
                    help="Context length used for KV-cache sizing (default: 128000). "
                         "Pass --ctx 0 to use the model's trained max context, or any other value "
                         "to stretch as far as VRAM allows.")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Print only the recommended llama-server flags")
    ap.add_argument("--json", action="store_true",
                    help="Output plan as JSON (compatible with --quiet)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print intermediate calculation steps (Task 10)")
    ap.add_argument("--gguf-py-path", default=None,
                    help="Path to the gguf-py directory inside an llama.cpp checkout (auto-detected if omitted)")
    ap.add_argument("--models-file", default=None,
                    help="Path to a file with one model path per line (overrides --scan glob, for --exclude filtering)")
    ap.add_argument("--compute-overhead", type=float, default=4000.0,
                    help="MiB to reserve for compute buffers (FlashAttention scratch, MTP draft, "
                         "context-length spikes). Default: 4000 MiB — empirically safe for 256k "
                         "context Qwen3 MoE on a Tesla T4. Pass 0 to disable the reserve.")
    args = ap.parse_args()

    # --check-avx: diagnostic only, exit immediately
    if args.check_avx:
        check_avx()

    # Validate cache types
    cache_type_k = _validate_cache_type(args.cache_type_k)
    cache_type_v = _validate_cache_type(args.cache_type_v)

    if not args.scan and not args.gguf and not args.models_file:
        ap.error("provide a gguf path, --scan DIR, or --models-file PATH")
    if args.scan and args.gguf:
        ap.error("--scan and a positional gguf are mutually exclusive")
    if args.models_file and args.gguf:
        ap.error("--models-file and a positional gguf are mutually exclusive")
    if args.models_file and args.scan:
        ap.error("--models-file and --scan are mutually exclusive")

    find_gguf_py(args.gguf_py_path)

    if args.scan or args.models_file:
        if args.models_file:
            # models-file mode: scan_path is irrelevant, pass a dummy
            scan_path = Path("/dev/null")
        else:
            scan_path = Path(args.scan).expanduser()
            if not scan_path.is_dir():
                raise SystemExit(f"--scan path is not a directory: {scan_path}")
        models_file = Path(args.models_file).expanduser() if args.models_file else None
        return scan_dir(scan_path, args.vram, args.ram,
                        args.ctx, args.quiet, args.json,
                        cache_type_k, cache_type_v, args.compute_overhead,
                        models_file)

    model = Path(args.gguf).expanduser()
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")

    plan = make_plan(model, args.vram, args.ram,
                     args.ctx, cache_type_k, cache_type_v,
                     verbose=args.verbose, compute_overhead_mib=args.compute_overhead)
    if args.json:
        # Task 23: --json --verbose outputs JSON to stdout, verbose info to stderr
        print(json.dumps(plan.to_dict(verbose=args.verbose), indent=2))
        return 0 if plan.fits else 1

    if args.quiet:
        print(flag_line(plan))
        return 0 if plan.fits else 1

    print_full_report(plan, args.vram, args.ram)
    return 0 if plan.fits else 1


if __name__ == "__main__":
    sys.exit(main())
