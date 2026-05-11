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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# gguf-py (vendored in the llama.cpp source tree) gives us GGUFReader, which
# exposes both KV metadata fields and per-tensor n_bytes (the actual on-disk
# byte size, accounting for the quant type). Without this we'd be reading the
# element count from gguf_dump's text output and mistaking it for bytes —
# which made every quant of a model look identical in size.
def _bootstrap_gguf_py() -> None:
    # Check CWD-relative locations that may hold gguf-py.
    candidates = [
        Path.cwd() / "llama.cpp" / "gguf-py",
        Path.cwd() / "gguf-py",
    ]
    for c in candidates:
        if (c / "gguf" / "__init__.py").is_file():
            sys.path.insert(0, str(c))
            return
    raise SystemExit(
        "Could not locate gguf-py under ./llama.cpp/gguf-py or ./gguf-py. "
        "Pass --gguf-py-path /path/to/gguf-py to point at it explicitly."
    )

MIB = 1024 ** 2
GIB = 1024 ** 3

# TurboQuant / TCQ KV compression factor (relative to fp16). The canonical KV
# config is `-ctk turbo3_tcq -ctv turbo3_tcq` from the spiritbuun/buun-llama-cpp
# fork: 3.25 bpv symmetric = 0.203x of fp16. If you switch to a different KV
# pair (e.g. turbo4/turbo4 = 0.266, turbo3_tcq/turbo2_tcq = 0.172,
# turbo2_tcq/turbo2_tcq = 0.141, or fall back to q8_0/q8_0 = 0.531), update
# this constant to match — sizing here drives the suggested `--n-cpu-moe`.
KV_TBQ_FACTOR = 0.203
KV_BYTES_PER_ELEM = 2  # fp16 baseline before the TCQ factor

# llama.cpp rounds n_ctx UP to a multiple of CTX_PAD when it allocates the KV
# cache (`cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256)` in src/llama-context.cpp).
# We round our suggested -c value DOWN to the same alignment so the value the
# user passes matches the value llama.cpp actually allocates — no surprise
# upward padding past the budgeted VRAM.
CTX_PAD = 256


def align_ctx_down(n: int) -> int:
    """Largest multiple of CTX_PAD that is <= n, with a CTX_PAD floor."""
    return max(CTX_PAD, (n // CTX_PAD) * CTX_PAD)


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
    gpu_experts: int
    cpu_experts: int
    vram_used_b: float
    vram_total_b: float
    vram_tax_b: float
    ram_budget_b: float
    cpu_expert_b: float
    gpu_expert_b: float
    ctx: int
    model_max_ctx: int
    fit_max_ctx: int
    rec_ctx: int

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
        return self.cpu_experts


def find_gguf_py(explicit: str | None) -> None:
    if explicit:
        p = Path(explicit).expanduser()
        # Accept the gguf-py dir itself, the script path the old README
        # documented (.../gguf-py/gguf/scripts/gguf_dump.py), or anything
        # else inside the package — walk up looking for gguf/__init__.py.
        for cand in (p, *p.parents):
            if (cand / "gguf" / "__init__.py").is_file():
                sys.path.insert(0, str(cand))
                return
        raise SystemExit(
            f"--gguf-py-path does not point at a gguf-py package: {p}\n"
            "Pass the gguf-py directory (e.g. .../llama.cpp/gguf-py) or any "
            "path inside it."
        )
    _bootstrap_gguf_py()


@dataclass
class KVShape:
    """Per-architecture KV-cache geometry. Models with interleaved sliding-
    window attention (Gemma 3/4, Cohere2, etc.) split layers into:

      * **full** layers whose KV grows linearly with `ctx`, and
      * **SWA** layers whose KV is bounded by a fixed window and may use
        different (typically smaller) per-head key/value dims.

    KV bytes at context `ctx` (TCQ-compressed) is therefore:

        swa_const_b  +  ctx * full_per_tok_b

    where `swa_const_b` is independent of `ctx`. For models without SWA the
    SWA fields are zero and the formula collapses to the old linear form.
    """
    n_full_layers: int
    n_swa_layers: int
    full_per_tok_per_layer_b: int   # fp16 bytes/token in one full layer
    swa_per_tok_per_layer_b: int    # fp16 bytes/token in one SWA layer
    sliding_window: int             # tokens of KV retained per SWA layer

    @property
    def swa_const_b(self) -> float:
        return (self.n_swa_layers * self.sliding_window
                * self.swa_per_tok_per_layer_b * KV_TBQ_FACTOR)

    @property
    def full_per_tok_b(self) -> float:
        return (self.n_full_layers
                * self.full_per_tok_per_layer_b * KV_TBQ_FACTOR)

    def bytes_at(self, ctx: int) -> float:
        return self.swa_const_b + ctx * self.full_per_tok_b


def parse_metadata(model: Path) -> tuple[int, int, int, int, int, int, KVShape]:
    """Return (layers, experts, active, dense_b, expert_b, model_max_ctx,
    KVShape).

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
    full_per_tok_per_layer = n_head_kv * (key_len + val_len) * KV_BYTES_PER_ELEM

    # SWA geometry. Models without SWA have no swa_layers array and no
    # sliding_window key, in which case all layers are treated as full.
    key_len_swa = field_uint("attention.key_length_swa", default=key_len)
    val_len_swa = field_uint("attention.value_length_swa", default=val_len)
    swa_per_tok_per_layer = n_head_kv * (key_len_swa + val_len_swa) * KV_BYTES_PER_ELEM
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

    kv_shape = KVShape(
        n_full_layers=n_full_layers,
        n_swa_layers=n_swa_layers,
        full_per_tok_per_layer_b=full_per_tok_per_layer,
        swa_per_tok_per_layer_b=swa_per_tok_per_layer,
        sliding_window=sliding_window,
    )

    dense_b = 0
    expert_b = 0
    for t in r.tensors:
        if "_exps" in t.name:
            expert_b += int(t.n_bytes)
        else:
            dense_b += int(t.n_bytes)

    return (layers, experts, active, dense_b, expert_b, model_max_ctx, kv_shape)


def kv_bytes(ctx: int, kv_shape: KVShape) -> float:
    return kv_shape.bytes_at(ctx)


def max_fit_ctx(kv_shape: KVShape, dense_b: float,
                vram_mib: int, tax_mib: int) -> int:
    """Largest context such that tax + dense + KV fits in VRAM. KV is
    `swa_const + ctx * full_per_tok` (linear in ctx), so closed-form."""
    vram_for_kv_b = vram_mib * MIB - tax_mib * MIB - dense_b - kv_shape.swa_const_b
    if vram_for_kv_b <= 0:
        return 0
    if kv_shape.full_per_tok_b <= 0:
        # No full-attention layers: KV is bounded by the SWA window for every
        # layer, so any ctx fits as far as KV is concerned. Return a sentinel
        # large enough that `model_max_ctx` and `--ctx` always clamp first.
        return 10 ** 9
    return int(vram_for_kv_b // kv_shape.full_per_tok_b)


def make_plan(model: Path, vram_mib: int, ram_mib: int,
              tax_mib: int, ctx: int | None) -> "Plan":
    """Build a sizing Plan with this VRAM allocation precedence:

      1. **Dense backbone:** all non-expert tensors are placed on the GPU.
      2. **KV cache:** target ctx is the user-supplied `--ctx` (or, if
         omitted, the model's trained `context_length`). It is capped to
         `model_max_ctx` and capped further to whatever fits in VRAM after
         the dense backbone, then rounded down to a multiple of CTX_PAD so
         the value matches what llama.cpp actually allocates.
      3. **Experts:** any VRAM left over after dense + KV is filled with as
         many experts as fit. The remaining experts live in RAM and run on
         CPU when routed.

    There is no expert floor: if the dense backbone + KV cache leave no
    room, all experts go to RAM and the model pages experts from CPU on
    every routed token. The verdict surfaces this via `gpu_experts == 0`.
    """
    (layers, experts, active, dense_b, expert_b, model_max_ctx,
     kv_shape) = parse_metadata(model)
    if experts <= 0:
        raise ValueError("expert_count is 0 — this script targets MoE models only.")

    per_expert_b = expert_b / experts

    # Step 2a: largest ctx that fits with just the dense backbone reserved.
    fit_ctx_raw = max_fit_ctx(kv_shape, dense_b, vram_mib, tax_mib)
    fit_max_ctx = align_ctx_down(min(model_max_ctx, max(0, fit_ctx_raw)))

    # Step 2b: cap the requested (or default) ctx by model_max and VRAM-fit.
    requested_ctx = ctx if ctx is not None else model_max_ctx
    chosen_ctx = align_ctx_down(min(requested_ctx, model_max_ctx, fit_ctx_raw))
    if ctx is not None and chosen_ctx < align_ctx_down(ctx):
        reasons = []
        if requested_ctx > model_max_ctx:
            reasons.append(f"model_max_ctx={model_max_ctx}")
        if requested_ctx > fit_ctx_raw:
            reasons.append(f"VRAM-fit max={fit_max_ctx}")
        print(f"NOTE: requested --ctx {ctx} clamped to {chosen_ctx} "
              f"({', '.join(reasons) or 'CTX_PAD alignment'}).",
              file=sys.stderr)

    rec_ctx = align_ctx_down(min(model_max_ctx, max(1, fit_ctx_raw)))

    kv_b = kv_bytes(chosen_ctx, kv_shape)

    vram_total_b = vram_mib * MIB
    vram_tax_b = tax_mib * MIB
    ram_budget_b = ram_mib * MIB

    # Step 3: fill remaining VRAM with experts.
    vram_after_kv_b = vram_total_b - vram_tax_b - dense_b - kv_b
    if vram_after_kv_b <= 0 or per_expert_b <= 0:
        gpu_experts = 0
    else:
        gpu_experts = max(0, min(experts, int(vram_after_kv_b / per_expert_b)))
    cpu_experts = experts - gpu_experts

    gpu_expert_b = gpu_experts * per_expert_b
    cpu_expert_b = cpu_experts * per_expert_b
    vram_used_b = dense_b + kv_b + gpu_expert_b + vram_tax_b

    return Plan(
        model=model, layers=layers, experts=experts, active=active,
        dense_b=dense_b, expert_b=expert_b, per_expert_b=per_expert_b,
        kv_b=kv_b, gpu_experts=gpu_experts, cpu_experts=cpu_experts,
        vram_used_b=vram_used_b, vram_total_b=vram_total_b, vram_tax_b=vram_tax_b,
        ram_budget_b=ram_budget_b, cpu_expert_b=cpu_expert_b,
        gpu_expert_b=gpu_expert_b, ctx=chosen_ctx,
        model_max_ctx=model_max_ctx, fit_max_ctx=fit_max_ctx, rec_ctx=rec_ctx,
    )


def fmt_mib(b: float) -> str:
    return f"{b / MIB:>9.2f} MiB"


def fmt_gib(b: float) -> str:
    return f"{b / GIB:>6.2f} GiB"


def flag_line(p: Plan) -> str:
    return f"--n-gpu-layers 999 --n-cpu-moe {p.n_cpu_moe} -c {p.ctx}"


def print_full_report(p: Plan, vram_mib: int, ram_mib: int, tax_mib: int) -> None:
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
    print(f"  KV cache (TCQ x{KV_TBQ_FACTOR}):  {fmt_mib(p.kv_b)}")
    print()
    print(f"=== VRAM plan (budget {vram_mib - tax_mib} MiB) ===")
    print(f"  Dense backbone:        {fmt_mib(p.dense_b)}")
    print(f"  KV cache:              {fmt_mib(p.kv_b)}")
    print(f"  Experts on GPU ({p.gpu_experts:>3}):  {fmt_mib(p.gpu_expert_b)}")
    print(f"  (precedence: dense -> KV cache (capped to fit) -> experts)")
    print(f"  -------------------------------------")
    used_no_tax = p.vram_used_b - p.vram_tax_b
    print(f"  Used:                  {fmt_mib(used_no_tax)}  ({fmt_gib(used_no_tax)})")
    print(f"  Headroom:              {fmt_mib(p.vram_total_b - p.vram_used_b)}")
    print()
    print(f"=== RAM plan (budget {int(p.ram_budget_b / MIB)} MiB) ===")
    print(f"  Experts on CPU ({p.cpu_experts:>3}):  {fmt_mib(p.cpu_expert_b)}")
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
              "(slower per-token compute). Reduce --ctx or use a smaller quant "
              "if you need more GPU experts.")
    print()
    print("=== llama-server flag ===")
    print(f"  {flag_line(p)}")


def scan_dir(scan_path: Path, vram: int, ram: int,
             tax: int, ctx: int | None, quiet: bool) -> int:
    ggufs = sorted(scan_path.glob("*.gguf"))
    if not ggufs:
        raise SystemExit(f"No .gguf files found in {scan_path}")

    rows = []
    for g in ggufs:
        try:
            p = make_plan(g, vram, ram, tax, ctx)
            rows.append((g, p, None))
        except Exception as e:  # noqa: BLE001
            rows.append((g, None, str(e)))

    fitting = [(g, p) for g, p, err in rows if p is not None and p.fits]
    fitting.sort(key=lambda gp: (gp[1].dense_b + gp[1].expert_b, -gp[1].cpu_experts), reverse=True)
    best = fitting[0] if fitting else None

    if quiet:
        if not best:
            print("# no model fits the budget", file=sys.stderr)
            return 1
        g, p = best
        print(f"# {g.name}")
        print(f"-m {g} {flag_line(p)}")
        return 0

    name_w = max(len(g.name) for g, _, _ in rows)
    ctx_label = "auto (per-model)" if ctx is None else str(ctx)
    print(f"Scanning {scan_path}  (vram={vram} MiB, ram={ram} MiB, tax={tax} MiB, ctx={ctx_label})")
    print()
    header = (f"{'MODEL'.ljust(name_w)}  {'CTX':>7}  {'MAX':>7}  "
              f"{'VRAM used':>11}  {'RAM used':>11}  {'GPU/CPU exp':>12}  FIT  FLAG")
    print(header)
    print("-" * len(header))
    for g, p, err in rows:
        if err:
            print(f"{g.name.ljust(name_w)}  {'-':>7}  {'-':>7}  {'-':>11}  "
                  f"{'-':>11}  {'-':>12}  ERR  {err}")
            continue
        fit = "OK " if p.fits else ("vram" if not p.vram_ok else "ram ")
        gpu_cpu = f"{p.gpu_experts}/{p.cpu_experts}"
        print(f"{g.name.ljust(name_w)}  {p.ctx:>7}  {p.model_max_ctx:>7}  "
              f"{p.vram_used_b/MIB:>8.0f} MiB  {p.cpu_expert_b/MIB:>8.0f} MiB  "
              f"{gpu_cpu:>12}  {fit:>3}  {flag_line(p)}")
    print()
    if best:
        g, p = best
        print(f"Best fit: {g.name}")
        print(f"  ./llama.cpp/build/bin/llama-server -m {g} {flag_line(p)} -ctk turbo3_tcq -ctv turbo3_tcq -fa on")
        return 0
    print("No model fits the VRAM+RAM budget at the given context length.", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", nargs="?", help="Path to a .gguf model file (omit when using --scan)")
    ap.add_argument("--scan", metavar="DIR",
                    help="Scan a directory of .gguf files and recommend the best-fitting model")
    ap.add_argument("--vram", type=int, default=6144,
                    help="Total VRAM in MiB (default: 6144 = 6 GiB)")
    ap.add_argument("--ram",  type=int, default=32768,
                    help="Total system RAM in MiB (default: 32768 = 32 GiB). "
                         "The script subtracts --os-reserve to get the budget "
                         "available to llama.cpp.")
    ap.add_argument("--tax",  type=int, default=650,
                    help="VRAM reserved for driver/runtime in MiB (default: 650)")
    ap.add_argument("--os-reserve", type=int, default=5120, dest="os_reserve",
                    help="RAM reserved for OS and other processes in MiB "
                         "(default: 5120 = 5 GiB). Subtracted from --ram to get "
                         "the RAM budget available to llama.cpp.")
    ap.add_argument("--ctx",  type=int, default=None,
                    help="Context length used for KV-cache sizing. Default: per-model recommended "
                         "= min(model trained max, largest VRAM-fitting context).")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Print only the recommended llama-server flags")
    ap.add_argument("--gguf-py-path", default=None,
                    help="Path to the gguf-py directory inside an llama.cpp checkout (auto-detected if omitted)")
    args = ap.parse_args()

    if not args.scan and not args.gguf:
        ap.error("provide a gguf path or --scan DIR")
    if args.scan and args.gguf:
        ap.error("--scan and a positional gguf are mutually exclusive")

    find_gguf_py(args.gguf_py_path)

    if args.scan:
        scan_path = Path(args.scan).expanduser()
        if not scan_path.is_dir():
            raise SystemExit(f"--scan path is not a directory: {scan_path}")
        return scan_dir(scan_path, args.vram, max(0, args.ram - args.os_reserve),
                        args.tax, args.ctx, args.quiet)

    model = Path(args.gguf).expanduser()
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")

    plan = make_plan(model, args.vram, max(0, args.ram - args.os_reserve),
                     args.tax, args.ctx)
    if args.quiet:
        print(flag_line(plan))
        return 0 if plan.fits else 1

    print_full_report(plan, args.vram, args.ram, args.tax)
    return 0 if plan.fits else 1


if __name__ == "__main__":
    sys.exit(main())
