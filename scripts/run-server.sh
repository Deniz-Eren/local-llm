#!/usr/bin/env bash
# Copyright (c) 2026 Deniz Eren
# Licensed under the MIT License. See the LICENSE file at the repository
# root for the full license text.
# run-server.sh — launch llama-server with MoE expert routing and turbo* KV.
#
# Usage:
#   ./scripts/run-server.sh --model <gguf> --n-cpu-moe N [options]
#
# Reads sizing flags from moe-configs.py when --n-cpu-moe is not supplied:
#   ./scripts/run-server.sh --model <gguf>
#   # (auto-derives --n-cpu-moe, -c, -ctk, -ctv from moe-configs.py)

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
LLAMA_CPP_DIR="./llama.cpp"
HOST="0.0.0.0"
PORT=8080
ALIAS=""
N_GPUS_LAYERS=999
N_CPU_MOE=""
CTX=""
CTK=""
CTV=""
THREADS=""
FA="on"
FIT="off"
NP=1
SPEC_TYPE=""
SPEC_DRAFT_N_MAX=""
NO_MMAP=""
MLOCK=""

# ── helpers ─────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") --model <gguf> [OPTIONS]

Run llama-server with MoE expert routing and TurboQuant KV cache.

Required:
  --model, -m PATH        Path to the .gguf model file

Options:
  --n-cpu-moe N           Number of experts to keep in RAM (CPU).
                          Omit to auto-detect from moe-configs.py.
  -c, --ctx N             Context length (default: 128000).
  -ctk, --cache-type-k T  KV cache type for keys (default: turbo4).
  -ctv, --cache-type-v T  KV cache type for values (default: turbo3_tcq).
  --threads N             CPU threads for expert MLPs.
                          Omit to auto-detect (P-cores for hybrid Intel).
  --alias NAME            Server alias for ANTHROPIC_MODEL resolution.
  --host HOST             Listen address (default: 0.0.0.0).
  --port PORT             Listen port (default: 8080).
  --llama-cpp-dir DIR     Path to llama.cpp checkout (default: ./llama.cpp).
  --no-mmap               Override: disable memory mapping.
                          (llama-server defaults to no-mmap; this flag
                          is redundant unless you want to be explicit.)
  --mlock                 Pin model weights in RAM (mlock).
  --quiet                 Run non-interactively (no server output).
  --fit off|on            Expert auto-fit mode (default: off).
  --spec-type TYPE        Speculative decoding type (e.g. draft-mtp).
                          Omit to skip speculative decoding entirely.
  --spec-draft-n-max N    Draft tokens per speculative step
                          (default: 3; requires --spec-type).
  --flash-attn on|off     Flash attention (default: on).

Examples:
  # Auto-size from GGUF, run on this machine's hardware defaults
  ./scripts/run-server.sh --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf

  # Explicit expert count, 128K context, custom port
  ./scripts/run-server.sh -m ~/models/Kimi-K2.6-UD-Q4_K_XL.gguf \
      --n-cpu-moe 382 --ctx 128000 --port 8081 --alias kimi-k2.6

  # Tighter KV, fewer threads for a hybrid CPU
  ./scripts/run-server.sh -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
      --n-cpu-moe 237 -ctk turbo3_tcq -ctv turbo3_tcq \
      --threads 8 --alias qwen3.6

  # Explicit mmap / mlock flags (Kimi K2.6 canonical)
  ./scripts/run-server.sh -m ~/models/Kimi-K2.6-UD-Q4_K_XL.gguf \
      --n-cpu-moe 382 --threads 28 --mlock --no-mmap \
      --alias kimi-k2.6

  # Try fit mode to let llama.cpp auto-fit experts to VRAM
  ./scripts/run-server.sh -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
      --fit on

  # Speculative decoding with MTP draft tokens
  ./scripts/run-server.sh -m ~/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
      --n-cpu-moe 86 --ctx 128000 --alias qwen3.6-ud \
      --spec-type draft-mtp --spec-draft-n-max 3
EOF
  exit 1
}

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)         MODEL="$2";        shift 2 ;;
    --n-cpu-moe)        N_CPU_MOE="$2";    shift 2 ;;
    --ctx|-c)           CTX="$2";          shift 2 ;;
    --cache-type-k|-ctk) CTK="$2";         shift 2 ;;
    --cache-type-v|-ctv) CTV="$2";         shift 2 ;;
    --threads)          THREADS="$2";      shift 2 ;;
    --alias)            ALIAS="$2";        shift 2 ;;
    --host)             HOST="$2";         shift 2 ;;
    --port)             PORT="$2";         shift 2 ;;
    --llama-cpp-dir)    LLAMA_CPP_DIR="$2"; shift 2 ;;
    --fit)              FIT="$2";          shift 2 ;;
    --flash-attn)       FA="$2";          shift 2 ;;
    --spec-type)        SPEC_TYPE="$2";    shift 2 ;;
    --spec-draft-n-max) SPEC_DRAFT_N_MAX="$2"; shift 2 ;;
    --quiet)            QUIET=true;        shift ;;
    --no-mmap)          NO_MMAP=1;         shift ;;
    --mlock)            MLOCK=1;           shift ;;
    --help|-h)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

# ── validation ──────────────────────────────────────────────────────────────
[[ -z "${MODEL:-}" ]] && echo "Error: --model is required" && exit 1

SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
[[ -x "$SERVER" ]] || { echo "Error: server not found at $SERVER"; exit 1; }

# Auto-detect expert count from moe-configs.py if not given
if [[ -z "$N_CPU_MOE" ]]; then
  echo "Auto-sizing from moe-configs.py ..."
  FLAGS=$(python3 "$(dirname "$0")/moe-configs.py" "$MODEL" --quiet --json 2>/dev/null) || { echo "Error: moe-configs.py failed"; exit 1; }

  # Extract fields from JSON via jq
  N_CPU_MOE=$(echo "$FLAGS" | jq -r '.n_cpu_moe')
  CTX=$(echo "$FLAGS"       | jq -r '.ctx')
  CTK=$(echo "$FLAGS"       | jq -r '.cache_type_k')
  CTV=$(echo "$FLAGS"       | jq -r '.cache_type_v')

  # Auto-detect threads (fall back to nproc).
  # For hybrid Intel CPUs, always set --threads to the P-core count explicitly.
fi

# Defaults if still empty
CTK="${CTK:-turbo4}"
CTV="${CTV:-turbo3_tcq}"
CTX="${CTX:-128000}"

# Auto-detect threads (P-cores for hybrid Intel)
if [[ -z "$THREADS" ]]; then
  if grep -q 'Intel' /proc/cpuinfo 2>/dev/null; then
    if command -v lscpu &>/dev/null; then
      # P-cores: physical cores × sockets
      THREADS=$(lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}
          /^Socket\(s\):/ {sockets=$NF}
          END {print cores*sockets}')
    else
      THREADS=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || nproc)
    fi
  else
    THREADS=$(nproc)
  fi
  echo "Auto-detected threads: $THREADS"
fi

# ── build command ───────────────────────────────────────────────────────────
CMD=(
  "$SERVER"
  -m "$MODEL"
  --n-gpu-layers "$N_GPUS_LAYERS"
  --n-cpu-moe "$N_CPU_MOE"
  -ctk "$CTK"
  -ctv "$CTV"
  -c "$CTX"
  -fa "$FA"
  --fit "$FIT"
  -np "$NP"
  --host "$HOST"
  --port "$PORT"
)
# Speculative decoding (MTP / draft tokens)
[[ -n "$SPEC_TYPE" ]]    && CMD+=(--spec-type "$SPEC_TYPE")
[[ -n "$SPEC_DRAFT_N_MAX" ]] && CMD+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")

# Optional flags
[[ -n "$ALIAS" ]]     && CMD+=(--alias "$ALIAS")
[[ -n "$THREADS" ]]   && CMD+=(--threads "$THREADS")
[[ -n "$NO_MMAP" ]]    && CMD+=(--no-mmap)
[[ -n "$MLOCK" ]]      && CMD+=(--mlock)

# ── launch ──────────────────────────────────────────────────────────────────
echo "Launching: ${CMD[*]}"
echo ""

if [[ "${QUIET:-false}" == "true" ]]; then
  exec "${CMD[@]}"
else
  "${CMD[@]}"
fi
