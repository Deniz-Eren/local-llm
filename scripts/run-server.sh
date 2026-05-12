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
  FLAGS=$(python3 "$(dirname "$0")/moe-configs.py" "$MODEL" --quiet 2>/dev/null)
  [[ $? -ne 0 ]] && { echo "Error: moe-configs.py failed"; exit 1; }

  # Extract --n-cpu-moe and -c from the flag line
  N_CPU_MOE=$(echo "$FLAGS" | grep -oP '(?<=--n-cpu-moe )\d+')
  CTX=$(echo "$FLAGS"   | grep -oP '(?<=-c )\d+')
  CTK=$(echo "$FLAGS"   | grep -oP '(?<=-ctk )\S+')
  CTV=$(echo "$FLAGS"   | grep -oP '(?<=-ctv )\S+')

  # Auto-detect threads (fall back to nproc).
  # For hybrid Intel CPUs, always set --threads to the P-core count explicitly.
fi

# Defaults if still empty
CTK="${CTK:-turbo4}"
CTV="${CTV:-turbo3_tcq}"
CTX="${CTX:-128000}"

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
