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
NO_SPEC_TYPE=""
NO_MMAP=""
MLOCK=""
API_KEY_FILE=""
DRY_RUN=false
QUIET=""

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
  --no-mmap               Disable memory mapping. (Passing it is redundant
                          since llama-server defaults to --no-mmap; included
                          for explicitness in examples and debugging.)
  --mlock                 Pin model weights in RAM (mlock).
  --api-key-file PATH     Path to a file containing the API key.
                          Reads the file content (stripped of trailing
                          whitespace/newlines) and passes it to
                          llama-server as --api-key.
  --dry-run               Compose the full command and print it without
                          launching. Useful for debugging or pasting into
                          another terminal/screen.
  --quiet                 Run non-interactively (no server output).
  --fit off|on            Expert auto-fit mode (default: off).
  --spec-type TYPE        Speculative decoding type (e.g. draft-mtp).
                          Omit to skip speculative decoding entirely.
  --spec-draft-n-max N    Draft tokens per speculative step
                          (default: 3; requires --spec-type).
  --no-spec-type          Disable speculative decoding even if model has MTP
                          weights. Useful when MTP causes instability.
  --flash-attn on|off     Flash attention (default: on).

Examples:
  # Auto-size from GGUF, run on this machine's hardware defaults
  ./scripts/run-server.sh --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf

  # Explicit expert count, 128K context, custom port
  ./scripts/run-server.sh -m ~/models/Kimi-K2.6-UD-Q4_K_XL.gguf \
      --n-cpu-moe 382 --ctx 128000 --port 8081 --alias Kimi-K2.6

  # Tighter KV, fewer threads for a hybrid CPU
  ./scripts/run-server.sh -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
      --n-cpu-moe 237 -ctk turbo3_tcq -ctv turbo3_tcq \
      --threads 8 --alias qwen3.6

  # Explicit mmap / mlock flags (Kimi K2.6 canonical)
  ./scripts/run-server.sh -m ~/models/Kimi-K2.6-UD-Q4_K_XL.gguf \
      --n-cpu-moe 382 --threads 14 --mlock --no-mmap \
      --alias Kimi-K2.6

  # Dry-run: preview the command without launching
  ./scripts/run-server.sh -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
      --dry-run

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
    --no-spec-type)     NO_SPEC_TYPE=1;    shift ;;
    --quiet)            QUIET=1;           shift ;;
    --dry-run)          DRY_RUN=true;      shift ;;
    --no-mmap)          NO_MMAP=1;         shift ;;
    --mlock)            MLOCK=1;           shift ;;
    --api-key-file)     API_KEY_FILE="$2"; shift 2 ;;
    --help|-h)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

# ── validation ──────────────────────────────────────────────────────────────
[[ -z "${MODEL:-}" ]] && echo "Error: --model is required" && exit 1

SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
# Skip server validation in dry-run mode — we're just composing the command
if [[ "$DRY_RUN" != "true" ]]; then
  [[ -x "$SERVER" ]] || { echo "Error: server not found at $SERVER"; exit 1; }
fi

# Auto-detect expert count from moe-configs.py if not given
if [[ -z "$N_CPU_MOE" ]]; then
  echo "Auto-sizing from moe-configs.py ..."
  # moe-configs.py exits 1 when the model doesn't fit — that's expected, not an error.
  # Exit code > 1 or stderr with "not found" means a real failure.
  _ERR=$(mktemp); _RC=0
  FLAGS=$(python3 "$(dirname "$0")/moe-configs.py" "$MODEL" --quiet --json 2>"$_ERR") || _RC=$?
  _STDERR=$(cat "$_ERR"); rm -f "$_ERR"
  if [[ $_RC -gt 1 ]] || echo "$_STDERR" | grep -qi "not found\|error"; then
    echo "Error: moe-configs.py failed: $_STDERR"; exit 1
  fi
  [[ $_RC -eq 1 ]] && echo "⚠ Model does not fit in configured VRAM/RAM budget (fits=false)"

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

# Auto-detect threads.
# For hybrid Intel (P/E cores): use P-cores only (E-cores bottleneck GEMM).
# Thread auto-detect: prefer physical cores for intra-GEMM parallelism.
# On Intel hybrid (P+E-core) CPUs, only use P-cores since E-cores stall
# the GEMM finish line and reduce per-core throughput.
if [[ -z "$THREADS" ]]; then
  if grep -q 'Intel' /proc/cpuinfo 2>/dev/null; then
    if command -v lscpu &>/dev/null; then
      # Detect hybrid via lscpu --extended: P-cores have higher MAXMHZ than E-cores.
      # Example (i7-13850HX): E-cores at 3800 MHz, P-cores at 5100–5300 MHz.
      # The lowest MAXMHZ identifies E-cores; everything above is P-cores.
      # Not all P-cores share the same MAXMHZ (power/thermal variance), so we
      # count unique CORE IDs (not threads) to get actual P-core count.
      EXTENDED=$(lscpu --extended=CPU,CORE,MAXMHZ 2>/dev/null | tail -n +2)
      if [[ -n "$EXTENDED" ]]; then
        E_CORE_FREQ=$(echo "$EXTENDED" | awk '{print $3}' | sort -n | head -1)
        if [[ -n "$E_CORE_FREQ" ]]; then
          # Count unique P-core IDs (cores with MAXMHZ strictly above E-core floor)
          P_CORE_COUNT=$(echo "$EXTENDED" | awk -v ef="$E_CORE_FREQ" '$3+0 > ef+0 {print $2}' | sort -un | wc -l)
          TOTAL_CORES=$(echo "$EXTENDED" | awk '{print $2}' | sort -un | wc -l)
          if [[ "$P_CORE_COUNT" -gt 0 && "$P_CORE_COUNT" -lt "$TOTAL_CORES" ]]; then
            THREADS=$P_CORE_COUNT
            echo "Hybrid Intel detected (P-cores only): $THREADS P-cores above E-core floor ($E_CORE_FREQ MHz)"
          fi
        fi
      fi
      # Fallback: non-hybrid (Xeon, older Core, AMD) or detection didn't find hybrid split
      if [[ -z "$THREADS" ]]; then
        THREADS=$(lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}
            /^Socket\(s\):/ {sockets=$NF}
            END {print cores*sockets}')
      fi
    else
      THREADS=$(nproc)
    fi
  else
    THREADS=$(nproc)
  fi
  echo "Auto-detected threads: $THREADS"
fi

# ── mlock-safe auto-guard (Task 13) ─────────────────────────────────────
# If --mlock was requested but the model doesn't fit in RAM, disable it
# to prevent OOM. We already have the JSON from auto-sizing.
if [[ -n "$MLOCK" && -n "${FLAGS:-}" ]]; then
  FITS=$(echo "$FLAGS" | jq -r '.fits' 2>/dev/null)
  if [[ "$FITS" != "true" ]]; then
    echo "⚠ RAM or VRAM over budget — disabling --mlock to prevent OOM"
    MLOCK=""
  fi
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
# Task 22: detect missing MTP weights and warn
if [[ -n "$SPEC_TYPE" ]]; then
  # Check if model has MTP tensors
  if command -v python3 &>/dev/null; then
    _HAS_MTP=$(python3 -c "
import sys
try:
    from gguf import GGUFReader
    r = GGUFReader('$MODEL')
    has_mtp = any('mtp_' in t.name or '_mtp' in t.name for t in r.tensors)
    print('yes' if has_mtp else 'no')
except Exception as e:
    print(f'error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null) || _HAS_MTP="error"
    if [[ "$_HAS_MTP" == "no" ]]; then
      echo "⚠ Model does not have MTP weights; --spec-type $SPEC_TYPE will be ignored"
    elif [[ "$_HAS_MTP" == "error" ]]; then
      echo "⚠ Could not verify MTP weights; proceeding with --spec-type $SPEC_TYPE"
    fi
  fi
  CMD+=(--spec-type "$SPEC_TYPE")
  [[ -n "$SPEC_DRAFT_N_MAX" ]] && CMD+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")
elif [[ -z "$NO_SPEC_TYPE" ]]; then
  # Auto-detect MTP and enable speculative decoding if available
  if command -v python3 &>/dev/null; then
    _HAS_MTP=$(python3 -c "
import sys
try:
    from gguf import GGUFReader
    r = GGUFReader('$MODEL')
    has_mtp = any('mtp_' in t.name or '_mtp' in t.name for t in r.tensors)
    print('yes' if has_mtp else 'no')
except:
    print('no')
" 2>/dev/null) || _HAS_MTP="no"
    if [[ "$_HAS_MTP" == "yes" ]]; then
      echo "ℹ MTP weights detected; enabling speculative decoding (use --no-spec-type to disable)"
      CMD+=(--spec-type draft-mtp --spec-draft-n-max 3)
    fi
  fi
fi

# Optional flags
[[ -n "$ALIAS" ]]     && CMD+=(--alias "$ALIAS")
[[ -n "$THREADS" ]]   && CMD+=(--threads "$THREADS")
[[ -n "$NO_MMAP" ]]    && CMD+=(--no-mmap)
[[ -n "$MLOCK" ]]      && CMD+=(--mlock)
# API key (read from file to avoid leaking into process listing)
if [[ -n "$API_KEY_FILE" ]]; then
  if [[ ! -f "$API_KEY_FILE" ]]; then
    echo "Error: api-key-file not found: $API_KEY_FILE"
    exit 1
  fi
  API_KEY=$(tr -d '\n\r' < "$API_KEY_FILE")
  [[ -n "$API_KEY" ]] && CMD+=(--api-key "$API_KEY")
fi

# ── launch ──────────────────────────────────────────────────────────────────
# --dry-run: compose and print the command without launching
if [[ "$DRY_RUN" == "true" ]]; then
  echo "# Would launch:"
  echo "${CMD[*]}"
  exit 0
fi

echo "Launching: ${CMD[*]}"
echo ""

if [[ -n "$QUIET" ]]; then
  exec "${CMD[@]}"
else
  "${CMD[@]}"
fi
