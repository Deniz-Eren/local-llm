#!/usr/bin/env bash
# Copyright (c) 2026 Deniz Eren
# Licensed under the MIT License. See the LICENSE file at the repository
# root for the full license text.
# scan-all.sh — scan all GGUFs in a directory across multiple KV cache
# configurations and output a combined table.
#
# Usage:
#   ./scripts/scan-all.sh <models-dir> [OPTIONS]
#
# The default scan uses the standard asymmetric KV pair
# (turbo4/turbo3_tcq) at the default context. Pass --configs to scan
# multiple KV configurations.

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
SCAN_DIR=""
VRAM=6144
RAM=32768
CTX=128000
MOE_CONFIGS=""
GGUF_PY_PATH=""
OUTPUT_FORMAT="markdown"  # markdown | csv

# KV configs to scan (name -> "K V")
DEFAULT_CONFIGS=("turbo4/turbo3_tcq" "turbo3_tcq/turbo3_tcq" "turbo4/turbo4" "turbo3_tcq/turbo2_tcq")
CONFIGS=("${DEFAULT_CONFIGS[@]}")

# ── helpers ─────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") <models-dir> [OPTIONS]

Scan all .gguf files in <models-dir> across multiple KV cache configurations
and print a combined table.

Required:
  models-dir              Directory containing .gguf files

Options:
  --vram MIB              Total VRAM in MiB (default: 6144)
  --ram MIB               RAM budget in MiB (default: 32768)
  --ctx N                 Context length (default: 128000)
  --configs CONFIGS       Comma-separated KV config names to scan
                          (default: turbo4/turbo3_tcq,turbo3_tcq/turbo3_tcq,turbo4/turbo4,turbo3_tcq/turbo2_tcq)
  --format fmt            Output format: markdown | csv (default: markdown)
  --moe-configs PATH      Path to moe-configs.py (default: alongside this script)
  --gguf-py-path PATH     Path to gguf-py directory
  --quiet                 Only print fitting models
  --help, -h              Show this help

Examples:
  # Scan with default asymmetric KV at 128K context
  ./scripts/scan-all.sh ~/models

  # Scan multiple KV configs, 16 GiB VRAM
  ./scripts/scan-all.sh ~/models --vram 16384 \\
      --configs "turbo4/turbo3_tcq,turbo4/turbo4,turbo3_tcq/turbo2_tcq"

  # CSV output for spreadsheet import
  ./scripts/scan-all.sh ~/models --format csv
EOF
  exit 1
}

# ── parse args ──────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
  usage
fi

SCAN_DIR="$1"; shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vram)           VRAM="$2";    shift 2 ;;
    --ram)            RAM="$2";     shift 2 ;;
    --ctx)            CTX="$2";     shift 2 ;;
    --configs)        IFS=',' read -ra CONFIGS <<< "$2"; shift 2 ;;
    --format)         OUTPUT_FORMAT="$2"; shift 2 ;;
    --moe-configs)      MOE_CONFIGS="$2";  shift 2 ;;
    --gguf-py-path)   GGUF_PY_PATH="$2"; shift 2 ;;
    --quiet)          QUIET=true;   shift ;;
    *)                echo "Unknown option: $1"; usage ;;
  esac
done

[[ -d "$SCAN_DIR" ]] || { echo "Error: $SCAN_DIR is not a directory"; exit 1; }

# ── run moe-configs.py for each config ──────────────────────────────────────
# Resolve moe-configs.py path: explicit flag > script directory
if [[ -n "$MOE_CONFIGS" ]]; then
  MOE_CONFIGS="$(cd "$(dirname "$MOE_CONFIGS")" && pwd)/$(basename "$MOE_CONFIGS")"
else
  MOE_CONFIGS="$(cd "$(dirname "$0")" && pwd)/moe-configs.py"
fi

PYTHON_ARGS=(
  python3 "$MOE_CONFIGS"
  --scan "$SCAN_DIR"
  --vram "$VRAM"
  --ram "$RAM"
  --ctx "$CTX"
)

[[ -n "$GGUF_PY_PATH" ]] && PYTHON_ARGS+=(--gguf-py-path "$GGUF_PY_PATH")

# Collect results into temp files
TMPDIR_SCAN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_SCAN"' EXIT

for config in "${CONFIGS[@]}"; do
  IFS='/' read -r k v <<< "$config"
  echo "Scanning: ${config} (K=$k, V=$v, ctx=$CTX, vram=${VRAM} MiB, ram=${RAM} MiB)"

  set +e
  OUTPUT=$("${PYTHON_ARGS[@]}" --cache-type-k "$k" --cache-type-v "$v" 2>&1)
  EXIT_CODE=$?
  set -e
  echo "$OUTPUT" > "${TMPDIR_SCAN}/${config//\//_}.txt"

  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: moe-configs.py failed for config $config (exit $EXIT_CODE)" >&2
    echo "$OUTPUT" >&2
    exit 1
  fi
done

# ── parse and combine ──────────────────────────────────────────────────────
# Parse the moe-configs.py scan output into a structured format.
# Each scan file has lines like:
#   MODEL_NAME     CTX  MAX  VRAM MiB  RAM MiB  GPU/CPU  FIT  FLAG...
# We extract: config, model, ctx, max_ctx, vram_mib, ram_mib, gpu_cpu, fit
parse_scan_file() {
  local file="$1"
  local config_name="$2"
  while IFS= read -r line; do
    # Skip header / dash / note lines, empty lines
    [[ "$line" =~ ^Scanning ]] && continue
    [[ "$line" =~ ^MODEL ]] && continue
    [[ "$line" =~ ^--- ]] && continue
    [[ "$line" =~ ^NOTE: ]] && continue
    [[ "$line" =~ ^$ ]] && continue

    # Skip error lines (contain "ERR" before any digit)
    [[ "$line" =~ "ERR" ]] && continue

    # Must start with a filename (contains ".gguf")
    [[ "$line" =~ \.gguf ]] || continue

    # Extract fields via regex: the scan format is:
    #   <name>.gguf  <ctx>  <max>  <vram> MiB  <ram> MiB  <gpu/cpu>  <fit>  ...
    # Use .* to match model names containing dots (e.g., Qwen3.6-...).
    if [[ "$line" =~ ([A-Za-z0-9_.-]+\.gguf)[[:space:]]+([0-9]+)[[:space:]]+([0-9]+)[[:space:]]+([0-9]+)[[:space:]]+MiB[[:space:]]+([0-9]+)[[:space:]]+MiB[[:space:]]+([0-9]+/[0-9]+)[[:space:]]+([A-Za-z]+) ]]; then
      local model="${BASH_REMATCH[1]}"
      local ctx="${BASH_REMATCH[2]}"
      local max_ctx="${BASH_REMATCH[3]}"
      local vram_mib="${BASH_REMATCH[4]}"
      local ram_mib="${BASH_REMATCH[5]}"
      local gpu_cpu="${BASH_REMATCH[6]}"
      local fit="${BASH_REMATCH[7]}"
      echo "${config_name}|${model}|${ctx}|${max_ctx}|${vram_mib}|${ram_mib}|${gpu_cpu}|${fit}"
    fi
  done < "$file"
}

# Build combined results
declare -A RESULTS
ALL_MODELS=()

for config in "${CONFIGS[@]}"; do
  IFS='/' read -r k v <<< "$config"
  config_short="${k}_${v}"
  file="${TMPDIR_SCAN}/${config_short}.txt"

  if [[ ! -f "$file" ]]; then
    echo "Warning: no output for config $config" >&2
    continue
  fi

  while IFS='|' read -r _config model ctx max_ctx vram_mib ram_mib gpu_cpu fit; do
    # Skip error entries
    [[ "$fit" == "ERR" ]] && continue

    if [[ ! " ${ALL_MODELS[*]:-} " =~ " ${model} " ]]; then
      ALL_MODELS+=("$model")
    fi

    RESULTS["${model}|${config_short}"]="${ctx}|${max_ctx}|${vram_mib}|${ram_mib}|${gpu_cpu}|${fit}"
  done < <(parse_scan_file "$file" "$config")
done

# ── output ──────────────────────────────────────────────────────────────────
if [[ "$OUTPUT_FORMAT" == "csv" ]]; then
  # CSV header
  echo "model,context,max_ctx,vram_used_mib,ram_used_mib,gpu_cpu,fit"
  for model in "${ALL_MODELS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      IFS='/' read -r k v <<< "$config"
      config_short="${k}_${v}"
      result="${RESULTS["${model}|${config_short}"]:-}"
      if [[ -n "$result" ]]; then
        IFS='|' read -r ctx max_ctx vram_mib ram_mib gpu_cpu fit <<< "$result"
        echo "${model},${ctx},${max_ctx},${vram_mib},${ram_mib},${gpu_cpu},${fit}"
      fi
    done
  done
elif [[ "$OUTPUT_FORMAT" == "markdown" ]]; then
  # Markdown table
  echo '| Model | Config | `ctx` | `max_ctx` | VRAM used | RAM used | GPU/CPU | FIT |'
  echo '|-------|--------|-------|-----------|-----------|----------|---------|-----|'
  for model in "${ALL_MODELS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      IFS='/' read -r k v <<< "$config"
      config_short="${k}_${v}"
      result="${RESULTS["${model}|${config_short}"]:-}"
      if [[ -n "$result" ]]; then
        IFS='|' read -r ctx max_ctx vram_mib ram_mib gpu_cpu fit <<< "$result"
        bold_fit="${fit}"
        [[ "$fit" == "OK" ]] && bold_fit="**${fit}**"
        echo "| ${model} | ${k}/${v} | ${ctx} | ${max_ctx} | ${vram_mib} MiB | ${ram_mib} MiB | ${gpu_cpu} | ${bold_fit} |"
      fi
    done
  done
fi
