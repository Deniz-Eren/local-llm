#!/usr/bin/env bash
# Copyright (c) 2026 Deniz Eren
# Licensed under the MIT License. See the LICENSE file at the repository
# root for the full license text.
# scan-all.sh — scan all GGUFs across multiple KV cache configurations.

set -euo pipefail

SCAN_DIR=""
VRAM=6144
RAM=32768
CTX=128000
MOE_CONFIGS=""
GGUF_PY_PATH=""
OUTPUT_FORMAT="markdown"
EXCLUDE=""

# Python interpreter: prefer venv, fall back to system python3.
if [[ -f "$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python3" ]]; then
  PYTHON="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python3"
else
  PYTHON="${PYTHON:-python3}"
fi

DEFAULT_CONFIGS=("turbo4/turbo3_tcq" "turbo3_tcq/turbo3_tcq" "turbo4/turbo4" "turbo3_tcq/turbo2_tcq")
CONFIGS=("${DEFAULT_CONFIGS[@]}")

usage() {
  cat <<EOF
Usage: $(basename "$0") <models-dir> [OPTIONS]

Scan all .gguf files across multiple KV cache configurations.

Options:
  --vram MIB              Total VRAM in MiB (default: 6144)
  --ram MIB               RAM budget in MiB (default: 32768)
  --ctx N                 Context length (default: 128000)
  --configs CONFIGS       Comma-separated KV config names to scan
  --exclude PATTERN       Glob pattern of model names to skip (e.g. "*.Q8_0*")
  --format fmt            Output format: markdown | csv (default: markdown)
  --moe-configs PATH      Path to moe-configs.py (default: alongside this script)
  --gguf-py-path PATH     Path to gguf-py directory
  --quiet                 Only print fitting models
  --help, -h              Show this help
EOF
  exit 1
}

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
    --exclude)        EXCLUDE="$2"; shift 2 ;;
    --format)         OUTPUT_FORMAT="$2"; shift 2 ;;
    --moe-configs)    MOE_CONFIGS="$2";  shift 2 ;;
    --gguf-py-path)   GGUF_PY_PATH="$2"; shift 2 ;;
    --quiet)          QUIET=true;   shift ;;
    *)                echo "Unknown option: $1"; usage ;;
  esac
done

[[ -d "$SCAN_DIR" ]] || { echo "Error: $SCAN_DIR is not a directory"; exit 1; }

# ── resolve moe-configs.py ─────────────────────────────────────────────────
if [[ -n "$MOE_CONFIGS" ]]; then
  MOE_CONFIGS="$(cd "$(dirname "$MOE_CONFIGS")" && pwd)/$(basename "$MOE_CONFIGS")"
else
  MOE_CONFIGS="$(cd "$(dirname "$0")" && pwd)/moe-configs.py"
fi

# ── temp dir ────────────────────────────────────────────────────────────────
TMPDIR_SCAN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_SCAN"' EXIT

# ── filter GGUFs ────────────────────────────────────────────────────────────
FILTERED_LIST="${TMPDIR_SCAN}/filtered.txt"
for gguf in "$SCAN_DIR"/*.gguf; do
  [[ -f "$gguf" ]] || continue
  if [[ -n "$EXCLUDE" ]]; then
    base=$(basename "$gguf")
    # shellcheck disable=SC2254
    case "$base" in
      $EXCLUDE) continue ;;
    esac
  fi
  echo "$gguf"
done > "$FILTERED_LIST"

if [[ ! -s "$FILTERED_LIST" ]]; then
  echo "No .gguf files found in $SCAN_DIR" >&2
  exit 1
fi

# ── scan each KV config (sequential — background jobs produce truncated output) ──
for config in "${CONFIGS[@]}"; do
  IFS='/' read -r k v <<< "$config"
  echo "Scanning: ${config} (K=$k, V=$v, ctx=$CTX, vram=${VRAM} MiB, ram=${RAM} MiB)"

  outfile="${TMPDIR_SCAN}/${config//\//_}.json"

  # Use moe-configs.py --scan --json for structured output (Task 3 cascade)
  # Sequential execution avoids background-job truncation bug
  # When --exclude is set, pass filtered model list via --models-file
  if [[ -n "$EXCLUDE" ]]; then
    "${PYTHON}" "$MOE_CONFIGS" --models-file "$FILTERED_LIST" \
      --vram "$VRAM" --ram "$RAM" \
      --ctx "$CTX" \
      --cache-type-k "$k" --cache-type-v "$v" \
      --json >"$outfile" 2>"${outfile}.err"
  else
    "${PYTHON}" "$MOE_CONFIGS" --scan "$SCAN_DIR" \
      --vram "$VRAM" --ram "$RAM" \
      --ctx "$CTX" \
      --cache-type-k "$k" --cache-type-v "$v" \
      --json >"$outfile" 2>"${outfile}.err"
  fi

  # Check that output was actually written
  if [[ ! -s "$outfile" ]]; then
    echo "ERROR: moe-configs.py produced no output for config $config" >&2
    cat "${outfile}.err" >&2
    exit 1
  fi

  # Validate JSON
  if ! jq empty "$outfile" 2>/dev/null; then
    echo "ERROR: invalid JSON output for config $config" >&2
    head -5 "$outfile" >&2
    cat "${outfile}.err" >&2
    exit 1
  fi
done

# ── parse and combine ──────────────────────────────────────────────────────
declare -A RESULTS
ALL_MODELS=()

for config in "${CONFIGS[@]}"; do
  IFS='/' read -r k v <<< "$config"
  config_short="${k}_${v}"
  json_file="${TMPDIR_SCAN}/${config_short}.json"

  if [[ ! -f "$json_file" ]]; then
    echo "Warning: no output for config $config" >&2
    continue
  fi

  # Parse JSON array with jq — extract fields for each model
  while IFS=$'\t' read -r model ctx max_ctx vram_mib ram_mib gpu_cpu fit; do
    [[ -z "$model" ]] && continue

    # Add to ALL_MODELS only if new (Task 5: safe init with set -u)
    if [[ ${#ALL_MODELS[@]} -eq 0 ]] || [[ ! " ${ALL_MODELS[*]} " =~ " ${model} " ]]; then
      ALL_MODELS+=("$model")
    fi

    RESULTS["${model}|${config_short}"]="${ctx}|${max_ctx}|${vram_mib}|${ram_mib}|${gpu_cpu}|${fit}"
  done < <(
    jq -r '.[] |
      if .error then empty
      else
        [( .model | split("/") | last),
         .ctx,
         .fit_max_ctx,
         (.vram_used_mib | tostring),
         (.cpu_expert_mib | tostring),
         ((.gpu_experts | tostring) + "/" + (.cpu_experts | tostring)),
         (if .fits then "OK" else "vram" end)] |
        join("\t")
      end' "$json_file" 2>/dev/null
  )
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
