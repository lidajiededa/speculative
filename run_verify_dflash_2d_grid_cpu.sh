#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-full}"

case "${MODE}" in
  smoke)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_cpu.py \
      --cases-per-combo 1 \
      --batch-sizes 1,8 \
      --spec-tokens 1,4 \
      --kv-block-sizes 16,128 \
      --triton-block-sizes 64,128 \
      --max-prompt-len 64
    ;;
  full)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_cpu.py
    ;;
  bench)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_cpu.py \
      --cases-per-combo 1 \
      --batch-sizes 1 \
      --spec-tokens 1 \
      --kv-block-sizes 128 \
      --triton-block-sizes 128 \
      --max-prompt-len 16 \
      --benchmark \
      --bench-warmup "${BENCH_WARMUP:-3}" \
      --bench-iters "${BENCH_ITERS:-10}" \
      --bench-batch-sizes "${BENCH_BATCH_SIZES:-32,64,128}" \
      --bench-prompt-lens "${BENCH_PROMPT_LENS:-256,1024}" \
      --bench-spec-tokens "${BENCH_SPEC_TOKENS:-4,8}" \
      --bench-kv-block-sizes "${BENCH_KV_BLOCK_SIZES:-128}" \
      --bench-triton-block-sizes "${BENCH_TRITON_BLOCK_SIZES:-64,128,256}"
    ;;
  *)
    echo "Usage: $0 [smoke|full|bench]" >&2
    echo "Optional env: PYTHON_BIN, BENCH_WARMUP, BENCH_ITERS, BENCH_*" >&2
    exit 2
    ;;
esac
