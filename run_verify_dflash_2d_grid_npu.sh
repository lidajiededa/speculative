#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-npu:0}"
MODE="${1:-smoke}"

# When running from the downloaded source trees in this workspace, these
# defaults make imports resolve without installing editable packages. Override
# VLLM_ASCEND_PATH or VLLM_PATH if your WSL layout is different.
VLLM_ASCEND_PATH="${VLLM_ASCEND_PATH:-${SCRIPT_DIR}/vllm-ascend-0.20.2rc1}"
VLLM_PATH="${VLLM_PATH:-${SCRIPT_DIR}/vllm-v0.20.2}"
export PYTHONPATH="${VLLM_ASCEND_PATH}:${VLLM_PATH}:${PYTHONPATH:-}"

case "${MODE}" in
  smoke)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_npu.py \
      --device "${DEVICE}" \
      --cases-per-combo 1 \
      --batch-sizes 1,8 \
      --spec-tokens 1,4 \
      --kv-block-sizes 16,128 \
      --triton-block-sizes 64,128 \
      --max-prompt-len 64
    ;;
  full)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_npu.py \
      --device "${DEVICE}"
    ;;
  long)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_npu.py \
      --device "${DEVICE}" \
      --cases-per-combo "${CASES_PER_COMBO:-3}" \
      --batch-sizes "${BATCH_SIZES:-64,128,256}" \
      --spec-tokens "${SPEC_TOKENS:-4,8}" \
      --kv-block-sizes "${KV_BLOCK_SIZES:-128}" \
      --triton-block-sizes "${TRITON_BLOCK_SIZES:-64,128,256}" \
      --max-prompt-len "${MAX_PROMPT_LEN:-4096}" \
      --max-position-base "${MAX_POSITION_BASE:-65536}"
    ;;
  bench)
    exec "${PYTHON_BIN}" verify_dflash_2d_grid_npu.py \
      --device "${DEVICE}" \
      --cases-per-combo 1 \
      --batch-sizes 1,8 \
      --spec-tokens 1,4 \
      --kv-block-sizes 16,128 \
      --triton-block-sizes 64,128 \
      --max-prompt-len 64 \
      --benchmark \
      --bench-warmup "${BENCH_WARMUP:-10}" \
      --bench-iters "${BENCH_ITERS:-50}" \
      --bench-batch-sizes "${BENCH_BATCH_SIZES:-32,64,128}" \
      --bench-prompt-lens "${BENCH_PROMPT_LENS:-256,1024,4096}" \
      --bench-spec-tokens "${BENCH_SPEC_TOKENS:-4,8}" \
      --bench-kv-block-sizes "${BENCH_KV_BLOCK_SIZES:-128}" \
      --bench-triton-block-sizes "${BENCH_TRITON_BLOCK_SIZES:-64,128,256}"
    ;;
  *)
    echo "Usage: $0 [smoke|full|long|bench]" >&2
    echo "Optional env: PYTHON_BIN, DEVICE, VLLM_ASCEND_PATH, VLLM_PATH" >&2
    echo "Optional bench env: BENCH_WARMUP, BENCH_ITERS, BENCH_*" >&2
    exit 2
    ;;
esac
