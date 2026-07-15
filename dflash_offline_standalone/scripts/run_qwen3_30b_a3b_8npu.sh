#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

NUM_NPUS=${1:-8}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-}
TRAIN_HIDDEN_PATH=${TRAIN_HIDDEN_PATH:-}
EVAL_HIDDEN_PATH=${EVAL_HIDDEN_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/qwen3-30b-a3b-dflash"}
DRAFT_CONFIG=${DRAFT_CONFIG:-"$ROOT_DIR/configs/qwen3_30b_a3b_dflash_npu.json"}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-3072}
NUM_ANCHORS=${NUM_ANCHORS:-128}
BLOCK_SIZE=${BLOCK_SIZE:-16}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-2}
REPORT_TO=${REPORT_TO:-none}

if [[ -z "$TARGET_MODEL_PATH" || ! -d "$TARGET_MODEL_PATH" ]]; then
  echo "TARGET_MODEL_PATH must point to the local Qwen3-30B-A3B model directory."
  exit 1
fi
if [[ -z "$TRAIN_HIDDEN_PATH" || ! -d "$TRAIN_HIDDEN_PATH" ]]; then
  echo "TRAIN_HIDDEN_PATH must point to the directory containing .ckpt files."
  exit 1
fi

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
if [[ "$NNODES" -eq 1 ]]; then
  DISTRIBUTED_ARGS=(--standalone --nproc_per_node "$NUM_NPUS")
else
  DISTRIBUTED_ARGS=(
    --nproc_per_node "$NUM_NPUS"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
  )
fi

EVAL_ARGS=()
if [[ -n "$EVAL_HIDDEN_PATH" ]]; then
  EVAL_ARGS=(--eval_hidden_path "$EVAL_HIDDEN_PATH")
fi

torchrun "${DISTRIBUTED_ARGS[@]}" "$ROOT_DIR/train.py" \
  --target_model_path "$TARGET_MODEL_PATH" \
  --draft_config "$DRAFT_CONFIG" \
  --train_hidden_path "$TRAIN_HIDDEN_PATH" \
  "${EVAL_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --torch_dtype bfloat16 \
  --attention_backend sdpa \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --block_size "$BLOCK_SIZE" \
  --num_anchors "$NUM_ANCHORS" \
  --loss_decay_gamma 7 \
  --num_train_epochs 12 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
  --learning_rate 6e-4 \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --optimizer_precision fp32_master \
  --logging_steps 50 \
  --save_steps 5000 \
  --dataloader_drop_last \
  --gradient_checkpointing \
  --fsdp "shard_grad_op auto_wrap" \
  --fsdp_config "$ROOT_DIR/configs/fsdp_config.json" \
  --report_to "$REPORT_TO" \
  --resume_from_checkpoint auto
