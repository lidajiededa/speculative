#!/usr/bin/env python3
"""Entry point for standalone DFlash offline training."""

import argparse
import inspect
import os
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from dflash_offline.configuration import load_draft_config
from dflash_offline.data import DFlashDataCollator, OfflineDFlashDataset
from dflash_offline.modeling_qwen3_dflash import QwenDFlashDraftModel
from dflash_offline.trainer import DFlashOfflineTrainer


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone offline DFlash training")
    parser.add_argument("--target_model_path", required=True) 
    parser.add_argument("--draft_config", required=True)
    parser.add_argument("--train_hidden_path", required=True)
    parser.add_argument("--eval_hidden_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embed_weight_key", default="model.embed_tokens.weight")
    parser.add_argument("--lm_head_key", default="lm_head.weight")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", choices=DTYPES, default="bfloat16")

    parser.add_argument("--block_size", type=int)
    parser.add_argument("--num_anchors", type=int)
    parser.add_argument("--loss_decay_gamma", type=float)
    parser.add_argument("--mask_token_id", type=int)
    parser.add_argument(
        "--attention_backend",
        choices=("sdpa", "eager", "flex_attention"),
        default=None,
    )
    parser.add_argument("--gamma_warmup", action="store_true")
    parser.add_argument("--gamma_warmup_step", type=float, default=0.5)

    parser.add_argument("--model_max_length", type=int, default=3072)
    parser.add_argument("--cache_in_memory", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--no_dataloader_pin_memory", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=12.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.04)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer_precision",
        choices=("fp32_master", "standard"),
        default="fp32_master",
    )
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_drop_last", action="store_true")
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp_config")
    parser.add_argument("--deepspeed")
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="auto, none, or a checkpoint directory",
    )
    return parser.parse_args()


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    reports = [] if args.report_to == "none" else args.report_to.split(",")
    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=(0.0 if args.optimizer_precision == "fp32_master" else args.max_grad_norm),
        lr_scheduler_type="cosine",
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        report_to=reports,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_drop_last=args.dataloader_drop_last,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=not args.no_dataloader_pin_memory,
        gradient_checkpointing=args.gradient_checkpointing,
        fsdp=args.fsdp,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )
    evaluation_value = "steps" if args.eval_hidden_path else "no"
    parameters = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = evaluation_value
    else:
        kwargs["evaluation_strategy"] = evaluation_value
    if args.fsdp_config:
        kwargs["fsdp_config"] = args.fsdp_config
    return TrainingArguments(**kwargs)


def resolve_resume(output_dir: str, value: str):
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        path = Path(output_dir)
        return get_last_checkpoint(str(path)) if path.is_dir() else None
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def main() -> None:
    args = parse_args()
    config = load_draft_config(args.draft_config)
    for name in ("block_size", "num_anchors", "loss_decay_gamma"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    config.gamma_warmup = args.gamma_warmup
    config.gamma_warmup_step = args.gamma_warmup_step
    config.attention_backend = args.attention_backend or getattr(
        config, "attention_backend", "sdpa"
    )
    if args.mask_token_id is not None:
        config.dflash_config["mask_token_id"] = args.mask_token_id

    dtype = DTYPES[args.torch_dtype]
    model = QwenDFlashDraftModel(config).to(dtype=dtype)
    expected_hidden_size = len(config.dflash_config["target_layer_ids"]) * config.hidden_size
    train_dataset = OfflineDFlashDataset(
        args.train_hidden_path,
        max_length=args.model_max_length,
        cache_in_memory=args.cache_in_memory,
        expected_hidden_size=expected_hidden_size,
    )
    eval_dataset = None
    if args.eval_hidden_path:
        eval_dataset = OfflineDFlashDataset(
            args.eval_hidden_path,
            max_length=args.model_max_length,
            cache_in_memory=args.cache_in_memory,
            expected_hidden_size=expected_hidden_size,
        )

    training_args = build_training_arguments(args)
    trainer = DFlashOfflineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DFlashDataCollator(),
        draft_config=config,
        target_model_path=args.target_model_path,
        target_dtype=dtype,
        embed_weight_key=args.embed_weight_key,
        lm_head_key=args.lm_head_key,
        trust_remote_code=not args.no_trust_remote_code,
        optimizer_precision=args.optimizer_precision,
        dflash_max_grad_norm=args.max_grad_norm,
    )

    if trainer.is_world_process_zero():
        print(f"device={training_args.device}, world_size={training_args.world_size}")
        print(f"draft_parameters={sum(p.numel() for p in model.parameters()):,}")
        print(f"train_samples={len(train_dataset):,}")
        print(
            f"backend={config.attention_backend}, block_size={config.block_size}, "
            f"num_anchors={config.num_anchors}, max_length={args.model_max_length}"
        )

    resume = resolve_resume(args.output_dir, args.resume_from_checkpoint)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
