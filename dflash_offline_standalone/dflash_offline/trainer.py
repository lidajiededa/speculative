"""Hugging Face Trainer integration for standalone offline DFlash."""

import math
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from .masks import create_dflash_block_mask, create_dflash_dense_mask
from .objective import (
    create_position_ids,
    create_targets_and_weights,
    sample_anchor_positions,
)
from .optim import FP32MasterWeightOptimizer, FP32StateAdamW
from .target_weights import TargetEmbeddingsAndHead


class DFlashOfflineTrainer(Trainer):
    def __init__(
        self,
        *args,
        draft_config,
        target_model_path: str,
        target_dtype: torch.dtype,
        embed_weight_key: str,
        lm_head_key: str,
        trust_remote_code: bool,
        optimizer_precision: str,
        dflash_max_grad_norm: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = int(draft_config.block_size)
        self.num_anchors = int(draft_config.num_anchors)
        self.loss_decay_gamma = getattr(draft_config, "loss_decay_gamma", None)
        self.gamma_warmup = bool(getattr(draft_config, "gamma_warmup", False))
        self.gamma_warmup_step = float(
            getattr(draft_config, "gamma_warmup_step", 0.5)
        )
        self.initial_gamma = self.loss_decay_gamma
        self.attention_backend = getattr(draft_config, "attention_backend", "sdpa")
        dflash_config = getattr(draft_config, "dflash_config", {}) or {}
        self.mask_token_id = dflash_config.get("mask_token_id")
        self.optimizer_precision = optimizer_precision
        self.dflash_max_grad_norm = dflash_max_grad_norm
        self._metric_loss = 0.0
        self._metric_accuracy = 0.0
        self._metric_count = 0

        self.model.config.attention_backend = self.attention_backend
        self.target_components = TargetEmbeddingsAndHead.from_pretrained(
            target_model_path,
            device=self.args.device,
            dtype=target_dtype,
            embed_key=embed_weight_key,
            lm_head_key=lm_head_key,
            trust_remote_code=trust_remote_code,
        )

    @property
    def target_embed_tokens(self) -> nn.Module:
        return self.target_components.embed_tokens

    @property
    def target_lm_head(self) -> nn.Module:
        return self.target_components.lm_head

    def create_optimizer(self, model=None):
        if self.optimizer is not None or self.optimizer_precision == "standard":
            return super().create_optimizer(model)
        if self.is_deepspeed_enabled:
            return super().create_optimizer(model)

        optimizer_model = self.model if model is None else model
        parameters = [
            parameter for parameter in optimizer_model.parameters() if parameter.requires_grad
        ]
        if self.is_fsdp_enabled:
            self.optimizer = FP32StateAdamW(
                [{"params": parameters}],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                max_grad_norm=self.dflash_max_grad_norm,
            )
        else:
            inner = torch.optim.AdamW(
                parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            self.optimizer = FP32MasterWeightOptimizer(
                parameters,
                inner,
                max_grad_norm=self.dflash_max_grad_norm,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        optimizer = optimizer or self.optimizer
        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return self.lr_scheduler

    def _save_optimizer_and_scheduler(self, output_dir: str):
        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if isinstance(optimizer, FP32MasterWeightOptimizer):
            if self.args.should_save:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                if self.lr_scheduler is not None:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
            return
        # FP32StateAdamW is attached to FSDP's real parameter objects, so let
        # Trainer/Accelerate gather its sharded state for a resumable checkpoint.
        super()._save_optimizer_and_scheduler(output_dir)

    def _current_gamma(self) -> Optional[float]:
        if not self.gamma_warmup or self.initial_gamma is None:
            return self.loss_decay_gamma
        epoch = int(self.state.epoch or 0)
        return self.initial_gamma + self.gamma_warmup_step * epoch

    def _create_noise_embedding(
        self,
        input_ids: torch.Tensor,
        anchors: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        num_blocks = anchors.shape[1]
        noise_ids = torch.full(
            (batch_size, num_blocks * self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        block_starts = (
            torch.arange(num_blocks, device=input_ids.device) * self.block_size
        ).expand(batch_size, -1)
        anchor_tokens = torch.gather(
            input_ids, 1, anchors.clamp(0, sequence_length - 1)
        )
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        noise_ids[batch_indices, block_starts] = torch.where(
            keep_mask,
            anchor_tokens,
            torch.full_like(anchor_tokens, self.mask_token_id),
        )
        return self.target_embed_tokens(noise_ids)

    def _compute_objective(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        loss_mask = inputs["loss_mask"]
        context_attention_mask = inputs["attention_mask"]
        _, sequence_length = input_ids.shape

        anchors, keep_mask = sample_anchor_positions(
            loss_mask,
            sequence_length,
            self.block_size,
            self.num_anchors,
        )
        if anchors is None:
            zero = sum(
                parameter.sum() * 0.0
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            return zero, torch.zeros((), device=input_ids.device)

        noise_embedding = self._create_noise_embedding(input_ids, anchors, keep_mask)
        position_ids = create_position_ids(
            anchors, sequence_length, self.block_size
        )
        if self.attention_backend == "flex_attention":
            attention_mask = create_dflash_block_mask(
                anchors, keep_mask, sequence_length, self.block_size
            )
        else:
            attention_mask = create_dflash_dense_mask(
                anchors,
                keep_mask,
                sequence_length,
                self.block_size,
                context_attention_mask=context_attention_mask,
            )

        model_dtype = next(model.parameters()).dtype
        output_hidden = model(
            noise_embedding=noise_embedding.to(model_dtype),
            target_hidden=hidden_states.to(model_dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.target_lm_head(
            output_hidden.to(self.target_lm_head.weight.dtype)
        )
        targets, weights, eval_weights = create_targets_and_weights(
            input_ids,
            loss_mask,
            anchors,
            keep_mask,
            self.block_size,
            self._current_gamma(),
        )
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets, reduction="none"
        )
        loss = (token_loss * weights).sum() / (weights.sum() + 1e-6)
        with torch.no_grad():
            predictions = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
            correct = (predictions == targets) & (eval_weights > 0.5)
            accuracy = correct.float().sum() / (eval_weights.sum() + 1e-6)
        return loss, accuracy

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        del num_items_in_batch
        loss, accuracy = self._compute_objective(model, inputs)
        self._metric_loss += float(loss.detach())
        self._metric_accuracy += float(accuracy.detach())
        self._metric_count += 1
        if return_outputs:
            return loss, {"loss": loss.detach(), "accuracy": accuracy.detach()}
        return loss

    def log(self, logs: dict, *args, **kwargs):
        if "loss" in logs and self._metric_count:
            logs = dict(logs)
            logs["dflash_loss"] = self._metric_loss / self._metric_count
            logs["dflash_accuracy"] = self._metric_accuracy / self._metric_count
            logs["loss_decay_gamma"] = self._current_gamma()
            self._metric_loss = 0.0
            self._metric_accuracy = 0.0
            self._metric_count = 0
        return super().log(logs, *args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _accuracy = self._compute_objective(model, inputs)
        return loss.detach(), None, None
