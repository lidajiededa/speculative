"""FP32 master/state AdamW optimizers for bf16 DFlash training."""

from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist


class FP32StateAdamW(torch.optim.Optimizer):
    """AdamW whose state and master parameter copies remain in fp32."""

    def __init__(
        self,
        params: Iterable,
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        super().__init__(params, defaults)
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["master_param"] = parameter.detach().clone().float()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        fp32_grads = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                for key in ("exp_avg", "exp_avg_sq", "master_param"):
                    if state[key].dtype != torch.float32:
                        state[key] = state[key].float()
                state["_fp32_grad"] = parameter.grad.detach().float()
                fp32_grads.append(state["_fp32_grad"])

        if self.max_grad_norm > 0 and fp32_grads:
            norm_squared = torch.stack(
                [gradient.pow(2).sum() for gradient in fp32_grads]
            ).sum()
            # SHARD_GRAD_OP keeps only a gradient shard on each rank. Reduce
            # the squared norm so every rank applies the same global clip.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(norm_squared, op=dist.ReduceOp.SUM)
            norm = norm_squared.sqrt()
            coefficient = min((self.max_grad_norm / (norm + 1e-6)).item(), 1.0)
            if coefficient < 1.0:
                for gradient in fp32_grads:
                    gradient.mul_(coefficient)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                gradient = state.pop("_fp32_grad")
                state["step"] += 1
                step = state["step"].item()
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if weight_decay:
                    master.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                correction1 = 1.0 - beta1**step
                correction2 = 1.0 - beta2**step
                denominator = (exp_avg_sq.sqrt() / correction2**0.5).add_(eps)
                master.addcdiv_(exp_avg, denominator, value=-(lr / correction1))
                parameter.copy_(master.to(parameter.dtype))
                parameter.grad = None
        return loss


class FP32MasterWeightOptimizer(torch.optim.Optimizer):
    """Wrap AdamW with fp32 parameter copies for non-FSDP training."""

    def __init__(
        self,
        model_params: list[torch.Tensor],
        inner_optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
    ) -> None:
        self.model_params = model_params
        self.master_params = [
            parameter.detach().clone().float().requires_grad_(True)
            for parameter in model_params
        ]
        if len(inner_optimizer.param_groups) != 1:
            raise ValueError("FP32MasterWeightOptimizer expects one parameter group")
        inner_optimizer.param_groups[0]["params"] = self.master_params
        inner_optimizer.state = defaultdict(dict)
        self.inner_optimizer = inner_optimizer
        self.max_grad_norm = max_grad_norm
        self._initializing = True
        super().__init__(self.master_params, inner_optimizer.defaults)
        self._initializing = False
        self.param_groups = inner_optimizer.param_groups
        self.state = inner_optimizer.state

    def step(self, closure=None):
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.master_params):
                master_param.grad = (
                    None if model_param.grad is None else model_param.grad.detach().float()
                )
                model_param.grad = None
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.master_params, self.max_grad_norm)
        loss = self.inner_optimizer.step(closure)
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.master_params):
                model_param.copy_(master_param.to(model_param.dtype))
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for parameter in self.model_params + self.master_params:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def state_dict(self):
        return self.inner_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.inner_optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        if getattr(self, "_initializing", True):
            return super().add_param_group(param_group)
        return self.inner_optimizer.add_param_group(param_group)
