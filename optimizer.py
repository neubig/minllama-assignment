from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            correct_bias = group["correct_bias"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                #raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment (m)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Second moment (v)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # -------------------
                # 1. Add weight decay (AdamW)
                # AdamWでは、L2正則化（重み減衰）を更新ステップの前、
                # かつ勾配ベースの更新とは独立して行います。
                if weight_decay != 0.0:
                    grad = grad + weight_decay * p.data

                # -------------------
                # 2. Update first and second moments of the gradients (EMA)
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                # torch.square は PyTorch 1.8.0 以降で利用可能
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # -------------------
                # 3. Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                
                if correct_bias:
                    # 1 - beta^t の計算
                    bias_correction1 = 1.0 - beta1 ** t
                    bias_correction2 = 1.0 - beta2 ** t
                    
                    # 補正されたモーメント
                    exp_avg_corrected = exp_avg / bias_correction1
                    exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                else:
                    exp_avg_corrected = exp_avg
                    exp_avg_sq_corrected = exp_avg_sq

                # -------------------
                # 4. Update parameters
                # θ_{t+1} = θ_t - lr * (m̂_t / (sqrt(v̂_t) + eps))
                
                # 分母の計算: sqrt(v̂_t) + eps
                # exp_avg_sq_corrected.sqrt().add_(eps)
                
                # Adamの更新ステップ
                # (m̂_t / (sqrt(v̂_t) + eps)) * lr
                step_size = lr
                
                denom = exp_avg_sq_corrected.sqrt().add_(eps)
                
                # 更新を適用
                # p.data -= step_size * exp_avg_corrected / denom
                p.data.addcdiv_(exp_avg_corrected, denom, value=-step_size)


                # Update first and second moments of the gradients

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                # Update parameters

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss