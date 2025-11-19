import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect

class NormalizedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=False, eps=1e-12):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]; mom = group["momentum"]; wd = group["weight_decay"]; nes = group["nesterov"]; eps = group["eps"]
            total_sq = torch.zeros((), device=group["params"][0].device)
            for p in group["params"]:
                if p.grad is not None:
                    total_sq += p.grad.detach().pow(2).sum()
            gn = total_sq.sqrt()
            scale = 1.0 / (gn + eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                g = g.mul(scale)
                state = self.state[p]
                buf = state.get("momentum_buffer", None)
                if mom != 0:
                    if buf is None:
                        buf = torch.clone(g).detach()
                    else:
                        buf.mul_(mom).add_(g)
                    state["momentum_buffer"] = buf
                    upd = g.add(buf, alpha=mom) if nes else buf
                else:
                    upd = g
                p.add_(upd, alpha=-lr)
        return loss

class BlockNormalizedSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=False,
        norm=2,
        eps=1e-12,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            norm=norm,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            nes = group["nesterov"]
            q = group["norm"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer", None)
                if mom != 0:
                    if buf is None:
                        buf = torch.clone(g).detach()
                    else:
                        buf.mul_(mom).add_(g)
                    state["momentum_buffer"] = buf
                    pre = g.add(buf, alpha=mom) if nes else buf
                else:
                    pre = g
                gn = pre.norm(p=q)
                pre = pre / (gn + eps)
                if wd != 0:
                    pre = pre.add(p, alpha=wd)
                p.add_(pre, alpha=-lr)

        return loss
