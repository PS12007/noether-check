"""One-step training and autoregressive rollouts."""
import math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .models import Surrogate
from .solvers import DEV, kdv_invariants, ns_invariants

SYSTEMS = {
    "kdv": dict(nd=1, inv=kdv_invariants, train="data/kdv_train.pt"),
    "ns": dict(nd=2, inv=ns_invariants, train="data/ns_train.pt"),
}
SIZES = {   # per system and architecture: small / base hyperparameters
    ("kdv", "fno"): dict(small=dict(width=32, modes=12, layers=4), base=dict(width=64, modes=24, layers=4)),
    ("kdv", "resnet"): dict(small=dict(width=32, blocks=4), base=dict(width=64, blocks=8)),
    ("kdv", "unet"): dict(small=dict(width=16, depth=3), base=dict(width=32, depth=3)),
    ("ns", "fno"): dict(small=dict(width=24, modes=8, layers=4), base=dict(width=32, modes=12, layers=4)),
    ("ns", "resnet"): dict(small=dict(width=32, blocks=4), base=dict(width=48, blocks=6)),
    ("ns", "unet"): dict(small=dict(width=16, depth=3), base=dict(width=32, depth=3)),
}


def pairs(U):
    """[B, T+1, ...] trajectories -> (inputs, targets) one-step pairs."""
    return U[:, :-1].reshape(-1, *U.shape[2:]), U[:, 1:].reshape(-1, *U.shape[2:])


def train(system, arch, size="base", seed=0, epochs=20, bs=128, lr=1e-3, noise=0.0, penalty=0.0,
          max_pairs=None, log=None):
    """noise: std of Gaussian input noise (relative to the state's std), a common stabiliser.
    penalty: weight of a soft invariant loss, mean relative mismatch of mass & momentum (KdV)
    or mean & energy (NS) between prediction and target."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    S = SYSTEMS[system]
    U = torch.load(S["train"])
    X, Y = pairs(U)
    if max_pairs:
        sel = torch.randperm(len(X))[:max_pairs]
        X, Y = X[sel], Y[sel]
    s_in, s_out = X.std().item(), (Y - X).std().item()
    model = Surrogate(arch, S["nd"], s_in, s_out, **SIZES[(system, arch)][size]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    keys = ("mass", "momentum") if system == "kdv" else ("mean", "energy")
    X, Y = X.to(DEV), Y.to(DEV)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x, y = X[idx], Y[idx]
            if noise > 0:
                x = x + noise * s_in * torch.randn_like(x)
            pred = model(x)
            loss = F.mse_loss((pred - x) / s_out, (y - x) / s_out)
            if penalty > 0:
                qp, qy = S["inv"](pred), S["inv"](y)
                loss = loss + penalty * sum(((qp[k] - qy[k]) / (qy[k].abs() + 1e-2)).pow(2).mean() for k in keys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if log:
            log(f"  ep {ep + 1}/{epochs} loss {tot / len(X):.2e} ({time.time() - t0:.0f}s)")
    model.eval()
    return model


@torch.no_grad()
def _step(model, u, bs=256):
    return torch.cat([model(u[i:i + bs]) for i in range(0, len(u), bs)])


def rollout(model, u0, steps, projector=None, blowup=1e3):
    """u0: [B, ...] on DEV. Returns [B, steps + 1, ...] on CPU. projector = (init, apply) or None.
    States that blow up are frozen at NaN so that they count as failed."""
    out = [u0.float().cpu()]
    u = u0.float()
    state = projector[0](u) if projector else None
    dead = torch.zeros(len(u), dtype=torch.bool, device=u.device)
    dims = tuple(range(1, u.dim()))
    for _ in range(steps):
        nxt = _step(model, u)
        if projector:
            with torch.enable_grad():
                nxt = projector[1](nxt, u, state)
        dead |= ~torch.isfinite(nxt).all(dim=dims) | (torch.nan_to_num(nxt, nan=1e9).abs().amax(dim=dims) > blowup)
        nxt[dead] = float("nan")
        u = torch.nan_to_num(nxt, nan=0.0)
        out.append(nxt.cpu())
    return torch.stack(out, 1)
