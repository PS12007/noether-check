"""Label-free drift of a rollout: how far its own conserved quantities (KdV) or budgets (NS)
have strayed from what physics requires. Needs only the rollout, never the reference solution.

KdV: |I(t) - I(0)| / scale, scale = int |u0| (mass), int u0^2 (momentum),
     int (u0_x^2 / 2 + |u0|^3) (energy), so that quantities near zero are not divided by zero.
NS:  mean vorticity: |mean(t) - mean(0)| / mean |w0|;
     energy budget:    |E(t) - E(0) + nu dt sum_i (Z_i + Z_{i+1})| / E(0)      (dE/dt = -2 nu Z)
     enstrophy budget: |Z(t) - Z(0) + nu dt sum_i (P_i + P_{i+1})| / Z(0)      (dZ/dt = -2 nu P)
"""
import math
import numpy as np
import torch

from .solvers import DEV, KdV, kdv_invariants, ns_invariants


def _chunks(fn, traj, bs=16):
    """traj [B, T+1, ...] (CPU) -> dict of [B, T+1] numpy arrays."""
    B, T = traj.shape[:2]
    flat = traj.reshape(B * T, *traj.shape[2:])
    parts = [fn(flat[i:i + bs * T].to(DEV)) for i in range(0, len(flat), bs * T)]
    return {k: torch.cat([p[k].cpu() for p in parts]).reshape(B, T).numpy() for k in parts[0]}


def kdv_scales(u):
    u = u.double()
    dx = KdV.L / KdV.N
    k = 2 * math.pi * torch.fft.rfftfreq(KdV.N, d=dx).to(u.device, torch.float64)
    ux = torch.fft.irfft(1j * k * torch.fft.rfft(u), n=KdV.N)
    return dict(mass=u.abs().sum(-1) * dx, momentum=(u * u).sum(-1) * dx,
                energy=(0.5 * ux * ux + u.abs() ** 3).sum(-1) * dx)


def kdv_drift(traj):
    q = _chunks(kdv_invariants, traj)
    s = {k: v.cpu().numpy() for k, v in kdv_scales(traj[:, 0].to(DEV)).items()}
    return {k: np.abs(q[k] - q[k][:, :1]) / s[k][:, None] for k in q}


def ns_drift(traj, nu=1e-3, dt=0.5):
    q = _chunks(lambda w: ns_invariants(w, nu), traj)
    out = {"mean": np.abs(q["mean"] - q["mean"][:, :1]) / np.abs(traj[:, 0].numpy()).mean((-2, -1))[:, None]}
    for key, rate in (("energy", "enstrophy"), ("enstrophy", "palinstrophy")):
        loss = np.concatenate([np.zeros_like(q[rate][:, :1]),
                               np.cumsum(nu * dt * (q[rate][:, 1:] + q[rate][:, :-1]), 1)], 1)
        out[key] = np.abs(q[key] - q[key][:, :1] + loss) / q[key][:, :1]
    return out


DRIFT = {"kdv": kdv_drift, "ns": ns_drift}
