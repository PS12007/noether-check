"""Noether projection: the smallest change to a predicted state that restores chosen invariants.

Given a state u and target values c_j for differentiable quantities I_j, solve
    min ||delta||^2  subject to  I_j(u + delta) = c_j
with a few Gauss-Newton steps, delta = J^T (J J^T)^-1 (c - I(u)), J_jk = dI_j/du_k.
Linear invariants (mass) are restored exactly in one step; quadratic and cubic ones in 2-4.
Works batched: the quantities of different samples are independent, so the gradient of
their sum gives every sample's Jacobian row at once.
"""
import torch

from .solvers import kdv_invariants, ns_invariants


def project(u, fn, keys, target, iters=4, eps=1e-10):
    """u: [B, ...]; fn(u) -> dict of [B] tensors; keys: quantities to restore; target: dict of [B]."""
    u = u.detach().double()
    B = u.shape[0]
    for _ in range(iters):
        x = u.clone().requires_grad_(True)
        q = fn(x)
        r = torch.stack([target[k].double() - q[k] for k in keys], 1)            # [B, m]
        J = torch.stack([torch.autograd.grad(q[k].sum(), x, retain_graph=True)[0].reshape(B, -1)
                         for k in keys], 1)                                         # [B, m, n]
        G = J @ J.transpose(1, 2) + eps * torch.eye(len(keys), device=u.device, dtype=u.dtype)
        lam = torch.linalg.solve(G, r.unsqueeze(-1))                                # [B, m, 1]
        u = (u + (J.transpose(1, 2) @ lam).squeeze(-1).reshape(u.shape)).detach()
    return u.float()


# ---------------------------------------------------------------- system-specific helpers
def kdv_projector(keys=("mass", "momentum")):
    """Restore KdV invariants to their initial values (they are exactly conserved)."""
    def init(u0):
        return {k: v for k, v in kdv_invariants(u0).items()}

    def apply(u_pred, u_prev, state):
        return project(u_pred, kdv_invariants, keys, state)
    return init, apply


def ns_projector(nu=1e-3, dt=0.5, budget=True, enstrophy=False):
    """Mean vorticity is conserved. Kinetic energy follows dE/dt = -2 nu Z; we impose the
    trapezoidal budget E_next = E_prev - nu dt (Z_prev + Z_pred), with Z_pred taken from the
    raw prediction. budget=False restores only the mean. enstrophy=True also imposes the
    enstrophy budget Z_next = Z_prev - nu dt (P_prev + P_pred), leaving nothing held out."""
    fn = lambda w: ns_invariants(w, nu)

    def init(w0):
        return {"mean": ns_invariants(w0, nu)["mean"]}

    def apply(w_pred, w_prev, state):
        tgt = {"mean": state["mean"]}
        keys = ["mean"]
        if budget:
            qp, qn = ns_invariants(w_prev, nu), ns_invariants(w_pred, nu)
            tgt["energy"] = qp["energy"] - nu * dt * (qp["enstrophy"] + qn["enstrophy"])
            keys.append("energy")
            if enstrophy:
                tgt["enstrophy"] = qp["enstrophy"] - nu * dt * (qp["palinstrophy"] + qn["palinstrophy"])
                keys.append("enstrophy")
        return project(w_pred, fn, keys, tgt)
    return init, apply
