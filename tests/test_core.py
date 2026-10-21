"""Fast checks of the core pieces (CPU, well under a minute, no data files needed).

    python -m pytest tests -q
"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.solvers import KdV, kdv_invariants, NS2D, ns_invariants
from noether.project import project
from noether.models import Surrogate
from noether.train import rollout


def soliton(c, x0):
    x = KdV.x()
    return sum(c / 2 / torch.cosh(math.sqrt(c) / 2 * (x - x0 + s)) ** 2 for s in (-KdV.L, 0, KdV.L))


def test_kdv_soliton_travels_at_its_speed_and_keeps_its_invariants():
    s = KdV(dt=0.005, device="cpu")
    u0 = soliton(1.0, 10.0)[None]
    U = s.simulate(u0, 5.0, 2)                      # speed c = 1: moves 10 units in t = 10
    x = KdV.x()
    assert abs(float(x[U[0, -1].argmax()]) - 20.0) < 0.4
    q0, q1 = kdv_invariants(U[:, 0]), kdv_invariants(U[:, -1])
    for k in q0:
        assert abs(float((q1[k] - q0[k]) / q0[k])) < 1e-4, k


def test_ns_energy_budget_holds():
    ns = NS2D(nu=1e-3, dt=0.01, device="cpu")
    w0 = NS2D.random_field(2, np.random.default_rng(0))
    W = ns.simulate(w0, 0.5, 4)
    q = ns_invariants(W)
    dE = (q["energy"][:, 1:] - q["energy"][:, :-1]) / 0.5
    pred = -1e-3 * (q["enstrophy"][:, 1:] + q["enstrophy"][:, :-1])
    assert torch.allclose(dE, pred, rtol=2e-3)
    assert float(q["mean"].abs().max()) < 1e-8


def test_projection_restores_chosen_invariants_and_moves_little():
    u = KdV.solitons(4, np.random.default_rng(1)).float()
    t = kdv_invariants(u)
    x = KdV.x().float()
    v = u * 1.03 + 0.02 * torch.sin(2 * math.pi * 3 * x / KdV.L)       # smooth perturbation
    p = project(v, kdv_invariants, ("mass", "momentum"), t)
    q = kdv_invariants(p)
    for k in ("mass", "momentum"):
        assert float(((q[k] - t[k]) / t[k]).abs().max()) < 1e-5
    assert float((p - v).norm()) < float((v - u).norm())


def test_models_are_shift_equivariant_and_rollout_marks_blowups():
    torch.manual_seed(0)
    m = Surrogate("resnet", 1, 1.0, 1.0, width=8, blocks=2).eval()
    x = torch.randn(3, 128)
    with torch.no_grad():
        assert torch.allclose(m(torch.roll(x, 5, -1)), torch.roll(m(x), 5, -1), atol=1e-5)
    boom = Surrogate("resnet", 1, 1.0, 1e3, width=8, blocks=2).eval()     # huge output scale: blows up
    P = rollout(boom, x, 5, blowup=10.0)
    assert P.shape == (3, 6, 128)
    assert torch.isnan(P[:, -1]).all()


def test_spectrum_parseval():
    """Shell-summed kinetic energy spectrum adds up to the kinetic energy (smooth field, no tail)."""
    from experiments.spectra import spectrum
    w = NS2D.random_field(2, np.random.default_rng(0), k0=3.0)
    E = spectrum(w).sum(-1)
    ref = ns_invariants(w)["energy"].cpu()
    assert torch.allclose(E, ref, rtol=1e-3)
