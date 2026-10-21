"""Generate all datasets with the reference solvers. Writes data/*.pt (about 0.5 GB, not committed).

KdV  (snapshots every 0.5 time units)
  kdv_train      1024 trajectories x 100 steps, sums of 1-4 solitons
  kdv_test        128 trajectories x 400 steps, same distribution, longer
  kdv_ood         128 trajectories x 400 steps, smooth random fields (shed solitons + radiation)
NS2D (snapshots every 0.5 time units, nu = 1e-3)
  ns_train        384 trajectories x 20 steps, starting after a 2-unit spin-up
  ns_test          64 trajectories x 100 steps
  ns_ood           64 trajectories x 100 steps, energy peaked at a higher wavenumber (k0 = 8)
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.solvers import KdV, NS2D

DT = 0.5


def batched(sim, u0, n, bs):
    return torch.cat([sim(u0[i:i + bs], DT, n) for i in range(0, len(u0), bs)])


def main(device=None):
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(0)
    kdv = KdV(dt=0.005, **({"device": device} if device else {}))
    jobs = [("kdv_train", lambda: KdV.solitons(1024, rng), 100),
            ("kdv_test", lambda: KdV.solitons(128, rng), 400),
            ("kdv_ood", lambda: KdV.random_field(128, rng), 400)]
    for name, init, n in jobs:
        if os.path.exists(f"data/{name}.pt"):
            continue
        t = time.time()
        U = batched(kdv.simulate, init(), n, 512)
        torch.save(U, f"data/{name}.pt")
        print(name, tuple(U.shape), f"{time.time() - t:.0f}s", flush=True)

    ns = NS2D(nu=1e-3, dt=0.01)
    jobs = [("ns_train", lambda: NS2D.random_field(384, rng), 20),
            ("ns_test", lambda: NS2D.random_field(64, rng), 100),
            ("ns_ood", lambda: NS2D.random_field(64, rng, k0=8.0), 100)]
    for name, init, n in jobs:
        if os.path.exists(f"data/{name}.pt"):
            continue
        t = time.time()
        w0 = batched(ns.simulate, init(), 4, 128)[:, -1]        # spin-up: drop the random-phase start
        W = batched(ns.simulate, w0, n, 128)
        torch.save(W, f"data/{name}.pt")
        print(name, tuple(W.shape), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)   # optional device, e.g. cpu
