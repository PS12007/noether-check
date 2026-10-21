"""Turbulence statistics after pointwise predictability is lost.

Chaotic 2D turbulence decorrelates from any reference within a few eddy turnovers, so pointwise
error stops being informative. The kinetic-energy spectrum E(k, t) is what a surrogate should still
get right. For the base models (all seeds, raw and projected) on the NS test and OOD sets we record
the spectrum error  mean_k |log10 E_pred(k, t) - log10 E_ref(k, t)|  over shells k = 1..21 (the
dealiased range), per trajectory and step.

Writes results/spectra_ns.json and runs/spectra_ns.npz; figure: experiments/figures.py spectra.
"""
import glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.solvers import DEV
from noether.project import ns_projector
from noether.train import rollout
from experiments.monitor_eval import load
from experiments.zoo import rel_err

KMAX = 21


def spectrum(w):
    """w: [..., N, N] vorticity -> [..., KMAX] shell-summed kinetic energy, k = 1..KMAX."""
    w = w.to(DEV, torch.float64)
    N = w.shape[-1]
    k = torch.fft.fftfreq(N, d=1.0 / N).to(DEV, torch.float64)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    wh = torch.fft.fft2(w) / N ** 2
    e = 0.5 * (wh.abs() ** 2) * torch.where(k2 > 0, 1 / k2, torch.zeros_like(k2))
    shell = torch.round(k2.sqrt()).long().flatten()
    flat = e.flatten(-2)
    out = torch.zeros(*flat.shape[:-1], KMAX + 1, dtype=torch.float64, device=DEV)
    out.index_add_(-1, shell.clamp(max=KMAX), flat * (shell <= KMAX))
    return out[..., 1:].cpu()


def spec_err(P, U, bs=8):
    errs = []
    for i in range(0, len(P), bs):
        Ep, Er = spectrum(torch.nan_to_num(P[i:i + bs], nan=0.0)), spectrum(U[i:i + bs])
        e = (torch.log10(Ep + 1e-30) - torch.log10(Er + 1e-30)).abs().mean(-1).numpy()
        e[torch.isnan(P[i:i + bs]).flatten(2).any(-1).numpy()] = np.nan
        errs.append(e)
    return np.concatenate(errs)


def main():
    sets = {k: torch.load(f"data/ns_{k}.pt") for k in ("test", "ood")}
    rows = [json.load(open(f)) for f in sorted(glob.glob("results/ns/*_base_s?.json"))]
    out, arrays = {}, {}
    for r in rows:
        m = load("ns", r)
        for s, U in sets.items():
            for variant in ("raw", "proj", "proj2"):
                proj = {"raw": None, "proj": ns_projector(), "proj2": ns_projector(enstrophy=True)}[variant]
                P = rollout(m, U[:, 0].to(DEV), U.shape[1] - 1, projector=proj)
                e = spec_err(P, U)
                err = rel_err(P, U)
                arrays[f"{r['name']}_{s}_{variant}"] = e.astype(np.float32)
                arrays[f"err_{r['name']}_{s}_{variant}"] = err.astype(np.float32)
                for h in (10, 20, 50, 100):
                    med = lambda a: float(np.nanmedian(np.where(np.isnan(a[:, h]), np.inf, a[:, h])))
                    out[f"{r['name']}_{s}_{variant}@{h}"] = med(e)
                    out[f"err_{r['name']}_{s}_{variant}@{h}"] = med(err)
                    out[f"blowup_{r['name']}_{s}_{variant}@{h}"] = float(np.isnan(err[:, h]).mean())
        print(r["name"], {k: round(v, 3) for k, v in out.items() if r["name"] in k and k.endswith("@100") and "_test_" in k}, flush=True)
    # spectrum error of the reference against itself shifted in time: how much the true spectrum changes
    for s, U in sets.items():
        e = spec_err(U[:, :1].expand_as(U).contiguous(), U)
        arrays[f"frozen_{s}"] = e.astype(np.float32)
        out[f"frozen_{s}@100"] = float(np.median(e[:, 100]))
    json.dump(out, open("results/spectra_ns.json", "w"), indent=1)
    np.savez_compressed("runs/spectra_ns.npz", **arrays)


if __name__ == "__main__":
    main()
