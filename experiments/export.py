"""Rollout gallery figures and the compact data file behind the web demo.

python experiments/export.py --system kdv|ns

Rolls out the three base seed-0 models (raw and projected) on fixed test / OOD trajectories,
draws docs/figures/gallery_<system>.png and writes docs/data/<system>.json (fields quantised to
uint8, base64, with per-array min/max).
"""
import argparse, base64, json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.solvers import DEV
from noether.project import kdv_projector, ns_projector
from noether.monitor import DRIFT
from noether.train import rollout
from experiments.monitor_eval import load
from experiments.zoo import rel_err
from experiments.figures import save, ANAME, COL, INK, INK2, MUTED

ARCHS = ("fno", "resnet", "unet")
PICK = {"kdv": dict(test=[0, 1, 2], ood=[0]), "ns": dict(test=[0], ood=[0])}
STRIDE = {"kdv": 2, "ns": 4}


def q8(a, rng=None):
    """rng: (lo, hi) to clip to, so that a blown-up rollout does not wash out the 8-bit range."""
    a = np.nan_to_num(np.asarray(a, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if rng is not None:
        a = np.clip(a, *rng)
    lo, hi = float(a.min()), float(a.max())
    b = np.round((a - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)
    return dict(shape=list(a.shape), lo=lo, hi=hi, b64=base64.b64encode(b.tobytes()).decode())


def r4(x):
    return [None if not np.isfinite(v) else float(f"{v:.4g}") for v in np.asarray(x, float)]


def main(system):
    sets = {k: torch.load(f"data/{system}_{k}.pt") for k in ("test", "ood")}
    out = dict(system=system, stride=STRIDE[system], dt=0.5, runs=[])
    keep = {}
    for set_name, idx in PICK[system].items():
        U = sets[set_name][idx]
        for arch in ARCHS:
            row = json.load(open(f"results/{system}/{arch}_base_s0.json"))
            m = load(system, row)
            for variant in ("raw", "proj"):
                proj = None if variant == "raw" else (kdv_projector(("mass", "momentum")) if system == "kdv" else ns_projector())
                P = rollout(m, U[:, 0].to(DEV), U.shape[1] - 1, projector=proj).cpu()
                err = rel_err(P, U)
                drift = DRIFT[system](P)
                keep[(set_name, arch, variant)] = (P.numpy(), err, drift)
        drift_ref = DRIFT[system](U)
        for j, i in enumerate(idx):
            T = U[j].numpy()
            pad = 0.5 * (T.max() - T.min())
            rng = (float(T.min() - pad), float(T.max() + pad))
            run = dict(set=set_name, index=int(i), truth=q8(T[::STRIDE[system]]),
                       truth_drift={k: r4(v[j]) for k, v in drift_ref.items()}, models={})
            for arch in ARCHS:
                for variant in ("raw", "proj"):
                    P, err, drift = keep[(set_name, arch, variant)]
                    run["models"][f"{arch}_{variant}"] = dict(
                        field=q8(P[j, ::STRIDE[system]], rng), err=r4(err[j]),
                        drift={k: r4(v[j]) for k, v in drift.items()},
                        dead=int(np.argmax(np.isnan(err[j]))) if np.isnan(err[j]).any() else None)
            out["runs"].append(run)
    os.makedirs("docs/data", exist_ok=True)
    json.dump(out, open(f"docs/data/{system}.json", "w"), separators=(",", ":"))
    print("wrote docs/data/%s.json (%.0f kB)" % (system, os.path.getsize(f"docs/data/{system}.json") / 1e3))
    (gallery_kdv if system == "kdv" else gallery_ns)(sets, keep)


def gallery_kdv(sets, keep):
    """Space-time plots: truth and the ResNet / FNO rollouts, raw and projected, test trajectory 0."""
    U = sets["test"][0].numpy()
    T = U.shape[0]
    panels = [("Reference solver", U)] + [
        (f"{ANAME[a]}, {'projected' if v == 'proj' else 'raw'}", keep[("test", a, v)][0][0]) for a in ("resnet", "fno") for v in ("raw", "proj")]
    fig, ax = plt.subplots(1, 5, figsize=(7.8, 2.7), sharey=True, layout="constrained")
    vmax = np.abs(U).max()
    for a, (title, F) in zip(ax, panels):
        F = np.where(np.isnan(F), np.nan, F)
        im = a.imshow(F, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=(-20, 20, 0, (T - 1) * 0.5), interpolation="nearest")
        a.set_title(title, fontsize=8); a.set_xlabel("x"); a.grid(False)
        dead = np.where(np.isnan(F).any(1))[0]
        if len(dead):
            a.axhline(dead[0] * 0.5, color=INK, lw=0.8, ls=":")
            a.text(0, dead[0] * 0.5 + 3, f"blew up, t={dead[0] * 0.5:.0f}", ha="center", fontsize=6.5, color=INK)
    ax[0].set_ylabel("time")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="u")
    save(fig, "gallery_kdv")


def gallery_ns(sets, keep):
    """Vorticity snapshots: truth vs each architecture raw / projected at three times."""
    U = sets["test"][0].numpy()
    times = [0, 20, 50, 100]
    rows = [("Reference", U)] + [(f"{ANAME[a]} {'proj.' if v == 'proj' else 'raw'}", keep[("test", a, v)][0][0])
                                 for a in ARCHS for v in ("raw", "proj")]
    fig, ax = plt.subplots(len(times), len(rows), figsize=(7.8, 4.8))
    vmax = np.abs(U[0]).max()
    for j, (title, F) in enumerate(rows):
        for i, t in enumerate(times):
            a = ax[i, j]
            a.imshow(F[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            if np.isnan(F[t]).any():
                a.text(0.5, 0.5, "blew up", transform=a.transAxes, ha="center", va="center", fontsize=8, color=INK2)
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
            if i == 0:
                a.set_title(title, fontsize=7.5)
            if j == 0:
                a.set_ylabel(f"t = {t * 0.5:.0f}", fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, "gallery_ns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="kdv")
    main(ap.parse_args().system)
