"""All figures, from results/*.json, runs/*.npz and (for rollout galleries) checkpoints.

python experiments/figures.py [name ...]
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.zoo import HORIZONS

OUT = "docs/figures"
COL = dict(fno="#2a78d6", resnet="#eb6834", unet="#1baf7a")
ANAME = dict(fno="FNO", resnet="ResNet", unet="U-Net")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SYS = dict(kdv="KdV solitons", ns="2D turbulence")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "lines.linewidth": 2,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.svg")
    plt.close(fig)
    print("wrote", name)


def rows(system):
    return [json.load(open(f)) for f in sorted(glob.glob(f"results/{system}/*.json"))]


def curves(system, name, s, variant):
    z = np.load(f"runs/{system}/{name}_{s}_{variant}.npz")
    return {k: z[k] for k in z.files}


def nanmed_curve(a):
    """Median over trajectories per step, counting blown-up (NaN) trajectories as +inf."""
    return np.median(np.where(np.isnan(a), np.inf, a), 0)


# ------------------------------------------------------------------------------------------
def fig_horizon(system):
    """Error and held-out drift vs rollout step, base models, raw (dashed) vs projected (solid)."""
    R = [r for r in rows(system) if r["size"] == "base" and not r["noise"] and not r["penalty"]]
    if not R:
        return
    held = {"kdv": "energy", "ns": "enstrophy"}[system]
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.6))
    for arch in COL:
        for variant, ls in (("raw", "--"), ("proj", "-")):
            E, D, Eo = [], [], []
            for r in R:
                if r["arch"] != arch:
                    continue
                c = curves(system, r["name"], "test", variant)
                E.append(nanmed_curve(c["err"]))
                D.append(nanmed_curve(c[f"drift_{held}"]))
                Eo.append(nanmed_curve(curves(system, r["name"], "ood", variant)["err"]))
            if not E:
                continue
            t = np.arange(len(E[0]))
            for a, Y in zip(ax, (E, D, Eo)):
                y = np.median(np.array(Y), 0)
                y = np.where(np.isinf(y), np.nan, y)
                a.plot(t[1:], y[1:], ls, color=COL[arch], lw=1.6 if variant == "proj" else 1.2,
                       label=f"{ANAME[arch]}, {'projected' if variant == 'proj' else 'raw'}")
    for a, title, yl in zip(ax, ("Error, in distribution", f"Held-out {held} drift", "Error, out of distribution"),
                            ("Relative L2 error", "Relative drift", "Relative L2 error")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(title); a.set_xlabel("Rollout step"); a.set_ylabel(yl)
    ax[2].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f"horizon_{system}")


def fig_rank(system):
    """One-step error vs long-horizon outcome, every model."""
    R = rows(system)
    if len(R) < 4:
        return
    H = HORIZONS[system]
    held = {"kdv": "energy", "ns": "enstrophy"}[system]
    MK = {"base": "o", "small": "s", "noise": "^", "pen": "D"}
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.6))
    for r in R:
        kind = "noise" if r["noise"] else "pen" if r["penalty"] else r["size"]
        x = r["test_one_step"]
        ys = (r["test_raw_valid_steps"], r[f"test_raw_err@{H[-1]}"], r[f"test_raw_drift_{held}@{H[-1]}"])
        for a, y in zip(ax, ys):
            y = min(y, 10.0) if a is not ax[0] else y
            a.plot(x, y, MK[kind], color=COL[r["arch"]], ms=5, mec="white", mew=0.6, alpha=0.9)
    sm = json.load(open(f"results/summary_{system}.json")) if os.path.exists(f"results/summary_{system}.json") else {}
    titles = (f"Valid steps (error < 0.5)", f"Error at step {H[-1]}", f"{held.capitalize()} drift at step {H[-1]}")
    keys = ("rho_onestep_valid_test", f"rho_onestep_err_test@{H[-1]}", f"rho_onestep_drift_{held}_test@{H[-1]}")
    for a, t, k in zip(ax, titles, keys):
        a.set_xscale("log")
        if a is not ax[0]:
            a.set_yscale("log")
            a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel("One-step test error")
        a.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        rho = sm.get(k)
        if rho is not None and a is ax[0]:
            rho = -rho  # summary ranks -valid; show the correlation in plotted terms
        a.set_title(t + (f"\nSpearman ρ = {rho:+.2f}" if rho is not None else ""), fontsize=8.5)
    handles = [plt.Line2D([], [], color=COL[a], lw=0, marker="o", ms=5, label=ANAME[a]) for a in COL]
    handles += [plt.Line2D([], [], color=MUTED, lw=0, marker=MK[k], ms=5, label=l)
                for k, l in (("base", "base"), ("small", "small"), ("noise", "noise-trained"), ("pen", "invariant penalty"))]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f"rank_{system}")


def fig_monitor(system):
    """AUROC of label-free monitors for 'error above 0.3', vs step."""
    path = f"results/monitor/{system}.json"
    if not os.path.exists(path):
        return
    M = json.load(open(path))
    H = HORIZONS[system][1:]
    q = {"kdv": ("mass", "momentum", "energy"), "ns": ("mean", "energy", "enstrophy")}[system]
    MC = {f"drift_{q[0]}": "#e87ba4", f"drift_{q[1]}": "#eda100", f"drift_{q[2]}": "#2a78d6", "ensemble": INK2}
    lab = {f"drift_{k}": f"{k} drift" if not (system == "ns" and k != "mean") else f"{k} budget" for k in q}
    lab["ensemble"] = "twin-model disagreement"
    fig, ax = plt.subplots(1, 4, figsize=(7.8, 2.4), sharey=True)
    for a, (s, v) in zip(ax, (("test", "raw"), ("test", "proj"), ("mix", "raw"), ("mix", "proj"))):
        for mon, c in MC.items():
            tgt = "now" if system == "kdv" else "wildnow"
            y = [M.get(f"{s}_{v}_{mon}_{tgt}@{h}", np.nan) for h in H]
            if v == "proj" and mon in (f"drift_{q[0]}", f"drift_{q[1]}"):
                continue  # these are fixed by the projection
            a.plot(range(len(H)), y, "-o", color=c, ms=3.5, lw=1.4, label=lab[mon])
        a.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        a.set_xticks(range(len(H))); a.set_xticklabels(H, fontsize=7)
        a.set_xlabel("Rollout step")
        a.set_title(f"{'In distribution' if s == 'test' else 'Test + OOD pooled'}\n{'raw' if v == 'raw' else 'projected'}", fontsize=8.5)
        a.set_ylim(0.3 if system == "kdv" else 0.0, 1.02)
    ax[0].set_ylabel("AUROC, error > 0.3" if system == "kdv" else "AUROC, unphysical (error > 2)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, f"monitor_{system}")


def fig_spectra(system):
    """NS only: energy-spectrum error vs step, base models, raw vs projected, test and OOD."""
    if system != "ns" or not os.path.exists("runs/spectra_ns.npz"):
        return
    z = np.load("runs/spectra_ns.npz")
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 2.7), sharey=True)
    for a, s in zip(ax, ("test", "ood")):
        for arch in COL:
            for variant, ls in (("raw", "--"), ("proj", "-")):
                ks = [k for k in z.files if k.startswith(f"{arch}_base_s") and k.endswith(f"_{s}_{variant}")]
                if not ks:
                    continue
                y = np.median([nanmed_curve(z[k]) for k in ks], 0)
                y = np.where(np.isinf(y), np.nan, y)
                a.plot(np.arange(len(y))[1:], y[1:], ls, color=COL[arch], lw=1.6 if variant == "proj" else 1.2,
                       label=f"{ANAME[arch]}, {'projected' if variant == 'proj' else 'raw'}")
        f = np.median(z[f"frozen_{s}"], 0)
        a.plot(np.arange(len(f))[1:], f[1:], ":", color=MUTED, lw=1.2, label="initial state, frozen")
        a.set_yscale("log"); a.set_xlabel("Rollout step")
        a.set_title("In distribution" if s == "test" else "Out of distribution (smaller eddies)")
    ax[0].set_ylabel("Spectrum error\nmean |log10 E(k) ratio|")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "spectra_ns")


FIGS = dict(horizon=fig_horizon, rank=fig_rank, monitor=fig_monitor, spectra=fig_spectra)

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for system in ("kdv", "ns"):
        for n in names:
            if n in FIGS:
                FIGS[n](system)
