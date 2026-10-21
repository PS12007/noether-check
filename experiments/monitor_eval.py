"""Can a rollout tell, without the reference solution, that it has gone wrong?

Monitors, all label-free:
  drift_<q>     drift of an invariant or budget (noether/monitor.py)
  ensemble      disagreement with a twin model (same configuration, next seed), the standard
                uncertainty baseline; it costs a second model and a second rollout.
Targets (need the reference, used only for grading):
  now           relative error at step h above TAU
  later         relative error at the final step above TAU, predicted from monitors at step h
                (early warning)
For projected rollouts the projected quantities are zero by construction, so only the held-out
ones (KdV energy; NS enstrophy budget) can warn.

Writes results/monitor/<system>.json and runs/monitor/<system>.npz (pooled per-trajectory values).
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.solvers import DEV
from noether.models import Surrogate
from noether.train import rollout, SIZES, SYSTEMS
from noether.project import kdv_projector, ns_projector
from noether.monitor import DRIFT
from noether.stats import auroc
from experiments.zoo import rel_err, HORIZONS

TAU = 0.3
WILD = 2.0   # error well above sqrt(2), the level of a decorrelated field with the right energy: the rollout is unphysical


def load(system, r):
    m = Surrogate(r["arch"], SYSTEMS[system]["nd"], 1, 1, **SIZES[(system, r["arch"])][r["size"]]).to(DEV)
    m.load_state_dict(torch.load(f"runs/{system}/{r['name']}.pt", map_location=DEV))
    return m.eval()


def score(system, flat):
    """AUROCs from pooled per-trajectory values. 'mix' pools test and OOD starts: a deployment
    that sees all of them (test, shift if present, OOD), where the monitor must separate failing from healthy rollouts."""
    H = HORIZONS[system][1:]
    groups = {}
    for key, v in flat.items():
        set_name, variant, mon = key.split("_", 2)
        groups.setdefault((set_name, variant), {})[mon] = v
    for variant in ("raw", "proj"):
        parts = [groups[(s, variant)] for s in ("test", "shift", "ood") if (s, variant) in groups]
        if len(parts) > 1:
            groups[("mix", variant)] = {k: np.concatenate([g[k] for g in parts]) for k in parts[0]}
    res = {}
    for (set_name, variant), P in groups.items():
        err = np.where(np.isnan(P["err"]), np.inf, P["err"])
        for mon in [k for k in P if k != "err"]:
            for h in H:
                for tgt, lab in (("now", err[:, h] > TAU), ("later", err[:, -1] > TAU),
                                 ("wildnow", err[:, h] > WILD), ("wildlater", err[:, -1] > WILD)):
                    res[f"{set_name}_{variant}_{mon}_{tgt}@{h}"] = auroc(P[mon][:, h], lab)
        for h in H:
            res[f"{set_name}_{variant}_failrate_now@{h}"] = float((err[:, h] > TAU).mean())
            res[f"{set_name}_{variant}_wildrate_now@{h}"] = float((err[:, h] > WILD).mean())
        res[f"{set_name}_{variant}_failrate_final"] = float((err[:, -1] > TAU).mean())
    os.makedirs("results/monitor", exist_ok=True)
    json.dump(res, open(f"results/monitor/{system}.json", "w"), indent=1)
    return res


def main(system, seeds=(0, 1, 2), set_names=("test", "ood")):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"results/{system}/*.json"))]
    by = {r["name"]: r for r in rows}
    sets = {k: torch.load(f"data/{system}_{k}.pt") for k in set_names}
    H = HORIZONS[system][1:]
    pooled = {}
    for r in rows:
        # twin: same configuration, the next seed that exists (cyclically)
        i = seeds.index(r["seed"])
        cands = [r["name"].replace(f"_s{r['seed']}", f"_s{seeds[(i + j) % len(seeds)]}") for j in range(1, len(seeds))]
        partner = next((c for c in cands if c in by), None)
        if partner is None:
            continue
        m1, m2 = load(system, r), load(system, by[partner])
        for set_name, U in sets.items():
            for variant in ("raw", "proj"):
                proj = None
                if variant == "proj":
                    proj = kdv_projector(("mass", "momentum")) if system == "kdv" else ns_projector()
                P1 = rollout(m1, U[:, 0].to(DEV), U.shape[1] - 1, projector=proj)
                P2 = rollout(m2, U[:, 0].to(DEV), U.shape[1] - 1, projector=proj)
                err = rel_err(P1, U)
                ens = rel_err(P2, torch.nan_to_num(P1, nan=0.0))
                ens[np.isnan(err)] = np.nan
                d = DRIFT[system](P1)
                key = (set_name, variant)
                pooled.setdefault(key, dict(err=[], ensemble=[], **{f"drift_{k}": [] for k in d}))
                pooled[key]["err"].append(err)
                pooled[key]["ensemble"].append(ens)
                for k, v in d.items():
                    pooled[key][f"drift_{k}"].append(v)
        print(r["name"], "done", flush=True)
    flat = {}
    old = f"runs/monitor/{system}.npz"
    if os.path.exists(old):           # keep values for sets not recomputed in this call
        z = np.load(old)
        flat = {k: z[k] for k in z.files if k.split("_", 1)[0] not in set_names}
    for (set_name, variant), P in pooled.items():
        for k, v in P.items():
            flat[f"{set_name}_{variant}_{k}"] = np.concatenate(v).astype(np.float32)
    os.makedirs("runs/monitor", exist_ok=True)
    np.savez_compressed(f"runs/monitor/{system}.npz", **flat)
    res = score(system, flat)
    for k, v in res.items():
        if "@" in k and k.endswith(f"@{H[1]}") and "now" in k:
            print(k, round(v, 3) if v == v else v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="kdv")
    ap.add_argument("--sets", default="test,ood")
    ap.add_argument("--rescore", action="store_true", help="recompute AUROCs from runs/monitor/<system>.npz")
    a = ap.parse_args()
    if a.rescore:
        z = np.load(f"runs/monitor/{a.system}.npz")
        score(a.system, {k: z[k] for k in z.files})
    else:
        main(a.system, set_names=tuple(a.sets.split(",")))
