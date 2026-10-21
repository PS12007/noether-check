"""Train a population of surrogates and roll each out with and without Noether projection.

For every model we record, on the in-distribution test set and an out-of-distribution set:
  one-step relative error (the usual training-time metric),
  relative L2 error to the reference solution after h steps, for every h,
  relative drift of each invariant after h steps.
Per-trajectory curves go to runs/<system>/<name>_<set>_<variant>.npz, summaries to
results/<system>/<name>.json. Checkpoints go to runs/<system>/<name>.pt.

python experiments/zoo.py --system kdv
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.solvers import DEV, kdv_invariants, ns_invariants
from noether.models import Surrogate
from noether.train import train, rollout, SIZES, SYSTEMS, pairs
from noether.project import kdv_projector, ns_projector
from noether.monitor import DRIFT

HORIZONS = {"kdv": [1, 10, 50, 100, 200, 400], "ns": [1, 5, 10, 20, 50, 100]}


def configs(system, seeds):
    out = []
    for arch in ("fno", "resnet", "unet"):
        for size in ("small", "base"):
            for s in seeds:
                out.append(dict(arch=arch, size=size, seed=s, noise=0.0, penalty=0.0))
        for s in seeds:
            out.append(dict(arch=arch, size="base", seed=s, noise=0.01, penalty=0.0))
            out.append(dict(arch=arch, size="base", seed=s, noise=0.0, penalty=0.1))
    return out


def name_of(c):
    n = f"{c['arch']}_{c['size']}_s{c['seed']}"
    if c["noise"]:
        n += f"_noise{c['noise']}"
    if c["penalty"]:
        n += f"_pen{c['penalty']}"
    return n


def rel_err(pred, ref):
    d = (pred - ref).flatten(2).norm(dim=-1)
    return (d / ref.flatten(2).norm(dim=-1)).numpy()                     # [B, T+1], NaN if blown up


def evaluate(system, model, U, variant):
    proj = None
    if variant == "proj":
        proj = kdv_projector(("mass", "momentum")) if system == "kdv" else ns_projector()
    P = rollout(model, U[:, 0].to(DEV), U.shape[1] - 1, projector=proj)
    return P, rel_err(P, U), DRIFT[system](P)


def eval_set(system, model, name, set_name, U, res):
    """One-step error plus raw / projected rollouts on one set; fills res and saves curves."""
    X, Y = pairs(U[:, :11])                       # one-step error on the first 10 steps of each test trajectory
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 256].to(DEV)).cpu() for i in range(0, len(X), 256)])
    res[f"{set_name}_one_step"] = float(((pred - Y).flatten(1).norm(dim=1) / Y.flatten(1).norm(dim=1)).mean())
    for variant in ("raw", "proj"):
        _, err, dr = evaluate(system, model, U, variant)
        np.savez_compressed(f"runs/{system}/{name}_{set_name}_{variant}.npz", err=err.astype(np.float32),
                            **{f"drift_{k}": v.astype(np.float32) for k, v in dr.items()})
        for h in HORIZONS[system]:
            e = err[:, h]
            res[f"{set_name}_{variant}_err@{h}"] = float(np.nanmedian(np.where(np.isnan(e), np.inf, e)))
            res[f"{set_name}_{variant}_blowup@{h}"] = float(np.isnan(e).mean())
            for k, v in dr.items():
                res[f"{set_name}_{variant}_drift_{k}@{h}"] = float(np.nanmedian(np.where(np.isnan(v[:, h]), np.inf, v[:, h])))
        # valid time: steps until relative error first exceeds 0.5 (median over trajectories)
        bad = np.where(np.isnan(err), np.inf, err) > 0.5
        first = np.where(bad.any(1), bad.argmax(1), err.shape[1])
        res[f"{set_name}_{variant}_valid_steps"] = float(np.median(first))


def run(system, c, sets, log):
    name = name_of(c)
    out = f"results/{system}/{name}.json"
    if os.path.exists(out):
        return
    os.makedirs(f"runs/{system}", exist_ok=True)
    os.makedirs(f"results/{system}", exist_ok=True)
    pt = f"runs/{system}/{name}.pt"
    t0 = time.time()
    if os.path.exists(pt):
        ck = torch.load(pt)
        model = Surrogate(c["arch"], SYSTEMS[system]["nd"], 1, 1, **SIZES[(system, c["arch"])][c["size"]]).to(DEV)
        model.load_state_dict(ck)
        model.eval()
    else:
        model = train(system, c["arch"], c["size"], c["seed"], epochs=EPOCHS[system], noise=c["noise"], penalty=c["penalty"])
        torch.save(model.state_dict(), pt)
    res = dict(system=system, name=name, **c, params=sum(p.numel() for p in model.parameters()),
               train_seconds=round(time.time() - t0, 1))
    for set_name, U in sets.items():
        eval_set(system, model, name, set_name, U, res)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)
    log(f"{name}: one-step {res['test_one_step']:.4f} | valid steps raw {res['test_raw_valid_steps']:.0f} proj {res['test_proj_valid_steps']:.0f} | ({res['seconds']:.0f}s)")


EPOCHS = {"kdv": 20, "ns": 30}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="kdv")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", default="", help="evaluate trained models on an extra set, e.g. shift")
    a = ap.parse_args()
    if a.extra:
        U = torch.load(f"data/{a.system}_{a.extra}.pt")
        for f in sorted(glob.glob(f"results/{a.system}/*.json")):
            res = json.load(open(f))
            if f"{a.extra}_one_step" in res or (a.only and a.only not in res["name"]):
                continue
            model = Surrogate(res["arch"], SYSTEMS[a.system]["nd"], 1, 1, **SIZES[(a.system, res["arch"])][res["size"]]).to(DEV)
            model.load_state_dict(torch.load(f"runs/{a.system}/{res['name']}.pt"))
            eval_set(a.system, model.eval(), res["name"], a.extra, U, res)
            json.dump(res, open(f, "w"), indent=1)
            print(res["name"], a.extra, "one-step %.4f valid raw %.0f proj %.0f" % (
                res[f"{a.extra}_one_step"], res[f"{a.extra}_raw_valid_steps"], res[f"{a.extra}_proj_valid_steps"]), flush=True)
        sys.exit()
    sets = {k: torch.load(f"data/{a.system}_{k}.pt") for k in ("test", "ood")}
    log = lambda s: print(s, flush=True)
    for c in configs(a.system, list(map(int, a.seeds.split(",")))):
        if a.only and a.only not in name_of(c):
            continue
        run(a.system, c, sets, log)
