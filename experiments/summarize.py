"""Population-level analysis of the zoo results. Writes results/summary_<system>.json and prints tables.

1. Does the one-step error (the usual metric) predict long-horizon behaviour?
   Spearman rank correlation across models between one-step error and error / drift at each horizon.
2. What does Noether projection buy?  Per model: raw vs projected error at each horizon.
3. Training variants (input noise, soft invariant penalty) against the plain base models.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noether.stats import spearman
from experiments.zoo import HORIZONS


def fin(x):
    x = np.asarray(x, float)
    return np.where(np.isinf(x), np.nan, x)


def main(system):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"results/{system}/*.json"))]
    H = HORIZONS[system]
    q = {"kdv": ("mass", "momentum", "energy"), "ns": ("mean", "energy", "enstrophy")}[system]
    held = {"kdv": "energy", "ns": "enstrophy"}[system]
    out = dict(n_models=len(rows))
    SETS = [s for s in ("test", "shift", "ood") if all(f"{s}_one_step" in r for r in rows)]
    print(f"{system}: {len(rows)} models")

    # 1. rank correlation of one-step error with later quantities (raw rollouts)
    for s in SETS:
        one = [r[f"{s}_one_step"] for r in rows]
        for h in H:
            # infinite (blown-up) medians are the worst rank: keep them as +inf for ranking
            e = [r[f"{s}_raw_err@{h}"] for r in rows]
            out[f"rho_onestep_err_{s}@{h}"] = spearman(one, [1e9 if np.isinf(v) else v for v in e])
            for k in q:
                d = [r[f"{s}_raw_drift_{k}@{h}"] for r in rows]
                out[f"rho_onestep_drift_{k}_{s}@{h}"] = spearman(one, [1e9 if np.isinf(v) else v for v in d])
        out[f"rho_onestep_valid_{s}"] = spearman(one, [-r[f"{s}_raw_valid_steps"] for r in rows])
        # drift at an early horizon as predictor of final error (across models)
        e_last = [1e9 if np.isinf(r[f"{s}_raw_err@{H[-1]}"]) else r[f"{s}_raw_err@{H[-1]}"] for r in rows]
        for k in q:
            d_early = [1e9 if np.isinf(r[f"{s}_raw_drift_{k}@{H[2]}"]) else r[f"{s}_raw_drift_{k}@{H[2]}"] for r in rows]
            out[f"rho_drift_{k}@{H[2]}_vs_err@{H[-1]}_{s}"] = spearman(d_early, e_last)
        one_early = [r[f"{s}_raw_err@{H[2]}"] for r in rows]
        out[f"rho_err@{H[2]}_vs_err@{H[-1]}_{s}"] = spearman([1e9 if np.isinf(v) else v for v in one_early], e_last)

    print("Spearman, one-step error vs error at horizon (test):",
          {h: round(out[f"rho_onestep_err_test@{h}"], 2) for h in H})
    print("  ... vs valid steps (test / ood):", round(out["rho_onestep_valid_test"], 2), round(out["rho_onestep_valid_ood"], 2))
    print("  early drift vs final error (test):", {k: round(out[f"rho_drift_{k}@{H[2]}_vs_err@{H[-1]}_test"], 2) for k in q},
          " early error vs final:", round(out[f"rho_err@{H[2]}_vs_err@{H[-1]}_test"], 2))

    # 2. projection, paired per model
    for s in SETS:
        for h in H:
            raw = fin([r[f"{s}_raw_err@{h}"] for r in rows])
            pr = fin([r[f"{s}_proj_err@{h}"] for r in rows])
            ratio = pr / raw
            out[f"proj_ratio_{s}@{h}_median"] = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
            out[f"proj_better_{s}@{h}"] = float(np.mean([(b < a) or (np.isinf(a) and not np.isinf(b))
                                                         for a, b in zip([r[f"{s}_raw_err@{h}"] for r in rows],
                                                                         [r[f"{s}_proj_err@{h}"] for r in rows])]))
            out[f"blowup_raw_{s}@{h}"] = float(np.mean([r[f"{s}_raw_blowup@{h}"] for r in rows]))
            out[f"blowup_proj_{s}@{h}"] = float(np.mean([r[f"{s}_proj_blowup@{h}"] for r in rows]))
        out[f"valid_raw_{s}_median"] = float(np.median([r[f"{s}_raw_valid_steps"] for r in rows]))
        out[f"valid_proj_{s}_median"] = float(np.median([r[f"{s}_proj_valid_steps"] for r in rows]))
        print(f"projection ({s}): median error ratio proj/raw", {h: round(out[f"proj_ratio_{s}@{h}_median"], 2) for h in H},
              "| better in", {h: round(out[f"proj_better_{s}@{h}"], 2) for h in H},
              "| blow-up raw/proj @end", round(out[f"blowup_raw_{s}@{H[-1]}"], 2), round(out[f"blowup_proj_{s}@{H[-1]}"], 2),
              "| valid steps raw/proj", out[f"valid_raw_{s}_median"], out[f"valid_proj_{s}_median"])

    # 3. per architecture / variant table
    table = {}
    for r in rows:
        key = f"{r['arch']}_{r['size']}" + ("_noise" if r["noise"] else "") + ("_pen" if r["penalty"] else "")
        table.setdefault(key, []).append(r)
    out["table"] = {}
    print(f"{'config':22s} {'1-step':>8s} {'err@'+str(H[3]):>9s} {'err@'+str(H[-1]):>9s} {'proj@'+str(H[-1]):>9s} {'valid':>6s} {'valid_p':>7s} {held+'@end':>11s}")
    for key in sorted(table):
        rs = table[key]
        med = lambda k: float(np.median([r[k] for r in rs]))
        t = dict(n=len(rs), one_step=med("test_one_step"), err_mid=med(f"test_raw_err@{H[3]}"),
                 err_end=med(f"test_raw_err@{H[-1]}"), proj_end=med(f"test_proj_err@{H[-1]}"),
                 valid=med("test_raw_valid_steps"), valid_proj=med("test_proj_valid_steps"),
                 held_end=med(f"test_raw_drift_{held}@{H[-1]}"), params=rs[0]["params"])
        out["table"][key] = t
        print(f"{key:22s} {t['one_step']:8.4f} {t['err_mid']:9.4f} {t['err_end']:9.3f} {t['proj_end']:9.3f} {t['valid']:6.0f} {t['valid_proj']:7.0f} {t['held_end']:11.4f}")
    json.dump(out, open(f"results/summary_{system}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="kdv")
    main(ap.parse_args().system)
