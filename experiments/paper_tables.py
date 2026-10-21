"""Print the HTML table rows used in docs/paper.html from the result files.

python experiments/paper_tables.py kdv|ns  config|monitor
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.zoo import HORIZONS

ANAME = dict(fno="FNO", resnet="ResNet", unet="U-Net")


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if np.isinf(x):
        return "∞"
    return f"{x:.{d}f}"


def config(system):
    S = json.load(open(f"results/summary_{system}.json"))
    out = []
    for key in sorted(S["table"], key=lambda k: (k.split("_")[0], ["small", "base"].index(k.split("_")[1]), k)):
        t = S["table"][key]
        arch, size, *rest = key.split("_")
        label = f"{ANAME[arch]} {size}" + {"noise": " + noise", "pen": " + penalty"}.get(rest[0] if rest else "", "")
        n = "" if t["n"] == 3 else f" ({t['n']} seeds)"
        out.append(f"<tr><td>{label}{n}</td><td class=\"n\">{t['params'] / 1e3:.0f}k</td><td class=\"n\">{t['one_step']:.1e}</td>"
                   f"<td class=\"n\">{f(t['err_mid'])}</td><td class=\"n\">{f(t['err_end'])}</td><td class=\"n\">{f(t['proj_end'])}</td>"
                   f"<td class=\"n\">{t['valid']:.0f}</td><td class=\"n\">{t['valid_proj']:.0f}</td></tr>")
    return "\n".join(out)


def monitor(system):
    M = json.load(open(f"results/monitor/{system}.json"))
    H = HORIZONS[system]
    early = H[2:5] if system == "kdv" else H[2:5]
    held = {"kdv": "energy", "ns": "enstrophy"}[system]
    other = {"kdv": "momentum", "ns": "energy"}[system]
    lab = {f"drift_{held}": f"{held} drift" if system == "kdv" else f"{held} budget",
           f"drift_{other}": f"{other} drift" if system == "kdv" else f"{other} budget",
           "ensemble": "twin-model disagreement"}
    out = []
    for s, sname in (("test", "in distribution"), ("mix", "test + OOD")):
        for v in ("raw", "proj"):
            for mon in (f"drift_{held}", f"drift_{other}", "ensemble"):
                if v == "proj" and mon == f"drift_{other}":
                    continue
                pre = "" if system == "kdv" else "wild"
                vals = [M.get(f"{s}_{v}_{mon}_{pre}later@{h}") for h in early] + [M.get(f"{s}_{v}_{mon}_{pre}now@{H[-1]}")]
                cells = "".join(f"<td class=\"n\">{f(x, 2)}</td>" for x in vals)
                out.append(f"<tr><td>{sname}, {'raw' if v == 'raw' else 'projected'}</td><td>{lab[mon]}</td>{cells}</tr>")
    return "\n".join(out)


if __name__ == "__main__":
    print({"config": config, "monitor": monitor}[sys.argv[2]](sys.argv[1]))
