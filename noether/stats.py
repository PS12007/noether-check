"""Small statistics helpers (numpy only)."""
import numpy as np


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    ra, rb = ranks(a[ok]), ranks(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(score, label):
    """P(score of a positive > score of a negative), ties count half. NaN scores rank highest."""
    s = np.where(np.isnan(score), np.inf, np.asarray(score, float))
    y = np.asarray(label, bool)
    if y.all() or not y.any():
        return float("nan")
    r = ranks(np.where(np.isinf(s), np.nanmax(np.where(np.isinf(s), np.nan, s)) + 1 if np.isfinite(s).any() else 0, s))
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def bootstrap_ci(values, n=2000, seed=0, stat=np.median):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(v, len(v))) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
