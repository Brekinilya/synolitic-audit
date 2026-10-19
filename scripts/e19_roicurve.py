"""E19: accuracy against the number of regions, for both readouts.

The published curve peaks near 50 regions and then declines, which the authors
attribute to noise contributed by weakly informative regions. The averaging
mechanism predicts something different and testable: node averaging projects the
d(d-1)/2 pairwise features onto the d-dimensional space of node sums, so the
fraction of edge directions that survive is about 2/(d-1) and the loss grows
with d. The node-summary readout should therefore peak and fall, while the flat
readout, which never aggregates, should keep rising or flatten out.

Regions are added in order of univariate informativeness, ranked on the training
fold only, which is what produces the "weakly informative tail" in the first
place.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"
DATA = Path("C:/Users/breki/abide_data")


def fit_predict(xtr, ytr, xte, C=1.0):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    return LogisticRegression(**kw).fit(sc.transform(xtr), ytr) \
        .decision_function(sc.transform(xte))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="10,25,50,100,150,200")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--order", default="ranked", choices=["ranked", "atlas"])
    ap.add_argument("--out", default="abide_roicurve.json")
    args = ap.parse_args()

    from e18_primary import load
    from fastpair import fit_pairwise, node_means, pairwise_logits

    ab = load()
    series, pheno = ab["rois_cc200"], ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])
    n, d_all = len(series), series[0].shape[1]
    vertex = np.empty((n, d_all, 2))
    edge = np.empty((n, d_all, d_all), dtype=np.float32)
    for k, s in enumerate(series):
        z = np.asarray(s, dtype=np.float64).T
        vertex[k, :, 0] = z.mean(axis=1)
        vertex[k, :, 1] = z.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            edge[k] = np.nan_to_num(np.corrcoef(z), nan=0.0)
    print(f"n={n} d_all={d_all}", flush=True)

    grid = [int(x) for x in args.grid.split(",")]
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                 random_state=0)
    rows: dict[int, dict[str, list]] = {k: {} for k in grid}
    t0 = time.time()
    for split, (tr, te) in enumerate(cv.split(vertex[:, :, 0], y)):
        if args.order == "atlas":
            order = np.arange(d_all)          # first k regions of the atlas
        else:
            strength = np.array([
                max(abs(roc_auc_score(y[tr], vertex[tr, i, 0]) - 0.5),
                    abs(roc_auc_score(y[tr], vertex[tr, i, 1]) - 0.5))
                for i in range(d_all)])
            order = np.argsort(-strength)
        for k in grid:
            reg = np.sort(order[:k])
            v = np.ascontiguousarray(vertex[:, reg])
            e = np.ascontiguousarray(edge[:, reg][:, :, reg].astype(np.float64))
            pi, pj = np.triu_indices(k, k=1)
            pairs = [(int(a), int(b)) for a, b in zip(pi, pj)]
            beta = fit_pairwise(v[tr], e[tr], y[tr], pairs, "mmssr")
            L = pairwise_logits(v, e, pairs, "mmssr", beta)
            wbar = node_means(np.tanh(L / 2), pairs, k)
            fc = e[:, pi, pj]
            out = {
                "SG: wbar": fit_predict(wbar[tr], y[tr], wbar[te]),
                "SG: flat": fit_predict(L[tr], y[tr], L[te]),
                "LR correlations": fit_predict(fc[tr], y[tr], fc[te]),
            }
            for key, s in out.items():
                rows[k].setdefault(key, []).append(
                    (float(roc_auc_score(y[te], s)),
                     float(balanced_accuracy_score(y[te], s > 0))))
        print(f"  split {split+1}/{args.repeats*args.folds} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'d':>5}  " + "".join(
        f"{k:>26}" for k in ["SG: wbar", "SG: flat", "LR correlations"]))
    summary = {}
    for k in grid:
        line = f"{k:>5}  "
        summary[k] = {}
        for key in ["SG: wbar", "SG: flat", "LR correlations"]:
            a = np.array(rows[k][key])
            summary[k][key] = dict(auc=float(a[:, 0].mean()),
                                   bacc=float(a[:, 1].mean()),
                                   bacc_sd=float(a[:, 1].std()))
            line += f"   AUC {a[:,0].mean():.4f} bacc {100*a[:,1].mean():.1f}%"
        print(line)

    (CACHE / args.out).write_text(json.dumps(
        dict(n=n, grid=grid, repeats=args.repeats, folds=args.folds,
             summary=summary), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
