"""E19b: accuracy against the number of regions, memory-lean rewrite.

Same question as e19 -- does the node-summary readout peak and fall while the
flat readout keeps rising -- but connectivity is held once as an (n, P) column
block instead of rebuilding dense (n, k, k) matrices for every grid point, which
is what exhausted memory in the first attempt.

--order ranked  regions added by univariate informativeness, ranked on the
                training fold (what produces a weakly informative tail)
--order atlas   the first k regions of CC200 in atlas order, which is how the
                published tangent-space analysis selected its 50 regions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="10,25,50,100,150,200")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--order", default="atlas", choices=["ranked", "atlas"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_name = args.out or f"abide_roicurve_{args.order}.json"

    from e8_mmp import fit_chunked, readouts
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def score2(xtr, ytr, xte, yte, C=1.0):
        sc = StandardScaler().fit(xtr)
        kw = dict(C=C, max_iter=5000)
        if xtr.shape[1] > 5000:
            kw.update(dual=True, solver='liblinear')
        clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
        s = clf.decision_function(sc.transform(xte))
        return (float(roc_auc_score(yte, s)),
                float(balanced_accuracy_score(yte, s > 0)))
    from e18_primary import load

    ab = load()
    series, pheno = ab["rois_cc200"], ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])
    n, d_all = len(series), series[0].shape[1]

    vertex = np.empty((n, d_all, 2))
    fi, fj = np.triu_indices(d_all, k=1)
    fc_all = np.empty((n, fi.size), dtype=np.float32)
    for k, s in enumerate(series):
        z = np.asarray(s, dtype=np.float64).T
        vertex[k, :, 0] = z.mean(axis=1)
        vertex[k, :, 1] = z.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.nan_to_num(np.corrcoef(z), nan=0.0)
        fc_all[k] = c[fi, fj]
    key = {(int(a), int(b)): t for t, (a, b) in enumerate(zip(fi, fj))}
    print(f"n={n} d_all={d_all} P_all={fi.size} order={args.order}", flush=True)

    grid = [int(x) for x in args.grid.split(",")]
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                 random_state=0)
    rows: dict[int, dict[str, list]] = {k: {} for k in grid}
    t0 = time.time()
    for split, (tr, te) in enumerate(cv.split(vertex[:, :, 0], y)):
        if args.order == "atlas":
            order = np.arange(d_all)
        else:
            strength = np.array([
                max(abs(roc_auc_score(y[tr], vertex[tr, i, 0]) - 0.5),
                    abs(roc_auc_score(y[tr], vertex[tr, i, 1]) - 0.5))
                for i in range(d_all)])
            order = np.argsort(-strength)
        for k in grid:
            reg = np.sort(order[:k])
            pi, pj = np.triu_indices(k, k=1)
            cols = np.array([key[(int(reg[a]), int(reg[b]))]
                             for a, b in zip(pi, pj)])
            v = np.ascontiguousarray(vertex[:, reg])
            fc = np.ascontiguousarray(fc_all[:, cols])
            beta = fit_chunked(v, fc, y, pi, pj, "mmssr", tr)
            _, wbar, _, flat = readouts(v, fc, pi, pj, "mmssr", beta, k,
                                        keep_flat=True)
            out = {
                "SG: wbar": score2(wbar[tr], y[tr], wbar[te], y[te]),
                "SG: flat": score2(flat[tr], y[tr], flat[te], y[te]),
                "LR correlations": score2(fc[tr], y[tr], fc[te], y[te]),
            }
            for key_name, pair in out.items():
                rows[k].setdefault(key_name, []).append(pair)
            del v, fc, flat, wbar, beta
        print(f"  split {split+1}/{args.repeats*args.folds} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        (CACHE / out_name).write_text(json.dumps(
            dict(n=n, grid=grid, order=args.order, splits_done=split + 1,
                 summary={str(k): {kk: dict(
                     auc=float(np.mean([a for a, _ in v])),
                     bacc=float(np.mean([b for _, b in v])))
                     for kk, v in rows[k].items()} for k in grid}), indent=2))

    cols_out = ["SG: wbar", "SG: flat", "LR correlations"]
    print(f"\n{'d':>5}" + "".join(f"{c:>28}" for c in cols_out))
    for k in grid:
        line = f"{k:>5}"
        for c in cols_out:
            a = np.array(rows[k][c])
            line += f"   AUC {a[:, 0].mean():.4f} bacc {100*a[:, 1].mean():.1f}%"
        print(line)
    print(f"\nwrote {CACHE / out_name}")


if __name__ == "__main__":
    main()
