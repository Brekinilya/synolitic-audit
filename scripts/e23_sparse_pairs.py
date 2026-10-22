"""E23: the regime where pairwise screening should win.

The earlier synthetic control put the whole signal in a global partition of the
features, which is the worst case for a pairwise decomposition: no single pair
is informative on its own, and a full-covariance estimator sees the structure
directly. Unsurprisingly RDA and Ledoit-Wolf QDA beat the pairwise graph there
at every sample size.

The opposite regime is the fair one. Here both classes have identical marginals
and both are Gaussian, but under class 1 a handful of disjoint feature pairs
become correlated while everything else stays independent. The signal is sparse
and strictly local: each informative pair carries it on its own, and a full
covariance has to be estimated to find k off-diagonal entries among d(d-1)/2.
If the pairwise construction has a regime of its own, this is it.

Compared, all under the same split:

    LogReg x              linear model on the features
    QDA-diag              per-class variances only, blind to correlation
    QDA-shrink            per-class Ledoit-Wolf covariance
    RDA                   Friedman (1989), lambda and gamma on a held-out slice
    L1 products           sparse logistic regression on all x_i * x_j
    syn-LogReg flat       the published edge construction
    syn-QDA flat / pool   pairwise QDA edges, flat and as an unweighted pool
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parent.parent / "results"


def sample_sparse(n, d, k, r, rng):
    """Class 1 correlates k disjoint pairs; class 0 is independent noise.

    Marginals are identical: every feature is standard normal in both classes.
    """
    y = rng.integers(0, 2, size=n)
    x = rng.standard_normal((n, d))
    pairs = [(2 * t, 2 * t + 1) for t in range(k)]
    one = y == 1
    for i, j in pairs:
        z1 = x[one, i]
        z2 = x[one, j]
        x[one, j] = r * z1 + np.sqrt(max(1.0 - r * r, 0.0)) * z2
    return x, y, pairs


def auc(xtr, ytr, xte, yte, C=1.0, l1=False):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if l1:
        kw.update(penalty="l1", solver="liblinear")
    elif xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r", type=float, default=0.6)
    ap.add_argument("--sizes", default="50,100,200,400,800")
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--regime", default="sparse",
                    choices=["sparse", "global"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_name = args.out or (f"{args.regime}_pairs_d{args.d}_"
                            f"k{args.k}.json")

    from e9_synth import logreg_edges, node_mean, qda_edges
    from e9_synth import sample as sample_global
    from e11_baselines import cov_diag, cov_ledoit, gauss_scores, rda_tuned

    d, k = args.d, args.k
    pi, pj = np.triu_indices(d, k=1)
    print(f"d={d} k={k} informative pairs of {pi.size} r={args.r}", flush=True)

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        acc: dict[str, list] = {}
        t0 = time.time()
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + n)
            if args.regime == "sparse":
                xtr, ytr, _ = sample_sparse(n, d, k, args.r, rng)
                xte, yte, _ = sample_sparse(args.n_test, d, k, args.r, rng)
            else:
                xtr, ytr = sample_global(n, d, d // 2, args.r, 0.0, rng)
                xte, yte = sample_global(args.n_test, d, d // 2, args.r,
                                         0.0, rng)
            res = {"LogReg x": auc(xtr, ytr, xte, yte)}
            for tag, fn in [("QDA-diag", cov_diag), ("QDA-shrink", cov_ledoit)]:
                try:
                    res[tag] = float(roc_auc_score(
                        yte, gauss_scores(xtr, ytr, xte, fn)))
                except np.linalg.LinAlgError:
                    res[tag] = float("nan")
            try:
                res["RDA"] = float(roc_auc_score(
                    yte, rda_tuned(xtr, ytr, xte, rng)))
            except np.linalg.LinAlgError:
                res["RDA"] = float("nan")
            prod_tr = np.hstack([xtr, xtr[:, pi] * xtr[:, pj]])
            prod_te = np.hstack([xte, xte[:, pi] * xte[:, pj]])
            res["L1 products"] = auc(prod_tr, ytr, prod_te, yte, C=0.1, l1=True)

            true_cols = [2 * t for t in range(k)]
            if args.regime == "sparse":
                res["oracle k products"] = auc(
                    (xtr[:, true_cols]
                     * xtr[:, [c + 1 for c in true_cols]]), ytr,
                    (xte[:, true_cols]
                     * xte[:, [c + 1 for c in true_cols]]), yte)

            ltr = logreg_edges(xtr, ytr, xtr, pi, pj)
            lte = logreg_edges(xtr, ytr, xte, pi, pj)
            res["syn-LogReg flat"] = auc(ltr, ytr, lte, yte)
            res["syn-LogReg L1"] = auc(ltr, ytr, lte, yte, C=0.1, l1=True)
            qtr = qda_edges(xtr, ytr, xtr, pi, pj)
            qte = qda_edges(xtr, ytr, xte, pi, pj)
            res["syn-QDA flat"] = auc(qtr, ytr, qte, yte)
            res["syn-QDA L1"] = auc(qtr, ytr, qte, yte, C=0.1, l1=True)
            res["syn-QDA nodes"] = auc(node_mean(qtr, pi, pj, d), ytr,
                                       node_mean(qte, pi, pj, d), yte)
            s_tr, s_te = qtr.sum(axis=1), qte.sum(axis=1)
            sgn = 1.0 if roc_auc_score(ytr, s_tr) >= 0.5 else -1.0
            res["syn-QDA pool"] = float(roc_auc_score(yte, sgn * s_te))

            # Screening: keep the edges that look informative on the training
            # fold alone, then fit on those. This is how a pairwise ensemble
            # would actually be used when the signal is sparse.
            strength = np.array([abs(roc_auc_score(ytr, qtr[:, c]) - 0.5)
                                 for c in range(qtr.shape[1])])
            for m in (20, 100):
                keep = np.argsort(-strength)[:m]
                res[f"syn-QDA top{m}"] = auc(qtr[:, keep], ytr,
                                             qte[:, keep], yte)
            if args.regime == "sparse":
                hit = len(set(np.argsort(-strength)[:k].tolist())
                          & {int(np.where((pi == a) & (pj == a + 1))[0][0])
                             for a in true_cols})
                res[f"top{k} edges recovered"] = hit / k
            for key, val in res.items():
                acc.setdefault(key, []).append(val)
        means = {key: float(np.nanmean(v)) for key, v in acc.items()}
        sds = {key: float(np.nanstd(v)) for key, v in acc.items()}
        rows.append(dict(n=n, mean=means, sd=sds))
        print(f"n={n:>5d}  " + "  ".join(f"{key}={means[key]:.3f}"
                                         for key in means)
              + f"  [{time.time()-t0:.0f}s]", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / out_name).write_text(json.dumps(
        dict(d=d, k=k, r=args.r, reps=args.reps, rows=rows), indent=2))
    print(f"\nwrote {OUT / out_name}")


if __name__ == "__main__":
    main()
