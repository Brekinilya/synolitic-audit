"""E11: does the pairwise-QDA graph beat the estimators people actually use?

The earlier run only compared against textbook QDA, which is undefined for
n < d and is a straw man. This adds the high-dimensional discriminant analysis
baselines and a direct sparse interaction model:

    shrinkage QDA     per-class Ledoit-Wolf covariance
    diagonal QDA      per-class variances only
    RDA               Friedman (1989), (lambda, gamma) chosen on a split
    L1 x_i*x_j        sparse logistic regression on all pairwise products

It also answers the objection that a non-linear meta-model could still read
interactions off generalized-linear edges: with centrally symmetric classes the
population optimum of every edge is the constant prior, so an MLP on those
edges has nothing to read. The MLP rows test that empirically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e9_synth import auc, logreg_edges, qda_edges, sample  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def gauss_scores(xtr, ytr, xte, cov_fn):
    """Generic Gaussian discriminant with a pluggable covariance estimator."""
    terms = []
    for c in (0, 1):
        z = xtr[ytr == c]
        mu = z.mean(axis=0)
        cov = cov_fn(z - mu)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 1e-6 * np.eye(cov.shape[0])
            sign, logdet = np.linalg.slogdet(cov)
        u = xte - mu
        sol = np.linalg.solve(cov, u.T).T
        terms.append(-0.5 * (u * sol).sum(axis=1) - 0.5 * logdet
                     + np.log(len(z) / len(xtr)))
    return terms[1] - terms[0]


def cov_ledoit(zc):
    return LedoitWolf(assume_centered=True).fit(zc).covariance_


def cov_diag(zc):
    v = (zc ** 2).mean(axis=0)
    return np.diag(np.maximum(v, 1e-8))


def rda_score(xtr, ytr, xte, lam, gam):
    d = xtr.shape[1]
    pooled = np.zeros((d, d))
    for c in (0, 1):
        z = xtr[ytr == c]
        zc = z - z.mean(axis=0)
        pooled += zc.T @ zc
    pooled /= max(len(xtr) - 2, 1)

    def make(zc):
        s = zc.T @ zc / max(len(zc) - 1, 1)
        s = (1 - lam) * s + lam * pooled
        return (1 - gam) * s + gam * (np.trace(s) / d) * np.eye(d)

    return gauss_scores(xtr, ytr, xte, make)


def rda_tuned(xtr, ytr, xte, rng):
    idx = rng.permutation(len(xtr))
    cut = max(int(0.7 * len(xtr)), 4)
    a, b = idx[:cut], idx[cut:]
    best, best_auc = (0.5, 0.1), -1.0
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        for gam in (0.001, 0.01, 0.1, 0.5):
            if len(np.unique(ytr[a])) < 2 or len(np.unique(ytr[b])) < 2:
                continue
            try:
                s = rda_score(xtr[a], ytr[a], xtr[b], lam, gam)
                v = roc_auc_score(ytr[b], s)
            except np.linalg.LinAlgError:
                continue
            if v > best_auc:
                best_auc, best = v, (lam, gam)
    return rda_score(xtr, ytr, xte, *best)


def products(x, pi, pj):
    return np.hstack([x, x[:, pi] * x[:, pj]])


def mlp_auc(xtr, ytr, xte, yte, seed):
    sc = StandardScaler().fit(xtr)
    clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=2000,
                        random_state=seed)
    clf.fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(xte))[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=40)
    ap.add_argument("--r", type=float, default=0.3)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--sizes", default="50,100,200,400,800")
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results_baselines.json")
    args = ap.parse_args()

    d, k = args.d, args.d // 2
    pi, pj = np.triu_indices(d, k=1)
    print(f"d={d} r={args.r} mu={args.mu} products={pi.size}\n", flush=True)

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        acc: dict[str, list] = {}
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + n)
            xtr, ytr = sample(n, d, k, args.r, args.mu, rng)
            xte, yte = sample(args.n_test, d, k, args.r, args.mu, rng)
            res = {}
            res["LogReg"] = auc(xtr, ytr, xte, yte)
            for tag, fn in [("QDA-shrink", cov_ledoit), ("QDA-diag", cov_diag)]:
                try:
                    res[tag] = float(roc_auc_score(
                        yte, gauss_scores(xtr, ytr, xte, fn)))
                except np.linalg.LinAlgError:
                    res[tag] = float("nan")
            try:
                res["RDA"] = float(roc_auc_score(yte, rda_tuned(xtr, ytr, xte,
                                                                rng)))
            except np.linalg.LinAlgError:
                res["RDA"] = float("nan")
            res["L1 products"] = float(roc_auc_score(
                yte,
                LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                   max_iter=5000)
                .fit(StandardScaler().fit_transform(products(xtr, pi, pj)), ytr)
                .decision_function(
                    StandardScaler().fit(products(xtr, pi, pj))
                    .transform(products(xte, pi, pj)))))

            ltr_lr = logreg_edges(xtr, ytr, xtr, pi, pj)
            lte_lr = logreg_edges(xtr, ytr, xte, pi, pj)
            ltr_q = qda_edges(xtr, ytr, xtr, pi, pj)
            lte_q = qda_edges(xtr, ytr, xte, pi, pj)
            res["syn-LR flat"] = auc(ltr_lr, ytr, lte_lr, yte)
            res["syn-QDA flat"] = auc(ltr_q, ytr, lte_q, yte)
            res["MLP on syn-LR"] = mlp_auc(np.tanh(ltr_lr / 2), ytr,
                                           np.tanh(lte_lr / 2), yte, rep)
            res["MLP on syn-QDA"] = mlp_auc(np.tanh(ltr_q / 2), ytr,
                                            np.tanh(lte_q / 2), yte, rep)
            res["MLP on raw x"] = mlp_auc(xtr, ytr, xte, yte, rep)
            for key, val in res.items():
                acc.setdefault(key, []).append(val)
        means = {key: float(np.nanmean(v)) for key, v in acc.items()}
        sds = {key: float(np.nanstd(v)) for key, v in acc.items()}
        rows.append(dict(n=n, mean=means, sd=sds))
        print(f"n={n:>5d}  " + "  ".join(f"{key}={means[key]:.3f}"
                                         for key in means), flush=True)

    (CACHE / args.out).write_text(json.dumps(
        dict(d=d, r=args.r, mu=args.mu, reps=args.reps, rows=rows), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
