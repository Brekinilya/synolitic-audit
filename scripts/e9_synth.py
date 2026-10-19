"""E3: positive control, and the pairwise-QDA fix.

Two Gaussian classes with *identical marginals* and *identical node-level mean
edge weights*: they differ only in which pairs co-vary. Class A correlates
within the blocks {0..k-1} and {k..2k-1}; class B correlates within the even and
the odd indices. Every node has k-1 positively and k negatively correlated
partners in both classes, so

    * the unary channel is null by construction,
    * the node-summary channel is null by construction,
    * all the signal sits in the identity of the correlated pairs.

With generalized-linear pairwise learners the edge log-odds is affine in
(x_i, x_j), so this signal is invisible no matter what graph machinery follows.
Pairwise QDA adds the x_i*x_j term and recovers it; its 2x2 class covariances
come straight out of the two full covariance matrices, so the whole ensemble
costs one covariance per class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).resolve().parent / "cache"
CHUNK = 4096


def correlation(d, k, r, scheme):
    idx = np.arange(d)
    block = (idx >= k).astype(int) if scheme == "A" else (idx % 2)
    s = np.where(block[:, None] == block[None, :], r, -r).astype(float)
    np.fill_diagonal(s, 1.0)
    return s


def sample(n, d, k, r, mu, rng):
    y = rng.integers(0, 2, size=n)
    x = np.empty((n, d))
    chol = {c: np.linalg.cholesky(correlation(d, k, r, "A" if c == 0 else "B"))
            for c in (0, 1)}
    for c in (0, 1):
        m = y == c
        x[m] = rng.standard_normal((m.sum(), d)) @ chol[c].T
    if mu:
        x[y == 1, :k] += mu
    return x, y


def logreg_edges(xtr, ytr, x, pairs_i, pairs_j, C=1.0):
    """Edge log-odds from a logistic model per pair, batched Newton."""
    out = np.empty((x.shape[0], pairs_i.size))
    lam = 1.0 / C
    pen = np.diag([1.0, 1.0, 0.0])
    yv = ytr.astype(float)
    for s in range(0, pairs_i.size, CHUNK):
        e = min(s + CHUNK, pairs_i.size)
        f = np.stack([xtr[:, pairs_i[s:e]], xtr[:, pairs_j[s:e]],
                      np.ones((xtr.shape[0], e - s))], axis=2)
        f = np.ascontiguousarray(np.transpose(f, (1, 0, 2)))
        b = np.zeros((e - s, 3))
        for _ in range(40):
            p = 1.0 / (1.0 + np.exp(-np.clip(np.einsum("knq,kq->kn", f, b),
                                             -500, 500)))
            w = np.clip(p * (1 - p), 1e-10, None)
            g = np.einsum("knq,kn->kq", f, p - yv) + lam * (b @ pen)
            h = np.einsum("kna,kn,knb->kab", f, w, f) + lam * pen
            h[:, [0, 1, 2], [0, 1, 2]] += 1e-9
            step = np.linalg.solve(h, g[:, :, None])[:, :, 0]
            b -= step
            if np.abs(step).max() < 1e-9:
                break
        g2 = np.stack([x[:, pairs_i[s:e]], x[:, pairs_j[s:e]],
                       np.ones((x.shape[0], e - s))], axis=2)
        out[:, s:e] = np.einsum("nkq,kq->nk", g2, b)
    return out


def qda_edges(xtr, ytr, x, pairs_i, pairs_j, reg=1e-3):
    """Edge log-odds from a 2-d QDA per pair, read off the class covariances."""
    mus, covs, logpri = {}, {}, {}
    for c in (0, 1):
        z = xtr[ytr == c]
        mus[c] = z.mean(axis=0)
        covs[c] = np.cov(z, rowvar=False) + reg * np.eye(z.shape[1])
        logpri[c] = np.log(len(z) / len(xtr))
    out = np.empty((x.shape[0], pairs_i.size))
    for s in range(0, pairs_i.size, CHUNK):
        e = min(s + CHUNK, pairs_i.size)
        i, j = pairs_i[s:e], pairs_j[s:e]
        quad = {}
        logdet = {}
        for c in (0, 1):
            a = covs[c][i, i]
            b = covs[c][i, j]
            dd = covs[c][j, j]
            det = np.maximum(a * dd - b * b, 1e-12)
            u = x[:, i] - mus[c][i]
            v = x[:, j] - mus[c][j]
            quad[c] = (dd * u * u - 2 * b * u * v + a * v * v) / det
            logdet[c] = np.log(det)
        out[:, s:e] = (-0.5 * (quad[1] - quad[0])
                       - 0.5 * (logdet[1] - logdet[0])
                       + (logpri[1] - logpri[0]))
    return out


def node_mean(L, pairs_i, pairs_j, d):
    out = np.empty((L.shape[0], d))
    for k in range(L.shape[0]):
        out[k] = (np.bincount(pairs_i, weights=L[k], minlength=d)
                  + np.bincount(pairs_j, weights=L[k], minlength=d)) / (d - 1)
    return out


def auc(xtr, ytr, xte, yte, C=1.0):
    if xtr.ndim == 1:
        xtr, xte = xtr[:, None], xte[:, None]
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=40)
    ap.add_argument("--r", type=float, default=0.3)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--sizes", default="50,100,200,400,800,1600")
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results_synth.json")
    args = ap.parse_args()

    d, k = args.d, args.d // 2
    pi, pj = np.triu_indices(d, k=1)
    for scheme in ("A", "B"):
        ev = np.linalg.eigvalsh(correlation(d, k, args.r, scheme))
        print(f"corr {scheme}: eigenvalues {ev.min():.4f}..{ev.max():.4f}")
    print(f"d={d} k={k} r={args.r} mu={args.mu} P={pi.size}\n")

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        acc: dict[str, list] = {}
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + n)
            xtr, ytr = sample(n, d, k, args.r, args.mu, rng)
            xte, yte = sample(args.n_test, d, k, args.r, args.mu, rng)
            res = {"raw LogReg": auc(xtr, ytr, xte, yte)}
            try:
                q = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(xtr, ytr)
                res["raw QDA (full d)"] = float(
                    roc_auc_score(yte, q.decision_function(xte)))
            except Exception:
                res["raw QDA (full d)"] = float("nan")
            for tag, fn in [("syn-LogReg", logreg_edges), ("syn-QDA", qda_edges)]:
                ltr = fn(xtr, ytr, xtr, pi, pj)
                lte = fn(xtr, ytr, xte, pi, pj)
                res[f"{tag}: lbar"] = auc(node_mean(ltr, pi, pj, d), ytr,
                                          node_mean(lte, pi, pj, d), yte)
                res[f"{tag}: flat"] = auc(ltr, ytr, lte, yte)
                s_tr, s_te = ltr.sum(axis=1), lte.sum(axis=1)
                sgn = 1.0 if roc_auc_score(ytr, s_tr) >= 0.5 else -1.0
                res[f"{tag}: pool"] = float(roc_auc_score(yte, sgn * s_te))
            for key, val in res.items():
                acc.setdefault(key, []).append(val)
        means = {key: float(np.mean(v)) for key, v in acc.items()}
        sds = {key: float(np.std(v)) for key, v in acc.items()}
        rows.append(dict(n=n, mean=means, sd=sds))
        print(f"n={n:>5d}  " + "  ".join(
            f"{key}={means[key]:.3f}" for key in means), flush=True)

    (CACHE / args.out).write_text(json.dumps(
        dict(d=d, r=args.r, mu=args.mu, reps=args.reps, rows=rows), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
