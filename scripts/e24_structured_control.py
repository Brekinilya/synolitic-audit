"""E24: supervision, or just the pairwise shape of the feature map?

The first control compared the fitted edges against a dense Gaussian embedding
of the same width. That is not a matched comparison: an edge reads exactly two
of the d features, a dense projection reads all of them. So the dense control
cannot separate "the edges were fitted on the labels" from "the feature map is
pairwise and sparse".

This replaces it with structured controls that keep the pairwise shape and drop
only the supervision:

    struct-linear     per pair, a x_i + b x_j + c with random coefficients
    struct-quad       per pair, a random quadratic form in (x_i, x_j), the same
                      functional class a two-dimensional QDA produces
    dense-linear      the original dense Gaussian embedding, for reference

Each map is read three ways -- ridge on everything, L1 on everything, and the
top-m edges screened on the training fold -- because the sparse regime showed
the readout matters as much as the map.

Also reported: the norm of the fitted edge coefficients against n. Under the
impossibility result the population optimum is zero, so with centrally
symmetric classes that norm should shrink as the sample grows, and whatever a
meta-model reads off such edges has to come from the scale-free direction that
standardisation preserves.
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


def auc(xtr, ytr, xte, yte, C=1.0, l1=False):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if l1:
        kw.update(penalty="l1", solver="liblinear")
    elif xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))


def readouts(name, ftr, fte, ytr, yte, res, top=20):
    res[f"{name} ridge"] = auc(ftr, ytr, fte, yte)
    res[f"{name} L1"] = auc(ftr, ytr, fte, yte, C=0.1, l1=True)
    strength = np.array([abs(roc_auc_score(ytr, ftr[:, c]) - 0.5)
                         for c in range(ftr.shape[1])])
    keep = np.argsort(-strength)[:top]
    res[f"{name} top{top}"] = auc(ftr[:, keep], ytr, fte[:, keep], yte)


def _restored(rng, state):
    """The same generator rewound, so a map draws identical coefficients."""
    rng.bit_generator.state = state
    return rng


def struct_linear(x, pi, pj, rng):
    a, b, c = (rng.standard_normal(pi.size) for _ in range(3))
    return x[:, pi] * a + x[:, pj] * b + c


def struct_quad(x, pi, pj, rng):
    a, b, g, dd, e, f = (rng.standard_normal(pi.size) for _ in range(6))
    xi, xj = x[:, pi], x[:, pj]
    return (a * xi * xi + b * xj * xj + g * xi * xj + dd * xi + e * xj + f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="sparse", choices=["sparse", "global"])
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r", type=float, default=0.6)
    ap.add_argument("--sizes", default="100,200,400,800")
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_name = args.out or f"structured_control_{args.regime}_d{args.d}.json"

    from e9_synth import logreg_edges, qda_edges
    from e9_synth import sample as sample_global
    from e23_sparse_pairs import sample_sparse

    d, k = args.d, args.k
    pi, pj = np.triu_indices(d, k=1)
    print(f"regime={args.regime} d={d} pairs={pi.size}", flush=True)

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
                xte, yte = sample_global(args.n_test, d, d // 2, args.r, 0.0,
                                         rng)
            res = {"raw x ridge": auc(xtr, ytr, xte, yte)}

            ltr = logreg_edges(xtr, ytr, xtr, pi, pj)
            lte = logreg_edges(xtr, ytr, xte, pi, pj)
            readouts("syn-LogReg", ltr, lte, ytr, yte, res)
            qtr = qda_edges(xtr, ytr, xtr, pi, pj)
            qte = qda_edges(xtr, ytr, xte, pi, pj)
            readouts("syn-QDA", qtr, qte, ytr, yte, res)

            # One draw of coefficients per repetition, applied unchanged to
            # train and test, so the map is fixed before any data is seen.
            lin_coef = np.random.default_rng(90000 + rep)
            lin_state = lin_coef.bit_generator.state
            readouts("struct-linear",
                     struct_linear(xtr, pi, pj, lin_coef),
                     struct_linear(xte, pi, pj,
                                   _restored(lin_coef, lin_state)),
                     ytr, yte, res)
            quad_coef = np.random.default_rng(80000 + rep)
            quad_state = quad_coef.bit_generator.state
            readouts("struct-quad",
                     struct_quad(xtr, pi, pj, quad_coef),
                     struct_quad(xte, pi, pj,
                                 _restored(quad_coef, quad_state)),
                     ytr, yte, res)
            proj = np.random.default_rng(70000 + rep).standard_normal(
                (d, pi.size)) / np.sqrt(d)
            readouts("dense-linear", xtr @ proj, xte @ proj, ytr, yte, res)

            # Scale of the fitted edge coefficients, excluding the intercept.
            coef = np.empty((pi.size, 2))
            step = 4096
            for s in range(0, pi.size, step):
                e = min(s + step, pi.size)
                sub = slice(s, e)
                f = np.stack([xtr[:, pi[sub]], xtr[:, pj[sub]],
                              np.ones((n, e - s))], axis=2)
                f = np.ascontiguousarray(np.transpose(f, (1, 0, 2)))
                b = np.zeros((e - s, 3))
                for _ in range(40):
                    p = 1 / (1 + np.exp(-np.clip(
                        np.einsum("knq,kq->kn", f, b), -500, 500)))
                    w = np.clip(p * (1 - p), 1e-10, None)
                    g = np.einsum("knq,kn->kq", f, p - ytr) \
                        + b @ np.diag([1.0, 1.0, 0.0])
                    h = np.einsum("kna,kn,knb->kab", f, w, f) \
                        + np.diag([1.0, 1.0, 1e-9])
                    st = np.linalg.solve(h, g[:, :, None])[:, :, 0]
                    b -= st
                    if np.abs(st).max() < 1e-9:
                        break
                coef[sub] = b[:, :2]
            res["mean |edge slope|"] = float(np.abs(coef).mean())

            for key, val in res.items():
                acc.setdefault(key, []).append(val)
        means = {key: float(np.nanmean(v)) for key, v in acc.items()}
        rows.append(dict(n=n, mean=means,
                         sd={key: float(np.nanstd(v)) for key, v in acc.items()}))
        print(f"\nn={n}  [{time.time()-t0:.0f}s]")
        for key in means:
            print(f"   {key:>22}: {means[key]:.4f}")

    OUT.mkdir(exist_ok=True)
    (OUT / out_name).write_text(json.dumps(
        dict(regime=args.regime, d=d, k=k, r=args.r, reps=args.reps,
             rows=rows), indent=2))
    print(f"\nwrote {OUT / out_name}")


if __name__ == "__main__":
    main()
