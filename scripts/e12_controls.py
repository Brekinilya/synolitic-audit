"""E12: why does an MLP on generalized-linear edges work at all?

The population statement is that every such edge collapses to the constant
prior under centrally symmetric classes. In a finite sample the fitted
coefficients are not zero, so each edge is a squashed noisy linear projection
of (x_i, x_j) and the graph becomes a non-linear embedding of x. The question
is whether anything in it beats an arbitrary embedding of the same width.

Controls, all read by the same MLP:
    raw x                       the features themselves
    syn-LR edges (logit)        pre-squash edge scores
    syn-LR edges (w)            the published edge weight, tanh(l/2)
    random linear embedding     a fixed Gaussian map of x to the same width
    random squashed embedding   tanh of that map, i.e. the same non-linearity
    syn-LR node summaries       the published readout

Also reports how close the edge weights are to an exact affine image of x.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e9_synth import auc, logreg_edges, node_mean, sample  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def mlp_auc(xtr, ytr, xte, yte, seed):
    sc = StandardScaler().fit(xtr)
    clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=2000,
                        random_state=seed)
    clf.fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(xte))[:, 1]))


def affine_r2(xtr, xte, ftr, fte):
    """R^2 of predicting edge features from x with a linear map."""
    pred = LinearRegression().fit(xtr, ftr).predict(xte)
    ss_res = ((fte - pred) ** 2).sum(axis=0)
    ss_tot = ((fte - fte.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-300)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=40)
    ap.add_argument("--r", type=float, default=0.3)
    ap.add_argument("--sizes", default="50,100,200,400,800")
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results_controls.json")
    args = ap.parse_args()

    d, k = args.d, args.d // 2
    pi, pj = np.triu_indices(d, k=1)
    width = pi.size
    print(f"d={d} r={args.r} width={width}\n", flush=True)

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        acc: dict[str, list] = {}
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + n)
            xtr, ytr = sample(n, d, k, args.r, 0.0, rng)
            xte, yte = sample(args.n_test, d, k, args.r, 0.0, rng)

            ltr = logreg_edges(xtr, ytr, xtr, pi, pj)
            lte = logreg_edges(xtr, ytr, xte, pi, pj)
            wtr, wte = np.tanh(ltr / 2), np.tanh(lte / 2)

            a = rng.standard_normal((d, width)) / np.sqrt(d)
            b = rng.standard_normal(width)
            rtr, rte = xtr @ a + b, xte @ a + b

            res = {
                "MLP raw x": mlp_auc(xtr, ytr, xte, yte, rep),
                "MLP syn-LR logit": mlp_auc(ltr, ytr, lte, yte, rep),
                "MLP syn-LR w": mlp_auc(wtr, ytr, wte, yte, rep),
                "MLP rand linear": mlp_auc(rtr, ytr, rte, yte, rep),
                "MLP rand tanh": mlp_auc(np.tanh(rtr / 2), ytr,
                                         np.tanh(rte / 2), yte, rep),
                "MLP syn-LR nodes": mlp_auc(node_mean(ltr, pi, pj, d), ytr,
                                            node_mean(lte, pi, pj, d), yte, rep),
                "LogReg syn-LR w": auc(wtr, ytr, wte, yte),
                "R2(x -> w)": affine_r2(xtr, xte, wtr, wte),
                "R2(x -> logit)": affine_r2(xtr, xte, ltr, lte),
            }
            for key, val in res.items():
                acc.setdefault(key, []).append(val)
        means = {key: float(np.nanmean(v)) for key, v in acc.items()}
        rows.append(dict(n=n, mean=means,
                         sd={key: float(np.nanstd(v)) for key, v in acc.items()}))
        print(f"n={n:>5d}  " + "  ".join(f"{key}={means[key]:.3f}"
                                         for key in means), flush=True)

    (CACHE / args.out).write_text(json.dumps(
        dict(d=d, r=args.r, reps=args.reps, rows=rows), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
