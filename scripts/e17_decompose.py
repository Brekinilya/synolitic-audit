"""E17: why node averaging keeps activation and discards connectivity.

The node summary of the edge log-odds splits exactly into two parts:

    lbar_i = (1/(d-1)) * sum_j [ a_ij m_i + b_ij m_j + c_ij s_i + e_ij s_j + k_ij ]
           + (1/(d-1)) * sum_j [ g_ij rho_ij ]

The moments of node i appear in all d-1 incident edges and add up coherently,
while each correlation appears in exactly one edge and is divided by d-1. So the
same averaging step that preserves an activation signal attenuates a
connectivity signal by a factor that grows with the number of regions.

This script measures both parts separately: how much each carries on its own,
and how much the pair part shrinks relative to the per-edge quantity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"
ROOT = Path(r"C:\Users\breki\Downloads\HCP_v4\yeo17_extracted\split_yeo17_lr")


def score(xtr, ytr, xte, yte, C=1.0):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))


def split_parts(vertex, edge, pairs, beta, d):
    """Node summaries of the unary part and of the correlation part."""
    pi = np.asarray([p[0] for p in pairs])
    pj = np.asarray([p[1] for p in pairs])
    n = vertex.shape[0]
    unary_edge = (vertex[:, pi, 0] * beta[:, 0] + vertex[:, pj, 0] * beta[:, 1]
                  + vertex[:, pi, 1] * beta[:, 2] + vertex[:, pj, 1] * beta[:, 3]
                  + beta[:, 5])
    pair_edge = edge[:, pi, pj] * beta[:, 4]
    ub = np.empty((n, d))
    pb = np.empty((n, d))
    for k in range(n):
        for src, dst in ((unary_edge, ub), (pair_edge, pb)):
            dst[k] = (np.bincount(pi, weights=src[k], minlength=d)
                      + np.bincount(pj, weights=src[k], minlength=d)) / (d - 1)
    return ub, pb, unary_edge, pair_edge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="EMOTION,GAMBLING,WM")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results_decompose.json")
    args = ap.parse_args()

    from fastpair import fit_pairwise
    from synolitic.setup2.data import build_task_features, scan_records

    out = {}
    for task in args.tasks.split(","):
        recs = [r for r in scan_records(ROOT, task) if r.direction == "RL"]
        tf = build_task_features(recs)
        vertex, edge, y = tf.vertex, tf.edge, tf.y
        groups = np.unique(np.array([r.subject for r in recs]),
                           return_inverse=True)[1]
        d = vertex.shape[1]
        pi, pj = np.triu_indices(d, k=1)
        pairs = [(int(a), int(b)) for a, b in zip(pi, pj)]

        rows: dict[str, list] = {}
        for tr, te in GroupKFold(n_splits=args.folds).split(y, y, groups):
            beta = fit_pairwise(vertex[tr], edge[tr], y[tr], pairs, "mmssr")
            ub, pb, ue, pe = split_parts(vertex, edge, pairs, beta, d)
            rows.setdefault("lbar: unary part", []).append(
                score(ub[tr], y[tr], ub[te], y[te]))
            rows.setdefault("lbar: pair part", []).append(
                score(pb[tr], y[tr], pb[te], y[te]))
            rows.setdefault("per-edge: pair part", []).append(
                score(pe[tr], y[tr], pe[te], y[te]))
            rows.setdefault("per-edge: unary part", []).append(
                score(ue[tr], y[tr], ue[te], y[te]))
            rows.setdefault("attenuation sd(pair node)/sd(pair edge)",
                            []).append(float(pb.std() / max(pe.std(), 1e-12)))
            rows.setdefault("attenuation sd(unary node)/sd(unary edge)",
                            []).append(float(ub.std() / max(ue.std(), 1e-12)))
        print(f"\n=== {task}  d={d}  1/sqrt(d-1)={1/np.sqrt(d-1):.4f} ===")
        res = {}
        for name, vals in rows.items():
            a = np.array(vals)
            res[name] = dict(mean=float(a.mean()), sd=float(a.std()))
            print(f"  {name:>40}  {a.mean():.4f} ({a.std():.4f})")
        out[task] = res

    (CACHE / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
