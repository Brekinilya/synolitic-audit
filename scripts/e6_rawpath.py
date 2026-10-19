"""E6: raw-path channel ablation with subject-safe, out-of-fold edge construction.

For one HCP task (yeo17, d=34) this builds three families of supervised pairwise
graphs and compares them with plain baselines under one identical protocol:

    full        pairwise model on (m_i, m_j, s_i, s_j, rho_ij)   -- Vlasenko
    unary       pairwise model on (m_i, m_j, s_i, s_j)           -- no connectivity
    pair        pairwise model on (rho_ij)                       -- connectivity only

Readouts per family: mean incident w (published baseline), mean incident
log-odds, the single-number pool sum(log-odds), all flat edges, and the two-way
interaction residual.

Baselines without any graph: LogReg on the raw moments (m, s), on vectorised
functional connectivity, and on both.

Edge models are fitted on the training fold only. The --leaky run repeats the
construction with edge models fitted on all records (train + test), which is how
the published bundle was produced, so the difference measures how much accuracy
comes from label leakage through edge construction.
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")
sys.path.insert(0, str(REPO / "src"))
CACHE = Path(__file__).resolve().parent / "cache"
ROOT = Path(r"C:\Users\breki\Downloads\HCP_v4\yeo17_extracted\split_yeo17_lr")
FAMILIES = {"full": "mmssr", "unary": "mmss", "pair": "r"}


def pair_features(vertex, edge, i, j, kind):
    cols = []
    if "m" in kind:
        cols += [vertex[:, i, 0], vertex[:, j, 0]]
    if "s" in kind:
        cols += [vertex[:, i, 1], vertex[:, j, 1]]
    if "r" in kind:
        cols += [edge[:, i, j]]
    return np.column_stack(cols)


def fit_edges(vertex, edge, y, pairs, kind):
    """Fit one logistic model per pair; return coefficients and intercepts."""
    coefs, inters = [], []
    for i, j in pairs:
        f = pair_features(vertex, edge, i, j, kind)
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(f, y)
        coefs.append(clf.coef_[0])
        inters.append(clf.intercept_[0])
    return np.asarray(coefs), np.asarray(inters)


def edge_logits(vertex, edge, pairs, kind, coefs, inters):
    out = np.empty((vertex.shape[0], len(pairs)))
    for k, (i, j) in enumerate(pairs):
        f = pair_features(vertex, edge, i, j, kind)
        out[:, k] = f @ coefs[k] + inters[k]
    return out


def decompose(L, pairs, d):
    """L (N,P) edge log-odds -> node effects alpha (N,d) and residual E (N,P)."""
    iu0 = np.array([p[0] for p in pairs])
    iu1 = np.array([p[1] for p in pairs])
    n = L.shape[0]
    lbar = np.empty((n, d))
    for k in range(n):
        lbar[k] = (np.bincount(iu0, weights=L[k], minlength=d)
                   + np.bincount(iu1, weights=L[k], minlength=d)) / (d - 1)
    mu = L.mean(axis=1)
    alpha = (d - 1) / (d - 2) * (lbar - mu[:, None])
    E = L - mu[:, None] - alpha[:, iu0] - alpha[:, iu1]
    return lbar, alpha, E


def score(xtr, ytr, xte, yte, C=1.0):
    if xtr.ndim == 1:
        xtr, xte = xtr[:, None], xte[:, None]
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(xtr), ytr)
    s = clf.decision_function(sc.transform(xte))
    return roc_auc_score(yte, s), float(np.mean((s > 0) == (yte == 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="EMOTION")
    ap.add_argument("--direction", default="RL")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--leaky", action="store_true",
                    help="also fit edge models on train+test (published recipe)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from synolitic.setup2.data import build_task_features, scan_records

    CACHE.mkdir(exist_ok=True)
    recs = [r for r in scan_records(ROOT, args.task)
            if args.direction in ("ALL", r.direction)]
    print(f"{args.task}/{args.direction}: {len(recs)} records, "
          f"{len({r.subject for r in recs})} subjects, "
          f"states {sorted({r.state for r in recs})}")

    t0 = time.time()
    tf = build_task_features(recs, n_timepoints=args.crop)
    vertex, edge, y = tf.vertex, tf.edge, tf.y
    subject = np.asarray([r.subject for r in recs])
    groups = np.unique(subject, return_inverse=True)[1]
    n, d, _ = vertex.shape
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    iu0 = np.array([p[0] for p in pairs])
    iu1 = np.array([p[1] for p in pairs])
    print(f"features: vertex {vertex.shape} edge {edge.shape} "
          f"crop T={tf.crop_timepoints} classes={np.bincount(y)} "
          f"[{time.time()-t0:.1f}s]\n")

    raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
    fc = edge[:, iu0, iu1]
    rows: dict[str, list] = {}

    def add(name, auc, acc):
        rows.setdefault(name, []).append((auc, acc))

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=args.folds)
                                    .split(raw, y, groups)):
        ft = time.time()
        add("raw (m,s)", *score(raw[tr], y[tr], raw[te], y[te]))
        add("FC vec(rho)", *score(fc[tr], y[tr], fc[te], y[te]))
        add("raw + FC", *score(np.hstack([raw, fc])[tr], y[tr],
                               np.hstack([raw, fc])[te], y[te]))

        conds = [("", tr)] + ([("leaky:", np.arange(n))] if args.leaky else [])
        for tag, fit_idx in conds:
            for fam, kind in FAMILIES.items():
                coefs, inters = fit_edges(vertex[fit_idx], edge[fit_idx],
                                          y[fit_idx], pairs, kind)
                L = edge_logits(vertex, edge, pairs, kind, coefs, inters)
                W = np.tanh(L / 2.0)                       # the published w
                lbar, alpha, E = decompose(L, pairs, d)
                wbar = np.empty((n, d))
                for k in range(n):
                    wbar[k] = (np.bincount(iu0, weights=W[k], minlength=d)
                               + np.bincount(iu1, weights=W[k], minlength=d)) / (d - 1)
                s = L.sum(axis=1)
                sgn = 1.0 if roc_auc_score(y[tr], s[tr]) >= 0.5 else -1.0
                add(f"{tag}{fam}: wbar", *score(wbar[tr], y[tr], wbar[te], y[te]))
                add(f"{tag}{fam}: lbar", *score(lbar[tr], y[tr], lbar[te], y[te]))
                add(f"{tag}{fam}: sum_l",
                    roc_auc_score(y[te], sgn * s[te]),
                    float(np.mean((sgn * s[te] > np.median(sgn * s[tr]))
                                  == (y[te] == 1))))
                add(f"{tag}{fam}: Lflat", *score(L[tr], y[tr], L[te], y[te]))
                add(f"{tag}{fam}: Eresid", *score(E[tr], y[tr], E[te], y[te]))
        print(f"fold {fold}: {time.time()-ft:.1f}s", flush=True)

    print(f"\n{'channel':>22s}   AUC (sd)        acc (sd)")
    out = []
    for name, vals in rows.items():
        a = np.array(vals)
        print(f"{name:>22s}   {a[:,0].mean():.4f} ({a[:,0].std():.4f})   "
              f"{a[:,1].mean():.4f} ({a[:,1].std():.4f})")
        out.append(dict(channel=name, auc=float(a[:, 0].mean()),
                        auc_sd=float(a[:, 0].std()), acc=float(a[:, 1].mean()),
                        acc_sd=float(a[:, 1].std())))
    name = args.out or f"results_raw_{args.task}_{args.direction}_T{tf.crop_timepoints}.json"
    (CACHE / name).write_text(json.dumps(
        dict(task=args.task, direction=args.direction, d=d, n=n,
             crop=tf.crop_timepoints, folds=args.folds, rows=out), indent=2))
    print(f"\nwrote {CACHE / name}")


if __name__ == "__main__":
    main()
