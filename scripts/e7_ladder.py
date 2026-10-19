"""E4/E7: learning curves in n and the dimensionality ladder, honest edges.

For every (fold, training size n, region subset of size d) the pairwise edge
models are refitted on *that* training subsample only, so the synolitic
representation never sees anything the competing baselines do not see. This is
the comparison that the "overcoming the curse of dimensionality" claim needs:
the synolitic advantage, if any, must show up as a gap that grows as n shrinks
or d grows.

Rows compared at every grid point:
    raw (m,s)        linear model on the moments
    FC vec(rho)      linear model on functional connectivity
    raw + FC         linear model on both  <- the baseline that matters
    syn full: wbar   published readout (mean incident weight)
    syn full: lbar   same in log-odds
    syn full: sum_l  unweighted pool, a single number
    syn full: Lflat  all edges
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"
ROOTS = {
    "yeo17": Path(r"C:\Users\breki\Downloads\HCP_v4\yeo17_extracted\split_yeo17_lr"),
    "mmp": Path(r"C:\Users\breki\Downloads\HCP_v4\mmp_extracted"),
}


def score(xtr, ytr, xte, yte, C=1.0):
    """AUC of an L2 logistic model; n << p uses the dual solver."""
    if xtr.ndim == 1:
        xtr, xte = xtr[:, None], xte[:, None]
    sc = StandardScaler().fit(xtr)
    if xtr.shape[1] > 5000:
        clf = LogisticRegression(C=C, dual=True, solver="liblinear",
                                 max_iter=5000)
    else:
        clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(xtr), ytr)
    s = clf.decision_function(sc.transform(xte))
    return float(roc_auc_score(yte, s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcel", default="yeo17")
    ap.add_argument("--task", default="EMOTION")
    ap.add_argument("--direction", default="RL")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--d-ladder", default="")
    ap.add_argument("--region-draws", type=int, default=1)
    ap.add_argument("--train-sizes", default="16,32,64,128,256,0")
    ap.add_argument("--family", default="mmssr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from fastpair import fit_pairwise, node_means, pairwise_logits
    from synolitic.setup2.data import build_task_features, scan_records

    CACHE.mkdir(exist_ok=True)
    root = ROOTS[args.parcel]
    sub = sorted(p for p in root.iterdir() if p.is_dir())
    if sub and not (root / args.task).exists():
        root = sub[0]
    recs = [r for r in scan_records(root, args.task)
            if args.direction in ("ALL", r.direction)]
    t0 = time.time()
    tf = build_task_features(recs)
    vertex_all, edge_all, y = tf.vertex, tf.edge, tf.y
    subject = np.asarray([r.subject for r in recs])
    groups = np.unique(subject, return_inverse=True)[1]
    n, d_all, _ = vertex_all.shape
    print(f"{args.parcel}/{args.task}/{args.direction}: n={n} d={d_all} "
          f"subjects={groups.max()+1} T={tf.crop_timepoints} "
          f"[{time.time()-t0:.1f}s]", flush=True)

    ladder = ([int(x) for x in args.d_ladder.split(",")] if args.d_ladder
              else [d_all])
    sizes = [int(x) for x in args.train_sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for d in ladder:
        for draw in range(args.region_draws if d < d_all else 1):
            regions = (np.sort(rng.choice(d_all, size=d, replace=False))
                       if d < d_all else np.arange(d_all))
            vertex = np.ascontiguousarray(vertex_all[:, regions])
            edge = np.ascontiguousarray(edge_all[:, regions][:, :, regions])
            pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
            iu0 = np.array([p[0] for p in pairs])
            iu1 = np.array([p[1] for p in pairs])
            raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
            fc = edge[:, iu0, iu1]
            rawfc = np.hstack([raw, fc])

            for fold, (tr, te) in enumerate(
                    GroupKFold(n_splits=args.folds).split(raw, y, groups)):
                tr_subj = np.unique(groups[tr])
                for size in sizes:
                    if size and size >= len(tr_subj):
                        continue
                    if size:
                        keep = rng.choice(tr_subj, size=size, replace=False)
                        sel = tr[np.isin(groups[tr], keep)]
                    else:
                        sel = tr
                    tag = dict(d=d, draw=draw, fold=fold,
                               n_subj=int(len(np.unique(groups[sel]))),
                               n_rec=int(len(sel)))
                    ts = time.time()
                    beta = fit_pairwise(vertex[sel], edge[sel], y[sel],
                                        pairs, args.family)
                    L = pairwise_logits(vertex, edge, pairs, args.family,
                                        beta).astype(np.float32)
                    W = np.tanh(L / 2.0).astype(np.float32)
                    lbar = node_means(L, pairs, d)
                    wbar = node_means(W, pairs, d)
                    s = L.sum(axis=1)
                    sgn = 1.0 if roc_auc_score(y[sel], s[sel]) >= 0.5 else -1.0
                    res = {
                        "raw (m,s)": score(raw[sel], y[sel], raw[te], y[te]),
                        "FC vec(rho)": score(fc[sel], y[sel], fc[te], y[te]),
                        "raw + FC": score(rawfc[sel], y[sel], rawfc[te], y[te]),
                        "syn: wbar": score(wbar[sel], y[sel], wbar[te], y[te]),
                        "syn: lbar": score(lbar[sel], y[sel], lbar[te], y[te]),
                        "syn: sum_l": float(roc_auc_score(y[te], sgn * s[te])),
                        "syn: Lflat": score(L[sel], y[sel], L[te], y[te]),
                    }
                    rows.append({**tag, **res, "secs": round(time.time() - ts, 1)})
                    print(f"d={d:>4d} draw={draw} fold={fold} "
                          f"n_subj={tag['n_subj']:>4d}  " +
                          "  ".join(f"{k}={v:.4f}" for k, v in res.items()) +
                          f"  [{rows[-1]['secs']}s]", flush=True)

    name = args.out or f"ladder_{args.parcel}_{args.task}_{args.direction}.json"
    (CACHE / name).write_text(json.dumps(
        dict(parcel=args.parcel, task=args.task, direction=args.direction,
             d_all=int(d_all), n=int(n), crop=int(tf.crop_timepoints),
             family=args.family, rows=rows), indent=2))
    print(f"\nwrote {CACHE / name}")


if __name__ == "__main__":
    main()
