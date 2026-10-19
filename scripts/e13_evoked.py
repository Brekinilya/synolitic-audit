"""E13: does the connectivity channel survive removal of the evoked response?

Cole and colleagues showed that block-locked task activation inflates estimated
task FC. Here the block-locked response is removed inside each record: for every
region the mean across that record's blocks is computed at each within-block
time point and subtracted, which uses no labels at all and therefore cannot leak
the class. Correlations are then recomputed on the residuals and the whole
channel comparison is rerun.

Two design audits run alongside, both label-free by construction:
    structure   block count and block lengths before cropping
    global      mean and std of the signal averaged over all regions
If those decode well, part of what the moments channel reports is design, not
regional physiology.
"""
from __future__ import annotations

import argparse
import json
import pickle
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


def load_blocks(path):
    with open(path, "rb") as fh:
        blocks = pickle.load(fh)
    return [np.asarray(b.to_numpy() if hasattr(b, "to_numpy") else b,
                       dtype=np.float64) for b in blocks]


def fir_residual(blocks):
    """Subtract the across-block mean at each within-block time point."""
    tmin = min(b.shape[1] for b in blocks)
    stack = np.stack([b[:, :tmin] for b in blocks])     # (B, d, tmin)
    resid = stack - stack.mean(axis=0, keepdims=True)
    return np.concatenate(list(resid), axis=1)


def moments_and_fc(series, crop):
    s = series[:, :crop]
    vertex = np.stack([s.mean(axis=1), s.std(axis=1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fc = np.corrcoef(s)
    return vertex, np.nan_to_num(fc, nan=0.0)


def score(xtr, ytr, xte, yte, C=1.0):
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
    ap.add_argument("--tasks", default="EMOTION,GAMBLING,RELATIONAL,WM")
    ap.add_argument("--direction", default="RL")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results_evoked.json")
    args = ap.parse_args()

    from fastpair import fit_pairwise, node_means, pairwise_logits
    from synolitic.setup2.data import scan_records, state_labels

    out = {}
    for task in args.tasks.split(","):
        recs = [r for r in scan_records(ROOT, task)
                if r.direction == args.direction]
        labels = state_labels(recs)
        y = np.array([labels[r.state] for r in recs])
        groups = np.unique(np.array([r.subject for r in recs]),
                           return_inverse=True)[1]

        raw_blocks = [load_blocks(r.path) for r in recs]
        plain = [np.concatenate(b, axis=1) for b in raw_blocks]
        resid = [fir_residual(b) for b in raw_blocks]
        crop_p = min(s.shape[1] for s in plain)
        crop_r = min(s.shape[1] for s in resid)
        d = plain[0].shape[0]

        feats = {}
        for tag, series, crop in [("plain", plain, crop_p),
                                  ("fir", resid, crop_r)]:
            v, e = zip(*[moments_and_fc(s, crop) for s in series])
            feats[tag] = (np.stack(v), np.stack(e))

        nblocks = np.array([[len(b)] + [b[i].shape[1] if i < len(b) else 0
                                        for i in range(4)]
                            for b in raw_blocks], dtype=float)
        gl = np.stack([np.stack([s[:, :crop_p].mean(axis=0).mean(),
                                 s[:, :crop_p].mean(axis=0).std()])
                       for s in plain])

        iu0, iu1 = np.triu_indices(d, k=1)
        pairs = [(int(a), int(b)) for a, b in zip(iu0, iu1)]
        rows: dict[str, list] = {}

        def add(name, val):
            rows.setdefault(name, []).append(val)

        for tr, te in GroupKFold(n_splits=args.folds).split(y, y, groups):
            add("structure (blocks)", score(nblocks[tr], y[tr], nblocks[te],
                                            y[te]))
            add("global signal (m,s)", score(gl[tr], y[tr], gl[te], y[te]))
            for tag in ("plain", "fir"):
                vertex, edge = feats[tag]
                raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
                fc = edge[:, iu0, iu1]
                add(f"{tag}: raw (m,s)", score(raw[tr], y[tr], raw[te], y[te]))
                add(f"{tag}: FC vec(rho)", score(fc[tr], y[tr], fc[te], y[te]))
                beta = fit_pairwise(vertex[tr], edge[tr], y[tr], pairs, "r")
                L = pairwise_logits(vertex, edge, pairs, "r", beta)
                wbar = node_means(np.tanh(L / 2), pairs, d)
                add(f"{tag}: pair-only wbar", score(wbar[tr], y[tr], wbar[te],
                                                    y[te]))
                add(f"{tag}: pair-only flat", score(L[tr], y[tr], L[te], y[te]))

        print(f"\n=== {task} (d={d}, crop plain {crop_p} / fir {crop_r}, "
              f"n={len(y)}) ===")
        res = {}
        for name, vals in rows.items():
            a = np.array(vals)
            res[name] = dict(auc=float(a.mean()), sd=float(a.std()))
            print(f"  {name:>22s}  {a.mean():.4f} ({a.std():.4f})")
        out[task] = res

    (CACHE / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
