"""E14: block-aligned cropping.

Cropping the concatenated series to a common length equalises total duration but
not the position of the block boundaries inside the window. In EMOTION the first
block is 24 frames for almost every neutral record and about 23 for fear, so
block phase alone separates the classes and survives the usual crop.

Block-aligned cropping takes the first K blocks of every record and cuts each to
the same length B before concatenating, which makes the design identical across
records by construction. The channel comparison is rerun on:

    concat   crop the concatenated series        (what the pipeline does now)
    block    block-aligned crop                   (the fix)
    block+fir  block-aligned crop, then removal of the block-locked response
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


def feats(series):
    vertex = np.stack([series.mean(axis=1), series.std(axis=1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fc = np.corrcoef(series)
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
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results_blockcrop.json")
    args = ap.parse_args()

    from fastpair import fit_pairwise, node_means, pairwise_logits
    from synolitic.setup2.data import scan_records, state_labels

    out = {}
    for task in args.tasks.split(","):
        recs = [r for r in scan_records(ROOT, task) if r.direction == "RL"]
        labels = state_labels(recs)
        y = np.array([labels[r.state] for r in recs])
        groups = np.unique(np.array([r.subject for r in recs]),
                           return_inverse=True)[1]
        blocks = [load_blocks(r.path) for r in recs]
        nb = min(len(b) for b in blocks)
        bl = min(x.shape[1] for b in blocks for x in b[:nb])
        d = blocks[0][0].shape[0]

        concat = [np.concatenate(b, axis=1) for b in blocks]
        tmin = min(s.shape[1] for s in concat)
        variants = {
            "concat": [s[:, :tmin] for s in concat],
            "block": [np.concatenate([x[:, :bl] for x in b[:nb]], axis=1)
                      for b in blocks],
        }
        aligned = np.stack([np.stack([x[:, :bl] for x in b[:nb]])
                            for b in blocks])            # (n, nb, d, bl)
        variants["block+fir"] = [np.concatenate(
            list(a - a.mean(axis=0, keepdims=True)), axis=1) for a in aligned]

        iu0, iu1 = np.triu_indices(d, k=1)
        pairs = [(int(a), int(b)) for a, b in zip(iu0, iu1)]
        rows: dict[str, list] = {}
        prepared = {}
        for tag, series in variants.items():
            v, e = zip(*[feats(s) for s in series])
            prepared[tag] = (np.stack(v), np.stack(e))

        for tr, te in GroupKFold(n_splits=args.folds).split(y, y, groups):
            for tag in variants:
                vertex, edge = prepared[tag]
                raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
                fc = edge[:, iu0, iu1]
                rows.setdefault(f"{tag}: raw (m,s)", []).append(
                    score(raw[tr], y[tr], raw[te], y[te]))
                rows.setdefault(f"{tag}: FC", []).append(
                    score(fc[tr], y[tr], fc[te], y[te]))
                beta = fit_pairwise(vertex[tr], edge[tr], y[tr], pairs, "mmssr")
                L = pairwise_logits(vertex, edge, pairs, "mmssr", beta)
                wbar = node_means(np.tanh(L / 2), pairs, d)
                rows.setdefault(f"{tag}: syn full wbar", []).append(
                    score(wbar[tr], y[tr], wbar[te], y[te]))

        print(f"\n=== {task}  d={d} blocks={nb}x{bl} concatT={tmin} "
              f"n={len(y)} ===")
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
