"""E22: is the GAMBLING jump in connectivity a denoising effect?

Removing the block-locked response raised the connectivity channel on GAMBLING
from 0.679 to 0.951, which was reported as "connectivity survives removal of the
evoked response". With two blocks per record that operation is not what it looks
like: the residual of block 1 is minus the residual of block 2, so the
concatenated residual series is [u, -u] and its correlation matrix is the
correlation matrix of the block difference u. That subtraction removes
everything common to both blocks -- the evoked response, slow drift and the
global signal alike.

So the jump may be ordinary denoising rather than evidence about connectivity.
Controls, all label-free:

    plain            correlations as computed in the main analysis
    fir              block-mean removal (the operation in question)
    gsr              the mean over regions regressed out of every region
    pc1              the first principal component removed
    diff             correlations of the explicit block difference

If gsr or pc1 reproduce the jump, the fir result is a denoising effect. The
label-permutation null (labels swapped within each subject's pair) must land at
chance for every variant; anything else would mean leakage.
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
OUT = Path(__file__).resolve().parent.parent / "results"
ROOT = Path(r"C:\Users\breki\Downloads\HCP_v4\yeo17_extracted\split_yeo17_lr")


def load_blocks(path):
    with open(path, "rb") as fh:
        blocks = pickle.load(fh)
    return [np.asarray(b.to_numpy() if hasattr(b, "to_numpy") else b,
                       dtype=np.float64) for b in blocks]


def variants(blocks):
    """Label-free transforms of one record's block list."""
    tmin = min(b.shape[1] for b in blocks)
    stack = np.stack([b[:, :tmin] for b in blocks])        # (B, d, tmin)
    plain = np.concatenate(list(stack), axis=1)

    resid = stack - stack.mean(axis=0, keepdims=True)
    fir = np.concatenate(list(resid), axis=1)

    g = plain.mean(axis=0, keepdims=True)                  # (1, T)
    gc = g - g.mean()
    denom = max(float((gc @ gc.T)[0, 0]), 1e-12)
    beta = (plain - plain.mean(axis=1, keepdims=True)) @ gc.T / denom
    gsr = plain - beta @ gc

    centred = plain - plain.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    pc1 = centred - np.outer(u[:, 0] * s[0], vt[0])

    diff = stack[0] - stack[1] if stack.shape[0] == 2 else resid[0]
    return {"plain": plain, "fir": fir, "gsr": gsr, "pc1": pc1, "diff": diff}


def fc_vector(series, iu0, iu1):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.corrcoef(series), nan=0.0)
    return c[iu0, iu1]


def auc_cv(x, y, groups, folds=5):
    out = []
    for tr, te in GroupKFold(n_splits=folds).split(x, y, groups):
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(x[tr]), y[tr])
        out.append(roc_auc_score(y[te], clf.decision_function(sc.transform(x[te]))))
    return float(np.mean(out)), float(np.std(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="GAMBLING,EMOTION")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--out", default="gambling_null.json")
    args = ap.parse_args()

    from synolitic.setup2.data import scan_records, state_labels

    report = {}
    for task in args.tasks.split(","):
        recs = [r for r in scan_records(ROOT, task) if r.direction == "RL"]
        labels = state_labels(recs)
        y = np.array([labels[r.state] for r in recs])
        subj = np.array([r.subject for r in recs])
        groups = np.unique(subj, return_inverse=True)[1]
        blocks = [load_blocks(r.path) for r in recs]
        nblocks = sorted({len(b) for b in blocks})
        d = blocks[0][0].shape[0]
        iu0, iu1 = np.triu_indices(d, k=1)

        feats = {k: [] for k in ["plain", "fir", "gsr", "pc1", "diff"]}
        for b in blocks:
            v = variants(b)
            for k in feats:
                feats[k].append(fc_vector(v[k], iu0, iu1))
        feats = {k: np.stack(v) for k, v in feats.items()}

        print(f"\n=== {task}  d={d} n={len(y)} blocks per record {nblocks} ===")
        res = {}
        for k, x in feats.items():
            m, s = auc_cv(x, y, groups)
            res[k] = dict(auc=m, sd=s)
            print(f"  {k:>7}: AUC {m:.4f} ({s:.4f})")

        # Label-permutation null: swap the two labels inside every subject.
        rng = np.random.default_rng(0)
        nulls = {k: [] for k in feats}
        for _ in range(args.perms):
            yp = y.copy()
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                if len(idx) == 2 and rng.random() < 0.5:
                    yp[idx] = yp[idx][::-1]
            for k, x in feats.items():
                nulls[k].append(auc_cv(x, yp, groups, folds=3)[0])
        print("  label-permutation null (should sit at 0.5):")
        for k in feats:
            a = np.array(nulls[k])
            res[k]["null_mean"] = float(a.mean())
            res[k]["null_max"] = float(a.max())
            print(f"  {k:>7}: null mean {a.mean():.4f}  max {a.max():.4f}")
        report[task] = res

    OUT.mkdir(exist_ok=True)
    (OUT / args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
