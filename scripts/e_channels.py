"""E1/E2: per-channel decoding of the HCP Emotion ensemble graphs.

Channels
    wbar    mean incident signed weight            (d)    published baseline
    lbar    mean incident edge log-odds            (d)
    absbar  mean incident |w| (magnitude only)     (d)
    alpha   node effects of the two-way fit        (d)
    Wflat   all signed edge weights                (P)
    Lflat   all edge log-odds                      (P)
    Eflat   two-way residual (interaction channel) (P)
    sum_w   single scalar, sum of w
    sum_l   single scalar, sum of log-odds (= pairwise composite LLR)

Splits are subject-safe: a subject contributes one fear and one neut graph, and
both always land in the same fold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).resolve().parent / "cache"
CGRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_FOLDS = 5


def make_clf(p: int, C: float) -> LogisticRegression:
    if p > 5000:  # n << p: dual formulation is much cheaper
        return LogisticRegression(C=C, penalty="l2", dual=True,
                                  solver="liblinear", max_iter=5000)
    return LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000)


def fit_eval(xtr, ytr, xte, yte, C):
    scaler = StandardScaler().fit(xtr)
    clf = make_clf(xtr.shape[1], C).fit(scaler.transform(xtr), ytr)
    s = clf.decision_function(scaler.transform(xte))
    return roc_auc_score(yte, s), float(np.mean((s > 0) == (yte == 1))), s


def eval_channel(x, y, groups, name, tune=True):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    aucs, accs, chosen = [], [], []
    outer = GroupKFold(n_splits=N_FOLDS)
    t0 = time.time()
    for tr, te in outer.split(x, y, groups):
        if tune and x.shape[1] > 1:
            inner, best = GroupKFold(n_splits=3), (-1.0, CGRID[0])
            for C in CGRID:
                sc = []
                for itr, ite in inner.split(x[tr], y[tr], groups[tr]):
                    sc.append(fit_eval(x[tr][itr], y[tr][itr],
                                       x[tr][ite], y[tr][ite], C)[0])
                m = float(np.mean(sc))
                if m > best[0]:
                    best = (m, C)
            C = best[1]
        else:
            C = 1.0
        auc, acc, _ = fit_eval(x[tr], y[tr], x[te], y[te], C)
        aucs.append(auc)
        accs.append(acc)
        chosen.append(C)
    res = dict(channel=name, dim=int(x.shape[1]),
               auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
               acc=float(np.mean(accs)), acc_sd=float(np.std(accs)),
               C=chosen, secs=round(time.time() - t0, 1))
    print(f"{name:>8s}  dim={res['dim']:>6d}  "
          f"AUC={res['auc']:.4f} (sd {res['auc_sd']:.4f})  "
          f"acc={res['acc']:.4f} (sd {res['acc_sd']:.4f})  "
          f"C={chosen}  [{res['secs']}s]", flush=True)
    return res


def eval_scalar(score, y, groups, name):
    """No fitting: a single statistic, orientation fixed on the training fold."""
    aucs, accs = [], []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(score[:, None], y, groups):
        sign = 1.0 if roc_auc_score(y[tr], score[tr]) >= 0.5 else -1.0
        s = sign * score
        thr = np.median(s[tr])
        aucs.append(roc_auc_score(y[te], s[te]))
        accs.append(float(np.mean((s[te] > thr) == (y[te] == 1))))
    res = dict(channel=name, dim=1, auc=float(np.mean(aucs)),
               auc_sd=float(np.std(aucs)), acc=float(np.mean(accs)),
               acc_sd=float(np.std(accs)), C=None, secs=0.0)
    print(f"{name:>8s}  dim={1:>6d}  AUC={res['auc']:.4f} (sd {res['auc_sd']:.4f})  "
          f"acc={res['acc']:.4f} (sd {res['acc_sd']:.4f})  [single statistic]",
          flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="scalars,nodes")
    ap.add_argument("--out", default="results_channels.json")
    args = ap.parse_args()
    want = set(args.channels.split(","))

    nz = np.load(CACHE / "nodes.npz", allow_pickle=True)
    y, subject = nz["y"], nz["subject"]
    groups = np.unique(subject, return_inverse=True)[1]
    print(f"n={len(y)}  classes={np.bincount(y)}  subjects={groups.max()+1}  "
          f"folds={N_FOLDS}\n")

    out = []
    if "scalars" in want:
        out.append(eval_scalar(nz["wsum"].astype(np.float64), y, groups, "sum_w"))
        out.append(eval_scalar(nz["lsum"].astype(np.float64), y, groups, "sum_l"))
    if "nodes" in want:
        for key in ["wbar", "lbar", "absbar", "alpha"]:
            out.append(eval_channel(nz[key], y, groups, key))
    for key, fname in [("Wflat", "W.npy"), ("Lflat", "L.npy"), ("Eflat", "E.npy")]:
        if key in want:
            arr = np.load(CACHE / fname, mmap_mode="r")
            out.append(eval_channel(np.asarray(arr), y, groups, key))
            del arr

    path = CACHE / args.out
    prev = json.loads(path.read_text()) if path.exists() else []
    merged = {r["channel"]: r for r in prev}
    merged.update({r["channel"]: r for r in out})
    path.write_text(json.dumps(list(merged.values()), indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
