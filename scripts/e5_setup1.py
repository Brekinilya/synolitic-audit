"""E5: Setup 1 (Transformer) under the logit view, on the fixed paper splits.

Rebuilds the dense pairwise ensemble from hidden_states.pt (train split only),
then evaluates the same channels as the HCP run. Two predictions are tested:

  1. span: the node-summary channel lbar is an affine image of X, so a linear
     model on lbar cannot beat a linear model on X -- expected AUC ~0.69,
     against the paper's 0.686 (raw X) and 0.699 (flat graph features).
  2. the magnitude channel used in the published Setup 1 (decisiveness) should
     be much weaker than the signed channel, as it is on HCP.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")
sys.path.insert(0, str(REPO / "src"))
CACHE = Path(__file__).resolve().parent / "cache"
SPLIT_TRAIN, SPLIT_TEST = 0, 2


def eval_fixed(xtr, ytr, xte, yte, C=1.0):
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(xtr), ytr)
    s = clf.decision_function(sc.transform(xte))
    return roc_auc_score(yte, s), float(np.mean((s > 0) == (yte == 1)))


def report(name, xtr, ytr, xte, yte):
    if xtr.ndim == 1:
        xtr, xte = xtr[:, None], xte[:, None]
    t0 = time.time()
    auc, acc = eval_fixed(xtr, ytr, xte, yte)
    print(f"{name:>10s}  dim={xtr.shape[1]:>5d}  AUC={auc:.4f}  acc={acc:.4f}"
          f"  [{time.time()-t0:.1f}s]", flush=True)
    return dict(channel=name, dim=int(xtr.shape[1]), auc=auc, acc=acc)


def main() -> None:
    from synolitic.stage2_graphs.build_graphs import fit_pairwise_ensemble

    CACHE.mkdir(exist_ok=True)
    hs = torch.load(REPO / "artifacts/hidden_states.pt", map_location="cpu",
                    weights_only=False)
    X = hs["X"].numpy().astype(np.float64)
    y = hs["is_correct"].numpy().astype(np.int64)
    split = hs["split"].numpy()
    conf = hs["confidence"].numpy().astype(np.float64)
    n, d = X.shape
    tr, te = split == SPLIT_TRAIN, split == SPLIT_TEST
    print(f"X {X.shape}  train {tr.sum()}  test {te.sum()}  "
          f"base acc {y.mean():.4f}\n")

    cache_file = CACHE / "setup1_logits.npy"
    if cache_file.exists():
        L = np.load(cache_file)
        pairs = list(itertools.combinations(range(d), 2))
        print(f"loaded cached logits {L.shape}")
    else:
        t0 = time.time()
        ens = fit_pairwise_ensemble(X[tr], y[tr])
        pairs = ens.pairs
        i_idx = np.array([p[0] for p in pairs])
        j_idx = np.array([p[1] for p in pairs])
        # Edge log-odds, exactly: logit p_ij = w_i x_i + w_j x_j + b.
        L = (X[:, i_idx] * ens.W[:, 0] + X[:, j_idx] * ens.W[:, 1]
             + ens.b).astype(np.float32)
        np.save(cache_file, L)
        print(f"fitted {len(pairs)} pairwise models, logits {L.shape} "
              f"[{time.time()-t0:.1f}s]")

    iu0 = np.array([p[0] for p in pairs])
    iu1 = np.array([p[1] for p in pairs])
    Lf = L.astype(np.float64)

    lsum = np.zeros((n, d))
    for k in range(n):
        lsum[k] = (np.bincount(iu0, weights=Lf[k], minlength=d)
                   + np.bincount(iu1, weights=Lf[k], minlength=d))
    lbar = lsum / (d - 1)
    absbar = np.zeros((n, d))
    A = np.abs(Lf)
    for k in range(n):
        absbar[k] = (np.bincount(iu0, weights=A[k], minlength=d)
                     + np.bincount(iu1, weights=A[k], minlength=d)) / (d - 1)
    del A

    mu = Lf.mean(axis=1)
    alpha = (d - 1) / (d - 2) * (lbar - mu[:, None])
    E = Lf - mu[:, None] - alpha[:, iu0] - alpha[:, iu1]

    # Span check: lbar must be an exact affine image of X.
    lin = LinearRegression().fit(X[tr], lbar[tr])
    pred = lin.predict(X[te])
    ss_res = ((lbar[te] - pred) ** 2).sum(axis=0)
    ss_tot = ((lbar[te] - lbar[te].mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-300)
    print(f"\nspan check  X -> lbar : min R^2 over {d} outputs = {r2.min():.10f}"
          f"  (mean {r2.mean():.10f})")
    lin_e = LinearRegression().fit(X[tr], E[tr][:, :200])
    r2e = 1 - (((E[te][:, :200] - lin_e.predict(X[te])) ** 2).sum(axis=0)
               / ((E[te][:, :200] - E[te][:, :200].mean(axis=0)) ** 2).sum(axis=0))
    print(f"span check  X -> E[:200]: min R^2 = {r2e.min():.10f}\n")

    out = []
    out.append(report("rawX", X[tr], y[tr], X[te], y[te]))
    out.append(report("confidence", conf[tr], y[tr], conf[te], y[te]))
    out.append(report("lbar", lbar[tr], y[tr], lbar[te], y[te]))
    out.append(report("alpha", alpha[tr], y[tr], alpha[te], y[te]))
    out.append(report("absbar", absbar[tr], y[tr], absbar[te], y[te]))
    out.append(report("Lflat", Lf[tr], y[tr], Lf[te], y[te]))
    out.append(report("Eflat", E[tr], y[tr], E[te], y[te]))
    s = Lf.sum(axis=1)
    sgn = 1.0 if roc_auc_score(y[tr], s[tr]) >= 0.5 else -1.0
    print(f"{'sum_l':>10s}  dim={1:>5d}  AUC={roc_auc_score(y[te], sgn*s[te]):.4f}"
          f"  [single statistic]")

    import json
    (CACHE / "results_setup1.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE / 'results_setup1.json'}")


if __name__ == "__main__":
    main()
