"""E15: the same diagnostics on ABIDE, where there is no task-activation channel.

ABIDE is resting state, so the moments channel carries no evoked response and
the discriminative information, if any, should sit in connectivity. This is the
setting where the published synolitic result is weakest (55.6 per cent balanced
accuracy against about 67 for correlation methods after residualisation), so it
is the place where a fix can show itself.

Channels: raw moments, functional connectivity, and supervised pairwise graphs
with generalized-linear and with QDA edges, each read both by node summaries and
by the flat edge vector -- the node-summary step is itself a candidate culprit.

Site is the dominant nuisance in ABIDE, so folds are grouped by site
(leave-sites-out) in addition to the plain subject-level split.
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"
DATA = Path("C:/Users/breki/abide_data")


def score(xtr, ytr, xte, yte, C=1.0):
    if xtr.ndim == 1:
        xtr, xte = xtr[:, None], xte[:, None]
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    s = clf.decision_function(sc.transform(xte))
    return float(roc_auc_score(yte, s)), float(np.mean((s > 0) == (yte == 1)))


def balanced_acc(yte, s):
    pred = s > 0
    out = []
    for c in (0, 1):
        m = yte == c
        if m.sum():
            out.append(float(np.mean(pred[m] == (c == 1))))
    return float(np.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derivative", default="rois_cc200")
    ap.add_argument("--crop", type=int, default=116)
    ap.add_argument("--max-regions", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results_abide.json")
    args = ap.parse_args()

    from nilearn.datasets import fetch_abide_pcp
    from e10_qda import qda_edges
    from fastpair import fit_pairwise, node_means, pairwise_logits

    DATA.mkdir(exist_ok=True)
    print("fetching ABIDE ...", flush=True)
    ab = None
    for attempt in range(1, 21):
        try:
            ab = fetch_abide_pcp(data_dir=str(DATA), pipeline="cpac",
                                 band_pass_filtering=True,
                                 global_signal_regression=False,
                                 derivatives=[args.derivative],
                                 quality_checked=True, verbose=0)
            break
        except Exception as exc:  # the S3 endpoint drops connections often
            print(f"  attempt {attempt} failed: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            time.sleep(5)
    if ab is None:
        raise SystemExit("could not fetch ABIDE after repeated attempts")
    series = ab[args.derivative]
    pheno = ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])
    site = np.asarray([str(s) for s in pheno["SITE_ID"]])
    lens = np.asarray([s.shape[0] for s in series])
    print(f"n={len(series)} classes={np.bincount(y)} sites={len(set(site))} "
          f"T: min={lens.min()} median={int(np.median(lens))} max={lens.max()}",
          flush=True)

    keep = lens >= args.crop
    series = [s for s, k in zip(series, keep) if k]
    y, site = y[keep], site[keep]
    print(f"after T>={args.crop}: n={len(series)} classes={np.bincount(y)} "
          f"sites={len(set(site))}", flush=True)

    d_full = series[0].shape[1]
    rng = np.random.default_rng(0)
    reg = (np.sort(rng.choice(d_full, args.max_regions, replace=False))
           if args.max_regions and args.max_regions < d_full
           else np.arange(d_full))
    d = len(reg)

    vertex = np.empty((len(series), d, 2))
    edge = np.empty((len(series), d, d))
    for k, s in enumerate(series):
        z = np.asarray(s[:args.crop, reg], dtype=np.float64).T
        vertex[k, :, 0] = z.mean(axis=1)
        vertex[k, :, 1] = z.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            edge[k] = np.nan_to_num(np.corrcoef(z), nan=0.0)
    iu0, iu1 = np.triu_indices(d, k=1)
    pairs = [(int(a), int(b)) for a, b in zip(iu0, iu1)]
    raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
    fc = edge[:, iu0, iu1]
    print(f"features: d={d} P={len(pairs)} crop={args.crop}", flush=True)

    schemes = {
        "site-grouped": list(GroupKFold(n_splits=args.folds).split(
            raw, y, np.unique(site, return_inverse=True)[1])),
        "subject-random": list(StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=0).split(raw, y)),
    }

    out = {}
    for scheme, splits in schemes.items():
        rows: dict[str, list] = {}

        def add(name, pair):
            rows.setdefault(name, []).append(pair)

        for tr, te in splits:
            add("raw (m,s)", score(raw[tr], y[tr], raw[te], y[te]))
            add("FC vec(rho)", score(fc[tr], y[tr], fc[te], y[te]))
            add("raw + FC", score(np.hstack([raw, fc])[tr], y[tr],
                                  np.hstack([raw, fc])[te], y[te]))
            for fam, kind in [("full", "mmssr"), ("pair", "r")]:
                beta = fit_pairwise(vertex[tr], edge[tr], y[tr], pairs, kind)
                L = pairwise_logits(vertex, edge, pairs, kind, beta)
                wbar = node_means(np.tanh(L / 2), pairs, d)
                add(f"syn {fam}: wbar", score(wbar[tr], y[tr], wbar[te], y[te]))
                add(f"syn {fam}: flat", score(L[tr], y[tr], L[te], y[te]))
            lq = qda_edges(vertex, fc, y, tr, np.asarray(iu0), np.asarray(iu1))
            wq = node_means(np.tanh(np.asarray(lq, dtype=np.float64) / 2),
                            pairs, d)
            add("syn QDA: wbar", score(wq[tr], y[tr], wq[te], y[te]))
            add("syn QDA: flat", score(lq[tr], y[tr], lq[te], y[te]))
        print(f"\n=== {scheme} ===")
        res = {}
        for name, vals in rows.items():
            a = np.array(vals)
            res[name] = dict(auc=float(a[:, 0].mean()), auc_sd=float(a[:, 0].std()),
                             acc=float(a[:, 1].mean()), acc_sd=float(a[:, 1].std()))
            print(f"  {name:>16s}  AUC {a[:,0].mean():.4f} ({a[:,0].std():.4f})"
                  f"   acc {a[:,1].mean():.4f} ({a[:,1].std():.4f})")
        out[scheme] = res

    (CACHE / args.out).write_text(json.dumps(
        dict(derivative=args.derivative, crop=args.crop, d=d, n=len(series),
             results=out), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
