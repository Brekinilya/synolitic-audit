"""E25: the dimensionality claim on biomedical tables with p much larger than n.

The n/d ladder so far says something about HCP, where p is a few hundred and n
a few hundred. The claim the method is sold on lives in the opposite corner:
tables with thousands of measurements and tens of samples.

One structural point decides how the comparison has to be set up. A pairwise
ensemble needs C(p, 2) models, which is 25 million at p = 7000, so on such
tables the graph cannot be built on the raw feature set at all -- features have
to be screened first. That screening is part of the method and is done inside
the training fold, never on the whole table. The competitors (L1, elastic net,
RDA, shrinkage QDA) are given the full feature set, which is the comparison
that favours them, and also the screened set, which is the matched one.

Readouts for the graph are the three from the sparse-regime study: ridge on all
edges, L1 on all edges, and the top-m edges screened inside the training fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parent.parent / "results"
CANDIDATES = [("leukemia", None), ("arcene", None), ("colon", None),
              ("DLBCL", None), ("CNS", None), ("SRBCT", None)]


def auc(xtr, ytr, xte, yte, C=1.0, l1=False, elastic=False):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if elastic:
        kw.update(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                  max_iter=3000)
    elif l1:
        kw.update(penalty="l1", solver="liblinear")
    elif xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))


def column_auc(x, y):
    """Univariate AUC of every column at once, via the rank identity."""
    from scipy.stats import rankdata
    ranks = rankdata(x, axis=0)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return np.full(x.shape[1], 0.5)
    return (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def screen(xtr, ytr, m):
    """Indices of the m features that separate best on the training fold."""
    return np.argsort(-np.abs(column_auc(xtr, ytr) - 0.5))[:m]


def load_tables(names):
    from sklearn.datasets import fetch_openml
    out = []
    for name, data_id in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kw = dict(data_id=data_id) if data_id else dict(name=name,
                                                                version=1)
                try:
                    ds = fetch_openml(as_frame=False, **kw)
                    x = np.asarray(ds.data, dtype=np.float64)
                except Exception:
                    # Some archives carry string-typed columns; read them as a
                    # frame and keep the numeric ones.
                    ds = fetch_openml(as_frame=True, **kw)
                    frame = ds.data.apply(lambda col: col.astype(str)
                                          .str.replace(",", ".", regex=False))
                    num = frame.apply(lambda col: np.asarray(
                        [float(v) if v not in ("", "nan", "?") else np.nan
                         for v in col]))
                    x = np.asarray(num.to_numpy(), dtype=np.float64)
            lab = np.asarray(ds.target)
            classes = np.unique(lab)
            if len(classes) != 2 or x.shape[0] < 30:
                print(f"  skip {name}: {len(classes)} classes, "
                      f"{x.shape[0]} rows", flush=True)
                continue
            y = (lab == classes[1]).astype(int)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            keep = x.std(axis=0) > 0
            out.append((name, x[:, keep], y))
            print(f"  {name}: n={x.shape[0]} p={int(keep.sum())} "
                  f"classes={np.bincount(y)}", flush=True)
        except Exception as exc:  # the archive is not always reachable
            print(f"  skip {name}: {type(exc).__name__}: {str(exc)[:80]}",
                  flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="biomed_pn.json")
    args = ap.parse_args()

    from e9_synth import logreg_edges, node_mean, qda_edges

    print("loading tables", flush=True)
    tables = load_tables(CANDIDATES)
    if not tables:
        raise SystemExit("no tables could be loaded")

    report = {}
    for name, x, y in tables:
        d = min(args.screen, x.shape[1])
        pi, pj = np.triu_indices(d, k=1)
        rows: dict[str, list] = {}
        t0 = time.time()
        cv = RepeatedStratifiedKFold(n_splits=args.folds,
                                     n_repeats=args.repeats, random_state=0)
        for tr, te in cv.split(x, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            xtr_full, xte_full = x[tr], x[te]
            res = {
                "L1 all p": auc(xtr_full, y[tr], xte_full, y[te], C=0.1,
                                l1=True),
                "elastic net all p": auc(xtr_full, y[tr], xte_full, y[te],
                                         C=0.1, elastic=True),
                "ridge all p": auc(xtr_full, y[tr], xte_full, y[te]),
            }
            sel = screen(xtr_full, y[tr], d)
            xtr, xte = xtr_full[:, sel], xte_full[:, sel]
            res["ridge screened"] = auc(xtr, y[tr], xte, y[te])
            res["L1 screened"] = auc(xtr, y[tr], xte, y[te], C=0.1, l1=True)
            try:
                from e11_baselines import cov_ledoit, gauss_scores, rda_tuned
                res["QDA-shrink screened"] = float(roc_auc_score(
                    y[te], gauss_scores(xtr, y[tr], xte, cov_ledoit)))
                res["RDA screened"] = float(roc_auc_score(
                    y[te], rda_tuned(xtr, y[tr], xte,
                                     np.random.default_rng(0))))
            except np.linalg.LinAlgError:
                res["QDA-shrink screened"] = float("nan")
                res["RDA screened"] = float("nan")

            ltr = logreg_edges(xtr, y[tr], xtr, pi, pj)
            lte = logreg_edges(xtr, y[tr], xte, pi, pj)
            res["syn-LogReg nodes"] = auc(node_mean(ltr, pi, pj, d), y[tr],
                                          node_mean(lte, pi, pj, d), y[te])
            res["syn-LogReg flat"] = auc(ltr, y[tr], lte, y[te])
            qtr = qda_edges(xtr, y[tr], xtr, pi, pj)
            qte = qda_edges(xtr, y[tr], xte, pi, pj)
            res["syn-QDA flat"] = auc(qtr, y[tr], qte, y[te])
            res["syn-QDA L1"] = auc(qtr, y[tr], qte, y[te], C=0.1, l1=True)
            strength = np.abs(column_auc(qtr, y[tr]) - 0.5)
            keep = np.argsort(-strength)[:args.top]
            res[f"syn-QDA top{args.top}"] = auc(qtr[:, keep], y[tr],
                                                qte[:, keep], y[te])
            for key, val in res.items():
                rows.setdefault(key, []).append(val)

        print(f"\n=== {name}  n={x.shape[0]} p={x.shape[1]} "
              f"screened to d={d}  [{time.time()-t0:.0f}s] ===")
        summary = {}
        for key, vals in rows.items():
            a = np.array(vals, dtype=float)
            summary[key] = dict(auc=float(np.nanmean(a)),
                                sd=float(np.nanstd(a)))
            print(f"  {key:>22}: {np.nanmean(a):.4f} ({np.nanstd(a):.4f})")
        report[name] = dict(n=int(x.shape[0]), p=int(x.shape[1]), d=int(d),
                            summary=summary)

    OUT.mkdir(exist_ok=True)
    (OUT / args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
