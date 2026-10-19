"""E20: is the weak channel motion, or site identity wearing motion's clothes?

ABIDE sites differ in scanner, in the age and motion of their participants, and
in their ASD/TC ratio, so in a stratified split that does not group by site any
channel sitting near 57 per cent may be partly reading site. This separates the
two:

    site -> diagnosis          what site identity alone buys in the same 10x5
    corr(sigma, FD) within     the association after site is projected out,
                               and site by site
    LR on sigma, site removed  the std channel once site is regressed out

It also checks the component that disagrees most with the published ablation:
the temporal means. Under a 0.01-0.1 Hz band-pass they should be numerically
negligible, so their magnitude is reported, together with how well means and
stds predict site.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"


def fit_predict(xtr, ytr, xte, C=1.0):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    return LogisticRegression(**kw).fit(sc.transform(xtr), ytr) \
        .decision_function(sc.transform(xte))


def nadeau_bengio(diffs, n_test, n_train, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    se = np.sqrt(max((1.0 / j + n_test / n_train) * d.var(ddof=1), 1e-300))
    tcrit = stats.t.ppf(1 - alpha / 2, j - 1)
    return dict(mean=float(d.mean()), lo=float(d.mean() - tcrit * se),
                hi=float(d.mean() + tcrit * se),
                p=float(2 * stats.t.sf(abs(d.mean() / se), j - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="abide_site.json")
    args = ap.parse_args()

    from e18_primary import load

    ab = load()
    series, pheno = ab["rois_cc200"], ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])
    site = np.asarray([str(s) for s in pheno["SITE_ID"]])
    sites, site_idx = np.unique(site, return_inverse=True)
    dummies = np.eye(len(sites))[site_idx]

    def column(name):
        v = np.asarray([float(x) if str(x) not in ("nan", "", "-9999")
                        else np.nan for x in pheno[name]], dtype=float)
        return np.nan_to_num(v, nan=float(np.nanmedian(v)))

    fd = column("func_mean_fd")
    n, d = len(series), series[0].shape[1]
    means = np.empty((n, d))
    stds = np.empty((n, d))
    for k, s in enumerate(series):
        z = np.asarray(s, dtype=np.float64).T
        means[k] = z.mean(axis=1)
        stds[k] = z.std(axis=1)
    sigma_bar = stds.mean(axis=1)

    print(f"n={n} d={d} sites={len(sites)}")
    print(f"\n--- magnitude of the temporal means (band-pass 0.01-0.1 Hz) ---")
    print(f"  mean |m| over all entries      = {np.abs(means).mean():.3e}")
    print(f"  median |m|                     = {np.median(np.abs(means)):.3e}")
    print(f"  mean sigma over all entries    = {stds.mean():.3e}")
    print(f"  ratio mean|m| / mean sigma     = "
          f"{np.abs(means).mean()/stds.mean():.3e}")

    print(f"\n--- motion versus site ---")
    r_raw = stats.pearsonr(sigma_bar, fd)
    res_s = sigma_bar - LinearRegression().fit(dummies, sigma_bar).predict(dummies)
    res_f = fd - LinearRegression().fit(dummies, fd).predict(dummies)
    r_part = stats.pearsonr(res_s, res_f)
    print(f"  corr(mean sigma, FD) raw           = {r_raw[0]:+.3f} "
          f"(p={r_raw[1]:.1e})")
    print(f"  corr(mean sigma, FD) | site        = {r_part[0]:+.3f} "
          f"(p={r_part[1]:.1e})")
    per = []
    for k, s in enumerate(sites):
        m = site_idx == k
        if m.sum() >= 20:
            per.append((s, int(m.sum()), float(stats.pearsonr(sigma_bar[m],
                                                              fd[m])[0])))
    print("  per site (n >= 20):")
    for s, cnt, r in sorted(per, key=lambda x: -x[1]):
        print(f"     {s:<14} n={cnt:<4} r={r:+.3f}")
    print(f"  median within-site r = "
          f"{np.median([r for _, _, r in per]):+.3f}")

    print(f"\n--- how well do the channels predict SITE (20 classes) ---")
    chance = float(np.bincount(site_idx).max() / n)
    for tag, x in [("means", means), ("stds", stds),
                   ("FD", fd[:, None]), ("mean sigma", sigma_bar[:, None])]:
        acc = cross_val_score(
            LogisticRegression(max_iter=3000),
            StandardScaler().fit_transform(x), site_idx, cv=5,
            scoring="accuracy").mean()
        print(f"  {tag:>12}: site accuracy {acc:.3f}  (chance {chance:.3f})")

    print(f"\n--- diagnosis, 10x5 stratified ---")
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                 random_state=0)
    per_fold: dict[str, list] = {}
    t0 = time.time()
    for tr, te in cv.split(means, y):
        n_test, n_train = len(te), len(tr)
        out = {
            "site one-hot": fit_predict(dummies[tr], y[tr], dummies[te]),
            "FD alone": fit_predict(fd[tr, None], y[tr], fd[te, None]),
            "LR stds": fit_predict(stds[tr], y[tr], stds[te]),
            "LR means": fit_predict(means[tr], y[tr], means[te]),
        }
        lin = LinearRegression().fit(dummies[tr], stds[tr])
        s_tr = stds[tr] - lin.predict(dummies[tr])
        s_te = stds[te] - lin.predict(dummies[te])
        out["LR stds, site removed"] = fit_predict(s_tr, y[tr], s_te)
        lin_f = LinearRegression().fit(dummies[tr], fd[tr, None])
        out["FD, site removed"] = fit_predict(
            fd[tr, None] - lin_f.predict(dummies[tr]), y[tr],
            fd[te, None] - lin_f.predict(dummies[te]))
        for key, s in out.items():
            per_fold.setdefault(key, []).append(
                (float(roc_auc_score(y[te], s)),
                 float(balanced_accuracy_score(y[te], s > 0))))
    print(f"  ({time.time()-t0:.0f}s)")
    print(f"{'channel':>24}   {'AUC':>14}   {'balanced acc':>14}")
    summary = {}
    for key, vals in per_fold.items():
        a = np.array(vals)
        summary[key] = dict(auc=float(a[:, 0].mean()), bacc=float(a[:, 1].mean()))
        print(f"{key:>24}   {a[:,0].mean():.4f} ({a[:,0].std():.3f})   "
              f"{100*a[:,1].mean():.1f}% ({100*a[:,1].std():.1f})")

    tests = {}
    for a_key, b_key in [("LR stds", "LR stds, site removed"),
                         ("LR stds", "site one-hot"),
                         ("FD alone", "FD, site removed")]:
        for metric, idx in (("AUC", 0), ("bacc", 1)):
            diffs = [x[idx] - z[idx] for x, z in zip(per_fold[a_key],
                                                     per_fold[b_key])]
            r = nadeau_bengio(diffs, n_test, n_train)
            tests[f"{a_key} - {b_key} [{metric}]"] = r
            star = "*" if r["p"] < 0.05 else " "
            print(f"  {a_key:>22} - {b_key:<24} {metric:>4}: {r['mean']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.4f}{star}")

    (CACHE / args.out).write_text(json.dumps(
        dict(n=n, d=d, sites=len(sites), corr_raw=float(r_raw[0]),
             corr_partial=float(r_part[0]),
             per_site=[dict(site=s, n=c, r=r) for s, c, r in per],
             mean_abs_m=float(np.abs(means).mean()),
             mean_sigma=float(stds.mean()),
             summary=summary, tests=tests), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
