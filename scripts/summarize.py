"""Aggregate every finished experiment into the tables for the write-up."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(np.asarray(x), size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ladder(path, keys, ref="raw (m,s)", cmp_key="syn: wbar"):
    obj = json.loads(Path(path).read_text())
    rows = obj["rows"]
    print(f"### {Path(path).name}  (d_all={obj.get('d_all')}, n={obj.get('n')})")
    for d in sorted({r["d"] for r in rows}):
        print(f"\n  d = {d}")
        print(f"  {'n_subj':>7s}" + "".join(f"{k:>14s}" for k in keys)
              + f"{'D(syn-raw)':>13s}{'95% CI':>20s}")
        for size in sorted({r["n_subj"] for r in rows if r["d"] == d}):
            sel = [r for r in rows if r["d"] == d and r["n_subj"] == size]
            if not sel:
                continue
            means = [np.mean([r[k] for r in sel]) for k in keys]
            gap = [r[cmp_key] - r[ref] for r in sel]
            lo, hi = boot_ci(gap)
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {size:>7d}" + "".join(f"{m:>14.4f}" for m in means)
                  + f"{np.mean(gap):>+12.4f}{star}  [{lo:+.4f},{hi:+.4f}]")
    print()


def qda_table(path):
    obj = json.loads(Path(path).read_text())
    rows = obj["rows"]
    keys = [k for k in rows[0] if k not in ("fold", "n_subj", "secs")]
    print(f"### {Path(path).name}  (d={obj['d']}, n={obj['n']}, "
          f"reg={obj['reg']})")
    print(f"  {'n_subj':>7s}" + "".join(f"{k:>15s}" for k in keys)
          + f"{'D(QDA-LR)':>12s}{'95% CI':>20s}")
    for size in sorted({r["n_subj"] for r in rows}):
        sel = [r for r in rows if r["n_subj"] == size]
        means = [np.mean([r[k] for r in sel]) for k in keys]
        gap = [r["syn-QDA: flat"] - r["syn-LR: flat"] for r in sel]
        lo, hi = boot_ci(gap)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {size:>7d}" + "".join(f"{m:>15.4f}" for m in means)
              + f"{np.mean(gap):>+11.4f}{star}  [{lo:+.4f},{hi:+.4f}]")
    print()


def synth(path):
    obj = json.loads(Path(path).read_text())
    keys = list(obj["rows"][0]["mean"].keys())
    print(f"### {Path(path).name}  (d={obj['d']}, r={obj['r']}, "
          f"mu={obj['mu']}, reps={obj['reps']})")
    print(f"  {'n':>6s}" + "".join(f"{k:>20s}" for k in keys))
    for r in obj["rows"]:
        print(f"  {r['n']:>6d}" + "".join(
            f"{r['mean'][k]:>13.3f}+-{r['sd'][k]:.3f}" for k in keys))
    print()


def main() -> None:
    keys = ["raw (m,s)", "FC vec(rho)", "raw + FC", "syn: wbar", "syn: lbar",
            "syn: sum_l"]
    for name in ["ladder_mmp_full.json", "ladder_yeo17_EMOTION_RL.json"]:
        p = CACHE / name
        if p.exists():
            k = keys + (["syn: Lflat"] if "Lflat" in
                        json.loads(p.read_text())["rows"][0] else [])
            ladder(p, k)
    for name in sorted(CACHE.glob("qda_*.json")):
        qda_table(name)
    for name in sorted(CACHE.glob("results_synth*.json")):
        synth(name)


if __name__ == "__main__":
    main()
