"""Aggregate ladder runs: per training size, mean AUC and the paired gap
between the synolitic readout and the raw+FC linear baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"
KEYS = ["raw (m,s)", "FC vec(rho)", "raw + FC", "syn: wbar", "syn: lbar",
        "syn: sum_l", "syn: Lflat"]


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(path):
    obj = json.loads(Path(path).read_text())
    rows = obj["rows"]
    print(f"{obj['parcel']}/{obj['task']} d_all={obj['d_all']} n={obj['n']} "
          f"T={obj['crop']} family={obj['family']}\n")
    for d in sorted({r["d"] for r in rows}):
        print(f"--- d = {d} ---")
        hdr = f"{'n_subj':>7s} {'n_rec':>6s} " + "".join(f"{k:>13s}" for k in KEYS)
        print(hdr + f"{'D(wbar-rawFC)':>16s}{'95% CI':>20s}")
        for size in sorted({r["n_subj"] for r in rows if r["d"] == d}):
            sel = [r for r in rows if r["d"] == d and r["n_subj"] == size]
            means = [np.mean([r[k] for r in sel]) for k in KEYS]
            gap = np.array([r["syn: wbar"] - r["raw + FC"] for r in sel])
            lo, hi = boot_ci(gap)
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"{size:>7d} {int(np.mean([r['n_rec'] for r in sel])):>6d} "
                  + "".join(f"{m:>13.4f}" for m in means)
                  + f"{gap.mean():>+15.4f}{star}"
                  + f"  [{lo:+.4f},{hi:+.4f}]")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else CACHE / "ladder_yeo17_EMOTION_RL.json")
