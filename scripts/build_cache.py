"""Build a memory-lean float32 feature cache for one parcellation/task.

build_task_features() materialises the whole (N, d, d) stack twice in float64,
which does not fit alongside the other running jobs at d=379. This writes the
same quantities straight into on-disk float32 arrays instead.

Cropping is identical to the repository path: every record is cut to the
shortest series in the task, so record length cannot leak the label.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")
sys.path.insert(0, str(REPO / "src"))
CACHE = Path(__file__).resolve().parent / "cache"
ROOTS = {
    "yeo17": Path(r"C:\Users\breki\Downloads\HCP_v4\yeo17_extracted\split_yeo17_lr"),
    "mmp": Path(r"C:\Users\breki\Downloads\HCP_v4\mmp_extracted\split"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcel", default="mmp")
    ap.add_argument("--task", default="EMOTION")
    ap.add_argument("--direction", default="RL")
    args = ap.parse_args()

    from synolitic.setup2.data import load_series, scan_records, state_labels

    CACHE.mkdir(exist_ok=True)
    recs = [r for r in scan_records(ROOTS[args.parcel], args.task)
            if args.direction in ("ALL", r.direction)]
    labels = state_labels(recs)
    print(f"{args.parcel}/{args.task}/{args.direction}: {len(recs)} records",
          flush=True)

    t0 = time.time()
    series = []
    for k, r in enumerate(recs):
        series.append(load_series(r.path).astype(np.float32))
        if k % 400 == 0:
            print(f"  loaded {k}/{len(recs)}  [{time.time()-t0:.0f}s]", flush=True)
    crop = min(s.shape[1] for s in series)
    d = series[0].shape[0]
    n = len(recs)
    print(f"d={d} n={n} crop T={crop}  [{time.time()-t0:.0f}s]", flush=True)

    stem = f"{args.parcel}_{args.task}_{args.direction}"
    vtx = np.lib.format.open_memmap(CACHE / f"{stem}_vertex.npy", mode="w+",
                                    dtype=np.float32, shape=(n, d, 2))
    edg = np.lib.format.open_memmap(CACHE / f"{stem}_edge.npy", mode="w+",
                                    dtype=np.float32, shape=(n, d, d))
    for k, s in enumerate(series):
        s = s[:, :crop].astype(np.float64)
        vtx[k, :, 0] = s.mean(axis=1)
        vtx[k, :, 1] = s.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(s)
        edg[k] = np.nan_to_num(c, nan=0.0)
        series[k] = None
        if k % 400 == 0:
            print(f"  featurised {k}/{n}  [{time.time()-t0:.0f}s]", flush=True)
    vtx.flush(); edg.flush()

    meta = dict(parcel=args.parcel, task=args.task, direction=args.direction,
                d=int(d), n=int(n), crop=int(crop), labels=labels,
                y=[int(labels[r.state]) for r in recs],
                subject=[r.subject for r in recs])
    (CACHE / f"{stem}_meta.json").write_text(json.dumps(meta))
    print(f"wrote {stem}_*  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
