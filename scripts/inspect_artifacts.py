"""Inventory of existing artifacts: shapes, splits, edge semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")


def show(name, obj, indent="  "):
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        print(f"{indent}{name}: tensor {tuple(obj.shape)} {obj.dtype} "
              f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g}")
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{name}: ndarray {obj.shape} {obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{indent}{name}: dict keys={list(obj.keys())}")
    elif isinstance(obj, list):
        print(f"{indent}{name}: list len={len(obj)} first={type(obj[0]).__name__}")
    else:
        print(f"{indent}{name}: {type(obj).__name__} = {obj!r}"[:300])


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    for rel in ["artifacts/hidden_states.pt",
                "artifacts/graphs.pt",
                "artifacts/setup2/emotion_ensemble_test.pt"]:
        p = REPO / rel
        print(f"\n=== {rel}  ({p.stat().st_size/1e6:.1f} MB) ===")
        try:
            obj = load(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
            continue

        if isinstance(obj, dict):
            for k, v in obj.items():
                show(k, v)
            meta = obj.get("meta")
            if isinstance(meta, dict):
                print(f"  meta = {meta}")
            graphs = obj.get("graphs")
            if isinstance(graphs, list) and graphs:
                g = graphs[0]
                print(f"  graphs[0]: {g}")
                for attr in ["x", "edge_index", "edge_attr", "y", "idx", "split"]:
                    if hasattr(g, attr) and getattr(g, attr) is not None:
                        show(f"graphs[0].{attr}", getattr(g, attr), indent="    ")
                splits = np.array([int(gg.split.item()) for gg in graphs])
                ys = np.array([float(gg.y.item()) for gg in graphs])
                print(f"  split codes: {np.unique(splits, return_counts=True)}")
                print(f"  y values:    {np.unique(ys, return_counts=True)}")
            for key in ["split", "is_correct", "y"]:
                v = obj.get(key)
                if isinstance(v, torch.Tensor):
                    a = v.numpy()
                    print(f"  {key}: unique={np.unique(a, return_counts=True)}")
        else:
            show("obj", obj, indent="  ")

    print("\n=== setup2 module files ===")
    for p in sorted((REPO / "src/synolitic/setup2").rglob("*.py")):
        print(" ", p.relative_to(REPO))


if __name__ == "__main__":
    sys.exit(main())
