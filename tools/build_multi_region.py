"""Build the multi-region inputs from the single-region spectra.

Each combined sample stacks one spectrum per region in the order
[thalamus, cortex, hippocampus] after asymmetric-least-squares baseline
subtraction, matching the pipeline used for the MoSe2 data shipped with this
repository. Region file lists are shuffled with a fixed seed so that the
pairing of spectra across regions is reproducible.

Usage (from the repository root):

    python tools/build_multi_region.py no mos2
"""
import os
import random
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REGIONS = ["thalamus", "cortex", "hippocampus"]
SEED = 43


def baseline_als(y, lam=10000, p=0.0001, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def build(material, split, cls):
    src = {r: os.path.join(DATA, "AD_processed_single", material, split, cls, r) for r in REGIONS}
    files = {}
    for r in REGIONS:
        names = sorted(f for f in os.listdir(src[r]) if f.endswith(".npy"))
        random.Random(SEED).shuffle(names)
        files[r] = names
    counts = {r: len(files[r]) for r in REGIONS}
    if min(counts.values()) == 0:
        print(f"skip {material}/{split}/{cls}: {counts}", flush=True)
        return 0

    out = os.path.join(DATA, "AD_processed_merge", material, split, cls)
    os.makedirs(out, exist_ok=True)
    max_n = max(counts.values())
    for i in range(max_n):
        stack = []
        for r in REGIONS:
            arr = np.load(os.path.join(src[r], files[r][i % counts[r]])).astype(np.float64)
            assert arr.shape == (1, 701), f"bad shape {arr.shape} in {r}"
            arr[0] = arr[0] - baseline_als(arr[0])
            stack.append(arr)
        np.save(os.path.join(out, f"combined_{i + 1}.npy"), np.concatenate(stack, axis=0))
    print(f"{material}/{split}/{cls}: {max_n} combined  (from {counts})", flush=True)
    return max_n


if __name__ == "__main__":
    materials = sys.argv[1:] or ["no", "mos2"]
    total = 0
    for material in materials:
        for split in ("train", "test"):
            for cls in ("positive", "negative"):
                total += build(material, split, cls)
    print("TOTAL", total, flush=True)
