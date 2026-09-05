"""Ablation on how Raman-MRNet merges the three regions.

The three variants share the encoder, the preprocessing and the training
schedule, and differ only in how the three regional feature vectors are merged
before the classifier: soft-attention weights them by content, mean pooling
gives all of them the same weight, concatenation stacks them.

On the split used in the main experiments every variant reaches 100% accuracy,
so the comparison carries no information; it is therefore run here, on the
animal- and substrate-disjoint split.

    python generalization/run_fusion_ablation.py
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import (ConcatMR, DEVICE, MeanPoolMR, RamanMRNet, SEEDS_NN, describe, fit_torch,
                    load_multi_region, metrics, pick_threshold, print_table, summarise)

EPOCHS = 30
VARIANTS = {
    "Mean pooling": MeanPoolMR,
    "Concatenation": ConcatMR,
    "Soft-attention (ours)": RamanMRNet,
}


def main():
    Xtr, ytr, Xte, yte = load_multi_region()
    print("device:", DEVICE, "| epochs:", EPOCHS, flush=True)
    print(describe(ytr, yte), flush=True)

    rows = []
    for name, model_cls in VARIANTS.items():
        runs = []
        for seed in SEEDS_NN:
            oof = np.zeros(len(ytr))
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
                oof[va] = fit_torch(model_cls, Xtr[tr], ytr[tr], seed, EPOCHS)(Xtr[va])
            thr = pick_threshold(ytr, oof)
            prob = fit_torch(model_cls, Xtr, ytr, seed, EPOCHS)(Xte)
            runs.append(metrics(yte, (prob >= thr).astype(int), prob))
        mean, std = summarise(runs)
        rows.append((name, mean, std))
        print("%-24s Acc %5.1f±%-4.1f  F1 %5.1f  AUC %5.1f"
              % (name, mean[0], std[0], mean[4], mean[5]), flush=True)

    print_table("Region fusion ablation, animal- and substrate-disjoint split, "
                "calibrated threshold, %d epochs, 5 runs" % EPOCHS, rows)


if __name__ == "__main__":
    main()
