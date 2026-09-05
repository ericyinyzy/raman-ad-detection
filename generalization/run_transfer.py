"""Transfer learning across brain regions.

For each target region Raman-Net is pretrained on the training spectra of the
other two regions and then fine-tuned on the target region, and compared with
the same architecture trained from scratch on the target region alone. Both
variants see the same target-region data and are tested on the same held-out
animals and substrate, so the only difference is the initialisation. A third
variant applies the pretrained model to the target region without fine-tuning.

    python generalization/run_transfer.py
"""
import copy

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import (DEVICE, METRIC_NAMES, REGIONS, RamanNet, SEEDS_NN, load_single_region, metrics,
                    pick_threshold, score_torch, summarise, train_torch)

EPOCHS = 15
VARIANTS = {
    "scratch": "Raman-Net (from scratch)",
    "direct": "Raman-Net (direct transfer)",
    "finetune": "Raman-Net (transfer + fine-tune)",
}


def fresh(state=None):
    m = RamanNet().to(DEVICE)
    if state is not None:
        m.load_state_dict(state)
    return m


def main():
    print("device:", DEVICE, "| epochs:", EPOCHS, flush=True)
    data = {r: load_single_region(r) for r in REGIONS}
    results = {}

    for region in REGIONS:
        Xtr, ytr, Xte, yte = data[region]
        source = [r for r in REGIONS if r != region]
        Xs = np.concatenate([data[r][0] for r in source])
        ys = np.concatenate([data[r][1] for r in source])
        print("\n[%s] target train %d (%d 5xFAD) | target test %d (%d 5xFAD) | pretrain on %s: %d"
              % (region, len(ytr), ytr.sum(), len(yte), yte.sum(), "+".join(source), len(ys)),
              flush=True)

        runs = {k: [] for k in VARIANTS}
        for seed in SEEDS_NN:
            # Trained from scratch on the target region.
            oof = np.zeros(len(ytr))
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
                m = train_torch(fresh(), Xtr[tr], ytr[tr], seed, EPOCHS)
                oof[va] = score_torch(m, Xtr[va])
            thr = pick_threshold(ytr, oof)
            prob = score_torch(train_torch(fresh(), Xtr, ytr, seed, EPOCHS), Xte)
            runs["scratch"].append(metrics(yte, (prob >= thr).astype(int), prob))

            # Pretrained on the other two regions.
            base = train_torch(fresh(), Xs, ys, seed, EPOCHS)
            state = copy.deepcopy(base.state_dict())

            # Applied to the target region as is. The threshold comes from the target
            # region's training spectra, which the pretrained model has not seen.
            prob = score_torch(base, Xte)
            thr = pick_threshold(ytr, score_torch(base, Xtr))
            runs["direct"].append(metrics(yte, (prob >= thr).astype(int), prob))

            # Fine-tuned on the target region.
            oof = np.zeros(len(ytr))
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
                m = train_torch(fresh(state), Xtr[tr], ytr[tr], seed, EPOCHS)
                oof[va] = score_torch(m, Xtr[va])
            thr = pick_threshold(ytr, oof)
            prob = score_torch(train_torch(fresh(state), Xtr, ytr, seed, EPOCHS), Xte)
            runs["finetune"].append(metrics(yte, (prob >= thr).astype(int), prob))

        for k in runs:
            results[(region, k)] = summarise(runs[k])
            mean = results[(region, k)][0]
            print("   %-10s Acc %5.1f  F1 %5.1f  AUC %5.1f" % (k, mean[0], mean[4], mean[5]),
                  flush=True)

    print("\n" + "=" * 116)
    print("Transfer learning, single region, animal- and substrate-disjoint split, "
          "calibrated threshold, %d epochs, 5 runs" % EPOCHS)
    print("=" * 116)
    print("%-13s %-33s %s" % ("Region", "Method", " ".join("%12s" % m for m in METRIC_NAMES)))
    for region in REGIONS:
        for k, label in VARIANTS.items():
            mean, std = results[(region, k)]
            print("%-13s %-33s %s" % (region, label,
                                      " ".join("%7.1f±%-4.1f" % (mean[i], std[i]) for i in range(6))))
    print("-" * 116)
    for k, label in VARIANTS.items():
        mean = np.stack([results[(r, k)][0] for r in REGIONS]).mean(0)
        std = np.stack([results[(r, k)][1] for r in REGIONS]).mean(0)
        print("%-13s %-33s %s" % ("Average", label,
                                  " ".join("%7.1f±%-4.1f" % (mean[i], std[i]) for i in range(6))))
    print("=" * 116)


if __name__ == "__main__":
    main()
