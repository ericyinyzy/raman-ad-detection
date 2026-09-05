"""Single-region half of the generalization table.

Each method is trained and evaluated separately on the thalamus, the
hippocampus and the cortex; the paper reports the average over the three
regions. Every method is run with five seeds.

    python generalization/run_single_region.py
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import (CNN1D, DEVICE, LSTMNet, METRIC_NAMES, REGIONS, RamanNet, SEEDS_NN, SEEDS_SK,
                    fit_torch, load_single_region, metrics, pick_threshold, summarise)

# The single-region models are trained for 15 epochs, the multi-region ones for 30;
# see run_multi_region.py.
EPOCHS = 15


def fit_svm(X, y, seed, epochs=None):
    m = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed,
            probability=True, class_weight="balanced").fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def make_xgb(**kw):
    def fit(X, y, seed, epochs=None):
        c = np.bincount(y, minlength=2)
        m = XGBClassifier(eval_metric="logloss", random_state=seed,
                          scale_pos_weight=c[0] / max(c[1], 1), **kw).fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]
    return fit


# "XGBoost" is the default configuration; "XGBoost-t" trades precision for recall
# with more, shallower and more heavily regularised trees.
FIT = {
    "SVM": fit_svm,
    "XGBoost": make_xgb(n_estimators=10, max_depth=6),
    "XGBoost-t": make_xgb(n_estimators=400, max_depth=2, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.3, reg_lambda=5.0),
    "LSTM": lambda X, y, s, ep: fit_torch(LSTMNet, X, y, s, ep),
    "1D-CNN": lambda X, y, s, ep: fit_torch(CNN1D, X, y, s, ep),
    "Raman-Net": lambda X, y, s, ep: fit_torch(RamanNet, X, y, s, ep),
}
ORDER = ["SVM", "XGBoost", "XGBoost-t", "LSTM", "1D-CNN", "Raman-Net"]
TORCH_METHODS = {"LSTM", "1D-CNN", "Raman-Net"}


def main():
    print("device:", DEVICE, "| epochs:", EPOCHS, flush=True)
    results = {}
    for region in REGIONS:
        Xtr, ytr, Xte, yte = load_single_region(region)
        print("\n[%s] train %d (%d 5xFAD) | test %d (%d 5xFAD)"
              % (region, len(ytr), ytr.sum(), len(yte), yte.sum()), flush=True)
        for name in ORDER:
            seeds = SEEDS_NN if name in TORCH_METHODS else SEEDS_SK
            runs, thresholds = [], []
            for seed in seeds:
                oof = np.zeros(len(ytr))
                for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
                    oof[va] = FIT[name](Xtr[tr], ytr[tr], seed, EPOCHS)(Xtr[va])
                thr = pick_threshold(ytr, oof)
                thresholds.append(thr)
                prob = FIT[name](Xtr, ytr, seed, EPOCHS)(Xte)
                runs.append(metrics(yte, (prob >= thr).astype(int), prob))
            results[(region, name)] = summarise(runs)
            mean = results[(region, name)][0]
            print("   %-11s thr=%.3f | Acc %5.1f  F1 %5.1f  AUC %5.1f"
                  % (name, np.mean(thresholds), mean[0], mean[4], mean[5]), flush=True)

    print("\n" + "=" * 112)
    print("Single-region, animal- and substrate-disjoint split, calibrated threshold, %d epochs, 5 runs" % EPOCHS)
    print("=" * 112)
    print("%-13s %-11s %s" % ("Region", "Method", " ".join("%12s" % m for m in METRIC_NAMES)))
    for region in REGIONS:
        for name in ORDER:
            mean, std = results[(region, name)]
            print("%-13s %-11s %s" % (region, name,
                                      " ".join("%7.1f±%-4.1f" % (mean[i], std[i]) for i in range(6))))
    print("-" * 112)
    for name in ORDER:
        mean = np.stack([results[(r, name)][0] for r in REGIONS]).mean(0)
        std = np.stack([results[(r, name)][1] for r in REGIONS]).mean(0)
        print("%-13s %-11s %s" % ("Average", name,
                                  " ".join("%7.1f±%-4.1f" % (mean[i], std[i]) for i in range(6))))
    print("=" * 112)


if __name__ == "__main__":
    main()
