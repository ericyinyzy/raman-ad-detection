"""Multi-region half of the generalization table.

Every method sees all three regions at once, as a (3, 701) tensor per sample.
Each method is run with five seeds.

    python generalization/run_multi_region.py
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import (CNN_MR, DEVICE, LSTM_MR, RamanMRNet, SEEDS_NN, SEEDS_SK, describe, fit_torch,
                    load_multi_region, metrics, pick_threshold, print_table, summarise)

EPOCHS = 30


def flat(X):
    return X.reshape(len(X), -1)


def fit_svm(X, y, seed, epochs=None):
    m = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed,
            probability=True, class_weight="balanced").fit(flat(X), y)
    return lambda Z: m.predict_proba(flat(Z))[:, 1]


def fit_xgb(X, y, seed, epochs=None):
    c = np.bincount(y, minlength=2)
    m = XGBClassifier(n_estimators=10, eval_metric="logloss", random_state=seed,
                      scale_pos_weight=c[0] / max(c[1], 1)).fit(flat(X), y)
    return lambda Z: m.predict_proba(flat(Z))[:, 1]


FIT = {
    "SVM": fit_svm,
    "XGBoost": fit_xgb,
    "LSTM": lambda X, y, s, ep: fit_torch(LSTM_MR, X, y, s, ep),
    "1D-CNN": lambda X, y, s, ep: fit_torch(CNN_MR, X, y, s, ep),
    "Raman-MRNet": lambda X, y, s, ep: fit_torch(RamanMRNet, X, y, s, ep),
}
ORDER = ["SVM", "XGBoost", "LSTM", "1D-CNN", "Raman-MRNet"]
TORCH_METHODS = {"LSTM", "1D-CNN", "Raman-MRNet"}


def main():
    Xtr, ytr, Xte, yte = load_multi_region()
    print("device:", DEVICE, "| epochs:", EPOCHS, flush=True)
    print(describe(ytr, yte), flush=True)

    rows = []
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
        mean, std = summarise(runs)
        rows.append((name, mean, std))
        print("%-12s thr=%.2f±%.2f | Acc %5.1f±%-4.1f  F1 %5.1f  AUC %5.1f"
              % (name, np.mean(thresholds), np.std(thresholds), mean[0], std[0], mean[4], mean[5]),
              flush=True)

    print_table("Multi-region, animal- and substrate-disjoint split, calibrated threshold, "
                "%d epochs, 5 runs" % EPOCHS, rows, label_width=14)


if __name__ == "__main__":
    main()
