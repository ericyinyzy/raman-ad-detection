"""Shared pieces for the generalization analysis.

Everything in this package is evaluated on the animal- and substrate-disjoint
split exported by `tools/export_generalization_split.py`: models are trained on
one 5xFAD mouse and one control mouse measured on bare quartz, and tested on a
different pair of mice measured on monolayer MoS2.

Two things differ from the main experiments of the paper and apply to every
script here:

* Spectra are rescaled by the standard normal variate transform, which removes
  the absolute intensity of each spectrum. Without it a classifier can separate
  the two genotypes from brightness alone, because each acquisition session
  covers a single animal.
* The decision threshold is chosen to maximise balanced accuracy on
  out-of-fold predictions over the training substrate, never on the test set.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "generalization")
CACHE = os.path.join(DATA, "multi_region_cache.npz")

REGIONS = ["thalamus", "hippocampus", "cortex"]
# Channel order of the multi-region tensors, matching data_merge.py.
MULTI_ORDER = ["thalamus", "cortex", "hippocampus"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS_NN = (42, 43, 44, 45, 46)
SEEDS_SK = (1337, 1338, 1339, 1340, 1341)
METRIC_NAMES = ["Acc.", "BAcc.", "Prec.", "Rec.", "F1", "AUC"]


# --------------------------------------------------------------------- data

def snv(X):
    """Standard normal variate: zero mean and unit standard deviation per spectrum."""
    mu = X.mean(-1, keepdims=True)
    sd = X.std(-1, keepdims=True)
    sd[sd == 0] = 1
    return ((X - mu) / sd).astype(np.float32)


def load_region(split, region):
    d = os.path.join(DATA, split, region)
    return np.load(os.path.join(d, "X.npy")), np.load(os.path.join(d, "y.npy"))


def load_single_region(region):
    """Single-region inputs: (Xtr, ytr, Xte, yte), SNV applied."""
    Xtr, ytr = load_region("train", region)
    Xte, yte = load_region("test", region)
    return snv(Xtr.astype(np.float32)), ytr, snv(Xte.astype(np.float32)), yte


def baseline_als(y, lam=10000, p=0.0001, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for _ in range(niter):
        W.setdiag(w)
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def load_multi_region(use_cache=True):
    """Multi-region inputs of shape (N, 3, 701), stacked in MULTI_ORDER.

    One sample pairs one spectrum from each region of the same animal. Regions
    contribute unequal numbers of spectra, so the shorter ones are cycled to the
    length of the longest. The pairing is drawn with a fixed seed, and the
    baseline is subtracted a second time as in `data_merge.py`. The result is
    cached because the baseline fit is the slow part.
    """
    if use_cache and os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    corrected = {}

    def per_mouse(split, region, label):
        key = (split, region, label)
        if key not in corrected:
            X, y = load_region(split, region)
            X = X[y == label]
            corrected[key] = np.stack([x - baseline_als(x) for x in X]).astype(np.float32)
        return corrected[key]

    def assemble(split, label, rng):
        parts = [per_mouse(split, r, label) for r in MULTI_ORDER]
        idx = [rng.permutation(len(p)) for p in parts]
        n = max(len(p) for p in parts)
        out = np.empty((n, 3, 701), dtype=np.float32)
        for i in range(n):
            for c, (p, ix) in enumerate(zip(parts, idx)):
                out[i, c] = p[ix[i % len(p)]]
        return out

    def build(split):
        rng = np.random.RandomState(0)
        ad = assemble(split, 1, rng)
        control = assemble(split, 0, rng)
        X = np.concatenate([ad, control])
        y = np.r_[np.ones(len(ad)), np.zeros(len(control))].astype(np.int64)
        return X, y

    Xtr, ytr = build("train")
    Xte, yte = build("test")
    Xtr, Xte = snv(Xtr), snv(Xte)
    np.savez(CACHE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def describe(ytr, yte):
    return ("train %d (%d 5xFAD / %d control)   test %d (%d 5xFAD / %d control)"
            % (len(ytr), ytr.sum(), len(ytr) - ytr.sum(),
               len(yte), yte.sum(), len(yte) - yte.sum()))


# ------------------------------------------------------------------ scoring

def pick_threshold(y, p):
    """Threshold maximising balanced accuracy, evaluated on training-side scores."""
    best, best_t = -1, 0.5
    for t in np.unique(np.round(p, 4)):
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        b = (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2
        if b > best:
            best, best_t = b, t
    return best_t


def metrics(y, pred, prob):
    """[accuracy, balanced accuracy, precision, recall, F1, AUC], in percent."""
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    acc = (pred == y).mean() * 100
    rec = tp / max(tp + fn, 1) * 100
    spec = tn / max(tn + fp, 1) * 100
    prec = tp / (tp + fp) * 100 if tp + fp > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return np.array([acc, (rec + spec) / 2, prec, rec, f1, roc_auc_score(y, prob) * 100])


def print_table(title, rows, label_width=24):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("%-*s %s" % (label_width, "Method", " ".join("%12s" % m for m in METRIC_NAMES)))
    for name, mean, std in rows:
        print("%-*s %s" % (label_width, name,
                           " ".join("%7.1f±%-4.1f" % (mean[i], std[i]) for i in range(6))))
    print("=" * 100)


def summarise(runs):
    a = np.stack(runs)
    return a.mean(0), a.std(0)


# ------------------------------------------------------------------- models

class RamanNet(nn.Module):
    """Single-region model of the paper: three fully connected layers on the raw spectrum."""

    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(701, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, 64)
        self.c = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.c(self.f3(self.f2(self.f1(x))))


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.c = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        o, _ = self.lstm(x.unsqueeze(-1))
        return self.c(o[:, -1, :])


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.c = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        return self.c(self.pool(o).squeeze(-1))


class Trunk(nn.Module):
    """Per-region encoder shared by every multi-region variant."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(701, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)

    def forward(self, x):
        return self.linear3(self.linear2(self.linear1(x)))


class RamanMRNet(nn.Module):
    """Multi-region model of the paper: region-wise soft-attention over the three encodings."""

    def __init__(self):
        super().__init__()
        self.t = Trunk()
        self.se_fc1 = nn.Linear(3, 16)
        self.se_fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        o = self.t(x)
        e = self.sigmoid(self.se_fc2(self.relu(self.se_fc1(o.mean(-1)))))
        return self.classifier((o * e.unsqueeze(-1)).sum(1))


class MeanPoolMR(nn.Module):
    """Ablation: average the three region encodings with equal weight."""

    def __init__(self):
        super().__init__()
        self.t = Trunk()
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.t(x).mean(1))


class ConcatMR(nn.Module):
    """Ablation: stack the three region encodings into one long vector."""

    def __init__(self):
        super().__init__()
        self.t = Trunk()
        self.classifier = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        o = self.t(x)
        return self.classifier(o.reshape(len(o), -1))


class LSTM_MR(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        o, _ = self.lstm(x.permute(0, 2, 1))
        return self.classifier(o[:, -1, :])


class CNN_MR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 5, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        return self.classifier(self.pool(o).squeeze(-1))


# ----------------------------------------------------------------- training

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _train_loop(model, X, y, epochs, batch_size=16, lr=1e-4):
    """Class-weighted cross entropy, Adam. Returns the model in eval mode.

    Draws from the global random state without reseeding, so that the caller
    controls whether the weight initialisation is part of the seeded sequence.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X).to(DEVICE)
    yt = torch.tensor(y).to(DEVICE)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1))).to(DEVICE)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx], weight=w).backward()
            opt.step()
    model.eval()
    return model


def score_torch(model, Z, batch_size=256):
    out = []
    with torch.no_grad():
        Zt = torch.tensor(Z).to(DEVICE)
        for i in range(0, len(Zt), batch_size):
            out.append(torch.softmax(model(Zt[i:i + batch_size]), 1).cpu().numpy()[:, 1])
    return np.concatenate(out)


def fit_torch(model_cls, X, y, seed, epochs):
    """Fit a fresh model and return a scoring function over the positive class.

    The seed is set before the model is built, so the initialisation is part of
    the seeded sequence. The transfer learning script initialises its models
    from a pretrained state instead and therefore calls `train_torch` directly.
    """
    set_seed(seed)
    model = model_cls().to(DEVICE)
    _train_loop(model, X, y, epochs)
    return lambda Z: score_torch(model, Z)


def train_torch(model, X, y, seed, epochs):
    """Train a model the caller has already built."""
    set_seed(seed)
    return _train_loop(model, X, y, epochs)
