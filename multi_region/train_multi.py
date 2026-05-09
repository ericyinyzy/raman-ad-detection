"""Train Raman-MRNet on multi-region spectra and report the 5 paper metrics.

Acc / Precision / Recall / F1 / AUC are printed every 3 epochs and at the end.
Training runs for 15 epochs with a fixed seed so the reported numbers are
reproducible across runs.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import baselineUtils
import individual_TF


SEED = 43
MAX_EPOCH = 15
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cal_acc(result, label):
    result = list(result.numpy())
    label = list(label.numpy())
    correct = 0
    bs = 0
    pos_correct = 0
    pos_all = 0
    for a, b in zip(result, label):
        if b == 1:
            pos_all += 1
            if a == 1:
                pos_correct += 1
        if a == b:
            correct += 1
        bs += 1
    return correct, bs, pos_correct, pos_all


def evaluate(model, test_dl):
    model.eval()
    total_num = 0
    correct_count = 0
    pos_correct_correct = 0
    pos_all_correct = 0
    total_pred_pos = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dl:
            inputs = batch['input'].to(device)
            inputs = inputs.squeeze(1)
            label = batch['label'][:, 0]
            out, _ = model(inputs)

            prob = torch.softmax(out.detach().cpu(), dim=1)[:, 1]
            result = torch.argmax(out.detach().cpu(), dim=1)

            correct, bs, pos_correct, pos_all = cal_acc(result, label.detach().cpu())
            correct_count += correct
            total_num += bs
            pos_correct_correct += pos_correct
            pos_all_correct += pos_all
            total_pred_pos += result.sum().item()
            all_probs.extend(prob.tolist())
            all_labels.extend(label.detach().cpu().tolist())

    acc = correct_count / total_num if total_num > 0 else 0.0
    precision = pos_correct_correct / total_pred_pos if total_pred_pos > 0 else 0.0
    recall = pos_correct_correct / pos_all_correct if pos_all_correct > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
    return acc, precision, recall, f1, auc


if __name__ == '__main__':
    set_seed(SEED)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, _, _ = baselineUtils.create_pie_dataset('train', device)
    test_dataset, _, _ = baselineUtils.create_pie_dataset('test', device)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=0)

    model = individual_TF.IndividualTF1().to(device)
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    print(f'start training (seed={SEED}, epochs={MAX_EPOCH})')
    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(tr_dl, desc=f'epoch {epoch}/{MAX_EPOCH}'):
            inputs = batch['input'].to(device).squeeze(1)
            label = batch['label'][:, 0]
            if int(inputs.shape[0]) == 1:
                continue
            optim.zero_grad()
            out, _ = model(inputs)
            loss = F.cross_entropy(out, label)
            train_loss += loss.item()
            loss.backward()
            optim.step()

        if epoch % 3 == 0 or epoch == MAX_EPOCH:
            acc, precision, recall, f1, auc = evaluate(model, test_dl)
            print(f'epoch {epoch} | acc {acc:.4f} | precision {precision:.4f} | '
                  f'recall {recall:.4f} | f1 {f1:.4f} | auc {auc:.4f}')

    acc, precision, recall, f1, auc = evaluate(model, test_dl)
    print('=' * 60)
    print('Final test metrics:')
    print(f'acc {acc:.4f} | precision {precision:.4f} | recall {recall:.4f} | '
          f'f1 {f1:.4f} | auc {auc:.4f}')
