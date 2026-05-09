"""Train Raman-MRNet and dump per-sample attention weights for AD samples.

The script trains the same model used by `train_multi.py` (same architecture,
seed and epoch budget) and then runs a forward pass over the entire test set.
For each AD-positive (class 1) sample we record the soft-attention weight that
the model assigned to each of the three brain regions. The result is written
to `weights_normalized.npy` (shape `(N_pos, 3)`), which `draw_heatmap.py`
consumes to render the attention heatmaps in the paper.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import baselineUtils
import individual_TF


SEED = 43
MAX_EPOCH = 15
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
WEIGHTS_OUT = 'weights_normalized.npy'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        for batch in tqdm(tr_dl, desc=f'epoch {epoch}/{MAX_EPOCH}'):
            inputs = batch['input'].to(device).squeeze(1)
            label = batch['label'][:, 0]
            if int(inputs.shape[0]) == 1:
                continue
            optim.zero_grad()
            out, _ = model(inputs)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optim.step()

    print('collecting test inputs and attention weights')
    model.eval()
    all_inputs = []
    all_labels = []
    for batch in tqdm(test_dl, desc='Collecting test data'):
        inputs = batch['input'].to(device).squeeze(1)
        label = batch['label'][:, 0].cpu().numpy()
        all_inputs.append(inputs)
        all_labels.append(label)
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = np.concatenate(all_labels)

    with torch.no_grad():
        _, weights = model(all_inputs)

    weights_normalized = torch.softmax(weights, dim=1)

    indices_class1 = np.where(all_labels == 1)[0]
    pos_weights = weights_normalized[indices_class1].cpu().numpy()
    np.save(WEIGHTS_OUT, pos_weights)

    region_scores = pos_weights.mean(axis=0)
    region_std = pos_weights.std(axis=0)
    print(f'Saved attention weights for {len(indices_class1)} AD samples '
          f'-> {WEIGHTS_OUT} (shape {pos_weights.shape})')
    print('Region importance (mean):', region_scores)
    print('Region importance (std):', region_std)
