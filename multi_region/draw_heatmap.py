"""Render the multi-region attention heatmaps used in the paper.

Auto-detects the most recently modified `weights_normalized*.npy` (produced by
`attention_analysis.py`), randomly samples 25 AD-positive cases and renders the
3xN attention heatmap shown in Fig. 6 / Fig. 7. A separate horizontal colorbar
is also exported so the heatmap panels can be assembled in the paper figure
without per-panel colorbars.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SEED = 43
NUM_HEATMAPS = 5
SAMPLES_PER_HEATMAP = 25
OUTPUT_DIR = 'heatmap_outputs'
REGION_LABELS = ['Cortex', 'Hippocampus', 'Thalamus']


def find_latest_weights() -> str:
    candidates = glob.glob('weights_normalized*.npy')
    if not candidates:
        raise FileNotFoundError(
            'No weights_normalized*.npy found. Run attention_analysis.py first.'
        )
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    weights_path = find_latest_weights()
    weights = np.load(weights_path)
    print(f'Loaded {weights_path} with shape {weights.shape}')

    n_samples = weights.shape[0]
    if n_samples < SAMPLES_PER_HEATMAP:
        raise ValueError(
            f'Need at least {SAMPLES_PER_HEATMAP} samples to make a heatmap, '
            f'got {n_samples}.'
        )

    for i in range(NUM_HEATMAPS):
        idx = rng.choice(n_samples, SAMPLES_PER_HEATMAP, replace=False)
        panel = weights[idx].T

        plt.figure(figsize=(12, 3))
        sns.heatmap(
            panel,
            cmap='coolwarm',
            cbar=False,
            xticklabels=False,
            yticklabels=REGION_LABELS,
            linewidths=0.5,
            linecolor='black',
        )
        plt.xlabel('Sample index')
        plt.title(f'Attention heatmap {i + 1} (3 x {SAMPLES_PER_HEATMAP})')

        save_path = os.path.join(OUTPUT_DIR, f'attention_heatmap_{i + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  saved {save_path}')

    fig, ax = plt.subplots(figsize=(12, 0.5))
    cmap = plt.get_cmap('coolwarm')
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax,
                      orientation='horizontal')
    cb.set_label('Attention weight')
    cbar_path = os.path.join(OUTPUT_DIR, 'colorbar.png')
    plt.savefig(cbar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  saved {cbar_path}')

    print(f'\nDone. {NUM_HEATMAPS} heatmaps + colorbar written to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
