# Raman-AD: Interpretable Predictive Modeling for Detecting Alzheimer's Disease Using Raman Spectra

This repository contains the code and processed data accompanying the paper:

> **Towards Interpretable Predictive Modeling for Detecting Alzheimer's Disease Using Raman Spectra**

The framework includes two models:

- **Raman-Net** — a single-region classifier (3 linear layers + ReLU + Softmax) that operates on Raman spectra collected from one brain region (cortex, hippocampus, or thalamus). Interpretation is performed with SHAP values over the 701 wavenumber points.
- **Raman-MRNet** — a multi-region classifier that takes Raman spectra from all three regions simultaneously and uses a region-wise soft-attention module to integrate them. Interpretation is performed by visualizing the attention weights as heatmaps.

The main experiments use the **MoSe₂** substrate dataset (5xFAD AD vs. healthy controls), which contains 2018 Raman spectra in total: 312 thalamus, 855 cortex, 851 hippocampus.

The generalization analysis reported in the paper uses two further substrates, both included here: **`no`** (quartz slide carrying no monolayer) for training and **`mos2`** (monolayer MoS₂) for testing. That split shares neither animals nor substrates between training and testing.

---

## Data

Each `.npy` file is a single Raman spectrum baseline-corrected to the wavenumber range **900–2000 cm⁻¹** (701 points).

| Region      | AD train | Non-AD train | AD test | Non-AD test | Total |
|-------------|---------:|-------------:|--------:|------------:|------:|
| Thalamus    | 144      | 105          | 36      | 27          | 312   |
| Cortex      | 144      | 540          | 36      | 135         | 855   |
| Hippocampus | 140      | 540          | 36      | 135         | 851   |

Per-substrate spectrum counts:

| Substrate | Single-region spectra | Multi-region samples |
|-----------|----------------------:|---------------------:|
| `mose2`   | 2018                  | 855                  |
| `no`      | 5946                  | 2071                 |
| `mos2`    | 1135                  | 478                  |

`data/AD_processed_single/<substrate>/<split>/<class>/<region>/*.npy` holds one spectrum per file, shaped `(1, 701)`. `data/AD_processed_merge/<substrate>/<split>/<class>/combined_*.npy` holds one multi-region sample per file, shaped `(3, 701)`, stacked in the order **thalamus, cortex, hippocampus**. The merged files for `no` and `mos2` are produced from the single-region files by `tools/build_multi_region.py`.

The wavenumber labels for the 701 points are stored as the column header of `wavenumbers.csv` (columns 9 through 709).

---

## Setup

```bash
git clone <this-repo>
cd raman-ad-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the training scripts default to `cuda:0`.

---

## Single-region pipeline (Raman-Net) — three scripts

The single-region workflow is split into three independent scripts. Pick a target region (`cortex`, `hippocampus`, or `thalamus`) and run them in any order — each one writes its own outputs.

### Step 0 — pick a region

Edit the four data paths in `single_region/baselineUtils_single.py` (lines 70-71 and 109-110):

```python
pos_data_dir = '../data/AD_processed_single/mose2/<split>/positive/<region>'
neg_data_dir = '../data/AD_processed_single/mose2/<split>/negative/<region>'
```

Replace `<region>` with `cortex`, `hippocampus`, or `thalamus`. The default in this repo is `hippocampus`.

### Script 1 — train the model and report the 5 paper metrics

```bash
cd single_region
python train_single.py
```

Trains Raman-Net for 15 epochs on the chosen region and prints the five paper metrics every 3 epochs:

```
epoch 3 | acc 1.0000 | precision 1.0000 | recall 1.0000 | f1 1.0000 | auc 1.0000
```

These are the same metrics reported in Table 2 of the paper: **Accuracy, Precision, Recall, F1, AUC**.

### Script 2 — run SHAP analysis

```bash
python shap_analysis.py
```

This is essentially `train_single.py` with the SHAP analysis appended. It:
1. Re-trains a model from scratch (so the SHAP run is fully reproducible from the `.npy` data alone),
2. Runs `shap.DeepExplainer` over the test set,
3. Writes per-wavenumber importance to `mose2_<sub_dir>/test/shap.json` and `prob.json`, plus 4 PNGs.

Before running, set `sub_dir` near line 372 of `shap_analysis.py` to match the region you set in Step 0:

```python
sub_dir = 'mose2_hippocampus'   # or mose2_cortex / mose2_thalamus
```

Otherwise the SHAP outputs will land in a sub-directory whose name doesn't match the data region.

### Script 3 — draw the spectral SHAP curve (paper Fig. 4)

```bash
python draw_wave.py
```

Auto-detects the most recently written `mose2_*/test/shap.json` and saves a salmon-colored line plot named `spectral_plot_salmon_<region>.png`. No editing needed — just run after Step 2.

---

## Multi-region pipeline (Raman-MRNet) — three scripts

Mirrors the single-region layout: one script trains and reports the paper
metrics, one script trains and dumps the attention map, and one script renders
the final figure. All three use `IndividualTF1` (the SE-attention variant
matching the paper) with seed `43` and 15 epochs so the results are
reproducible across runs.

### Script 1 — train the model and report the 5 paper metrics

```bash
cd multi_region
python train_multi.py
```

Trains Raman-MRNet on cortex + hippocampus + thalamus jointly and prints the
five paper metrics every 3 epochs and at the end of training:

```
epoch 15 | acc 1.0000 | precision 1.0000 | recall 1.0000 | f1 1.0000 | auc 1.0000
```

These match the metrics reported in Table 4 of the paper (Acc / Precision /
Recall / F1 / AUC).

### Script 2 — extract per-sample attention weights

```bash
python attention_analysis.py
```

Re-trains Raman-MRNet with the same seed/epochs and then runs a forward pass
over the test set, recording the soft-attention weight that the model assigns
to each of the three brain regions for every AD-positive (class 1) sample.
Output:

```
weights_normalized.npy    # shape (N_pos, 3)
```

The script also prints the per-region mean and standard deviation
(corresponds to the values quoted in the attention-analysis section of the
paper).

### Script 3 — draw the attention heatmap (paper Fig. 6 / Fig. 7)

```bash
python draw_heatmap.py
```

Auto-detects the most recent `weights_normalized*.npy`, randomly draws 5
panels of 25 AD-positive samples each, and writes them to
`heatmap_outputs/attention_heatmap_*.png` together with a separate
horizontal `colorbar.png`.

---

## Generalization analysis — four scripts

These scripts reproduce the generalization tables of the paper. They train on one 5xFAD mouse and one control mouse measured on bare quartz, and test on a **different** pair of mice measured on monolayer MoS₂, so the training and test sets share neither animals nor substrate. The four animals are renumbered 1 to 4 in the paper; the identifiers in the source table are:

| Split | Substrate | 5xFAD | Control | Spectra (thalamus / hippocampus / cortex) |
|-------|-----------|------:|--------:|-------------------------------------------|
| train | `no` (bare quartz) | mouse 4 | mouse 2 | 1045 / 927 / 953 |
| test  | `mos2` (monolayer MoS₂) | mouse 12 | mouse 6 | 478 / 331 / 326 |

Two settings apply to all four scripts and differ from the main experiments above:

- Each spectrum is rescaled by the **standard normal variate** transform. Without it the two genotypes are separable from brightness alone, because each acquisition session covers a single animal, so absolute intensity is confounded with the label.
- The **decision threshold** is chosen to maximise balanced accuracy on out-of-fold predictions over the training substrate. No test label is used at any point.

The split is shipped as `data/generalization/<split>/<region>/{X.npy,y.npy}`, in the row order of the source table (the training scripts shuffle with a fixed seed, so a different ordering would change the mini-batch composition and the numbers). `tools/export_generalization_split.py` regenerates it from the full spectral table for anyone who has that table.

```bash
python generalization/run_single_region.py     # single-region rows of the generalization table
python generalization/run_multi_region.py      # multi-region rows of the same table
python generalization/run_transfer.py          # cross-region pretraining vs. training from scratch
python generalization/run_fusion_ablation.py   # soft-attention vs. mean pooling vs. concatenation
```

Each script prints per-region results followed by the average, over five seeds. The multi-region scripts pair one spectrum per region into a `(3, 701)` sample and cache the result in `data/generalization/multi_region_cache.npz` on first run, since the baseline fit is the slow part. A GPU is used when available; the models are small enough to run on CPU.

---

## Notes

- `wavenumbers.csv` is a single header row, not a data file. It exists only to expose the 701 wavenumber labels used for plot axis ticks; the spectra themselves are the `.npy` files under `data/`, which are the actual model inputs for every number reported in the paper.
- Both `single_region/mose2_*/test/` and `multi_region/heatmap_outputs/` start empty — they are populated by Script 2 / Script 3 respectively.

---

## Citation

If you find this code useful, please cite the paper.
