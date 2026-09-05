"""Export the animal- and substrate-disjoint split used in the generalization analysis.

Training draws on the quartz substrate carrying no monolayer (`no`), testing on
monolayer MoS2 (`mos2`), with a different 5xFAD mouse and a different control
mouse on each side, so the two sets share neither animals nor substrate:

    train  ->  material `no`    5xFAD mouse 4    control mouse 2
    test   ->  material `mos2`  5xFAD mouse 12   control mouse 6

The paper renumbers these four animals 1 to 4 in order of appearance.

For each split and region the spectra are written as a single array in the row
order of the source table, together with their labels:

    data/generalization/<split>/<region>/X.npy   (N, 701) float64
    data/generalization/<split>/<region>/y.npy   (N,)     int64   1 = 5xFAD

The arrays keep the double precision of the source table. The single-region
scripts cast to single precision before normalising, the multi-region ones fit
the baseline in double precision, which is what the reported runs did.

Keeping the original row order matters: the training scripts shuffle with a
fixed seed, so a different ordering would change the mini-batch composition and
the numbers would no longer match the ones reported in the paper.

This script needs the full spectral table, which is not redistributed here (see
the Data availability section of the paper). The exported arrays it produces are
included in the repository, so the scripts under `generalization/` run without
it.

Usage:  python tools/export_generalization_split.py path/to/AD_2023_processed.csv
"""
import os
import sys

import numpy as np
import pandas as pd

META = ["adlabel", "sheet", "region", "side", "side2", "material", "sample", "animal", "date"]
REGIONS = ["thalamus", "hippocampus", "cortex"]
SPLITS = {
    "train": ("no", "4", "2"),
    "test": ("mos2", "12", "6"),
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generalization")


def main(csv_path):
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    spec = [c for c in df.columns if c not in META]
    df["sample"] = df["sample"].astype(str)
    assert len(spec) == 701, f"expected 701 spectral columns, found {len(spec)}"

    for split, (material, ad_mouse, control_mouse) in SPLITS.items():
        for region in REGIONS:
            g = df[(df.material == material) & (df.region == region)
                   & (df["sample"].isin([ad_mouse, control_mouse]))]
            X = g[spec].to_numpy(np.float64)
            y = g.adlabel.to_numpy(np.int64)
            d = os.path.join(OUT, split, region)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "X.npy"), X)
            np.save(os.path.join(d, "y.npy"), y)
            print(f"{split:5s} {region:12s} {len(y):5d} spectra  ({y.sum()} 5xFAD, {len(y) - y.sum()} control)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
