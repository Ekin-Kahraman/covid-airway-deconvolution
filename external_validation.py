"""External validation of the deconvolution model on GSE163151 (Ng et al. 2021).

Applies the trained PyTorch ensemble model (trained on GSE152075 with Ziegler et al. reference)
to an independent nasopharyngeal RNA-seq cohort. Tests whether the model generalizes beyond
the training dataset.

GSE163151: 404 NP samples — 145 COVID-19, 31 controls, 140 other viral, 82 non-viral, 6 bacterial.
"""

import json

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import mannwhitneyu

RESULTS_DIR = Path("results/external_validation")
DATA_DIR = Path("data/gse163151")
MODEL_PATH = Path("results/deconvolution_model.pt")
MODEL_METADATA_PATH = Path("results/model_metadata.json")

CELL_TYPES = [
    "B Cells", "Basal Cells", "Ciliated Cells", "Dendritic Cells",
    "Deuterosomal Cells", "Developing Ciliated Cells",
    "Developing Secretory and Goblet Cells", "Goblet Cells",
    "Ionocytes", "Macrophages", "Mitotic Basal Cells",
    "Secretory Cells", "Squamous Cells", "T Cells"
]


class DeconvNet(nn.Module):
    def __init__(self, n_genes, n_types, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_types),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class EnsembleDeconvNet(nn.Module):
    def __init__(self, n_genes, n_types, hidden_dims=(128, 256, 512)):
        super().__init__()
        self.models = nn.ModuleList([
            DeconvNet(n_genes, n_types, h) for h in hidden_dims
        ])

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0)


def get_hvg_list():
    """Re-derive the HVG list from the Ziegler reference (same as training)."""
    print("Loading reference to get HVG list...")
    adata_ref = sc.read_h5ad("data/ziegler2021_nasopharyngeal.h5ad")

    for col in ["Coarse_Cell_Annotations", "cell_type", "CellType"]:
        if col in adata_ref.obs.columns:
            ct_col = col
            break

    adata_ref = adata_ref[~adata_ref.obs[ct_col].isin(
        ["nan", "Unknown", "Doublet", "unassigned", ""]
    )]

    # Get shared genes with the new bulk data without loading the full matrix.
    new_bulk = pd.read_csv(DATA_DIR / "count_matrix.csv", index_col=0, usecols=[0])
    ref_genes = set(adata_ref.var_names)
    bulk_genes = set(new_bulk.index)
    shared = sorted(ref_genes & bulk_genes)
    print(f"  Shared genes (ref and new bulk): {len(shared)}")

    adata_sub = adata_ref[:, [g for g in shared if g in adata_ref.var_names]].copy()
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000)
    hvg = adata_sub.var_names[adata_sub.var.highly_variable].tolist()
    print(f"  HVGs selected: {len(hvg)}")
    return hvg


def deconvolve(model, counts, hvg):
    """Apply model to a count matrix using same preprocessing as training."""
    bulk_sub = counts.loc[counts.index.isin(hvg)].T
    for g in hvg:
        if g not in bulk_sub.columns:
            bulk_sub[g] = 0
    bulk_sub = bulk_sub[hvg]

    X = np.log2(bulk_sub.values.astype(np.float32) + 1)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums

    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X)).numpy()
    return pred


def load_model_bundle():
    """Load model weights plus the saved gene and cell-type contract."""
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text())
        hvg = metadata["hvg"]
        cell_types = metadata["cell_types"]
        hidden_dims = tuple(metadata.get("model", {}).get("hidden_dims", [128, 256, 512]))
        n_genes = len(hvg)
        n_types = len(cell_types)
    else:
        print("No model_metadata.json found; falling back to derived HVGs and hard-coded cell types.")
        hvg = get_hvg_list()
        cell_types = CELL_TYPES
        hidden_dims = (128, 256, 512)
        n_genes = 2000
        n_types = len(CELL_TYPES)

    model = EnsembleDeconvNet(n_genes, n_types, hidden_dims=hidden_dims)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, hvg, cell_types


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading trained model...")
    model, hvg, cell_types = load_model_bundle()

    # Load new data
    print("\nLoading GSE163151 count matrix...")
    counts = pd.read_csv(DATA_DIR / "count_matrix.csv", index_col=0)
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    print(f"  Shape: {counts.shape}")

    # Deconvolve
    print("\nDeconvolving 404 samples...")
    bulk_sub = counts.loc[counts.index.isin(hvg)].T
    for g in hvg:
        if g not in bulk_sub.columns:
            bulk_sub[g] = 0
    bulk_sub = bulk_sub[hvg]

    X = np.log2(bulk_sub.values.astype(np.float32) + 1)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums

    with torch.no_grad():
        proportions = model(torch.FloatTensor(X)).numpy()

    prop_df = pd.DataFrame(proportions, columns=cell_types)
    prop_df["condition"] = meta["disease state"].values
    prop_df["pathogen"] = meta["pathogen"].values
    prop_df.index = counts.columns
    prop_df.to_csv(RESULTS_DIR / "gse163151_proportions.csv")
    print(f"  Done: {prop_df.shape}")

    # COVID vs Controls
    print("\n=== COVID-19 vs Donor Controls ===")
    covid = prop_df[prop_df["condition"] == "COVID-19"]
    control = prop_df[prop_df["condition"] == "Donor control"]
    print(f"  COVID: n={len(covid)}, Controls: n={len(control)}")

    comparison = []
    for ct in cell_types:
        diff = covid[ct].mean() - control[ct].mean()
        _, pval = mannwhitneyu(covid[ct], control[ct], alternative="two-sided")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        comparison.append({
            "cell_type": ct, "covid_mean": round(covid[ct].mean(), 4),
            "control_mean": round(control[ct].mean(), 4),
            "difference": round(diff, 4), "p_value": pval, "significant": sig
        })
        print(f"  {ct:>42}: {diff:>+7.3f} (p={pval:.2e}) {sig}")

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(RESULTS_DIR / "covid_vs_control.csv", index=False)

    # Cross-cohort concordance with GSE152075
    print("\n=== Cross-Cohort Concordance (GSE163151 vs GSE152075) ===")
    orig_props = pd.read_csv("results/cell_type_proportions.csv", index_col=0)
    if "condition" in orig_props.columns:
        orig_covid = orig_props[orig_props["condition"] == "positive"][cell_types].mean()
        orig_control = orig_props[orig_props["condition"] == "negative"][cell_types].mean()
        orig_diff = orig_covid - orig_control

        new_diff = covid[cell_types].mean() - control[cell_types].mean()

        concordant = sum(1 for ct in cell_types if np.sign(orig_diff[ct]) == np.sign(new_diff[ct]))
        print(f"  Direction concordance: {concordant}/{len(cell_types)} ({concordant/len(cell_types)*100:.0f}%)")

        from scipy.stats import pearsonr
        r, p = pearsonr(orig_diff.values, new_diff.values)
        print(f"  Effect size correlation: r={r:.3f}, p={p:.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
