#!/usr/bin/env python3
"""
COVID-19 Airway Cell Type Deconvolution

Predicts cell type proportions in bulk nasopharyngeal RNA-seq (GSE152075,
n=484) using a neural network trained on pseudo-bulk mixtures generated
from tissue-matched single-cell data (Ziegler et al. 2021, Cell).

The original Lieberman et al. (2020) analysis used CIBERSORTx with a
blood-derived immune reference (LM22) — a poor match for nasopharyngeal
tissue dominated by epithelial cells. This project uses a tissue-matched
scRNA-seq reference to deconvolve BOTH epithelial and immune cell types.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, mannwhitneyu

RANDOM_STATE = 42
N_PSEUDO_BULK = 10000     # more training data = better generalisation
CELLS_PER_SAMPLE = 500
N_TOP_GENES = 2000
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
DATA_DIR = Path("data")

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def load_reference():
    """Load Ziegler et al. nasopharyngeal scRNA-seq with cell type labels."""
    path = DATA_DIR / "ziegler2021_nasopharyngeal.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download with:\n"
            "wget -O data/ziegler2021_nasopharyngeal.h5ad "
            '"https://covid19.cog.sanger.ac.uk/submissions/release2/'
            '20210217_NasalSwab_Broad_BCH_UMMC_to_CZI.h5ad"'
        )
    print("Loading scRNA-seq reference...")
    adata = sc.read_h5ad(path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    cell_type_col = None
    for col in ["Coarse_Cell_Annotations", "cell_type", "CellType", "celltype",
                 "Cell_type", "annotated_cell_type", "cell_ontology_class",
                 "Annotation", "ann_level_1", "ann_finest_level"]:
        if col in adata.obs.columns:
            cell_type_col = col
            break

    if cell_type_col is None:
        print(f"  Available obs columns: {adata.obs.columns.tolist()}")
        raise ValueError("No cell type annotation column found")

    adata.obs["cell_type"] = adata.obs[cell_type_col].astype(str)

    # Remove unassigned / doublet cells
    mask = ~adata.obs["cell_type"].isin(["nan", "Unknown", "Doublet", "unassigned", ""])
    adata = adata[mask].copy()

    # Exclude cell types unsuitable for deconvolution:
    # - Rare types (<50 cells): pseudo-bulk training just recycles the same cells
    # - Erythroblasts: blood contamination in nasal swabs, not airway cells
    MIN_CELLS = 50
    type_counts = adata.obs["cell_type"].value_counts()
    rare = type_counts[type_counts < MIN_CELLS].index.tolist()
    blood_contam = ["Erythroblasts"]
    exclude = list(set(rare + blood_contam))
    if exclude:
        print(f"  Excluding {len(exclude)} types: {exclude}")
        adata = adata[~adata.obs["cell_type"].isin(exclude)].copy()

    cell_types = adata.obs["cell_type"].value_counts()
    print(f"  Cell types ({len(cell_types)}):")
    for ct, count in cell_types.items():
        print(f"    {ct}: {count} ({count/adata.n_obs*100:.1f}%)")

    return adata


def load_bulk():
    """Load GSE152075 bulk RNA-seq count matrix."""
    cache = DATA_DIR / "GSE152075_raw_counts_GEO.txt.gz"
    if cache.exists():
        print(f"Loading cached bulk data from {cache}")
        counts = pd.read_csv(cache, sep=r"\s+", index_col=0, compression="gzip")
    else:
        print("Downloading GSE152075 from GEO...")
        url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152075/suppl/"
               "GSE152075_raw_counts_GEO.txt.gz")
        counts = pd.read_csv(url, sep=r"\s+", index_col=0, compression="gzip")
        cache.parent.mkdir(exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, cache)

    print(f"  Bulk data: {counts.shape[0]} genes x {counts.shape[1]} samples")
    conditions = pd.Series(
        ["positive" if s.startswith("POS") else "negative" for s in counts.columns],
        index=counts.columns, name="condition"
    )
    print(f"  {(conditions == 'positive').sum()} positive, "
          f"{(conditions == 'negative').sum()} negative")
    return counts, conditions


def prepare_gene_space(adata_ref, bulk_counts):
    """Find shared highly variable genes between reference and bulk."""
    ref_genes = set(adata_ref.var_names)
    bulk_genes = set(bulk_counts.index)
    shared = sorted(ref_genes & bulk_genes)
    print(f"  Shared genes: {len(shared)}")

    adata_sub = adata_ref[:, adata_ref.var_names.isin(shared)].copy()
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=min(N_TOP_GENES, len(shared)))
    hvg = adata_sub.var_names[adata_sub.var.highly_variable].tolist()
    print(f"  HVGs selected: {len(hvg)}")
    return hvg


def generate_pseudo_bulk(adata_ref, hvg, n_samples=N_PSEUDO_BULK):
    """Create synthetic bulk samples by mixing single cells in known proportions.

    Each pseudo-bulk sample draws a random Dirichlet proportion vector
    across cell types, samples cells accordingly, and sums their profiles.
    This generates diverse training data spanning the full simplex of
    possible cell compositions.
    """
    print(f"Generating {n_samples} pseudo-bulk samples...")
    cell_types = sorted(adata_ref.obs["cell_type"].unique())
    n_types = len(cell_types)

    X = adata_ref[:, hvg].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    expr = pd.DataFrame(X, index=adata_ref.obs_names, columns=hvg)

    type_indices = {}
    for ct in cell_types:
        mask = adata_ref.obs["cell_type"] == ct
        type_indices[ct] = expr.index[mask].tolist()

    bulk_profiles = np.zeros((n_samples, len(hvg)))
    true_fractions = np.zeros((n_samples, n_types))

    # Dirichlet alpha weighted by reference prevalence — generates
    # realistic proportions where common types (ciliated) dominate and
    # rare types (DCs) appear infrequently, matching real tissue composition.
    ref_prevalence = np.array([len(type_indices[ct]) for ct in cell_types], dtype=float)
    ref_prevalence = ref_prevalence / ref_prevalence.sum()
    alpha = ref_prevalence * n_types  # scale so mean matches reference

    for i in range(n_samples):
        props = np.random.dirichlet(alpha)
        true_fractions[i] = props

        mixed = np.zeros(len(hvg))
        for j, ct in enumerate(cell_types):
            n_cells = int(props[j] * CELLS_PER_SAMPLE)
            if n_cells == 0:
                continue
            n_cells = min(n_cells, len(type_indices[ct]))
            sampled = np.random.choice(type_indices[ct], size=n_cells, replace=True)
            mixed += expr.loc[sampled].sum(axis=0).values

        bulk_profiles[i] = mixed

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_samples}")

    # Augment with realistic noise to improve generalisation to real bulk data.
    # Pseudo-bulk is clean; real bulk has dropout, library size variation, and
    # technical noise that the model needs to tolerate.

    # 1. Gene dropout: randomly zero out 2-8% of genes per sample
    #    (real bulk has some genes below detection limit)
    for i in range(n_samples):
        dropout_rate = np.random.uniform(0.02, 0.08)
        dropout_mask = np.random.random(len(hvg)) > dropout_rate
        bulk_profiles[i] *= dropout_mask

    # 2. Library size variation: scale by a log-normal factor
    #    (real samples vary in sequencing depth)
    lib_factors = np.random.lognormal(mean=0, sigma=0.15, size=n_samples)
    bulk_profiles = bulk_profiles * lib_factors[:, np.newaxis]

    # 3. Gaussian noise: small additive noise (technical variation)
    noise_scale = np.std(bulk_profiles[bulk_profiles > 0]) * 0.05
    noise = np.random.normal(0, noise_scale, bulk_profiles.shape)
    bulk_profiles = np.maximum(bulk_profiles + noise, 0)

    # Log-transform and normalise
    bulk_profiles = np.log2(bulk_profiles + 1)
    row_sums = bulk_profiles.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bulk_profiles = bulk_profiles / row_sums

    print(f"  Done: {n_samples} samples, {n_types} cell types (with noise augmentation)")
    return bulk_profiles, true_fractions, cell_types


class DeconvNet(nn.Module):
    """Feedforward network for cell type deconvolution.

    Deliberately simple: 2 hidden layers with dropout. With n=5000
    training samples and ~2000 gene features, deeper architectures
    overfit without improving accuracy.
    """
    def __init__(self, n_genes, n_types, hidden_dim=HIDDEN_DIM):
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
        logits = self.net(x)
        return torch.softmax(logits, dim=1)


def train_model(X_train, y_train, X_val, y_val, n_types):
    """Train the deconvolution neural network."""
    n_genes = X_train.shape[1]
    model = DeconvNet(n_genes, n_types)

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    PATIENCE = 20
    train_losses, val_losses = [], []

    print(f"\nTraining ({EPOCHS} epochs, {n_genes} genes, {n_types} cell types)...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(torch.clamp(pred, min=1e-8).log(), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(X_tr)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(torch.clamp(val_pred, min=1e-8).log(), y_v).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — train: {train_loss:.4f}, val: {val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    return model, train_losses, val_losses


def deconvolve_bulk(model, bulk_counts, hvg):
    """Apply trained model to real GSE152075 bulk samples."""
    print("\nDeconvolving bulk samples...")
    bulk_sub = bulk_counts.loc[bulk_counts.index.isin(hvg)].T
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

    print(f"  Deconvolved {pred.shape[0]} samples into {pred.shape[1]} cell types")
    return pred


def plot_training(train_losses, val_losses):
    """Plot training curves."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Training", linewidth=2)
    ax.plot(val_losses, label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence Loss")
    ax.set_title("Deconvolution Model Training")
    ax.legend()
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_model(model, X_val, y_val, cell_types):
    """Measure accuracy on held-out pseudo-bulk samples."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_val)).numpy()

    correlations = {}
    for i, ct in enumerate(cell_types):
        r, p = pearsonr(y_val[:, i], pred[:, i])
        correlations[ct] = {"r": r, "p": p}

    overall_r, _ = pearsonr(y_val.flatten(), pred.flatten())
    rmse = np.sqrt(np.mean((y_val - pred) ** 2))

    print(f"\n  Validation Pearson r = {overall_r:.3f}, RMSE = {rmse:.4f}")
    for ct, vals in sorted(correlations.items(), key=lambda x: -x[1]["r"]):
        print(f"    {ct}: r = {vals['r']:.3f}")

    # Scatter plots — 4 columns, enough rows for all cell types
    n_types = len(cell_types)
    cols = 4
    rows = (n_types + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if n_types == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for i, ct in enumerate(cell_types):
        axes[i].scatter(y_val[:, i], pred[:, i], s=5, alpha=0.3, color="#1976D2")
        axes[i].plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.5)
        axes[i].set_xlabel("True", fontsize=8)
        axes[i].set_ylabel("Predicted", fontsize=8)
        axes[i].set_title(f"{ct}\nr={correlations[ct]['r']:.3f}", fontsize=8, pad=4)
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].tick_params(labelsize=7)

    for j in range(n_types, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Pseudo-bulk Validation — r = {overall_r:.3f}, RMSE = {rmse:.4f}",
                 fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_DIR / "validation_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return correlations, overall_r


def analyse_results(proportions, cell_types, conditions, sample_ids):
    """Analyse and visualise deconvolution results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    prop_df = pd.DataFrame(proportions, columns=cell_types, index=sample_ids)
    prop_df["condition"] = conditions.values
    prop_df.to_csv(RESULTS_DIR / "cell_type_proportions.csv")

    # Grouped bar chart — side by side comparison
    neg_means = prop_df.loc[prop_df["condition"] == "negative", cell_types].mean()
    pos_means = prop_df.loc[prop_df["condition"] == "positive", cell_types].mean()

    # Sort by absolute difference
    order = (pos_means - neg_means).abs().sort_values(ascending=True).index
    neg_sorted = neg_means[order]
    pos_sorted = pos_means[order]

    y = np.arange(len(order))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y - height/2, neg_sorted.values, height, label=f"Negative (n=54)",
            color="#2196F3", alpha=0.85)
    ax.barh(y + height/2, pos_sorted.values, height, label=f"Positive (n=430)",
            color="#E53935", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Mean proportion")
    ax.set_title("Cell Type Composition — COVID-19 Positive vs Negative", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.1, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "composition_by_condition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Boxplots of top changing cell types
    pos_means = prop_df.loc[prop_df["condition"] == "positive", cell_types].mean()
    neg_means = prop_df.loc[prop_df["condition"] == "negative", cell_types].mean()
    diff = (pos_means - neg_means).abs().sort_values(ascending=False)
    top_changed = diff.head(min(8, len(diff))).index.tolist()

    n_plots = len(top_changed)
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes).flatten() if n_plots > 1 else [axes]

    for i, ct in enumerate(top_changed):
        neg_vals = prop_df.loc[prop_df["condition"] == "negative", ct]
        pos_vals = prop_df.loc[prop_df["condition"] == "positive", ct]
        bp = axes[i].boxplot([neg_vals, pos_vals], tick_labels=["Neg", "Pos"],
                            patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#90CAF9")
        bp["boxes"][1].set_facecolor("#EF9A9A")
        axes[i].set_title(ct, fontsize=10)
        axes[i].set_ylabel("Proportion")

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cell Type Proportions — Positive vs Negative", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplots_by_condition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary with statistical tests
    summary = prop_df.groupby("condition")[cell_types].mean().T
    if "positive" in summary.columns and "negative" in summary.columns:
        summary["diff"] = summary["positive"] - summary["negative"]

        # Mann-Whitney U test for each cell type
        pvals = {}
        for ct in cell_types:
            neg = prop_df.loc[prop_df["condition"] == "negative", ct]
            pos = prop_df.loc[prop_df["condition"] == "positive", ct]
            _, p = mannwhitneyu(neg, pos, alternative="two-sided")
            pvals[ct] = p
        summary["p_value"] = pd.Series(pvals)
        summary["significant"] = summary["p_value"] < 0.05
        summary = summary.sort_values("diff", ascending=False)
    summary.to_csv(RESULTS_DIR / "mean_proportions_by_condition.csv")

    print(f"\n  Results saved to {RESULTS_DIR}/")
    return prop_df


def main():
    print("=" * 60)
    print("COVID-19 Airway Cell Type Deconvolution")
    print("=" * 60)

    print("\n--- Load reference scRNA-seq ---")
    adata_ref = load_reference()

    print("\n--- Load bulk RNA-seq ---")
    bulk_counts, conditions = load_bulk()

    print("\n--- Prepare gene space ---")
    hvg = prepare_gene_space(adata_ref, bulk_counts)

    print("\n--- Generate pseudo-bulk training data ---")
    X_pseudo, y_pseudo, cell_types = generate_pseudo_bulk(adata_ref, hvg)

    X_train, X_val, y_train, y_val = train_test_split(
        X_pseudo, y_pseudo, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n--- Train neural network ---")
    model, train_losses, val_losses = train_model(
        X_train, y_train, X_val, y_val, n_types=len(cell_types)
    )
    plot_training(train_losses, val_losses)

    print("\n--- Validate on pseudo-bulk ---")
    correlations, overall_r = validate_model(model, X_val, y_val, cell_types)

    print("\n--- Deconvolve GSE152075 ---")
    proportions = deconvolve_bulk(model, bulk_counts, hvg)

    print("\n--- Analyse results ---")
    analyse_results(proportions, cell_types, conditions, bulk_counts.columns)

    torch.save(model.state_dict(), RESULTS_DIR / "deconvolution_model.pt")

    print("\n" + "=" * 60)
    print(f"Complete. Validation r = {overall_r:.3f}")
    print(f"{len(cell_types)} cell types, {proportions.shape[0]} bulk samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
