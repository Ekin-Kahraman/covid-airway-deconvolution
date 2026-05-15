import json

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import deconvolve


def make_reference():
    rng = np.random.default_rng(deconvolve.RANDOM_STATE)
    cell_types = ["Basal Cells", "Ciliated Cells", "T Cells"]
    obs = pd.DataFrame(
        {"cell_type": np.repeat(cell_types, 8)},
        index=[f"cell_{i}" for i in range(24)],
    )
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(12)])
    X = rng.poisson(lam=5, size=(24, 12)).astype(np.float32)
    return sc.AnnData(X=X, obs=obs, var=var)


def test_pseudo_bulk_and_model_simplex():
    adata = make_reference()
    hvg = adata.var_names.tolist()

    X_pseudo, y_pseudo, cell_types = deconvolve.generate_pseudo_bulk(
        adata, hvg, n_samples=20
    )

    assert X_pseudo.shape == (20, len(hvg))
    assert y_pseudo.shape == (20, len(cell_types))
    assert np.isfinite(X_pseudo).all()
    assert np.isfinite(y_pseudo).all()
    assert np.allclose(y_pseudo.sum(axis=1), 1.0)

    model = deconvolve.EnsembleDeconvNet(
        n_genes=len(hvg), n_types=len(cell_types), hidden_dims=(8, 12)
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_pseudo[:4])).numpy()

    assert pred.shape == (4, len(cell_types))
    assert np.all(pred >= 0)
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)


def test_analyse_results_writes_expected_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(deconvolve, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(deconvolve, "FIG_DIR", tmp_path / "figures")

    cell_types = ["Basal Cells", "Ciliated Cells", "T Cells"]
    sample_ids = ["neg1", "neg2", "pos1", "pos2"]
    conditions = pd.Series(["negative", "negative", "positive", "positive"])
    proportions = np.array(
        [
            [0.45, 0.45, 0.10],
            [0.40, 0.50, 0.10],
            [0.20, 0.35, 0.45],
            [0.25, 0.30, 0.45],
        ]
    )

    prop_df = deconvolve.analyse_results(proportions, cell_types, conditions, sample_ids)

    assert prop_df.shape == (4, 4)
    assert (tmp_path / "cell_type_proportions.csv").exists()
    assert (tmp_path / "mean_proportions_by_condition.csv").exists()
    assert (tmp_path / "figures" / "composition_difference.png").exists()
    assert (tmp_path / "figures" / "composition_by_condition.png").exists()
    assert (tmp_path / "figures" / "boxplots_by_condition.png").exists()


def test_model_metadata_records_reuse_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(deconvolve, "RESULTS_DIR", tmp_path)

    metadata = deconvolve.save_model_metadata(
        cell_types=["Basal Cells", "T Cells"],
        hvg=["GENE1", "GENE2", "GENE3"],
        correlations={
            "Basal Cells": {"r": 0.9, "p": 0.01},
            "T Cells": {"r": 0.8, "p": 0.02},
        },
        overall_r=0.85,
        rmse=0.05,
        mean_r=0.84,
        std_r=0.01,
        mean_rmse=0.06,
        nnls_r=0.5,
        nnls_rmse=0.12,
    )

    path = tmp_path / "model_metadata.json"
    on_disk = json.loads(path.read_text())
    assert metadata == on_disk
    assert on_disk["model"]["n_genes"] == 3
    assert on_disk["cell_types"] == ["Basal Cells", "T Cells"]
