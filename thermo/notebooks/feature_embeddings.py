"""
This notebook plots the Magpie feature space for the Gaultois database with
several dimensional reduction algorithms (t-SNE, UMAP) to check for clustering.
It also plots the prevalence of different chemical elements in the Gaultois
database in a histogram and onto the periodic table.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly import express as px
from pymatgen import Composition
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from thermo.data import load_gaultois
from thermo.plots import (
    hist_elemental_prevalence,
    ptable_elemental_prevalence,
    show_bar_values,
)
from thermo.utils import ROOT

# %%
features, labels = load_gaultois(
    target_cols=["formula", "rho", "seebeck", "kappa", "zT"]
)
DIR = f"{ROOT}/results/feature_embeddings"

# %% [markdown]
# # TSNE feature space embedding


# %% 2D
tsne_2d = TSNE(n_components=2).fit_transform(features)

tsne_cols = ["tsne_1", "tsne_2"]
tsne_2d = pd.DataFrame(tsne_2d, columns=tsne_cols)


# %%
tsne_2d[labels.columns] = labels
px.scatter(tsne_2d, *tsne_cols, hover_data=labels.columns)


# %% 3D
tsne_cols = ["tsne_1", "tsne_2", "tsne_3"]
tsne_3d = TSNE(n_components=3).fit_transform(features)
tsne_3d = pd.DataFrame(tsne_3d, columns=tsne_cols)


# %%
# Consider using IHS transformation to shrink outlier spread.
# tsne_3d = np.arcsinh(tsne_3d)

tsne_3d[labels.columns] = labels
px.scatter_3d(tsne_3d, *tsne_cols, hover_data=labels.columns)


# %% [markdown]
# # UMAP feature space embedding


# %% 2D
umap_2d = UMAP(n_components=2).fit_transform(features)

umap_cols = ["umap_1", "umap_2"]
umap_2d = pd.DataFrame(umap_2d, columns=umap_cols)


# %%
umap_2d[labels.columns] = labels
px.scatter(umap_2d, *umap_cols, hover_data=labels.columns)


# %% 3D
umap_cols = ["umap_1", "umap_2", "umap_3"]
umap_3d = UMAP(n_components=3).fit_transform(features)
umap_3d = pd.DataFrame(umap_3d, columns=umap_cols)


# %%
umap_3d[labels.columns] = labels
px.scatter_3d(umap_3d, *umap_cols, hover_data=labels.columns)


# %% [markdown]
# # PCA feature space embedding


# %% 2D
pca_2d = PCA(n_components=2).fit_transform(features)

pca_cols = ["pca_1", "pca_2"]
pca_2d = pd.DataFrame(pca_2d, columns=pca_cols)


# %%
pca_2d[labels.columns] = labels
px.scatter(pca_2d, *pca_cols, hover_data=labels.columns)


# %% 3D
pca_cols = ["pca_1", "pca_2", "pca_3"]
pca_3d = PCA(n_components=3).fit_transform(features)
pca_3d = pd.DataFrame(pca_3d, columns=pca_cols)


# %%
pca_3d[labels.columns] = labels
px.scatter_3d(pca_3d, *pca_cols, hover_data=labels.columns)


# %%
ptable_elemental_prevalence(labels.formula.values, log_scale=True)
plt.savefig(f"{DIR}/ptable-elemental-prevalence.pdf", bbox_inches="tight")


# %%
hist_elemental_prevalence(labels.formula.values, keep_top=20)
plt.savefig(f"{DIR}/hist-elemental-prevalence.pdf", bbox_inches="tight")


# %%
labels["composition"] = [Composition(x) for x in labels.formula]
# Histogram of the number of elements in each composition
x_labels, y_counts = np.unique(
    labels.composition.apply(lambda x: len(x.elements)), return_counts=True
)
ax = plt.bar(x_labels, y_counts, align="center")
plt.xticks(x_labels)
plt.xlabel("number of elements in composition")
plt.ylabel("sample count")
show_bar_values(plt.gca())
plt.savefig(f"{DIR}/hist-number-of-elements-in-composition.pdf", bbox_inches="tight")
