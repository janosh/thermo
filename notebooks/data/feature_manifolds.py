"""This notebook plots the Magpie feature space for the Gaultois database with
several dimensional reduction algorithms (t-SNE, UMAP, PCA) to check for clustering.
"""


# %%
import matplotlib.pyplot as plt
import pandas as pd
from plotly import express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from thermo.data import load_gaultois


# %%
magpie_features, targets = load_gaultois(
    target_cols=["T", "formula", "rho", "seebeck", "kappa", "zT"]
)
# temperature is by far the most indicative feature for thermoelectric performance
# but not specific to any material, consider removing it from clustering
magpie_features = magpie_features.drop(columns="T")


# %% [markdown]
# # TSNE feature space embedding


# %% 2D
tsne_2d = TSNE(n_components=2).fit_transform(magpie_features)

tsne_cols = ["tsne_1", "tsne_2"]
tsne_2d = pd.DataFrame(tsne_2d, columns=tsne_cols)


# %%
tsne_2d[targets.columns] = targets
px.scatter(tsne_2d, *tsne_cols, hover_data=targets.columns)


# %%
tsne_2d.assign(**targets).dropna().plot.scatter(
    x="tsne_1", y="tsne_2", c="zT", cmap="hot", alpha=0.5, figsize=[12, 8], s=100
)


# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("2d TSNE of Magpie features for Gaultois dataset by temperatures")
for ax, [key, group] in zip(
    axs.flat, tsne_2d.assign(**targets).dropna().groupby("T", as_index=False)
):
    ax.scatter(group["tsne_1"], group["tsne_2"], c=group["zT"], cmap="hot", alpha=0.5)
    ax.set_title(f"{key} K")

# plt.savefig("2d-tsne-of-magpie-features-for-gaultois-dataset-by-temps.pdf")


# %% 3D
tsne_cols = ["tsne_1", "tsne_2", "tsne_3"]
tsne_3d = TSNE(n_components=3).fit_transform(magpie_features)
tsne_3d = pd.DataFrame(tsne_3d, columns=tsne_cols)


# %%
# Consider using IHS transformation to shrink outlier spread.
# tsne_3d = np.arcsinh(tsne_3d)

tsne_3d[targets.columns] = targets
px.scatter_3d(tsne_3d, *tsne_cols, hover_data=targets.columns)


# %% [markdown]
# # UMAP feature space embedding


# %% 2D
umap_2d = UMAP(n_components=2).fit_transform(magpie_features)

umap_cols = ["umap_1", "umap_2"]
umap_2d = pd.DataFrame(umap_2d, columns=umap_cols)


# %%
umap_2d[targets.columns] = targets
px.scatter(umap_2d, *umap_cols, hover_data=targets.columns)


# %% 3D
umap_cols = ["umap_1", "umap_2", "umap_3"]
umap_3d = UMAP(n_components=3).fit_transform(magpie_features)
umap_3d = pd.DataFrame(umap_3d, columns=umap_cols)


# %%
umap_3d[targets.columns] = targets
px.scatter_3d(umap_3d, *umap_cols, hover_data=targets.columns)


# %% [markdown]
# # PCA feature space embedding


# %% 2D
pca_2d = PCA(n_components=2).fit_transform(magpie_features)

pca_cols = ["pca_1", "pca_2"]
pca_2d = pd.DataFrame(pca_2d, columns=pca_cols)


# %%
pca_2d[targets.columns] = targets
px.scatter(pca_2d, *pca_cols, hover_data=targets.columns)


# %% 3D
pca_cols = ["pca_1", "pca_2", "pca_3"]
pca_3d = PCA(n_components=3).fit_transform(magpie_features)
pca_3d = pd.DataFrame(pca_3d, columns=pca_cols)


# %%
pca_3d[targets.columns] = targets
px.scatter_3d(pca_3d, *pca_cols, hover_data=targets.columns)
