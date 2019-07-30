# %%
import pandas as pd
from plotly import express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from data import load_gaultois

# %%
features, labels = load_gaultois(
    target_cols=["formula", "rho", "seebeck", "kappa", "zT"]
)


# %% [markdown]
# # TSNE feature space embedding


# %% 2D
tsne_2d = TSNE(n_components=2).fit_transform(features)

tsne_cols = [f"tsne_{i}" for i in range(1, 4)]
tsne_2d = pd.DataFrame(tsne_2d, columns=tsne_cols[:-1])


# %%
tsne_2d[labels.columns] = labels
px.scatter(tsne_2d, *tsne_cols[:-1], hover_data=labels.columns)


# %% 3D
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

umap_cols = [f"umap_{i}" for i in range(1, 4)]
umap_2d = pd.DataFrame(umap_2d, columns=umap_cols[:-1])


# %%
umap_2d[labels.columns] = labels
px.scatter(umap_2d, *umap_cols[:-1], hover_data=labels.columns)


# %% 3D
umap_3d = UMAP(n_components=3).fit_transform(features)
umap_3d = pd.DataFrame(umap_3d, columns=umap_cols)


# %%
umap_3d[labels.columns] = labels
px.scatter_3d(umap_3d, *umap_cols, hover_data=labels.columns)


# %% [markdown]
# # PCA feature space embedding


# %% 2D
pca_2d = PCA(n_components=2).fit_transform(features)

pca_cols = [f"pca_{i}" for i in range(1, 4)]
pca_2d = pd.DataFrame(pca_2d, columns=pca_cols[:-1])


# %%
umap_2d[labels.columns] = labels
px.scatter(pca_2d, *pca_cols[:-1], hover_data=labels.columns)


# %% 3D
pca_3d = PCA(n_components=3).fit_transform(features)
pca_3d = pd.DataFrame(pca_3d, columns=pca_cols)


# %%
pca_3d[labels.columns] = labels
px.scatter_3d(pca_3d, *pca_cols, hover_data=labels.columns)
