"""This notebook screens synthesizable materials from ICSD and COD
for viable thermoelectrics.
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatviz as pmv
import tensorflow as tf
from gurobipy import GRB, Model, quicksum
from tf_mnf.models import MNFFeedForward
from tqdm import trange

from thermo.data import dropna, load_gaultois, load_screen, normalize, train_test_split
from thermo.plots import plot_output
from thermo.utils import ROOT


# %%
tf.random.set_seed(0)

gaultois_magpie_feas, targets = load_gaultois(target_cols=["formula", "zT", "T"])
targets, gaultois_magpie_feas = dropna(targets, gaultois_magpie_feas)

targets.groupby("T", as_index=False)["zT"].mean().plot(x="T", y="zT", ylim=[0, None])
plt.title("mean zT by temperature")

zT = targets.pop("zT")

candidates, screen_features = load_screen()
screen_features, [screen_mean, screen_std] = normalize(screen_features)

# normalize temperature separately as it not included in screening features yet
screen_mean = pd.Series({"T": gaultois_magpie_feas["T"].mean()}).append(screen_mean)
screen_std = pd.Series({"T": gaultois_magpie_feas["T"].std()}).append(screen_std)

gaultois_magpie_feas, _ = normalize(
    gaultois_magpie_feas, mean=screen_mean, std=screen_std
)

[X_train, zT_train], [X_test, zT_test] = train_test_split(gaultois_magpie_feas, zT)


# %% duplicate each row with new column temperature set to 700 K or 1000 K
candidates = candidates.assign(T=700).append(candidates.assign(T=1000))
candidates = candidates.set_index(["id", "T"], append=True)

# temperature needs to be in front in screen_features
# assign same normalized temperature values used during training
screen_features.insert(0, "T", 0.532785)
screen_features = screen_features.append(screen_features.assign(T=1.687777))


# %% ExtraTreesRegressor feature importance not used here since it hurts performance
# zT MAE with important features: 0.072, will all Magpie features: 0.064

# from sklearn.ensemble import ExtraTreesRegressor
# forest = ExtraTreesRegressor(n_estimators=250, random_state=0)

# forest.fit(magpie_features, zT)
# importance_idx = np.argsort(forest.feature_importances_)[::-1]
# # keep only the 30 most relevant features
# magpie_features = magpie_features.iloc[:, importance_idx[:30]]
# magpie_features.columns.to_list()


# %% [markdown]
# # Multiplicative Normalizing Flow


# %%
mnf_model = MNFFeedForward(layer_sizes=(100, 50, 10, 1))
optimizer = tf.optimizers.Adam()


# %%
def loss_factory(model):
    def loss_fn(y_true, y_pred):
        mse = tf.metrics.mse(y_true, y_pred)
        # KL div is reweighted such that it's applied once per epoch
        kl_loss = model.kl_div() / len(X_train) * 1e-3
        return mse + kl_loss

    return loss_fn


# %%
mnf_model.compile(optimizer, loss=loss_factory(mnf_model), metrics=["mae"])


# %%
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae", patience=50, restore_best_weights=True
)


# %% train with train-test split to find good number of epochs for screening
batch = 32
train_hist = mnf_model.fit(
    X_train.astype("float32"),
    zT_train,
    validation_data=(X_test, zT_test),
    batch_size=batch,
    epochs=300,
    callbacks=[stop_early],
)

pd.DataFrame(train_hist.history).plot(
    xlabel="epoch", ylabel=r"$|zT_\mathrm{true} - zT_\mathrm{pred}|$"
)


# %%
n_preds = 500
mnf_preds = mnf_model(X_test.values.repeat(n_preds, axis=0))
mnf_preds = mnf_preds.numpy().reshape(-1, n_preds).T


# %%
# Use pmv.qq_gaussian which checks std calibration by comparing quantiles of
# z_score = (y_true - y_pred) / y_std against those of a Gaussian. Used
# here to eye-ball scaling factor for NF uncertainties.
plt.title("scaled by 1.5")
pmv.qq_gaussian(zT_test.values, mnf_preds.mean(0), 1.5 * mnf_preds.std(0))
zT_pred = mnf_preds.mean(0)
zT_std = 1.5 * mnf_preds.std(0)


# %%
plot_output(zT_test.values, zT_pred, zT_std, title="Magpie + MNF")


# %% [markdown]
# # Screening


# %%
screen_model = MNFFeedForward(layer_sizes=(100, 50, 10, 1))


# %%
screen_model.compile(optimizer, loss=loss_factory(screen_model), metrics=["mae"])


# %%
epochs = 140
screen_hist = screen_model.fit(
    gaultois_magpie_feas, zT, batch_size=batch, epochs=epochs
)

pd.DataFrame(screen_hist.history).plot(
    xlabel="epoch", ylabel=r"$|zT_\mathrm{true} - zT_\mathrm{pred}|$"
)


# %%
screen_preds, n_preds = [], 50
for _ in trange(n_preds):
    preds = screen_model.predict(screen_features.astype("float32"))
    screen_preds.append(preds)

screen_preds = np.array(screen_preds).squeeze()


# %%
screen_preds_df = pd.DataFrame(
    screen_preds.T,
    columns=[f"pred_{idx}" for idx in range(n_preds)],
    index=candidates.index,
)
screen_preds_df.to_csv(
    f"screen_preds-{epochs=}-{batch=}-{n_preds=}.csv", float_format="%.3g"
)


# %%
candidates["zT_pred"] = zT_pred = screen_preds.mean(0)
candidates["zT_std"] = zT_std = screen_preds.std(0) * 1.5


# %%
candidates.sort_values(by="zT_pred", ascending=False).to_csv(
    f"greedy-candidates-{epochs=}-{batch=}-{n_preds=}.csv"
)
# candidates = pd.read_csv(
#     "greedy-candidates-epochs=140-batch=32-n_preds=50.csv", index_col=[0, "id", "T"]
# )
# zT_pred = candidates["zT_pred"]
# zT_std = candidates["zT_std"]


# %%
pmv.density_hexbin_with_hist(candidates.zT_std, candidates.zT_pred)


# %%
min_zT_pred, max_zT_std = 1.5, 0.15
_, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))
plt.subplots_adjust(wspace=0.1)

for ax, cmap, (temp, group) in zip(
    axs, ["Blues", "Reds"], candidates.reset_index().groupby("T")
):
    pmv.density_scatter(
        xs=group.zT_std.values,
        ys=group.zT_pred.values,
        label=f"{temp} K",
        identity=False,
        stats=False,
        color_map=cmap,
        alpha=0.3,
        ax=ax,
    )
    ax.set_xlabel(r"$zT_\mathrm{std}$")
    ax.set_ylabel(r"$zT_\mathrm{pred}$")

    # ensure v/h lines reach to edge of plot
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)

    # add v/h lines to indicate boundaries of low-risk high-return selection
    ax.vlines(x=max_zT_std, ymin=min_zT_pred, ymax=ymax, color="gray")
    ax.vlines(
        x=max_zT_std, ymin=ymin, ymax=min_zT_pred, color="gray", linestyles="dashed"
    )

    ax.hlines(y=min_zT_pred, xmin=xmin, xmax=max_zT_std, color="gray")
    ax.hlines(
        y=min_zT_pred, xmin=max_zT_std, xmax=xmax, color="gray", linestyles="dashed"
    )

    ax.fill_between([xmin, max_zT_std], min_zT_pred, ymax, alpha=0.2)
    ax.legend()

    # add axes legend
    leg = ax.get_legend()
    leg.legend_handles[0].set_color(cmap[:-1])

ax.text(max_zT_std, ymin, r"max $zT_\mathrm{std}$", ha="left", va="bottom")
ax.text(xmax, min_zT_pred, r"min $zT_\mathrm{pred}$", ha="right", va="top")

plt.suptitle("Mean vs std.dev. of predicted zT at different temperatures")

pmv.save_fig(ax, "zT_pred-vs-zT_std.png", bbox_inches="tight", dpi=300)


# %%
lrhr_idx = np.logical_and(zT_std < max_zT_std, zT_pred > min_zT_pred)
print(lrhr_candidates := candidates[lrhr_idx])


# %%
corr_mat = np.corrcoef(screen_preds.T[lrhr_candidates.index.get_level_values(0)])
corr_mat = pd.DataFrame(
    corr_mat, index=lrhr_candidates.index, columns=lrhr_candidates.formula
)


# %%
# corr_mat.to_csv("correlation_matrix.csv", float_format="%g")
corr_mat = pd.read_csv("correlation_matrix.csv", index_col=[0, "id", "T"])


# %%
color_ax = plt.matshow(corr_mat)
plt.colorbar(color_ax, fraction=0.047, pad=0.02)
plt.gcf().set_size_inches(12, 12)
plt.title("MNF correlation matrix")
# pmv.save_fig(color_ax, "correlation_matrix_mnf.png", bbox_inches="tight", dpi=200)


# %%
# Ensure the correlation matrix contains significant eigenvalues larger than
# the maximum expected in a random matrix based on Marchenko_pastur distribution
ax = pmv.marchenko_pastur(corr_mat, gamma=len(corr_mat) / n_preds)
p, N = n_preds, len(corr_mat)
ax.set_title(
    "Marchenko-Pastur distribution of the MNF zT correlation matrix\n"
    f"{p = } MNF preds, {N = } candidate materials, gamma = p / N = {p / N:.2f}"
)
ax.set(yscale="log")
pmv.save_fig(ax, "corr-mat-marchenko-pastur-dist.png")


# %%
# Set environment variable GRB_LICENSE_FILE so that Gurobi finds its license file.
# An academic license can be obtained for free at
# https://gurobi.com/downloads/end-user-license-agreement-academic.
os.environ["GRB_LICENSE_FILE"] = f"{ROOT}/hpc/gurobi.lic"
# Create a model for solving the quadratic optimization problem of selecting p out of n
# materials with least pairwise correlation according to the correlation matrix zT_corr.
grb_model = Model("quadratic_problem")
grb_model.params.LogFile = "gurobi.log"
# os.remove("gurobi.log")


# %%
# Create decision variables.
dec_vars = grb_model.addVars(len(lrhr_candidates), vtype=GRB.BINARY).values()


# %%
# Define the model objective to minimize the sum of pairwise correlations.
obj = corr_mat.dot(dec_vars).dot(dec_vars)
grb_model.setObjective(obj, GRB.MINIMIZE)


# %%
# Add L1 constraint on dec_vars so that the optimization returns at least
# n_select formulas. If the model finds more at equal correlation, even better.
n_select = 250
constr = grb_model.addConstr(quicksum(dec_vars) >= n_select, "l1_norm")


# %%
grb_model.optimize()


# %%
# Save selected materials to dataframe and CSV file.
dec_vals = [bool(var.x) for var in dec_vars]
# printed 0.907
print(f"final objective value: {corr_mat.dot(dec_vals).dot(dec_vals) = :.3f}")
gurobi_candidates = lrhr_candidates[dec_vals]
gurobi_candidates.to_csv("gurobi_candidates.csv")


# %%
n_candidates = len(gurobi_candidates)
greedy_candidates = candidates.sort_values("zT_pred").tail(n_candidates)


# %%
ax = pmv.ptable_heatmap(gurobi_candidates.formula)
ax.set_title(f"elemental prevalence among {n_candidates} gurobi candidates")
pmv.save_fig(ax, "gurobi-ptable-elements.pdf", bbox_inches="tight")


ax = pmv.ptable_heatmap(greedy_candidates.formula)
ax.set_title(f"elemental prevalence among {n_candidates} greedy candidates")
pmv.save_fig(ax, "greedy-ptable-elements.pdf", bbox_inches="tight")


# %%
high_zT_idx = lrhr_candidates.sort_values("zT_pred").tail(n_candidates).index

high_zT_mask = lrhr_candidates.index.isin(high_zT_idx)
# printed 195.417
print(f"greedy objective value: {corr_mat.dot(high_zT_mask).dot(high_zT_mask) = :.3f}")
