"""
This notebook screens synthesizable materials from ICSD and COD
for viable thermoelectrics
"""
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gurobipy import GRB, Model, quicksum
from matplotlib import colors
from mlmatrics import (
    marchenko_pastur,
    ptable_elemental_prevalence,
    qq_gaussian,
)
from tf_mnf.models import MNFFeedForward
from tqdm import trange

from thermo.data import (
    dropna,
    load_gaultois,
    load_screen,
    normalize,
    train_test_split,
)
from thermo.evaluate import filter_low_risk_high_ret
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
candidates.set_index(["id", "T"], inplace=True, append=True)

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


# %%
mnf_model.fit(
    X_train.astype("float32"),
    zT_train,
    validation_data=(X_test, zT_test),
    batch_size=32,
    epochs=300,
    callbacks=[stop_early],
)


# %%
n_preds = 500
mnf_preds = mnf_model(X_test.values.repeat(n_preds, axis=0))
mnf_preds = mnf_preds.numpy().reshape(-1, n_preds).T


# %%
# Use qq_gaussian which checks std calibration by comparing quantiles of
# z_score = (y_true - y_pred) / y_std against those of a Gaussian. Used
# here to eye-ball scaling factor for NF uncertainties.
plt.title("scaled by 1.5")
qq_gaussian(zT_test.values, mnf_preds.mean(0), 1.5 * mnf_preds.std(0))
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
screen_model.fit(gaultois_magpie_feas, zT, batch_size=32, epochs=140)


# %%
screen_preds, n_preds = [], 50
for _ in trange(n_preds):
    preds = screen_model.predict(screen_features.astype("float32"))
    screen_preds.append(preds)

screen_preds = np.array(screen_preds).squeeze()


# %%
candidates["zT_pred"] = screen_preds.mean(0)
candidates["zT_std"] = screen_preds.std(0) * 1.5


# %%
candidates.sort_values(by="zT_pred", ascending=False).to_csv(
    "mnf-screen-epochs=150-batch=32-preds=50.csv"
)


# %%
cmap = colors.ListedColormap(["blue", "red"])
bounds = [700, 850, 1000]
norm = colors.BoundaryNorm(bounds, cmap.N)
candidates.reset_index().plot.scatter(
    x="zT_std", y="zT_pred", c="T", cmap=cmap, norm=norm
)
plt.savefig("zT_pred-vs-zT_std.png", bbox_inches="tight", dpi=200)


# %%
lrhr_idx = filter_low_risk_high_ret(
    candidates.zT_pred, candidates.zT_std, return_percentile=0.8, risk_percentile=0.4
)
lrhr_candidates = candidates.loc[lrhr_idx]


# %%
corr_mat = np.corrcoef(screen_preds.T[lrhr_candidates.index.get_level_values(0)])
corr_mat = pd.DataFrame(
    corr_mat, index=lrhr_candidates.index, columns=lrhr_candidates.formula
)


# %%
# corr_mat.to_csv("correlation_matrix.csv", float_format="%g")
corr_mat = pd.read_csv("correlation_matrix.csv", index_col=[0, "id", "T"])


# %%
plt.figure(figsize=[10, 10])
plt.matshow(corr_mat.iloc[:200, :200])
plt.title("First 200 compositions")
plt.savefig("correlation_matrix_mnf.png", bbox_inches="tight", dpi=200)


# %%
# Ensure the correlation matrix contains significant eigenvalues larger than
# the maximum expected in a random matrix based on Marchenko_pastur distribution
marchenko_pastur(corr_mat, gamma=len(corr_mat) / n_preds)
p, N = n_preds, len(corr_mat)
plt.title(f"{p = }, {N = }, gamma = p / N = {p / N:.2f}")

plt.yscale("log")
plt.savefig("marchenko-pastur-dist.png")


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
# printed 0.887
print(f"final objective value: {corr_mat.dot(dec_vals).dot(dec_vals) = :.3f}")
gurobi_candidates = lrhr_candidates[dec_vals]
gurobi_candidates.to_csv("gurobi_candidates.csv", index=False)


# %%
greedy_ids = candidates.sort_values("zT_pred").tail(211)


# %%
ptable_elemental_prevalence(gurobi_candidates.formula)
plt.savefig("gurobi_candidates_ptable_elemental_prevalence.pdf", bbox_inches="tight")


ptable_elemental_prevalence(greedy_ids.formula)
plt.savefig("greedy_candidates_ptable_elemental_prevalence.pdf", bbox_inches="tight")


# %%
high_zT_idx = lrhr_candidates.sort_values("zT_pred").tail(211).index

high_zT_mask = lrhr_candidates.index.isin(high_zT_idx)
# printed 185.798
print(f"greedy objective value: {corr_mat.dot(high_zT_mask).dot(high_zT_mask) = :.3f}")


# %%
candidates.zT_pred.hist(bins=1000, log=True)

# %%
candidates.plot.scatter(x="zT_std", y="zT_pred")
# %%

# %%
screen_features.describe()
