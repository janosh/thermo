"""
This notebook screens a combined list of synthesizable materials from ICSD and
COD databases for promising thermoelectric candidates using random forest
regression.
"""


# %%
import os

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
from plotly import express as px

from thermo.data import dropna, load_gaultois, load_screen
from thermo.evaluate import filter_low_risk_high_ret
from thermo.rf import rf_predict
from thermo.utils import ROOT
from thermo.utils.correlation import expected_rand_obj_val, rand_obj_val_avr

# %%
features, targets = load_gaultois()
targets, features = dropna(targets, features)
# magpie_screen are the Magpie features for the screening set without temperature yet
screen_ids, magpie_screen = load_screen()


# %%
# Form Cartesian product between screen features and the 4 temperatures ([300, 400, 700,
# 1000] Kelvin) found in Gaultois' database. We'll predict each material at all 4 temps.
# Note: None of the composition are predicted to achieve high zT at 300, 400 Kelvin.
# Remove those to save time.
temps = (700, 1000)
temps_col = np.array(temps).repeat(len(magpie_screen))

screen_features = magpie_screen.loc[magpie_screen.index.repeat(len(temps))]
screen_features.insert(0, "T", temps_col)

candidates = screen_ids.loc[screen_ids.index.repeat(len(temps))]
candidates.insert(0, "T", temps_col)


# %% [markdown]
# # Random Forest Regression


# %%
zT_pred, zT_var, forest = rf_predict(features, targets.zT, screen_features)


# %%
candidates["zT_pred"] = zT_pred
candidates["zT_std"] = zT_var ** 0.5


# %% [markdown]
# # Coarse triaging


# %%
lrhr_idx = filter_low_risk_high_ret(candidates.zT_pred, candidates.zT_std)

lrhr_candidates = candidates.loc[lrhr_idx]


# %%
# Get the 20 materials predicted to have the highest zT with no concern for estimated
# uncertainty to compare if uncertainty estimation reduces the false positive rate.
high_risk_candidates = candidates.sort_values("zT_pred", ascending=False)[:20]
high_risk_candidates[["formula", "database", "id", "T", "zT_pred", "zT_std"]].to_csv(
    "high_risk_candidates.csv", index=False, float_format="%g"
)


# %%
# lrhr_candidates.to_csv("lrhr_materials.csv", index=False, float_format="%g")
zT_std_lt_half = lrhr_candidates[lrhr_candidates.zT_std < 0.5]
zT_std_lt_half.plot.scatter(x="zT_std", y="zT_pred")
plt.xlabel("")
plt.ylabel("")
plt.savefig(
    "lrhr_materials.pdf",
    bbox_inches="tight",
    transparent=True,
)
pearson = zT_std_lt_half[["zT_std", "zT_pred"]].corr().iloc[0, 1]
print(f"Pearson corr.: {pearson:.4g}")


# %%
px.scatter(lrhr_candidates, x="zT_std", y="zT_pred", hover_data=lrhr_candidates.columns)


# %% [markdown]
# # Compute correlations between low-risk high-return materials


# %%
zT_corr = forest.get_corr(screen_features.iloc[lrhr_candidates.index])
zT_corr = pd.DataFrame(
    zT_corr, columns=lrhr_candidates.formula, index=lrhr_candidates.formula
)


# %%
# zT_corr.to_csv("correlation_matrix.csv", float_format="%g")
zT_corr = pd.read_csv("correlation_matrix.csv", index_col="formula")


plt.figure(figsize=[10, 10])
plt.pcolormesh(zT_corr.iloc[:200, :200])
plt.title("First 200 compositions")
plt.savefig("correlation_matrix_rf.png", bbox_inches="tight", dpi=200)


# %%
# unlike np.linalg.eig, eigh assumes the matrix is symmetric and as a result is faster
zT_corr_evals, zT_corr_evecs = np.linalg.eigh(zT_corr)

zT_corr_evecs = zT_corr_evecs[:, zT_corr_evals.argsort()[::-1]]

plt.scatter(zT_corr_evecs[:, 0], zT_corr_evecs[:, 1])


# %% [markdown]
# # Fine Triaging
# Helpful links for the discrete constrained optimization problem of
# finding the p least correlated materials out of n predictions:
# - [Find k of n items with least pairwise correlations](
# https://stats.stackexchange.com/q/73125)
# - [Least correlated subset of random variables from a correlation matrix](
# https://stats.stackexchange.com/q/110426)


# %%
# The idea for this way of reducing correlation came from
# https://stats.stackexchange.com/a/327822. Taking the element-wise
# absolute value (rather than squaring) and then summing gives similar results.
greedy_candidates = lrhr_candidates.copy(deep=True)

greedy_candidates["rough_correlation"] = (zT_corr ** 2).sum().values

greedy_candidates = (
    greedy_candidates.reset_index()
    .rename(columns={"index": "orig_index"})
    .sort_values(by="rough_correlation")
)

greedy_candidates.to_csv("greedy_candidates.csv", index=False, float_format="%g")


# %%
# Set environment variable GRB_LICENSE_FILE so that Gurobi finds its license file.
# An academic license can be obtained for free at
# https://gurobi.com/downloads/end-user-license-agreement-academic.
os.environ["GRB_LICENSE_FILE"] = f"{ROOT}/hpc/gurobi.lic"
# Create a model for solving the quadratic optimization problem of selecting p out of n
# materials with least pairwise correlation according to the correlation matrix zT_corr.
grb_model = Model("quadratic_problem")
grb_model.params.LogFile = "gurobi.log"
os.remove("gurobi.log")


# %%
n_select = 20
# Create decision variables.
dec_vars = grb_model.addVars(len(lrhr_candidates), vtype=GRB.BINARY).values()


# %%
# Define the model objective to minimize the sum of pairwise correlations.
obj = zT_corr.dot(dec_vars).dot(dec_vars)
grb_model.setObjective(obj, GRB.MINIMIZE)


# %%
# Add L1 constraint on dvar so that the optimization returns at least n_select formulas.
constr = grb_model.addConstr(quicksum(dec_vars) >= n_select, "l1_norm")


# %%
grb_model.optimize()


# %%
# Save selected materials to dataframe and CSV file.
gurobi_candidates = lrhr_candidates.iloc[[bool(var.x) for var in dec_vars]]
gurobi_candidates.to_csv("gurobi_candidates.csv", index=False)


# %%
for name, df in zip(
    ["gurobi_candidates", "greedy_candidates"],
    [gurobi_candidates, greedy_candidates.iloc[:20]],
):
    df.sort_values(["formula", "T"]).to_latex(
        f"{name}.tex",
        columns=["formula", "database", "id", "T", "zT_pred", "zT_std"],
        float_format="%.3g",
        index=False,
    )


# %% [markdown]
# # Comparing greedy and Gurobi solution


# %%
# greedy_candidates contains all low-risk high-return materials sorted by their sum of
# squared correlations with all other materials. If either the greedy or Gurobi method
# (or both) picked materials entirely at random, we would expect formulas chosen by
# Gurobi to have an average index in the list equal to the total list's average index.
# The degree to which the average index of Gurobi materials in the greedy list is lower
# than average list index is an indicator of agreement between the two methods.
gurobi_in_greedy = greedy_candidates.orig_index.isin(gurobi_candidates.index)
greedy_avg_index = greedy_candidates[gurobi_in_greedy].index.to_series().mean()

print(
    "Average index of materials chosen by Gurobi in the list\n"
    f"sorted according to least squared correlation: {greedy_avg_index}\n"
    f"vs the average index of the total list: {(len(lrhr_candidates) + 1) / 2}"
)


# %%
greedy_indices_in_corr_mat = lrhr_candidates.index.isin(
    greedy_candidates.orig_index[:n_select]
)
greedy_obj_val = zT_corr.values.dot(greedy_indices_in_corr_mat).dot(
    greedy_indices_in_corr_mat
)

avr_rand_obj_val = rand_obj_val_avr(zT_corr, n_select, (n_repeats := 50))

# If len(zT_corr) >> 500, expected_rand_obj_val will take a long time due to cubic
# scaling. Consider decreasing max_risk or increasing min_ret in
# filter_low_risk_high_ret to decrease len(zT_corr).
exp_rand_obj_val = expected_rand_obj_val(zT_corr, n_select)

print(
    f"objective values:\n- Gurobi: {grb_model.objVal:.4g}\n"
    f"- greedy: {greedy_obj_val:.4g}\n"
    f"- average of {n_repeats} random draws: {avr_rand_obj_val:.4g}\n"
    f"- expectation value of random solution: {exp_rand_obj_val:.4g}"
)


# %% [markdown]
# # Correlation between low-risk high-return materials


# %%
dft_seebeck = pd.read_csv("dft/gurobi_seebeck.csv")[
    ["formula", "300", "400", "700", "1000"]
].set_index("formula")


# %%
rf_seebeck = (
    screen_features[["formula", "seebeck_abs_pred", "T"]]
    .loc[screen_features.formula.isin(dft_seebeck.index)]
    .pivot(index="formula", columns="T", values="seebeck_abs_pred")
) * 1e6  # conversion from SI to common units (V/K -> uV/K)

# rf_seebeck.seebeck_abs_pred = rf_seebeck.seebeck_abs_pred * 1e6


# %%
dft_seebeck.columns = rf_seebeck.columns
methods = ["pearson", "spearman"]
dft_rf_seebeck_abs_corr = pd.concat(
    [rf_seebeck.corrwith(abs(dft_seebeck), axis=1, method=m) for m in methods],
    axis=1,
    keys=methods,
)
dft_rf_seebeck_abs_corr.mean()


# %%
seebeck_abs_preds_and_corrs = pd.concat(
    [dft_rf_seebeck_abs_corr, rf_seebeck, abs(dft_seebeck)],
    axis=1,
    keys=["correlation", "RF", "DFT"],
)
seebeck_abs_preds_and_corrs.loc["mean"] = seebeck_abs_preds_and_corrs.mean()
seebeck_abs_preds_and_corrs.to_latex(
    "rf_seebeck_abs_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)
