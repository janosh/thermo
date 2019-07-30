# %%
import os
import random

import pandas as pd
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
from plotly import express as px

from data import load_gaultois, load_screening, normalize
from rf import rf_predict
from utils import ROOT, predict_multiple_labels
from utils.evaluate import back_transform_labels

# %%
features, labels = load_gaultois()
screening_features = load_screening()


# %%
# Form Cartesian product between the screening features and the
# 4 temperatures found in Gaultois' db. We'll predict each
# material at all 4 temps.
temps = pd.DataFrame([300, 400, 700, 1000], columns=["T"])

# Create a temporary common key around which to compute the Cartesian product.
screening_features["key"] = temps["key"] = 0

# Perform Cartesian product.
screening_features = temps.merge(screening_features, how="outer")
for df in [temps, screening_features]:
    df.drop(columns=["key"], inplace=True)

formulas = pd.concat([df.pop(x) for x in ["formula", "database", "id"]], 1)
formulas["T"] = screening_features["T"]


# %%
screening_features, [X_mean, X_std] = normalize(screening_features)
features, _ = normalize(features, X_mean, X_std)
scd_labels, [y_mean, y_std] = normalize(labels)


# %% [markdown]
# ## Random Forest Regression


# %%
rf_y_preds_scd, rf_y_vars_scd, rf_models = predict_multiple_labels(
    rf_predict, features, scd_labels.T, screening_features
)


# %%
rf_y_preds, rf_y_vars = back_transform_labels(
    y_mean, y_std, rf_y_preds_scd, rf_y_vars_scd, to="orig"
)


# %%
formulas[[ln + "_pred" for ln in labels.columns]] = rf_y_preds
formulas[[ln + "_var" for ln in labels.columns]] = rf_y_vars

# %% [markdown]
# ## Coarse triaging


# %%
def filter_low_risk_high_ret(
    *dfs, max_risk=0.25, min_ret=1, cols=["zT_pred", "zT_var"]
):
    y_pred, y_var = dfs[0][cols].values.T
    y_pred_low_risk = y_pred[y_var < max_risk]
    dfs = [df[y_var < max_risk] for df in dfs]
    dfs = [df[y_pred_low_risk > min_ret] for df in dfs]
    return dfs


# %%
[formulas_lrhr] = filter_low_risk_high_ret(formulas)

# %%
# Get the 20 materials predicted to have the highest zT with no concern for estimated
# uncertainty to compare if uncertainty estimation reduces the false positive rate.
formulas_hr = formulas.sort_values("zT_pred", ascending=False)[:20]
formulas_hr[["formula", "database", "id", "T", "zT_pred", "zT_var"]].to_csv(
    ROOT + "/results/screening/hr-materials.csv", index=False, float_format="%g"
)

# %%
# formulas_lrhr.to_csv(
#     "results/screening/lrhr_materials.csv", index=False, float_format="%g"
# )
# plt.rcParams.update(plt.rcParamsDefault)
zT_var_lt_half = formulas_lrhr[formulas_lrhr.zT_var < 0.5]
zT_var_lt_half.plot.scatter(x="zT_var", y="zT_pred")
plt.xlabel("")
plt.ylabel("")
plt.savefig(
    ROOT + "/results/screening/lrhr_materials.pdf",
    bbox_inches="tight",
    transparent=True,
)
pearson = zT_var_lt_half[["zT_var", "zT_pred"]].corr().iloc[0, 1]
print(f"Pearson corr.: {pearson:.4g}")

# %%
px.scatter(formulas_lrhr, x="zT_var", y="zT_pred", hover_data=formulas_lrhr.columns)

# %% [markdown]
# ## Compute correlations between low-risk high-return materials

# %%
zT_forest = rf_models["zT_log_scd"]
zT_corr = zT_forest.get_corr(screening_features.iloc[formulas_lrhr.index])
zT_corr = pd.DataFrame(
    zT_corr, columns=formulas_lrhr.formula, index=formulas_lrhr.formula
)


# %%
zT_corr.to_csv("results/screening/correlation_matrix.csv", float_format="%g")

# %%
zT_corr_evals, zT_corr_evecs = pd.np.linalg.eig(zT_corr)
zT_corr_evecs = zT_corr_evecs[zT_corr_evals.argsort()[::-1]]
plt.scatter(zT_corr_evecs[0], zT_corr_evecs[1])


# %% [markdown]
# ## Fine Triaging
# Helpful links for the discrete constrained optimization problem of
# finding the p least correlated materials out of n predictions:
# - [Find k of n items with least pairwise correlations](
# https://stats.stackexchange.com/questions/73125)
# - [Least correlated subset of random variables from a correlation matrix](
# https://stats.stackexchange.com/questions/110426)

# %%
# The idea for this way of reducing correlation came from
# https://stats.stackexchange.com/a/327822/226996. Taking the element-wise
# absolute value (rather than squaring) and then summing gives similar results.
greedy_candidates = formulas_lrhr.copy(deep=True)
greedy_candidates["correlation"] = (zT_corr ** 2).sum().values
greedy_candidates = (
    greedy_candidates.reset_index()
    .rename(columns={"index": "orig_index"})
    .sort_values(by="correlation")
)
greedy_candidates.to_csv(
    "results/screening/greedy_candidates.csv", index=False, float_format="%g"
)

# %%
# Set environment variable GRB_LICENSE_FILE so that Gurobi finds its license file.
# An academic license can be obtained for free at
# https://www.gurobi.com/downloads/end-user-license-agreement-academic.
os.environ["GRB_LICENSE_FILE"] = "hpc/gurobi.lic"
# Create a model for solving the quadratic optimization problem of selecting p out of n
# materials with least pairwise correlation according to the correlation matrix zT_corr.
model = Model("quadratic_problem")
model.params.LogFile = "results/screening/gurobi.log"
os.remove("gurobi.log")

n_vars, n_select = len(formulas_lrhr), 20

# %%
# Create decision variables.
dvar = model.addVars(n_vars, vtype=GRB.BINARY).values()

# %%
# Define the model objective to minimize the sum of pairwise correlations.
obj = zT_corr.dot(dvar).dot(dvar)
model.setObjective(obj, GRB.MINIMIZE)

# %%
# Add L1 constraint on dvar so that the optimization returns at least n_select formulas.
constr = model.addConstr(quicksum(dvar) >= n_select, "l1_norm")

# %%
model.optimize()

# %%
# Save selected materials to dataframe and CSV file.
gurobi_candidates = formulas_lrhr.iloc[[bool(var.x) for var in dvar]]
gurobi_candidates.to_csv("results/screening/gurobi_candidates.csv", index=False)

# %%
for name, df in zip(
    ["gurobi_candidates", "greedy_candidates"],
    [gurobi_candidates, greedy_candidates.iloc[:20]],
):
    df.sort_values(["formula", "T"]).to_latex(
        f"results/screening/{name}.tex",
        columns=["formula", "database", "id", "T", "zT_pred", "zT_var"],
        float_format="%.3g",
        index=False,
    )

# %% [markdown]
# ## Comparing greedy and Gurobi solution

# %%
# greedy_candidates contains all low-risk high-return materials sorted by their
# sum of squared correlations with all other materials. Assuming that either
# the greedy method or Gurobi (or both) picked materials entirely at random,
# we would expect formulas choosen by Gurobi to have an average index in the list
# equal to the total list's average index. The degree to which this is not the case
# is hence an indicator of agreement between the two methods.
gurobi_in_greedy = greedy_candidates.orig_index.isin(gurobi_candidates.index)
greedy_avg_index = (
    greedy_candidates[gurobi_in_greedy]
    .drop_duplicates(keep="first", subset="formula")
    .reset_index()["index"]
    .mean()
)

print(
    "Average index of materials chosen by Gurobi in the list\n"
    f"sorted according to least squared correlation: {greedy_avg_index}\n"
    f"vs the average index of the total list: {(n_vars + 1)/2}"
)


# %%
greedy_indices_in_corr_mat = formulas_lrhr.index.isin(
    greedy_candidates.orig_index[:n_select]
)
greedy_obj_val = zT_corr.values.dot(greedy_indices_in_corr_mat).dot(
    greedy_indices_in_corr_mat
)


def random_obj_val_avr(n=50):
    avr = 0
    for i in range(n):
        rand_seq = [1] * n_select + [0] * (n_vars - n_select)
        random.shuffle(rand_seq)
        avr += zT_corr.dot(rand_seq).dot(rand_seq)
    return avr / n


def expec_random_obj_val():
    # See https://math.stackexchange.com/questions/3315535.
    zT_chol = pd.np.linalg.cholesky(zT_corr.values)
    res = 0
    for j in range(n_vars):
        for i in range(j, n_vars):
            res += zT_chol[i][j] ** 2
            temp = 0
            for k in range(i + 1, n_vars):
                temp += zT_chol[i][j] * zT_chol[k][j]
            res += 2 * (n_select - 1) / (n_vars - 1) * temp

    return n_select / n_vars * res


print(
    f"objective value of the greedy solution: {greedy_obj_val:.4g}'\n"
    f"vs the Guroby solution: {model.objVal:.4g}'\n"
    f"vs average of 50 random solutions: {random_obj_val_avr():.4g}'\n"
    f"vs expectation value of random solution: {expec_random_obj_val():.4g}"
)

# %% [markdown]
# ## Computing correlation between random forest and DFT predictions Seebeck coefficient

# %%
dft_seebeck = pd.read_csv("results/screening/dft/gurobi_seebeck.csv")[
    ["formula", "300", "400", "700", "1000"]
].set_index("formula")

# %%
rf_seebeck = (
    formulas[["formula", "seebeck_abs_pred", "T"]]
    .loc[formulas.formula.isin(dft_seebeck.index)]
    .pivot(index="formula", columns="T", values="seebeck_abs_pred")
) * 1e6  # conversion from SI to common units (uV/K -> V/K)

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
    "results/screening/rf_seebeck_abs_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)
