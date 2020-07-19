# %%
import os

import pandas as pd
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
from plotly import express as px

from data import load_gaultois, load_screen
from rf.forest import RandomForestRegressor
from utils import ROOT
from utils.amm import MatPipe, featurize, fit_pred_pipe
from utils.correlation import expected_rand_obj_val, rand_obj_val_avr
from utils.evaluate import filter_low_risk_high_ret

DIR = ROOT + "/results/screen/amm+rf/"
os.makedirs(DIR, exist_ok=True)


# %%
_, train_df = load_gaultois(target_cols=["formula", "zT", "T"])
screen_df, _ = load_screen()

for df in [train_df, screen_df]:
    df.rename(columns={"formula": "composition"}, inplace=True)


# %%
# Form Cartesian product between screen features and the 4 temperatures ([300, 400, 700,
# 1000] Kelvin) found in Gaultois' database. We'll predict each material at all 4 temps.
# Note: None of the composition are predicted to achieve high zT at 300, 400 Kelvin.
# Remove those to cut computation time in half.
screen_df = (
    pd.DataFrame({"T": [700, 1000], "key": 1})
    .merge(screen_df.assign(key=1), on="key")
    .drop("key", axis=1)
)


# %%
mat_pipe_zT, zT_pred = fit_pred_pipe(train_df, screen_df, "zT")
os.remove(os.path.dirname(__file__) + "automatminer.log")


# %%
mat_pipe_zT = MatPipe.save(DIR + "mat.pipe")


# %%
mat_pipe_zT = MatPipe.load(DIR + "mat.pipe")


# %%
train_features = featurize(mat_pipe_zT, train_df[["T", "composition"]])

screen_features = featurize(mat_pipe_zT, screen_df[["T", "composition"]])


# %%
# add composition column for duplicate detection so we save features for
# every material only once
screen_features["composition"] = screen_df.composition
screen_features.drop_duplicates(subset=["composition"]).to_csv(
    DIR + "screen_features.csv", float_format="%g", index=False
)


# %%
train_features = pd.read_csv(DIR + "train_features.csv")

screen_features = pd.read_csv(DIR + "screen_features.csv")
del screen_features["composition"]

# reinstate temperature column, removed to save disk space
screen_features = (
    pd.DataFrame({"T": [700, 1000], "key": 1})
    .merge(screen_features.assign(key=1), on="key")
    .drop("key", axis=1)
)

# %%
rf_zT = RandomForestRegressor()
rf_zT.fit(train_features.iloc[train_df.dropna().index], train_df.zT.dropna())

zT_pred, zT_var = rf_zT.predict(screen_features)

screen_df["zT_pred"] = zT_pred
screen_df["zT_var"] = zT_var


# %% [markdown]
# # Coarse triaging


# %%
# Save to CSV the 20 materials predicted to have the highest zT with no concern for
# estimated uncertainty. Baseline comparison to check if uncertainty estimation reduces
# the false positive rate.
screen_df.sort_values("zT_pred", ascending=False)[:20].to_csv(
    ROOT + "/results/screen/hr-materials.csv", index=False, float_format="%g"
)


# %%
formulas_lrhr = filter_low_risk_high_ret(screen_df, min_ret=1.3)


# %%
px.scatter(formulas_lrhr, x="zT_var", y="zT_pred", hover_data=formulas_lrhr.columns)


# %% [markdown]
# # Correlation between low-risk high-return materials


# %%
zT_corr = rf_zT.get_corr(screen_features.iloc[formulas_lrhr.index])
zT_corr = pd.DataFrame(
    zT_corr, columns=formulas_lrhr.composition, index=formulas_lrhr.composition
)


# %%
zT_corr.to_csv(DIR + "correlation_matrix.csv", float_format="%g")


# %%
zT_corr_evals, zT_corr_evecs = pd.np.linalg.eig(zT_corr)
zT_corr_evecs = zT_corr_evecs[zT_corr_evals.argsort()[::-1]]
plt.scatter(zT_corr_evecs[0], zT_corr_evecs[1])


# %% [markdown]
# # Fine Triaging
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

greedy_candidates["rough_correlation"] = (zT_corr ** 2).sum().values

greedy_candidates = (
    greedy_candidates.reset_index()
    .rename(columns={"index": "orig_index"})
    .sort_values(by="rough_correlation")
)


# %%
# Set environment variable GRB_LICENSE_FILE so that Gurobi finds its license.
# An academic license can be obtained for free at
# https://www.gurobi.com/downloads/end-user-license-agreement-academic.
os.environ["GRB_LICENSE_FILE"] = ROOT + "/hpc/gurobi.lic"
# Create a model for solving the quadratic optimization problem of selecting p out of n
# materials with least pairwise correlation according to the correlation matrix zT_corr.
grb_model = Model("quadratic_problem")
grb_model.params.TimeLimit = 300  # in sec
grb_model.params.LogFile = DIR + "gurobi.log"
os.remove(DIR + "gurobi.log")


# %%
n_select = 20
# Create decision variables.
dvar = grb_model.addVars(len(formulas_lrhr), vtype=GRB.BINARY).values()


# %%
# Define the model objective to minimize the sum of pairwise correlations.
obj = zT_corr.dot(dvar).dot(dvar)
grb_model.setObjective(obj, GRB.MINIMIZE)


# %%
# Add L1 constraint on dvar so that the optimization returns at least n_select formulas.
constr = grb_model.addConstr(quicksum(dvar) >= n_select, "l1_norm")


# %%
grb_model.optimize()


# %%
# Save selected materials to dataframe and CSV file.
assert (
    sum([var.x for var in dvar]) == n_select
), "Gurobi selected a different number of materials than specified by n_select"

gurobi_candidates = formulas_lrhr.iloc[[bool(var.x) for var in dvar]]


# %%
gurobi_candidates.to_csv(DIR + "gurobi_candidates.csv", float_format="%g")
greedy_candidates.to_csv(DIR + "greedy_candidates.csv", float_format="%g")


# %%
gurobi_candidates = pd.read_csv(DIR + "gurobi_candidates.csv", index_col=0)
greedy_candidates = pd.read_csv(DIR + "greedy_candidates.csv", index_col=0)


# %%
for name, df in zip(
    ["gurobi_candidates", "greedy_candidates"],
    [gurobi_candidates, greedy_candidates.iloc[:20]],
):
    df.sort_values(["formula", "T"]).to_latex(
        f"{DIR}{name}.tex",
        columns=["formula", "database", "id", "T", "zT_pred", "zT_var"],
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
    f"vs the average index of the total list: {(len(formulas_lrhr) + 1) / 2}"
)


# %%
greedy_indices_in_corr_mat = formulas_lrhr.index.isin(
    greedy_candidates.orig_index[:n_select]
)
greedy_obj_val = zT_corr.values.dot(greedy_indices_in_corr_mat).dot(
    greedy_indices_in_corr_mat
)

avr_rand_obj_val = rand_obj_val_avr(zT_corr, n_select)

# If len(zT_corr) >> 500, expected_rand_obj_val will take a long time due to cubic
# scaling. Consider decreasing max_risk or increasing min_ret in
# filter_low_risk_high_ret to decrease len(zT_corr).
exp_rand_obj_val = expected_rand_obj_val(zT_corr, n_select)

print(
    f"objective values:\n- Gurobi: {grb_model.objVal:.4g}\n"
    f"- greedy: {greedy_obj_val:.4g}\n"
    f"- average of 50 random draws: {avr_rand_obj_val:.4g}\n"
    f"- expectation value of random solution: {exp_rand_obj_val:.4g}"
)
