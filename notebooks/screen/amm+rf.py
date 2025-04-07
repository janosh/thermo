# %%
import os

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
from plotly import express as px

from thermo.correlation import expected_rand_obj_val, rand_obj_val_avr
from thermo.data import dropna, load_gaultois, load_screen, train_test_split
from thermo.evaluate import filter_low_risk_high_ret, plot_output
from thermo.rf import RandomForestRegressor
from thermo.utils import ROOT
from thermo.utils.amm import MatPipe, featurize, fit_pred_pipe


OUT_DIR = f"{ROOT}/results/screen/amm+rf"
os.makedirs(OUT_DIR, exist_ok=True)


# %%
magpie_features, gaultois_df = load_gaultois(target_cols=["formula", "zT", "T"])
screen_df, _ = load_screen()

for df_data in [gaultois_df, screen_df]:
    df_data = df_data.rename(columns={"formula": "composition"})


# %%
# Form Cartesian product between screen features and the 4 temperatures ([300, 400, 700,
# 1000] Kelvin) found in Gaultois' database. We'll predict each material at all 4 temps.
# Note: None of the composition are predicted to achieve high zT at 300, 400 Kelvin.
# Remove those to cut computation time in half.
temps = (700, 1000)
temps_col = np.array(temps).repeat(len(screen_df))

screen_df = screen_df.loc[screen_df.index.repeat(len(temps))]
screen_df.insert(0, "T", temps_col)


# %%
mat_pipe_zT, zT_pred = fit_pred_pipe(gaultois_df, screen_df, "zT")


# %%
mat_pipe_zT = MatPipe.save(f"{OUT_DIR}/mat.pipe")


# %%
mat_pipe_zT = MatPipe.load(f"{OUT_DIR}/mat.pipe")


# %%
amm_train_features = featurize(mat_pipe_zT, gaultois_df[["T", "composition"]])

amm_screen_features = featurize(mat_pipe_zT, screen_df[["T", "composition"]])


# %%
# add composition column for duplicate detection so we save features for
# every material only once
amm_screen_features["composition"] = screen_df.composition
amm_screen_features.drop_duplicates(subset=["composition"]).to_csv(
    f"{OUT_DIR}/amm_screen_features.csv", float_format="%g", index=False
)

amm_train_features.to_csv(
    f"{OUT_DIR}/amm_train_features.csv", float_format="%g", index=False
)


# %%
amm_train_features = pd.read_csv(f"{OUT_DIR}/amm_train_features.csv")

amm_screen_features = pd.read_csv(f"{OUT_DIR}/amm_screen_features.csv")
del amm_screen_features["composition"]

# add temperature column to AMM features
amm_screen_features = amm_screen_features.loc[
    amm_screen_features.index.repeat(len(temps))
]
amm_screen_features.insert(0, "T", temps_col)


# %% [markdown]
# # Check AMM+RF performance on Gaultois data
# Running cells in this section shows automatminer (AMM) features (which are just a
# subset of less correlated MagPie features) performs about the same as the complete
# MagPie set in accuracy but slightly better in uncertainty.


# %%
zT_series, magpie_features, check_features = dropna(
    gaultois_df.zT, magpie_features, amm_train_features
)

[X_tr_amm, X_tr_magpie, y_tr], [X_test_amm, X_test_magpie, y_test] = train_test_split(
    check_features, magpie_features, zT_series
)


# %%
amm_rf_zT = RandomForestRegressor()

amm_rf_zT.fit(X_tr_amm, y_tr)

amm_check_pred, amm_check_var = amm_rf_zT.predict(X_test_amm)

plot_output(y_test.values, amm_check_pred, amm_check_var)


# %%
magpie_rf_zT = RandomForestRegressor()

magpie_rf_zT.fit(X_tr_magpie, y_tr)

magpie_check_pred, magpie_check_var = magpie_rf_zT.predict(X_test_magpie)

plot_output(y_test.values, magpie_check_pred, magpie_check_var)


# %% [markdown]
# # Train AMM+RF on entire Gaultois data, then screen ICSD+COD


# %%
rf_zT = RandomForestRegressor()
rf_zT.fit(amm_train_features.iloc[gaultois_df.dropna().index], gaultois_df.zT.dropna())

zT_pred, zT_var = rf_zT.predict(amm_screen_features)

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
lrhr_idx = filter_low_risk_high_ret(screen_df.zT_pred, screen_df.zT_var, min_ret=1.3)

lrhr_candidates = screen_df[lrhr_idx]


# %%
px.scatter(lrhr_candidates, x="zT_var", y="zT_pred", hover_data=lrhr_candidates.columns)


# %% [markdown]
# # Correlation between low-risk high-return materials


# %%
zT_corr = rf_zT.get_corr(amm_screen_features.iloc[lrhr_candidates.index])
zT_corr = pd.DataFrame(
    zT_corr, columns=lrhr_candidates.composition, index=lrhr_candidates.composition
)


# %%
zT_corr.to_csv(f"{OUT_DIR}/correlation_matrix.csv", float_format="%g")


# %%
zT_corr_evals, zT_corr_evecs = np.linalg.eig(zT_corr)
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
greedy_candidates = lrhr_candidates.copy(deep=True)

greedy_candidates["rough_correlation"] = (zT_corr**2).sum().values

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
grb_model.params.LogFile = f"{OUT_DIR}/gurobi.log"
os.remove(f"{OUT_DIR}/gurobi.log")


# %%
n_select = 20
# Create decision variables.
dvar = grb_model.addVars(len(lrhr_candidates), vtype=GRB.BINARY).values()


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
assert sum(var.x for var in dvar) == n_select, (
    "Gurobi selected a different number of materials than specified by n_select"
)

gurobi_candidates = lrhr_candidates.iloc[[bool(var.x) for var in dvar]]


# %%
gurobi_candidates.to_csv(f"{OUT_DIR}/gurobi_candidates.csv", float_format="%g")
greedy_candidates.to_csv(f"{OUT_DIR}/greedy_candidates.csv", float_format="%g")


# %%
gurobi_candidates = pd.read_csv(f"{OUT_DIR}/gurobi_candidates.csv", index_col=0)
greedy_candidates = pd.read_csv(f"{OUT_DIR}/greedy_candidates.csv", index_col=0)


# %%
for name, df_cand in zip(
    ["gurobi_candidates", "greedy_candidates"],
    [gurobi_candidates, greedy_candidates.iloc[:20]],
):
    df_cand.sort_values(["formula", "T"]).to_latex(
        f"{OUT_DIR}{name}.tex",
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
