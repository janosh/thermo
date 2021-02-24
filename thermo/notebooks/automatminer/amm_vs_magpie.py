# %%
import os

import pandas as pd
from scipy.stats import pearsonr

from thermo.data import dropna, load_gaultois, train_test_split
from thermo.evaluate import mae, rmse
from thermo.plots import plot_output
from thermo.rf.forest import RandomForestRegressor
from thermo.utils import ROOT, amm

# %%
SAVE_TO = ROOT + "/results/amm/amm_vs_magpie/"
os.makedirs(SAVE_TO, exist_ok=True)
r_pearson = lambda *args: pearsonr(*args)[0]


def print_err(true, pred):
    print(f"- RMSE: {rmse(true,pred):.3g}")
    print(f"- MAE: {mae(true,pred):.3g}")
    print(f"- Pearson: {r_pearson(true,pred):.3g}")


# %%
magpie_features, targets = load_gaultois(target_cols=["T", "formula", "zT"])

targets, magpie_features = dropna(targets, magpie_features)

targets.rename(columns={"formula": "composition"}, inplace=True)

[magpie_train, train_df], [magpie_test, test_df] = train_test_split(
    magpie_features, targets
)


# %%
# !%%capture
# setting max_time_mins to 10 or 30 (minutes) makes no difference here.
# In both cases, we get
# - RMSE: 0.121
# - MAE: 0.0564
# - Pearson: 0.914
mat_pipe, amm_pred = amm.fit_pred_pipe(train_df, test_df, "zT")


# %%
# mat_pipe.save(SAVE_TO + "mat.pipe")


# %%
mat_pipe = amm.MatPipe.load(SAVE_TO + "mat.pipe")
amm_pred = mat_pipe.predict(test_df, output_col="zT_pred")


# %%
print("Automatminer")
print_err(test_df.zT, amm_pred.zT_pred)


# %%
plot_output(test_df.zT.values, amm_pred.zT_pred.values)


# %%
# !%%capture
amm_train_fea = amm.featurize(mat_pipe, train_df[["T", "composition"]])
amm_test_fea = amm.featurize(mat_pipe, test_df[["T", "composition"]])


# %%
rf_amm = RandomForestRegressor()
rf_amm.fit(amm_train_fea, train_df.zT)
rf_amm_pred, rf_amm_var = rf_amm.predict(amm_test_fea)


# %%
print("Random Forest trained on automatminer features")
print_err(test_df.zT, rf_amm_pred)


# %%
plot_output(test_df.zT.values, rf_amm_pred, rf_amm_var)


# %%
rf_magpie = RandomForestRegressor()
rf_magpie.fit(magpie_train, train_df.zT)
rf_magpie_pred, rf_magpie_var = rf_magpie.predict(magpie_test)


# %%
print("Random Forest trained on Magpie features")
print_err(test_df.zT, rf_magpie_pred)


# %%
plot_output(test_df.zT.values, rf_magpie_pred, rf_magpie_var)


# %%
errors = [
    [fn(test_df.zT, x) for fn in [mae, rmse, r_pearson]]
    for x in [rf_magpie_pred, rf_amm_pred, amm_pred.zT_pred]
]
errors = pd.DataFrame(
    errors, index=["RF+MAGPIE", "RF+AMM", "AMM"], columns=["RMSE", "MAE", "$r_P$"]
).T


# %%
for idx, [c1, c2] in enumerate([[0, 1], [1, 2], [0, 2]]):
    errors[f"{c1+1} to {c2+1}"] = -100 * (1 - errors.iloc[:, c2] / errors.iloc[:, c1])
errors


# %%
errors.to_latex(f"{SAVE_TO}errors.tex", float_format="%.3g", escape=False)
