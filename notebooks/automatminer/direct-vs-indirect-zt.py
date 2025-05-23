# %%
import os

from thermo.data import dropna, load_gaultois, train_test_split
from thermo.evaluate import mae, rmse
from thermo.plots import plot_output
from thermo.rf import RandomForestRegressor
from thermo.utils import ROOT
from thermo.utils.amm import MatPipe, featurize, fit_pred_pipe


SAVE_TO = ROOT + "/results/amm/zT_direct_vs_indirect/"
os.makedirs(SAVE_TO, exist_ok=True)


# %%
_, targets = load_gaultois(target_cols=["T", "formula", "zT", "kappa", "power_factor"])

targets = dropna(targets)

targets = targets.rename(columns={"formula": "composition", "power_factor": "PF"})

train_df, test_df = train_test_split(targets)


# %%
# !%%capture
mat_pipe_zT, pred_df_zT = fit_pred_pipe(train_df, test_df, "zT")


# %%
# !%%capture
mat_pipe_PF, pred_df_PF = fit_pred_pipe(train_df, test_df, "PF")


# %%
# !%%capture
mat_pipe_kappa, pred_df_kappa = fit_pred_pipe(train_df, test_df, "kappa")


# %%
mat_pipe_zT.summarize(SAVE_TO + "pipe_summary.yml")
mat_pipe_zT.inspect(SAVE_TO + "pipe_details.yml")
mat_pipe_zT.save(SAVE_TO + "mat.pipe")


# %%
loaded_mat_pipe = MatPipe.load(SAVE_TO + "mat.pipe")


# %%
"""
Directly predicting zT incurs higher MSE and MAE compared to predicting kappa and
power factor individually but of course requires less compute. Running this
notebook gave zT_direct: (MSE: 0.01977, MAE: 0.07446), zT_computed: (MSE: 0.01591,
MAE: 0.0644).

If running without **pipe_config() (see amm.fit_pred_pipe), predicting zT directly
actually performs slightly better. zT_direct: (MSE: 0.02031, MAE: 0.06547),
zT_computed: (MSE: 0.02541, MAE: 0.06609). This may be due to the directly fitted
TPOTAdapter running to completion while those for PF and kappa timed out after 60 min.

If running with preset "debug" (optimized for speed), predicting zT directly noticeably
outperforms kappa & power factor individually. zT_direct: (MSE: 0.02084, MAE: 0.06684),
zT_computed: (MSE: 0.05633, MAE: 0.08102)

automatminer's feature relevance determination selected the following descriptors:
- T
- Magpie mean Number
- Magpie mean MendeleevNumber
- Magpie avg_dev MendeleevNumber
- Magpie maximum AtomicWeight
- Magpie range AtomicWeight
- Magpie avg_dev AtomicWeight
- Magpie minimum MeltingT
- Magpie range MeltingT
- Magpie mean MeltingT
- Magpie avg_dev MeltingT
- Magpie avg_dev Row
- Magpie range CovalentRadius
- Magpie mean CovalentRadius
- Magpie avg_dev CovalentRadius
- Magpie maximum Electronegativity
- Magpie range Electronegativity
- Magpie mean Electronegativity
- Magpie avg_dev NpValence
- Magpie mean NdValence
- Magpie mode NdValence
- Magpie avg_dev NfValence
- Magpie maximum NValence
- Magpie range NValence
- Magpie mean NValence
- Magpie avg_dev NValence
- Magpie mode NValence
- Magpie mean NpUnfilled
- Magpie avg_dev NpUnfilled
- Magpie maximum NUnfilled
- Magpie mean NUnfilled
- Magpie avg_dev NUnfilled
- Magpie minimum GSvolume_pa
- Magpie mean GSvolume_pa
- Magpie avg_dev GSvolume_pa
- Magpie maximum SpaceGroupNumber
- Magpie mean SpaceGroupNumber
- Magpie avg_dev SpaceGroupNumber
"""
print("predicting zT directly")
print(f"- RMSE: {rmse(test_df.zT, pred_df_zT.zT_pred):.3g}")
print(f"- MAE: {mae(test_df.zT, pred_df_zT.zT_pred):.3g}")


# %%
pred_df_zT["zT_pred_computed"] = (
    pred_df_PF.PF_pred / pred_df_kappa.kappa_pred * test_df["T"]
)
print("predicting power factor and kappa individually")
print(f"- RMSE: {rmse(test_df.zT, pred_df_zT.zT_pred_computed):.3g}")
print(f"- MAE: {mae(test_df.zT, pred_df_zT.zT_pred_computed):.3g}")


# %%
plot_output(test_df.zT.values, pred_df_zT.zT_pred.values)


# %%
# !%%capture
amm_train_fea = featurize(mat_pipe_zT, train_df[["T", "composition"]])
amm_test_fea = featurize(mat_pipe_zT, test_df[["T", "composition"]])


# %%
rf = RandomForestRegressor()
rf.fit(amm_train_fea, train_df.zT)
rf_y_pred, rf_y_var = rf.predict(amm_test_fea)


# %%
plot_output(test_df.zT.values, rf_y_pred, rf_y_var)
