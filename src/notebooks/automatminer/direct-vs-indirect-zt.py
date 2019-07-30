# %%
import os

import automatminer as amm
import matminer as mm
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import ROOT
from utils.evaluate import plot_output

# %%
# Check os.getenv("save_to_disk") == "True" to see if
# logs, plots and other results should be saved to disk.
os.environ["save_to_disk"] = "True"

SAVE_TO = ROOT + "/results/amm/zT_direct_vs_indirect"


# %%
labels_df = pd.read_csv("data/gaultois_labels.csv", header=1)[
    ["T", "formula", "zT", "kappa", "power_factor"]
].dropna(subset=["zT"])
labels_df.rename(columns={"formula": "composition", "power_factor": "PF"}, inplace=True)
train_df, test_df = train_test_split(labels_df, random_state=0)

# %%
featurizers = {
    "composition": [mm.featurizers.composition.ElementProperty.from_preset("magpie")],
    "structure": [],
}
good_regressors = (
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
)
tpot_config = amm.automl.config.tpot_configs.TPOT_REGRESSOR_CONFIG
# Only retain regressors that were among the best in unrestricted benchmarking.
tpot_config = {
    key: val for key, val in tpot_config.items() if key.endswith(good_regressors)
}

# pipe_config needs to be a function rather than a dict. Else differrent pipes
# will all receive a reference to the same learner and overwrite each other's fitting.
pipe_config = lambda: {
    **amm.get_preset_config(),
    "autofeaturizer": amm.AutoFeaturizer(
        # preset="express",
        featurizers=featurizers,
        guess_oxistates=False,
    ),
    "learner": amm.TPOTAdaptor(max_time_mins=10),
    # "learner": amm.SinglePipelineAdaptor(
    #     regressor=forest.RandomForestRegressor(), classifier=None
    # ),
}


# %%
def fit_pred_pipe(target):
    mat_pipe = amm.MatPipe(**pipe_config())
    mat_pipe.fit(train_df[["T", "composition", target]], target)
    pred_df = mat_pipe.predict(test_df[["T", "composition"]])
    return mat_pipe, pred_df


# %%
# !%%capture
mat_pipe_zT, pred_df_zT = fit_pred_pipe("zT")

# %%
# !%%capture
mat_pipe_PF, pred_df_PF = fit_pred_pipe("PF")

# %%
# !%%capture
mat_pipe_kappa, pred_df_kappa = fit_pred_pipe("kappa")

# %%
mat_pipe_zT.summarize(SAVE_TO + "/pipe_summary.yml")
mat_pipe_zT.inspect(SAVE_TO + "/pipe_details.yml")
mat_pipe_zT.save(SAVE_TO + "/mat.pipe")

# # %%
# # Load previously fitted MatPipe.
# mat_pipe = amm.MatPipe.load("logs/automatminer/10-07@15:35:51/mat.pipe")

# %%
"""
Directly predicting zT incurs higher MSE and MAE compared to predicting kappa and
power factor individually but of course requires less compute. Running this
notebook gave zT_direct: (MSE: 0.01977, MAE: 0.07446), zT_computed: (MSE: 0.01591,
MAE: 0.0644).

If running without **pipe_config(), predicting zT directly actually performs slightly
better. zT_direct: (MSE: 0.02031, MAE: 0.06547), zT_computed: (MSE: 0.02541,
MAE: 0.06609). This may be due to the directly fitted TPOTAdapter ran to completion
while those for PF and kappa both timed out after 60 min.

If running with preset "debug" (optimized for speed), predicting zT directly noticeably
outperforms kappa & power factor individually. zT_direct: (MSE: 0.02084, MAE: 0.06684),
zT_computed: (MSE: 0.05633, MAE: 0.08102)
"""
print("automatminer predicting zT for Gaultois data")
print("zT directly")
print(f"- MSE: {((test_df.zT - pred_df_zT['zT predicted']) ** 2).mean():.4g}")
print(f"- MAE: {(test_df.zT - pred_df_zT['zT predicted']).abs().mean():.4g}")
pred_df_zT["zT_pred_computed"] = (
    pred_df_PF["PF predicted"] / pred_df_kappa["kappa predicted"] * test_df["T"]
)
print("power factor and kappa individually")
print(f"- MSE: {((test_df.zT - pred_df_zT.zT_pred_computed) ** 2).mean():.4g}")
print(f"- MAE: {(test_df.zT - pred_df_zT.zT_pred_computed).abs().mean():.4g}")

# %%
plot_output(
    test_df.zT.values, pred_df_zT["zT predicted"].values, pd.np.ones(len(pred_df_zT)),
)
