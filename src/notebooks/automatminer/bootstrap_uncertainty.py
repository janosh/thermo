# %%
import automatminer as amm
import matminer as mm
import pandas as pd

from data import dropna, load_gaultois, train_test_split
from utils import ROOT
from utils.evaluate import plot_output

# %%
_, labels = load_gaultois(target_cols=["T", "formula", "zT", "kappa", "power_factor"])

labels = dropna(labels)

labels.rename(columns={"formula": "composition", "power_factor": "PF"}, inplace=True)

[train_df], [test_df] = train_test_split(labels)


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
    pred_df = mat_pipe.predict(
        test_df[["T", "composition"]], output_col=target + "_pred"
    )
    return mat_pipe, pred_df


# %%
# !%%capture
n_pipes = 5
pipes, pred_dfs = zip(*[fit_pred_pipe("zT") for _ in range(n_pipes)])


# %%
zT_stats_df = pd.DataFrame(
    [df.zT_pred for df in pred_dfs], index=[f"zT_pred_{i}" for i in range(n_pipes)]
).T
col_names = zT_stats_df.columns
zT_stats_df["mean"] = zT_stats_df[col_names].mean(axis=1)
zT_stats_df["std"] = zT_stats_df[col_names].std(axis=1)
zT_stats_df["zT_true"] = test_df.zT


# %%
plot_output(*zT_stats_df[["zT_true", "mean", "std"]].values.T, title="zT")


# %%
zT_stats_df.to_csv(ROOT + "/results/amm/zT_stats.csv", index=False, float_format="%g")
