# %%
import pandas as pd

from thermo.data import dropna, load_gaultois, train_test_split
from thermo.plots import plot_output
from thermo.utils import ROOT
from thermo.utils.amm import fit_pred_pipe


# %%
_, targets = load_gaultois(target_cols=["T", "formula", "zT", "kappa", "power_factor"])

targets = dropna(targets)

targets.rename(columns={"formula": "composition", "power_factor": "PF"}, inplace=True)

train_df, test_df = train_test_split(targets)


# %%
# !%%capture
n_pipes = 5
pipes, pred_dfs = zip(*(fit_pred_pipe(train_df, test_df, "zT") for _ in range(n_pipes)))


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
