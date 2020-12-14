# %%
import pandas as pd

from data import dropna, load_gaultois, train_test_split
from plots import plot_output
from utils import ROOT
from utils.amm import fit_pred_pipe

# %%
_, labels = load_gaultois(target_cols=["T", "formula", "zT", "kappa", "power_factor"])

labels = dropna(labels)

labels.rename(columns={"formula": "composition", "power_factor": "PF"}, inplace=True)

train_df, test_df = train_test_split(labels)


# %%
# !%%capture
n_pipes = 5
pipes, pred_dfs = zip(*[fit_pred_pipe(train_df, test_df, "zT") for _ in range(n_pipes)])


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
