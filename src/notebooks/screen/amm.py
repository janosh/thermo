# %%
import os

import pandas as pd

from data import load_gaultois, load_screen
from utils import ROOT
from utils.amm import MatPipe, pipe_config

DIR = ROOT + "/results/amm/screen/"
os.makedirs(DIR, exist_ok=True)


# %%
_, train_df = load_gaultois(target_cols=["T", "formula", "zT"])

screen_df, _ = load_screen()

for df in [train_df, screen_df]:
    df.rename(columns={"formula": "composition"}, inplace=True)


# %%
# Form Cartesian product between the screen features and the
# 4 temperatures found in Gaultois' db. Then predict each
# material at all 4 temps.
temps = pd.DataFrame([300, 400, 700, 1000], columns=["T"])
# Create a temporary common key to perform the product.
screen_df["key"] = temps["key"] = 0
screen_df = temps.merge(screen_df, how="outer")
for df in [temps, screen_df]:
    df.drop(columns=["key"], inplace=True)

# screen_df["T"] = 1000


# %%
pipe = MatPipe(**pipe_config())


# %%
# !%%capture
# Use `# !%%capture` to hide progress bar as it can crash the VS Code Python extension
# on long runs. Update: Seems to be fixed but still a way to decrease verbosity.
pipe.fit(train_df, "zT")


# %%
# pipe.summarize(DIR + "pipe_summary.yml")
# pipe.inspect(DIR + "pipe_inspection.yml")
# pipe.save(DIR + "mat.pipe")


# %%
pipe = MatPipe.load(DIR + "mat.pipe")


# %%
# !%%capture
pred_df = pipe.predict(screen_df, ignore=["database", "id"], output_col="zT_pred")


# %%
pred_df["composition"] = screen_df.composition

pred_df = pred_df[["composition", "T", "database", "id", "zT_pred"]]

pred_df.to_csv(DIR + "amm_preds.csv", index=False, float_format="%g")

pred_df = pred_df.sort_values(by="zT_pred", ascending=False)

pred_df[:1000].to_csv(DIR + "amm_top_preds.csv", index=False, float_format="%g")
