# %%
import os

import numpy as np

from thermo.data import load_gaultois, load_screen
from thermo.utils import ROOT
from thermo.utils.amm import MatPipe, pipe_config

DIR = ROOT + "/results/amm/screen/"
os.makedirs(DIR, exist_ok=True)


# %%
_, train_df = load_gaultois(target_cols=["T", "formula", "zT"])

screen_df, _ = load_screen()

for df in [train_df, screen_df]:
    df.rename(columns={"formula": "composition"}, inplace=True)


# %%
# Form Cartesian product between screen features and the 4 temperatures ([300, 400, 700,
# 1000] Kelvin) found in Gaultois' database. We'll predict each material at all 4 temps.
# Note: None of the composition are predicted to achieve high zT at 300, 400 Kelvin.
# Remove those to save time.
temps = (700, 1000)
temps_col = np.array(temps).repeat(len(screen_df))

screen_df = screen_df.loc[screen_df.index.repeat(len(temps))]
screen_df.insert(0, "T", temps_col)


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
