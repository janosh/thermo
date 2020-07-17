# %%
import automatminer as amm
import matminer as mm
import pandas as pd

# from forest import RandomForestRegressor


# %%
train_df = pd.read_csv("../../../data/gaultois_labels.csv", header=1)[
    ["T", "formula", "zT"]
].dropna(subset=["zT"])

screen_df = pd.read_csv("../../../data/screening_formulas.csv", header=2)

for df in [train_df, screen_df]:
    df.rename(columns={"formula": "composition"}, inplace=True)


# %%
# Form Cartesian product between the screening features and the
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
featurizers = {
    "composition": [mm.featurizers.composition.ElementProperty.from_preset("magpie")],
    "structure": [],
}
pipe_config = lambda: {
    **amm.get_preset_config("express"),
    "autofeaturizer": amm.AutoFeaturizer(
        # preset="express",
        featurizers=featurizers,
        guess_oxistates=False,
    ),
    "learner": amm.TPOTAdaptor(max_time_mins=5),
    # "learner": amm.SinglePipelineAdaptor(
    #     regressor=RandomForestRegressor(), classifier=None
    # ),
}

pipe = amm.MatPipe(**pipe_config())


# %%
# !%%capture
# Don't show progress bar as it can crash the VSCode Python extension on
# extended calculations.
pipe.fit(train_df, "zT")

# %%
# pipe.inspect("pipe_inspection.yml")
# pipe.save("mat.pipe")


# %%
pipe = amm.MatPipe.load("mat.pipe")


# %%
# !%%capture
pred_df = pipe.predict(screen_df, ignore=["database", "id"], output_col="zT_pred")


# %%
pred_df["composition"] = screen_df.composition
pred_df = pred_df[["composition", "T", "database", "id", "zT_pred"]]
pred_df.to_csv("amm_preds.csv", index=False, float_format="%g")
pred_df.sort_values(by="zT_pred", ascending=False)[:1000].to_csv(
    "amm_top_preds.csv", index=False, float_format="%g"
)
pred_df.sort_values(by="zT_pred", ascending=False)
