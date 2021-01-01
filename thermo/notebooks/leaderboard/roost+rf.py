# %%
import pandas as pd
import torch
from roost.core import Normalizer
from roost.roost.data import CompositionData, collate_batch
from roost.roost.model import Roost
from torch.utils.data import DataLoader, random_split

from thermo.data import dropna, load_gaultois, normalize, train_test_split
from thermo.rf.forest import RandomForestRegressor
from thermo.utils import ROOT
from thermo.utils.evaluate import mae, rmse

# %%
magpie_features, targets = load_gaultois(target_cols=["icsd_id", "formula", "T", "zT"])

targets, magpie_features = dropna(targets, magpie_features)


# %%
[X_train_mp, y_train_mp], [X_test_mp, y_test_mp] = train_test_split(
    magpie_features, targets
)


# %%
task = "regression"

data_params = {
    "batch_size": 32,
    "pin_memory": False,
    "shuffle": True,
    "collate_fn": collate_batch,
}

targets.to_csv(ROOT + "/data/for_roost.csv", index=False, float_format="%g")

gaultois_db = CompositionData(
    data_path=ROOT + "/data/for_roost.csv",
    fea_path=ROOT + "/data/matscholar-embedding.json",
    task=task,
)

# standardize/z-score temperatures
gaultois_db.df["T"] = (
    gaultois_db.df["T"] - gaultois_db.df["T"].mean()
) / gaultois_db.df["T"].std()

# train_set = Subset(gaultois_db, y_train_mp.index.to_list())
# test_set = Subset(gaultois_db, y_test_mp.index.to_list())

train_set, test_set = random_split(
    gaultois_db, [round(len(gaultois_db) * 0.9), round(len(gaultois_db) * 0.1)]
)

train_generator = DataLoader(train_set, **data_params)
test_generator = DataLoader(test_set, **data_params)


# %%
roost_featurizer = Roost(
    task=task,
    robust=False,  # whether to use robust_mse as a training criterion
    n_targets=1,  # number of targets to fit roost to, just 1 usually
    elem_emb_len=200,  # default element embedding length for use with matscholar
    device="cpu",
    non_elem_fea_len=1,  # accommodate temperature as a non-message-passed input
    # feature to the residual net
)


# %%
learning_rate = 3e-4
weight_decay = 1e-6

optimizer = torch.optim.AdamW(
    roost_featurizer.parameters(), lr=learning_rate, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [])


epochs = 20
# criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()  # L2 loss
normalizer = Normalizer()


# %%
roost_featurizer.fit(
    train_generator=train_generator,
    val_generator=None,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    criterion=criterion,
    normalizer=normalizer,
    model_name="roost",
    run_id="test",
    # writer=writer,
)


# %%
*_, zT_train, zT_pred_train = roost_featurizer.predict(train_generator)

print(f"MAE={mae(zT_train, zT_pred_train):.3g}")
print(f"RMSE={rmse(zT_train, zT_pred_train):.3g}")


# %%
*_, zT_test, zT_pred_test = roost_featurizer.predict(test_generator)

print(f"MAE={mae(zT_test, zT_pred_test):.3g}")
print(f"RMSE={rmse(zT_test, zT_pred_test):.3g}")


# %%
roost_features = roost_featurizer.featurize(generator=train_generator)


# %%
targets["T_norm"] = normalize(targets["T"])

roost_features = pd.DataFrame(roost_features).join(
    targets["T_norm"].reset_index(drop=True)
)

[X_train_rst, y_train_rst], [X_test_rst, y_test_rst] = train_test_split(
    roost_features, targets
)


# %%
rf_rst = RandomForestRegressor()
rf_rst.fit(X_train_rst, y_train_rst.zT)


# %%
rf_pred_rst, rf_var_rst = rf_rst.predict(X_test_rst)


# %%
print(f"MAE={mae(y_test_rst.zT, rf_pred_rst):.3g}")
print(f"RMSE={rmse(y_test_rst.zT, rf_pred_rst):.3g}")


# %%
rf_magpie = RandomForestRegressor()
rf_magpie.fit(X_train_mp, y_train_mp.zT)


# %%
rf_pred_mp, rf_var_mp = rf_magpie.predict(X_test_mp)


# %%
print(f"MAE={mae(y_test_mp.zT, rf_pred_mp):.3g}")
print(f"RMSE={rmse(y_test_mp.zT, rf_pred_mp):.3g}")
