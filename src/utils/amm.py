import os

import automatminer as amm
import matminer as mm
from automatminer import MatPipe  # make MatPipe importable from this file

try:
    os.remove(os.getcwd() + "/automatminer.log")  # delete since not needed
except FileNotFoundError:
    pass

featurizers = {
    "composition": [mm.featurizers.composition.ElementProperty.from_preset("magpie")],
    "structure": [],
}

# To save time and computation, we only retain regressors that were among the best in
# prior unrestricted benchmarking. See https://tinyurl.com/y27hajoy and "Customizing
# TPOT's operators and parameters" at https://epistasislab.github.io/tpot/using.
good_regressors = (
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
)

# Don't modify TPOT_REGRESSOR_CONFIG in place. Would throw: 'RuntimeError: There was an
# error in the TPOT optimization process. This could be because...'
tpot_config = amm.automl.config.tpot_configs.TPOT_REGRESSOR_CONFIG.copy()

for key in list(tpot_config):
    if not key.endswith(good_regressors):
        del tpot_config[key]

# pipe_config needs to be a function rather than a dict. Else multiple pipes
# running concurrently will receive a handle to the same learner (pipe.learner)
# and overwrite each other's fitting.
pipe_config = lambda preset="express": {
    **amm.get_preset_config(preset),
    "autofeaturizer": amm.AutoFeaturizer(
        # preset="express",
        featurizers=featurizers,
        guess_oxistates=False,
    ),
    "learner": amm.TPOTAdaptor(max_time_mins=10),  # config_dict=tpot_config
}


def fit_pred_pipe(train_df, test_df, target):
    mat_pipe = MatPipe(**pipe_config())
    mat_pipe.fit(train_df[["T", "composition", target]], target)
    pred_df = mat_pipe.predict(
        test_df[["T", "composition"]], output_col=target + "_pred"
    )
    return mat_pipe, pred_df


def featurize(pipe, df):
    """Use a fitted MatPipe, specifically its autofeaturizer, cleaner and
    reducer components, to featurize a dataframe. Can be used in combination
    with custom models that don't fit into the Automatminer API like
    multi-output regressors.
    """
    df = pipe.autofeaturizer.transform(df, pipe.target)
    df = pipe.cleaner.transform(df, pipe.target)
    df = pipe.reducer.transform(df, pipe.target)
    return df