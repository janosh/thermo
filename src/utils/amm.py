import automatminer as amm
import matminer as mm
from automatminer import MatPipe  # make MatPipe importable from this file

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

# pipe_config needs to be a function rather than a dict. Else multiple pipes
# running concurrently will receive a handle to the same learner (pipe.learner)
# and overwrite each other's fitting.
pipe_config = lambda: {
    **amm.get_preset_config(),
    "autofeaturizer": amm.AutoFeaturizer(
        # preset="express",
        featurizers=featurizers,
        guess_oxistates=False,
    ),
    "learner": amm.TPOTAdaptor(max_time_mins=10),
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
    with custom models that don't fit into the Automatminer API, e.g.
    multi-output regressors.
    """
    df = pipe.autofeaturizer.transform(df, pipe.target)
    df = pipe.cleaner.transform(df, pipe.target)
    df = pipe.reducer.transform(df, pipe.target)
    return df
