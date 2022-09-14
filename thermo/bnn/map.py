import numpy as np
import tensorflow as tf

import thermo.bnn as bnn


def get_map_trace(target_log_prob_fn, state, n_iter=1000, save_every=10, callbacks=()):
    optimizer = tf.optimizers.Adam()

    @tf.function
    def minimize():
        optimizer.minimize(lambda: -target_log_prob_fn(*state), state)

    state_trace, cb_trace = [], [[] for _ in callbacks]
    for i in range(n_iter):
        if i % save_every == 0:
            state_trace.append(state)
            for trace, cb in zip(cb_trace, callbacks):
                trace.append(cb(state).numpy())
        minimize()

    return state_trace, cb_trace


def get_best_map_state(map_trace, map_log_probs):
    # map_log_probs[0/1]: train/test log probability
    test_set_max_log_prob_idx = np.argmax(map_log_probs[1])
    # Return MAP params that achieved highest test set likelihood.
    return map_trace[test_set_max_log_prob_idx]


def get_nodes_per_layer(n_features, net_taper=(1, 0.5, 0.2, 0.1)):
    nodes_per_layer = [int(n_features * x) for x in net_taper]
    # Ensure the last layer has two nodes so that output can be split into
    # predictive mean and learned loss attenuation (see eq. (7) of
    # https://arxiv.org/abs/1703.04977) which the network learns individually.
    nodes_per_layer.append(2)
    return nodes_per_layer


def map_predict(weight_prior, bias_prior, X_train, y_train, X_test, y_test):
    """Generate maximum a posteriori neural network predictions.

    Args:
        weight_prior (tfp.distribution): Prior probability for the weights
        bias_prior (tfp.distribution): Prior probability for the biases
        [X/y_train/test] (np.arrays): Train and test sets
    """

    log_prob_tracers = (
        bnn.tracer_factory(X_train, y_train),
        bnn.tracer_factory(X_test, y_test),
    )

    n_features = X_train.shape[-1]
    nodes = get_nodes_per_layer(n_features, net_taper=(1, 0.5, 0.3))
    random_initial_state = bnn.get_random_initial_state(weight_prior, bias_prior, nodes)

    trace, log_probs = get_map_trace(
        bnn.target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train),
        random_initial_state,
        n_iter=5000,
        callbacks=log_prob_tracers,
    )
    # Initial configuration for HMC.
    best_params = get_best_map_state(trace, log_probs)

    model = bnn.build_net(best_params)
    y_pred, y_var = model(X_test, training=False)
    return y_pred.numpy(), y_var.numpy(), log_probs, best_params
