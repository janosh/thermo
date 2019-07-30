import functools as ft

import tensorflow as tf
import tensorflow_probability as tfp


def dense(inputs, w, b, activation=tf.identity):
    return activation(tf.matmul(inputs, w) + b)


def build_net(params, activation=tf.nn.relu):
    def model(X, training=True):
        for w, b in params[:-1]:
            X = dense(X, w, b, activation)
        # final linear layer
        X = dense(X, *params[-1])
        y_pred, y_log_var = tf.unstack(X, axis=-1)
        y_var = tf.exp(y_log_var)
        if training:
            return tfp.distributions.Normal(loc=y_pred, scale=tf.sqrt(y_var))
        return y_pred, y_var

    return model


def bnn_log_prob_fn(X, y, params, get_mean=False):
    """Compute log likelihood of predicted labels y given features X and params.

    Args:
        X (np.array): 2d feature values.
        y (np.array): 1d labels (ground truth).
        params (list): [[w1, b1], ...] containing 2d/1d arrays for weights/biases.
        get_mean (bool, optional): Whether to return the mean log
            probability over all labels for diagnostics, e.g. to
            compare train and test set performance.

    Returns:
        tf.tensor: Sum or mean of log probabilities of all labels.
    """
    net = build_net(params)
    labels_dist = net(X)
    if get_mean:
        return tf.reduce_mean(labels_dist.log_prob(y))
    return tf.reduce_sum(labels_dist.log_prob(y))


def prior_log_prob_fn(w_prior, b_prior, params):
    log_prob = 0
    for w, b in params:
        log_prob += tf.reduce_sum(w_prior.log_prob(w))
        log_prob += tf.reduce_sum(b_prior.log_prob(b))
    return log_prob


def target_log_prob_fn_factory(w_prior, b_prior, X_train, y_train):
    # This signature is forced by TFP's HMC kernel which calls log_prob_fn(*chains).
    def target_log_prob_fn(*params):
        if not isinstance(params[0], (list, tuple)):
            params = chunks(params, 2)
        log_prob = prior_log_prob_fn(w_prior, b_prior, params)
        log_prob += bnn_log_prob_fn(X_train, y_train, params)
        return log_prob

    return target_log_prob_fn


def tracer_factory(X, y):
    return lambda params: ft.partial(bnn_log_prob_fn, X, y, get_mean=True)(params)


def chunks(lst, n):
    # Subdivide lst into successive n-sized chunks.
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def get_random_initial_state(w_prior, b_prior, nodes_per_layer, overdisp=1.0):
    """Draw random samples for weights and biases of a NN according to some
    specified priors. distributions. This set of parameter values can serve as a
    starting point for MCMC or gradient descent training.
    """
    init_state = []
    for n1, n2 in zip(nodes_per_layer, nodes_per_layer[1:]):
        w_shape, b_shape = [n1, n2], n2
        # Use overdispersion > 1 for better R-hat statistics.
        w = w_prior.sample(w_shape) * overdisp
        b = b_prior.sample(b_shape) * overdisp
        init_state.append([tf.Variable(w), tf.Variable(b)])
    return init_state
