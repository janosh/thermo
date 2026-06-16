"""Hamiltonian Monte Carlo sampling and prediction for BNNs."""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from thermo.bnn import build_net, chunks, target_log_prob_fn_factory


def pre_train_nn(x_train, y_train, nodes_per_layer, epochs=100) -> tuple:
    """Pre-train NN to get good starting point for HMC.

    Args:
        x_train (Tensor or np.array): training samples
        y_train (Tensor or np.array): training targets
        nodes_per_layer (list): the number of nodes in each dense layer
        epochs (int): number of training epochs. Defaults to 100.

    Returns:
        Tensor: list of arrays specifying the weights of the trained network
        model: Keras Sequential model
    """
    layers = [tf.keras.layers.Dense(n, activation="relu") for n in nodes_per_layer]
    layers[-1].activation = tf.identity  # Make last layer linear.
    model = tf.keras.Sequential(layers)

    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs=epochs, verbose=0)
    return model.get_weights(), model


def trace_fn(current_state, kernel_results, summary_freq=10, callbacks=()) -> tuple:
    """Can be passed to the HMC kernel to obtain a trace of intermediate
    kernel results and histograms of the network parameters in Tensorboard.
    """
    step = kernel_results.step
    with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
        for idx, tensor in enumerate(current_state):
            count = idx // 2 + 1
            name = ("w" if idx % 2 == 0 else "b") + str(count)
            tf.summary.histogram(name, tensor, step=step)
        return kernel_results, [cb(*current_state) for cb in callbacks]


@tf.function(experimental_compile=True)
def sample_chain(*args: Any, **kwargs: Any) -> Any:
    """Compile static graph for sample_chain to improve performance."""
    return tfp.mcmc.sample_chain(*args, **kwargs)


def run_hmc(
    target_log_prob_fn,
    step_size=0.01,
    num_leapfrog_steps=10,
    num_burn_in_steps=1000,
    num_results=1000,
    current_state=None,
    resume=None,
    log_dir="logs/hmc/",
    sampler="nuts",
    step_size_adapter="dual_averaging",
    **kwargs: Any,
) -> tuple:
    """Use adaptive HMC to generate a Markov chain of length num_results.

    Args:
        target_log_prob_fn (callable): Determines the stationary distribution
            the Markov chain should converge to.
        step_size (float): Initial leapfrog step size. Defaults to 0.01.
        num_leapfrog_steps (int): Number of leapfrog steps per HMC proposal
            (only used by the "hmc" sampler). Defaults to 10.
        num_burn_in_steps (int): Number of warm-up steps used for step size
            adaptation and discarded from the chain. Defaults to 1000.
        num_results (int): Number of samples to keep after burn-in. Defaults to 1000.
        current_state (list, optional): Initial parameter state to start sampling
            from. Required unless resume is given.
        resume (tuple, optional): Previous (chain, trace, kernel_results) to
            continue sampling from. Required unless current_state is given.
        log_dir (str): Directory for TensorBoard summaries. Defaults to "logs/hmc/".
        sampler (str): Either "nuts" or "hmc". Defaults to "nuts".
        step_size_adapter (str): Either "simple" or "dual_averaging". Defaults to
            "dual_averaging".
        **kwargs: Additional keyword arguments forwarded to sample_chain.

    Returns:
        burn_in(s): Discarded samples generated during warm-up
        chain(s): Markov chain(s) of samples distributed according to
            target_log_prob_fn (if converged)
        trace: the data collected by trace_fn
        final_kernel_result: kernel results of the last step (in case the
            computation needs to be resumed)
    """
    err = "Either current_state or resume is required when calling run_hmc"
    assert current_state is not None or resume is not None, err

    summary_writer = tf.summary.create_file_writer(log_dir)

    step_size_adapter = {
        "simple": tfp.mcmc.SimpleStepSizeAdaptation,
        "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
    }[step_size_adapter]
    if sampler == "nuts":
        kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size)
        adaptive_kernel = step_size_adapter(
            kernel,
            num_adaptation_steps=num_burn_in_steps,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                step_size=new_step_size
            ),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )
    else:  # sampler == "hmc"
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )
        adaptive_kernel = step_size_adapter(
            kernel, num_adaptation_steps=num_burn_in_steps
        )

    if resume:
        prev_chain, prev_trace, prev_kernel_results = resume
        step = len(prev_chain)
        current_state = tf.nest.map_structure(lambda chain: chain[-1], prev_chain)
    else:
        prev_kernel_results = adaptive_kernel.bootstrap_results(current_state)
        step = 0

    tf.summary.trace_on(graph=True, profiler=False)

    chain, trace, final_kernel_results = sample_chain(
        kernel=adaptive_kernel,
        current_state=current_state,
        previous_kernel_results=prev_kernel_results,
        num_results=num_burn_in_steps + num_results,
        trace_fn=trace_fn,
        return_final_kernel_results=True,
        **kwargs,
    )

    with summary_writer.as_default():
        tf.summary.trace_export(name="hmc_trace", step=step)
    summary_writer.close()

    if resume:
        chain = nest_concat(prev_chain, chain)
        trace = nest_concat(prev_trace, trace)
    burn_in, samples = zip(
        *((t[:-num_results], t[-num_results:]) for t in chain), strict=True
    )
    return burn_in, samples, trace, final_kernel_results


def predict_from_chain(chain, x_test, uncertainty="aleatoric_epistemic") -> tuple:
    """Takes a Markov chain of NN configurations and does the actual
    prediction on a test set x_test including aleatoric and optionally
    epistemic uncertainty estimation.
    """
    if uncertainty == "aleatoric":
        post_params = [tf.reduce_mean(t, axis=0) for t in chain]
        post_model = build_net(post_params)
        y_pred, y_var = post_model(x_test, training=False)

        return y_pred.numpy(), y_var.numpy()

    if uncertainty == "aleatoric_epistemic":
        restructured_chain = [
            [tensor[i] for tensor in chain] for i in range(len(chain[0]))
        ]

        def predict(params) -> tuple[tf.Tensor, tf.Tensor]:
            post_model = build_net(params)
            y_pred, y_var = post_model(x_test, training=False)
            return y_pred, y_var

        preds = [predict(chunks(params, 2)) for params in restructured_chain]
        y_pred_mc_samples, y_var_mc_samples = tf.unstack(preds, axis=1)
        y_pred, y_var_epist = tf.nn.moments(y_pred_mc_samples, axes=0)
        y_var_aleat = tf.reduce_mean(y_var_mc_samples, axis=0)
        y_var_tot = y_var_epist + y_var_aleat
        return y_pred, y_var_tot

    raise ValueError(f"unrecognized uncertainty type: {uncertainty}")


def hmc_predict(
    weight_prior,
    bias_prior,
    init_state,
    x_train,
    y_train,
    x_test,
    _y_test=None,
    **kwds: Any,
) -> tuple:
    """Top-level function that ties together run_hmc and predict_from_chain by accepting
    a train and test set plus parameter priors to construct the BNN's log probability
    function given the training data x_train, y_train.
    """
    bnn_log_prob_fn = target_log_prob_fn_factory(
        weight_prior, bias_prior, x_train, y_train
    )
    # Flatten init_state since TFP's sample_chain can't handle sublists.
    init_state = [i for sublist in init_state for i in sublist]
    _burn_in, samples, _trace, final_kernel_results = run_hmc(
        bnn_log_prob_fn, current_state=init_state, **kwds
    )

    y_pred, y_var = predict_from_chain(samples, x_test)
    return y_pred.numpy(), y_var.numpy(), final_kernel_results


def nest_concat(*args: Any, axis=0) -> Any:
    """Utility function for concatenating a new Markov chain or trace with
    older ones when resuming a previous calculation.
    """
    return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)


def ess(chains, **kwargs: Any) -> Any:
    """Estimate effective sample size of Markov chain(s).

    Args:
        chains (Tensor or list of Tensors): If list, first
            dimension should index identically distributed states.
        **kwargs: Additional keyword arguments forwarded to
            tfp.mcmc.effective_sample_size.
    """
    return tfp.mcmc.effective_sample_size(chains, **kwargs)


def r_hat(tensors) -> list:
    """TFP docs https://www.tensorflow.org/probability."""
    return [tfp.mcmc.diagnostic.potential_scale_reduction(t) for t in tensors]


def count_chains(target_log_prob_fn, current_state) -> Any:
    """Check how many chains your kernel thinks it's dealing
    with. Can help with debugging.
    """
    return tf.size(target_log_prob_fn(current_state)).numpy()
