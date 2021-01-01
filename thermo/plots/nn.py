from matplotlib import pyplot as plt


def loss_history(hist):
    fig = plt.figure(figsize=[12, 5])
    for key, data in hist.items():
        if "loss" in key:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(data, label=key)
            ax1.set(xlabel="epoch")
        else:  # plot other metrics like accuracy or loss without
            # regularizers on a separate axis
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(data, label=key)
            ax2.set(xlabel="epoch")

    [ax.legend() for ax in fig.axes]

    plt.show()


def log_probs(log_probs, text=None, title=None):
    for legend_label, log_prob in log_probs:
        plt.plot(log_prob, label=legend_label)
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.title(title)
    if text:
        plt.gcf().text(*(0.5, -0.1), text, horizontalalignment="center")

    plt.show()
