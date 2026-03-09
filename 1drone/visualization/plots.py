import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None


def plot_statistics(stats):
    epochs = np.arange(len(stats["mean"]))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, stats["mean"], label="Mean Robustness", color="blue", lw=2)
    plt.plot(epochs, stats["median"], label="Median Robustness", color="green", linestyle="--")
    plt.fill_between(
        epochs,
        np.array(stats["mean"]) - np.array(stats["std"]),
        np.array(stats["mean"]) + np.array(stats["std"]),
        color="blue",
        alpha=0.2,
        label="Std Dev",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Robustness")
    plt.title("Robustness over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_robustness_dist(rob_values):
    plt.figure(figsize=(8, 4))
    if sns is not None:
        sns.histplot(rob_values, kde=True, color="purple", bins=20)
    else:
        plt.hist(rob_values, bins=20, color="purple", alpha=0.7)
    plt.axvline(np.mean(rob_values), color="red", linestyle="dashed", linewidth=1, label=f"Mean: {np.mean(rob_values):.2f}")
    plt.title("Robustness Distribution (Histogram and KDE)")
    plt.legend()
    plt.show()


def plot_training_curves(stats_history, epoch):
    plt.figure(figsize=(10, 6))
    epochs = np.arange(len(stats_history["mean"]))
    loss_arr = np.array(stats_history["loss"])

    plt.plot(epochs, stats_history["mean"], color="tab:blue", lw=2, label="Mean Robustness")
    plt.plot(epochs, -loss_arr, color="tab:red", linestyle="--", alpha=0.8, label="-Loss (mirrored)")
    plt.plot(epochs, loss_arr, color="tab:orange", linestyle=":", alpha=0.9, label="Loss")

    plt.axhline(0, color="black", lw=1, alpha=0.3)
    plt.xlabel("Epochs")
    plt.ylabel("Scalar Value (common scale)")
    plt.title(f"Training Curves - Epoch {epoch}\n(Robustness vs Loss)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_robustness_comparison(current_rob_trajectory, historical_avg_value):
    plt.figure(figsize=(10, 4))
    t_steps = np.arange(current_rob_trajectory.shape[1])

    for i in range(current_rob_trajectory.shape[0]):
        plt.plot(t_steps, current_rob_trajectory[i], label=f"D{i+1} Current", alpha=0.8)

    plt.axhline(y=historical_avg_value, color="black", linestyle="--", label=f"Historical Avg: {historical_avg_value:.3f}")
    plt.title("Instantaneous Robustness vs Historical Average")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Robustness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
