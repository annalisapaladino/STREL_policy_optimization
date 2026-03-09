import numpy as np


def append_epoch_stats(stats_history, epoch_rob_np, loss_value):
    stats_history["mean"].append(float(np.mean(epoch_rob_np)))
    stats_history["std"].append(float(np.std(epoch_rob_np)))
    stats_history["median"].append(float(np.median(epoch_rob_np)))
    stats_history["loss"].append(float(loss_value))
