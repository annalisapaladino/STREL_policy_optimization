import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from IPython.display import HTML, display, Image
except ImportError:
    HTML = None
    display = None
    Image = None

from drone.config import ScenarioCfg


def visualize_enhanced_results(executed_traj, robustness_history, epoch_label, cfg: ScenarioCfg, title_prefix="Execution"):
    data = torch.stack(executed_traj).permute(1, 2, 0).cpu().numpy()
    num_steps = data.shape[2]

    plt.figure(figsize=(10, 2))
    batt_data = data[cfg.drone_slice, 4, :]
    if sns is not None:
        sns.heatmap(
            batt_data,
            annot=False,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            yticklabels=[f"D{i+1}" for i in range(cfg.n_drones)],
            cbar_kws={"label": "Batt %"},
        )
    else:
        plt.imshow(batt_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.yticks(np.arange(cfg.n_drones), [f"D{i+1}" for i in range(cfg.n_drones)])
        plt.colorbar(label="Batt %")
    plt.title(f"Battery Level - Epoch {epoch_label}")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 7))
    drone_colors = ["red", "blue", "magenta", "cyan", "orange", "brown"]

    def animate(t):
        ax.clear()
        ax.set_xlim(0, cfg.grid_side + 1)
        ax.set_ylim(0, cfg.grid_side + 1)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhspan(8.5, cfg.grid_side + 1, color="yellow", alpha=0.1, label="Target Area")

        for i in range(cfg.n_bases):
            idx = cfg.base_start + i
            bx, by = data[idx, 0, t], data[idx, 1, t]
            is_occupied = data[idx, 4, t] > 0.8
            color = "orange" if is_occupied else "lightgreen"
            ax.scatter(bx, by, c=color, marker="s", s=250, edgecolors="black", zorder=4)
            circle = plt.Circle((bx, by), 0.8, color=color, fill=True, alpha=0.2, linestyle="--")
            ax.add_patch(circle)

        too_close = False
        d12 = None
        if cfg.n_drones == 2:
            i0, i1 = cfg.drone_start, cfg.drone_start + 1
            d12 = np.linalg.norm(data[i0, 0:2, t] - data[i1, 0:2, t])
            too_close = d12 < 1.0

        for i in range(cfg.n_drones):
            idx = cfg.drone_start + i
            px, py = data[idx, 0, t], data[idx, 1, t]
            batt = data[idx, 4, t]
            rob = robustness_history[i, t] if t < robustness_history.shape[1] else 0.0
            c = "purple" if (too_close and i < 2) else drone_colors[i % len(drone_colors)]
            ax.scatter(px, py, c=c, s=130, edgecolors="black", zorder=3)
            ax.text(
                px,
                py + 0.5,
                f"D{i+1}\nB:{batt:.2f}\nR:{rob:.2f}",
                color=c,
                ha="center",
                weight="bold",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

        if d12 is not None:
            ax.text(0.5, cfg.grid_side + 0.6, f"D1-D2 distance: {d12:.2f}", fontsize=9, color="purple" if too_close else "black")

        ax.set_title(f"{title_prefix} Epoch {epoch_label} | Step {t}")

    ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=150)
    if display is not None and HTML is not None:
        display(HTML(ani.to_jshtml()))

    gif_filename = f"trajectory_epoch_{epoch_label}.gif"
    try:
        ani.save(gif_filename, writer="pillow", fps=7)
        print(f"GIF saved: {gif_filename}")
        if display is not None and Image is not None:
            display(Image(filename=gif_filename))
    except Exception as exc:
        print(f"Skipping GIF export: {exc}")
    plt.close(fig)
