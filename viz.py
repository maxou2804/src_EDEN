import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from matplotlib.animation import FFMpegWriter

cmap = ListedColormap(["white", "royalblue", "crimson"])

def plot_metrics(metrics):
    fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    ax[0].plot(metrics["urban_fraction"], marker="o")
    ax[0].set_ylabel("Urban fraction")
    ax[1].plot(metrics["citycore_fraction"], marker="o", color="crimson")
    ax[1].set_ylabel("City core fraction")
    ax[2].plot(metrics["n_clusters"], marker="o", color="royalblue")
    ax[2].set_ylabel("Number of clusters")
    ax[2].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()


def animate(history, interval=500, save_path=None, cmap="viridis"):
    """
    Animate a sequence of 2D grids with a time legend and optionally save as GIF.

    Parameters
    ----------
    history : list of 2D arrays
        Sequence of frames to animate.
    interval : int, optional
        Delay between frames in milliseconds. Default = 500.
    save_path : str, optional
        Path (including filename) to save the gif. 
        Example: "outputs/animation.gif"
    cmap : str, optional
        Colormap for imshow. Default = "viridis".
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=2)
    ax.axis("off")

    # Add text to display time
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        color="white", fontsize=12, ha="left", va="top",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3")
    )

    def update(frame_idx):
        im.set_data(history[frame_idx])
        time_text.set_text(f"Time: {frame_idx}")
        return [im, time_text]

    ani = FuncAnimation(
        fig, update, frames=len(history),
        blit=True, interval=interval, repeat=False
    )

    # Save as GIF if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer=FFMpegWriter(fps=1000//interval))
        print(f"Animation saved to {save_path}")

    plt.show()
    return ani

