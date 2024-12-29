import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import from_levels_and_colors
from scipy.signal import convolve, windows
from tqdm import tqdm

c = 0.25  # wave speed
hs = 1  # spatial step width
ts = 1  # time step width
tau = (c * ts / hs) ** 2

sim_size = np.array([500, 500])
num_steps = 50
sim_arr = np.zeros((num_steps, *sim_size), dtype=float)

# laplace_xyt = (
#     np.array(
#         [
#             [
#                 [0, 0, 0],
#                 [0, -1, 0],
#                 [0, 0, 0],
#             ],
#             [
#                 [0.25, 0.5, 0.25],
#                 [0.5, -1, 0.5],
#                 [0.25, 0.5, 0.25],
#             ],
#             [
#                 [0, 0, 0],
#                 [0, -1, 0],
#                 [0, 0, 0],
#             ],
#         ]
#     )
# )

laplace_xyt = np.zeros((3, 7, 7))
laplace_xyt[1] = (
    np.array(
        [
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, -27, 0, 0, 0],
            [0, 0, 0, 270, 0, 0, 0],
            [2, -27, 270, -980, 270, -27, 2],
            [0, 0, 0, 270, 0, 0, 0],
            [0, 0, 0, -27, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
        ]
    )
    / 180.0
)


def gkern(kernlen=5, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def init_simulation(sim_arr):
    pos = sim_size // 2
    s = 6
    sim_arr[0, pos[0] - s : pos[0] + s + 1, pos[1] - s : pos[1] + s + 1] = gkern(
        2 * s + 1
    )


def step(sim_arr, t):
    sim_arr[t - 1 : t + 1] = convolve(sim_arr[t - 2 : t], laplace_xyt, mode="same")
    return tau * sim_arr


def setup_graphics():
    fig, ax = plt.subplots()
    heatmap = ax.imshow(sim_arr[0], interpolation="nearest", cmap="jet")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax, heatmap


def animation_step(frame):
    heatmap.set_data(sim_arr[frame])
    return [heatmap]


if __name__ == "__main__":
    init_simulation(sim_arr)

    for t in tqdm(range(2, num_steps)):
        sim_arr = step(sim_arr, t)

    fig, ax, heatmap = setup_graphics()
    anim = FuncAnimation(fig, animation_step, frames=num_steps, interval=1, blit=False)
    plt.show()
