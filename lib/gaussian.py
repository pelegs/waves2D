import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import gaussian

sys.path.insert(1, os.path.abspath("."))
from lib.defs import AX, npdarr, npiarr


def gaussian_kernel(sidelen: int = 5, sigma: float = 3):
    """Generate a 2D Gaussian kernel."""
    gkern1d = gaussian(sidelen, std=sigma).reshape(sidelen, 1)
    return np.outer(gkern1d, gkern1d)


def add_gaussian(
    mat: npdarr,
    center: npiarr,
    sidelen: int,
    amplitude: float = 1.0,
):
    """Add a gaussian kernel to a 2D matrix at a specified position."""
    ll_idx = center - sidelen
    rh_idx = center + sidelen + 1
    mat[
        ll_idx[AX.X] : rh_idx[AX.X], ll_idx[AX.Y] : rh_idx[AX.Y]
    ] += amplitude * gaussian_kernel(sidelen=sidelen * 2 + 1, sigma=np.sqrt(sidelen))
    return mat


if __name__ == "__main__":
    mat = np.zeros((400, 400))
    mat = add_gaussian(mat, center=np.array([200, 200]), sidelen=50, amplitude=1)

    # testing kernel
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mat)
    plt.show()
