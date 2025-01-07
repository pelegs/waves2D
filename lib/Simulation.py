import os
import sys

import numpy as np
from scipy.signal import convolve
from tqdm import tqdm

sys.path.insert(1, os.path.abspath("."))
from lib.defs import AX, npdarr, npiarr
from lib.gaussian import add_gaussian
from lib.laplacians import laplace2d


class Simulation:
    """Docstring for Simulation."""

    def __init__(
        self,
        dimensions: npiarr,
        num_steps: int = 100,
        laplacian: npdarr = laplace2d["h2"],
        init_state: npdarr | float | None = None,
        velocities: npdarr | float | None = None,
        boundary_type: str = "absorbing",
    ) -> None:
        self.dimensions: npiarr = dimensions
        self.num_steps: int = num_steps
        self.laplacian: npdarr = laplacian
        self.gap: int = (self.laplacian.shape[0] - 1) // 2

        if isinstance(init_state, np.ndarray):
            self.init_state: npdarr = init_state
        elif init_state is None:
            self.init_state: npdarr = np.zeros((dimensions[0], dimensions[1]))
        else:
            raise ValueError(
                f"Type '{type(init_state)}' is not supported for initial state matrix"
            )

        if isinstance(velocities, np.ndarray):
            self.velocities: npdarr = velocities
        elif isinstance(velocities, float):
            self.velocities = velocities * np.ones((dimensions[0], dimensions[1]))
        elif velocities is None:
            self.velocities: npdarr = np.zeros((dimensions[0], dimensions[1]))
        else:
            raise ValueError(
                f"Type '{type(velocities)}' is not supported for velocities matrix"
            )

        self.boundary_type: str = boundary_type

        self.step = 2
        self.data_array = np.zeros((num_steps, self.dimensions[0], self.dimensions[1]))

    def load_init_state_from_file(self, filename: str) -> None:
        pass

    def load_velocities_matrix_from_file(self, filename: str) -> None:
        pass

    def calc_step(self) -> None:
        self.data_array[
            self.step,
            self.gap : self.dimensions[AX.X] - self.gap,
            self.gap : self.dimensions[AX.Y] - self.gap,
        ] = (
            self.velocities[
                self.gap : self.dimensions[AX.X] - self.gap,
                self.gap : self.dimensions[AX.Y] - self.gap,
            ]
            * convolve(
                self.data_array[
                    self.step - 1,
                    self.gap : self.dimensions[AX.X] - self.gap,
                    self.gap : self.dimensions[AX.Y] - self.gap,
                ],
                self.laplacian,
                mode="same",
            )
            + 2
            * self.data_array[
                self.step - 1,
                self.gap : self.dimensions[AX.X] - self.gap,
                self.gap : self.dimensions[AX.Y] - self.gap,
            ]
            - self.data_array[
                self.step - 2,
                self.gap : self.dimensions[AX.X] - self.gap,
                self.gap : self.dimensions[AX.Y] - self.gap,
            ]
        )

    def run(self) -> None:
        self.data_array[0] = self.data_array[1] = self.init_state
        for step in tqdm(range(2, self.num_steps - 1), desc="Running simulation"):
            self.step = step
            self.calc_step()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    dims: npiarr = np.array([300, 300])
    init: npdarr = np.zeros((dims[0], dims[1]))
    init = add_gaussian(init, center=dims // 2, sidelen=50, amplitude=20)
    sim = Simulation(dimensions=dims, num_steps=500, init_state=init, velocities=0.35)
    sim.run()

    def update_anim(step):
        heatmap.set_data(sim.data_array[step])
        return [heatmap]

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    heatmap = ax.imshow(
        sim.init_state,
        cmap="jet",
        interpolation="nearest",
        vmin=-10,
        vmax=10,
    )

    anim = FuncAnimation(
        fig=fig, func=update_anim, frames=sim.num_steps - 2, interval=1, blit=True
    )
    plt.show()
