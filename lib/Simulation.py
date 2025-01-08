import os
import sys

import numpy as np
from PIL import Image
from scipy.signal import convolve
from tqdm import tqdm

sys.path.insert(1, os.path.abspath("."))
from lib.defs import AX, npdarr, npiarr
from lib.gaussian import add_gaussian
from lib.laplacians import laplace2d
from lib.Oscillator import Oscillator


class Simulation:
    """Docstring for Simulation."""

    def __init__(
        self,
        dimensions: npiarr,
        num_steps: int = 100,
        t_max: float = 1.0,
        laplacian: npdarr = laplace2d["h4"],
        init_state: npdarr | float | None = None,
        velocities: npdarr | float | None = None,
        oscillators: list[Oscillator] = list(),
        boundary_type: str = "absorbing",
    ) -> None:
        self.dimensions: npiarr = dimensions
        self.width, self.height = dimensions
        self.num_steps: int = num_steps
        self.laplacian: npdarr = laplacian
        self.gap: int = (self.laplacian.shape[0] - 1) // 2

        # Initial form
        if isinstance(init_state, np.ndarray):
            self.init_state: npdarr = init_state
        elif init_state is None:
            self.init_state: npdarr = np.zeros((self.width, self.height))
        else:
            raise ValueError(
                f"Type '{type(init_state)}' is not supported for initial state matrix"
            )

        # Velocity array
        if isinstance(velocities, np.ndarray):
            self.velocities: npdarr = velocities
        elif isinstance(velocities, float):
            self.velocities = velocities * np.ones((self.width, self.height))
        elif velocities is None:
            self.velocities: npdarr = np.zeros((self.width, self.height))
        else:
            raise ValueError(
                f"Type '{type(velocities)}' is not supported for velocities matrix"
            )

        # Oscillators
        self.oscillators: list[Oscillator] = oscillators

        # Boundaries
        self.boundary_type: str = boundary_type

        # Time related stuff
        self.step = 2
        self.time_series = np.linspace(0, t_max, self.num_steps)
        self.time = self.time_series[self.step]

        # Set up data array
        self.data_array = np.zeros((num_steps, self.width, self.height))

    def load_init_state_from_file(self, filename: str) -> None:
        pass

    def load_velocities_matrix_from_file(
        self, filename: str, vmin: float = 0.0, vmax: float = 1.0
    ) -> None:
        try:
            img = Image.open(filename).convert("L")
        except OSError as err:
            raise err
        width, height = img.size
        if width != self.width or height != self.height:
            raise ValueError(
                f"Velocity matrix should have exact same dimensions as simulation area: ({self.width},{self.height}) != ({width},{height})"
            )

        # Set up array from image
        arr: npdarr = (
            np.array(img.getdata(), dtype=np.uint8)
            .reshape((self.width, self.height))
            .astype(float)
        )

        # Adjust array to fit given paramters
        self.velocities = arr / 256 * (vmax - vmin) + vmin

    def advance_oscillators(self) -> None:
        for oscillator in self.oscillators:
            self.data_array[self.step] += oscillator.get_shape(self.time)

    def calc_step(self) -> None:
        self.data_array[
            self.step,
            self.gap : self.dimensions[AX.X] - self.gap,
            self.gap : self.dimensions[AX.Y] - self.gap,
        ] += (
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
        for step, time in enumerate(
            tqdm(self.time_series[2:-1], desc="Running simulation"), start=2
        ):
            self.step = step
            self.time = time
            self.advance_oscillators()
            self.calc_step()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    side: int = 500
    dims: npiarr = np.array([side, side])

    osc1: Oscillator = Oscillator(
        sim_sides=dims,
        pos=np.array([150, 250]),
        sidelen=50,
        amplitude=1.0,
        freq=np.pi / 2,
        type="sin",
    )
    osc2: Oscillator = Oscillator(
        sim_sides=dims,
        pos=np.array([350, 250]),
        sidelen=50,
        amplitude=1.0,
        freq=np.pi / 2,
        type="sin",
    )
    osc_list: list[Oscillator] = [osc1, osc2]
    # num_oscillators: int = 5
    # osc_list: list[Oscillator] = [
    #     Oscillator(
    #         sim_sides=side * np.ones(2, dtype=int),
    #         pos=np.random.randint(100, side - 100, size=2),
    #         sidelen=np.random.randint(5, 15),
    #         amplitude=np.random.uniform(0.1, 2.0),
    #         type="sin",
    #         freq=np.random.uniform(1, 5 * np.pi),
    #     )
    #     for _ in range(num_oscillators)
    # ]

    sim = Simulation(
        dimensions=dims,
        num_steps=1000,
        t_max=10.0,
        init_state=None,
        velocities=0.3,
        oscillators=osc_list,
    )
    sim.run()

    def update_anim(step):
        heatmap.set_data(sim.data_array[step])
        frame_count.set_text(f"Frame: {step}/{sim.num_steps}")
        return [heatmap, frame_count]

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    heatmap = ax.imshow(
        sim.init_state,
        cmap="seismic",
        interpolation="nearest",
        vmin=-100,
        vmax=100,
    )
    frame_count = ax.annotate("Frame: 0/0", (10, 10))

    anim = FuncAnimation(
        fig=fig, func=update_anim, frames=sim.num_steps - 2, interval=1, blit=True
    )
    plt.show()
