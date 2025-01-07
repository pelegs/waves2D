import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(1, os.path.abspath("."))
from lib.defs import AX, npdarr, npiarr
from lib.gaussian import add_gaussian

periodics = {
    "sin": lambda A, f, t: A * np.sin(2 * np.pi * f * t),
    "cos": lambda A, f, t: A * np.cos(2 * np.pi * f * t),
}


class Oscillator:
    """Docstring for Oscillator."""

    def __init__(
        self,
        sim_sides: npiarr,
        pos: npiarr,
        sidelen: int,
        type: str,
        amplitude: float = 1.0,
        freq: float = 2 * np.pi,
    ) -> None:
        self.sim_sides: npiarr = sim_sides
        self.pos: npiarr = pos
        self.sidelen: int = sidelen
        self.amplitude: float = amplitude
        self.freq: float = freq

        try:
            self.func: Callable[[float, float, float], float] = periodics[type]
        except:
            raise KeyError(f"Type '{type}' for an oscillator is unknown")

        self.template: npdarr = np.zeros((sim_sides[AX.X], sim_sides[AX.Y]))
        self.generate_template()

    def generate_template(self) -> None:
        self.template = add_gaussian(
            mat=self.template,
            center=self.pos,
            sidelen=self.sidelen,
            amplitude=self.amplitude,
        )

    def get_func_val(self, time: float) -> float:
        return self.func(self.amplitude, self.freq, time)

    def get_shape(self, time: float) -> npdarr:
        return self.template * self.get_func_val(time)


if __name__ == "__main__":
    from random import choice

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    A: float = 10.0
    side: int = 500

    num_oscillators: int = 20
    osc_list: list[Oscillator] = [
        Oscillator(
            sim_sides=side * np.ones(2, dtype=int),
            pos=np.random.randint(100, side - 100, size=2),
            sidelen=np.random.randint(10, 50),
            type=choice(["sin", "cos"]),
            amplitude=np.random.uniform(0.1, 10.0),
            freq=np.random.uniform(1, 5 * np.pi),
        )
        for _ in range(num_oscillators)
    ]
    osc_arr: npdarr = np.zeros((num_oscillators, side, side))
    osc_arr_sum: npdarr = np.zeros((side, side))

    def update_data(time) -> None:
        for i, _ in enumerate(osc_list):
            osc_arr[i] = osc_list[i].get_shape(time)
        osc_arr_sum[:, :] = np.sum(osc_arr, axis=0)

    update_data(0.0)

    def update_anim(step):
        time: float = step * 0.01
        update_data(time)
        heatmap.set_data(osc_arr_sum)
        return [heatmap]

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    heatmap = ax.imshow(
        osc_arr_sum,
        cmap="jet",
        interpolation="nearest",
        vmin=-A,
        vmax=A,
    )

    anim = FuncAnimation(fig=fig, func=update_anim, frames=650, interval=1, blit=True)
    plt.show()
