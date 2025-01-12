import os
import sys
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

sys.path.insert(1, os.path.abspath("."))
from lib.defs import AX, npdarr, npiarr

periodic_funcs = {
    "sin": lambda A, f, t: A * np.sin(2 * np.pi * f * t),
    "cos": lambda A, f, t: A * np.cos(2 * np.pi * f * t),
}


class Source(ABC):
    """
    Summary:
        This class represents a field source. It can either be static,
    instantaneous or periodic (each as an inhereted class).

    Args:
        sim_sides: the side lengths of the simulation area.
        pos: the position of the source (its center).

    Attributes:
        mat: hold the shape of the source at any given time.
    """

    mat: npdarr

    def __init__(self, sim_sides: npiarr, pos: npiarr) -> None:
        self.sim_sides: npiarr = sim_sides
        self.pos: npiarr = pos
        self.init_steps()

    @abstractmethod
    def init_steps(self) -> None:
        """
        Summary:
            Initialization steps for the source. Exists to provide uiform interface for
            all types of sources (e.g. periodic ones have a time argument).

        Note:
            This function is called at initialization.
        """
        pass

    @abstractmethod
    def get_mat(self) -> npdarr:
        """
        Summary:
            Returns the source's matrix.
        """
        pass
