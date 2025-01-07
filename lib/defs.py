from enum import IntEnum

import numpy as np

# Types (for hints)
npdarr = np.typing.NDArray[np.float64]
npfarr = np.typing.NDArray[np.float32]
npiarr = np.typing.NDArray[np.int8]


# Axes (for easy following?)
class AX(IntEnum):
    X = 0
    Y = 1
    # Z = 2
