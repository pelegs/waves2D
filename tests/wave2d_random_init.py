import math
import random
import time

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.signal.windows import gaussian

hs = 1  # spatial step width
ts = 1  # time step width
dimx = 500  # width of the simulation domain
dimy = 500  # height of the simulation domain
cellsize = 1  # display size of a cell in pixel

laplace_2 = np.array(
    [
        [0.25, 0.5, 0.25],
        [0.5, -3.0, 0.5],
        [0.25, 0.5, 0.25],
    ]
)
laplace_4 = (
    np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 16, 0, 0],
            [-1, 16, -60, 16, -1],
            [0, 0, 16, 0, 0],
            [0, 0, -1, 0, 0],
        ]
    )
    / 12
)
laplace_6 = (
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
    / 180
)
laplace_8 = np.array(
    [
        [0, 0, 0, 0, -1 / 560, 0, 0, 0, 0],
        [0, 0, 0, 0, 8 / 315, 0, 0, 0, 0],
        [0, 0, 0, 0, -1 / 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 8 / 5, 0, 0, 0, 0],
        [
            -1 / 560,
            8 / 315,
            -1 / 5,
            8 / 5,
            -410 / 72,
            8 / 5,
            -1 / 5,
            8 / 315,
            -1 / 560,
        ],
        [0, 0, 0, 0, 8 / 5, 0, 0, 0, 0],
        [0, 0, 0, 0, -1 / 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 8 / 315, 0, 0, 0, 0],
        [0, 0, 0, 0, -1 / 560, 0, 0, 0, 0],
    ]
)


def gkern(kernlen: int = 5, std: float = 3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def create_arrays():
    global velocity
    global tau
    global kappa
    global gauss_peak
    global u

    # The three dimensional simulation grid
    u = np.zeros((3, dimx, dimy))

    # A field containing the velocity for each cell
    velocity = np.zeros((dimx, dimy))

    # A field containing the factor for the Laplace Operator that  combines Velocity and Grid Constants for the Wave Equation
    tau = np.zeros((dimx, dimy))

    # A field containing the factor for the Laplace Operator that combines Velocity and Grid Constants for the Boundary Condition
    kappa = np.zeros((dimx, dimy))

    # Create a template for a gauss peak to use as a rain drop model
    sigma = 2.4
    gauss_peak = 300 / (sigma * np.sqrt(2 * np.pi)) * gkern(kernlen=10, std=sigma)


def set_initial_conditions():
    global velocity
    global tau
    global kappa
    global gauss_peak

    velocity[:, :] = 0.5

    # compute tau and kappa from the velocity field
    tau = ((velocity * ts) / hs) ** 2
    kappa = ts * velocity / hs

    # Place a single gaussian peak at the center of the simulation
    u[:, :] = np.random.normal(0, 1000, size=(dimx, dimy))
    u[:, :] = gaussian_filter(u, sigma=10)
    return u


def update(u: any, method: int):
    u[2] = u[1]
    u[1] = u[0]

    if method == 0:
        boundary_size = 1

        # This is the second order scheme with a laplacian that takes the diagonals into account.
        # The resulting wave shape will look a bit better under certain conditions but the accuracy
        # is still low. In most cases you will hardly see a difference to #1
        u[0, 1 : dimx - 1, 1 : dimy - 1] = (
            tau[1 : dimx - 1, 1 : dimy - 1]
            * convolve(u[1], laplace_2, mode="same")[1 : dimx - 1, 1 : dimy - 1]
            # * (
            #     0.25 * u[1, 0 : dimx - 2, 0 : dimy - 2]  # c-1, r-1 =>  1
            #     + 0.5 * u[1, 1 : dimx - 1, 0 : dimy - 2]  # c,   r-1 =>  1
            #     + 0.25 * u[1, 2:dimx, 0 : dimy - 2]  # c+1, r-1 =>  1
            #     + 0.5 * u[1, 0 : dimx - 2, 1 : dimy - 1]  # c-1, r =>  1
            #     - 3 * u[1, 1 : dimx - 1, 1 : dimy - 1]  # c,   r => -8
            #     + 0.5 * u[1, 2:dimx, 1 : dimy - 1]  # c+1, r =>  1
            #     + 0.25 * u[1, 0 : dimx - 2, 2:dimy]  # c-1, r+1 =>  1
            #     + 0.5 * u[1, 1 : dimx - 1, 2:dimy]  # c,   r+1 =>  1
            #     + 0.25 * u[1, 2:dimx, 2:dimy]  # c+1, r+1 =>  1
            # )
            + 2 * u[1, 1 : dimx - 1, 1 : dimy - 1]
            - u[2, 1 : dimx - 1, 1 : dimy - 1]
        )
    elif (
        method == 1
    ):  # ok, (4)th Order https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 2
        u[0, 2 : dimx - 2, 2 : dimy - 2] = (
            tau[2 : dimx - 2, 2 : dimy - 2]
            * convolve(u[1], laplace_4, mode="same")[2 : dimx - 2, 2 : dimy - 2]
            # * (
            #     -1 * u[1, 2 : dimx - 2, 0 : dimy - 4]  # c    , r-2 => -1
            #     + 16 * u[1, 2 : dimx - 2, 1 : dimy - 3]  # c    , r-1 => 16
            #     - 1 * u[1, 0 : dimx - 4, 2 : dimy - 2]  # c - 2, r => -1
            #     + 16 * u[1, 1 : dimx - 3, 2 : dimy - 2]  # c - 1, r => 16
            #     - 60 * u[1, 2 : dimx - 2, 2 : dimy - 2]  # c    , r => -60
            #     + 16 * u[1, 3 : dimx - 1, 2 : dimy - 2]  # c+1  , r => 16
            #     - 1 * u[1, 4:dimx, 2 : dimy - 2]  # c+2  , r => -1
            #     + 16 * u[1, 2 : dimx - 2, 3 : dimy - 1]  # c    , r+1 => 16
            #     - 1 * u[1, 2 : dimx - 2, 4:dimy]  # c    , r+2 => -1
            # )
            # / 12
            + 2 * u[1, 2 : dimx - 2, 2 : dimy - 2]
            - u[2, 2 : dimx - 2, 2 : dimy - 2]
        )
    elif (
        method == 2
    ):  # ok, (6th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 3
        u[0, 3 : dimx - 3, 3 : dimy - 3] = (
            tau[3 : dimx - 3, 3 : dimy - 3]
            * convolve(u[1], laplace_6, mode="same")[3 : dimx - 3, 3 : dimy - 3]
            # * (
            #     2 * u[1, 3 : dimx - 3, 0 : dimy - 6]  # c,   r-3
            #     - 27 * u[1, 3 : dimx - 3, 1 : dimy - 5]  # c,   r-2
            #     + 270 * u[1, 3 : dimx - 3, 2 : dimy - 4]  # c,   r-1
            #     + 2 * u[1, 0 : dimx - 6, 3 : dimy - 3]  # c - 3, r
            #     - 27 * u[1, 1 : dimx - 5, 3 : dimy - 3]  # c - 2, r
            #     + 270 * u[1, 2 : dimx - 4, 3 : dimy - 3]  # c - 1, r
            #     - 980 * u[1, 3 : dimx - 3, 3 : dimy - 3]  # c    , r
            #     + 270 * u[1, 4 : dimx - 2, 3 : dimy - 3]  # c + 1, r
            #     - 27 * u[1, 5 : dimx - 1, 3 : dimy - 3]  # c + 2, r
            #     + 2 * u[1, 6:dimx, 3 : dimy - 3]  # c + 3, r
            #     + 270 * u[1, 3 : dimx - 3, 4 : dimy - 2]  # c  , r+1
            #     - 27 * u[1, 3 : dimx - 3, 5 : dimy - 1]  # c  , r+2
            #     + 2 * u[1, 3 : dimx - 3, 6:dimy]  # c  , r+3
            # )
            # / 180
            + 2 * u[1, 3 : dimx - 3, 3 : dimy - 3]
            - u[2, 3 : dimx - 3, 3 : dimy - 3]
        )
    elif (
        method == 3
    ):  # ok, (8th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 4
        u[0, 4 : dimx - 4, 4 : dimy - 4] = (
            tau[4 : dimx - 4, 4 : dimy - 4]
            * convolve(u[1], laplace_8, mode="same")[4 : dimx - 4, 4 : dimy - 4]
            + 2 * u[1, 4 : dimx - 4, 4 : dimy - 4]
            - u[2, 4 : dimx - 4, 4 : dimy - 4]
        )

    # Absorbing Boundary Conditions:
    mur = True
    if mur == True:
        update_boundary(u, boundary_size)


def update_boundary(u, sz) -> None:
    """Update the boundary cells.

    Implement MUR boundary conditions. This represents an open boundary
    were waves can leave the simulation domain with little remaining
    reflection artifacts. Although this is of a low error order it is
    good enough for this simulation.
    """
    c = dimx - 1
    u[0, dimx - sz - 1 : c, 1 : dimy - 1] = u[
        1, dimx - sz - 2 : c - 1, 1 : dimy - 1
    ] + (kappa[dimx - sz - 1 : c, 1 : dimy - 1] - 1) / (
        kappa[dimx - sz - 1 : c, 1 : dimy - 1] + 1
    ) * (
        u[0, dimx - sz - 2 : c - 1, 1 : dimy - 1]
        - u[1, dimx - sz - 1 : c, 1 : dimy - 1]
    )

    c = 0
    u[0, c:sz, 1 : dimy - 1] = u[1, c + 1 : sz + 1, 1 : dimy - 1] + (
        kappa[c:sz, 1 : dimy - 1] - 1
    ) / (kappa[c:sz, 1 : dimy - 1] + 1) * (
        u[0, c + 1 : sz + 1, 1 : dimy - 1] - u[1, c:sz, 1 : dimy - 1]
    )

    r = dimy - 1
    u[0, 1 : dimx - 1, dimy - 1 - sz : r] = u[
        1, 1 : dimx - 1, dimy - 2 - sz : r - 1
    ] + (kappa[1 : dimx - 1, dimy - 1 - sz : r] - 1) / (
        kappa[1 : dimx - 1, dimy - 1 - sz : r] + 1
    ) * (
        u[0, 1 : dimx - 1, dimy - 2 - sz : r - 1]
        - u[1, 1 : dimx - 1, dimy - 1 - sz : r]
    )

    r = 0
    u[0, 1 : dimx - 1, r:sz] = u[1, 1 : dimx - 1, r + 1 : sz + 1] + (
        kappa[1 : dimx - 1, r:sz] - 1
    ) / (kappa[1 : dimx - 1, r:sz] + 1) * (
        u[0, 1 : dimx - 1, r + 1 : sz + 1] - u[1, 1 : dimx - 1, r:sz]
    )


def put_gauss_peak(u, x: int, y: int, height):
    """Place a gauss shaped peak into the simulation domain.

    This function will put a gauss shaped peak at position x,y of the
    simulation domain.
    """
    w, h = gauss_peak.shape
    w = int(w / 2)
    h = int(h / 2)

    use_multipole = False
    if use_multipole:
        # Multipole
        dist = 3
        u[0:2, x - w - dist : x + w - dist, y - h : y + h] += height * gauss_peak
        u[0:2, x - w : x + w, y - h + dist : y + h + dist] -= height * gauss_peak
        u[0:2, x - w + dist : x + w + dist, y - h : y + h] += height * gauss_peak
        u[0:2, x - w : x + w, y - h - dist : y + h - dist] -= height * gauss_peak
    else:
        # simple peak
        u[0:2, x - w : x + w, y - h : y + h] += height * gauss_peak


def place_raindrops(u):
    if random.random() < 0.003:
        w, h = gauss_peak.shape
        x = int(random.randrange(w + w // 2, dimx - h - h // 2))
        y = int(random.randrange(w + w // 2, dimy - h - h // 2))

        height = 2
        put_gauss_peak(u, x, y, height)


def draw_waves(display, u, data, offset):
    global velocity
    global font

    data[1:dimx, 1:dimy, 0] = 255 - np.clip(
        (u[0, 1:dimx, 1:dimy] > 0) * 10 * u[0, 1:dimx, 1:dimy]
        + u[1, 1:dimx, 1:dimy]
        + u[2, 1:dimx, 1:dimy],
        0,
        255,
    )
    data[1:dimx, 1:dimy, 1] = 255 - np.clip(np.abs(u[0, 1:dimx, 1:dimy]) * 10, 0, 255)
    data[1:dimx, 1:dimy, 2] = 255 - np.clip(
        (u[0, 1:dimx, 1:dimy] <= 0) * -10 * u[0, 1:dimx, 1:dimy]
        + u[1, 1:dimx, 1:dimy]
        + u[2, 1:dimx, 1:dimy],
        0,
        255,
    )

    surf = pygame.surfarray.make_surface(data)
    display.blit(
        pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), offset
    )


def draw_text(display, fps, tick):
    global font

    text_surface = font.render(
        "2D Wave Equation - Explicit Euler (Radiating Boundary Conditions)",
        True,
        (0, 0, 40),
    )
    display.blit(text_surface, (5, 5))

    text_surface = font.render(
        f"FPS: {fps:.1f}; t={tick*ts:.2f} s; area={dimx*hs}x{dimy*hs} m",
        True,
        (0, 0, 40),
    )
    display.blit(text_surface, (5, dimy * cellsize - 20))


def main():
    global font

    pygame.init()
    pygame.font.init()

    font = pygame.font.SysFont("Consolas", 15)
    display = pygame.display.set_mode((dimx * cellsize, dimy * cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    create_arrays()
    u = set_initial_conditions()

    image1data = np.zeros((dimx, dimy, 3), dtype=np.uint8)

    tick = 0
    last_tick = 0
    fps = 0
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        tick = tick + 1

        current_time = time.time()
        if current_time - start_time > 0.5:
            fps = (tick - last_tick) / (current_time - start_time)
            start_time = time.time()
            last_tick = tick

        update(u, 3)
        draw_waves(display, u, image1data, (0, 0))
        draw_text(display, fps, tick)

        pygame.display.update()


if __name__ == "__main__":
    main()
