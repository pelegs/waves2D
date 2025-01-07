import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

N = 500
arr = np.random.normal(0, 1, size=(N, N))
arr_blur = gaussian_filter(arr, sigma=10)

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(arr_blur)
plt.show()
